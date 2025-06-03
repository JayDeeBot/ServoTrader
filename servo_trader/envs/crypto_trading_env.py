"""
crypto_trading_env.py

A custom OpenAI Gym environment for reinforcement learning-based cryptocurrency trading.

This environment simulates a trading agent operating on a selection of 100 cryptocurrencies
using historical OHLCV (Open, High, Low, Close, VWAP, Volume, Count) data. The agent is tasked
with learning an optimal trading strategy through Proximal Policy Optimization (PPO) or other 
RL algorithms.

Features:
- Discrete action space: {0: Hold, 1-100: Buy crypto[i-1], 101: Sell}
- Normalized multi-symbol OHLCV data as observations
- Reward engineering with:
    - Non-linear profit/loss scaling
    - Volatility-aware (Sharpe-style) penalties
    - Penalties for holding break-even positions
    - Transaction cost modeling
- Per-episode trading loop with optional timeouts
- Structured for simulated training and future live deployment

Dependencies:
- gym
- numpy
- pandas

Author: Jarred Deluca
Created: 2025
License: MIT
"""

# envs/crypto_trading_env.py

import gym
import numpy as np
import pandas as pd
from gym import spaces # Necessary for specifying valid actions & observations

class CryptoTradingEnv(gym.Env):
    """
    Custom Gym environment for PPO-based crypto trading with Buy, Sell, Hold actions.
    """
    metadata = {'render.modes': ['human']}

    # Constructor
    # Inits: data, crypto_codes, timeout_steps, current_step, start_index, active_crypto_index, buy_price, hold_duration, 
    # cash_balance, portfolio_value, crypto_codes, data, num_timesteps
    # Arguments: data = historical data, crypto_codes = 100 cryptos we are observing, episode_timeout = max steps possible before each episode times out
    def __init__(self, data, crypto_codes, episode_timeout=15):
        super(CryptoTradingEnv, self).__init__()

        self.data = data # Historical Data used for training
        self.crypto_codes = crypto_codes # Codes for the 100 Cryptos we are observing
        self.timeout_steps = episode_timeout # The maximum length in minutes that an episode will be allowed
        self.current_step = 0 # Time step of the current episode
        self.start_index = 0 # The first step index of each episode
        self.active_crypto_index = None # Index of crypto purchased in the current episode
        self.buy_price = 0.0 # Buy price of the current episode
        self.hold_duration = 0 # Duration crypto has been held for (steps)
        self.cash_balance = 1.0  # Start with $1.00 virtual capital
        self.portfolio_value = self.cash_balance # Total portfolio value at each time step
        self.break_even_steps = 0 # Member to track how long we've been near break-even after a buy has been made

        # Store symbol list, preprocess raw dataframe
        self.crypto_codes = sorted(crypto_codes)
        self.data = self.preprocess_data(data)  # Now self.data is [T, N, F] tensor

        self.num_timesteps, self.num_cryptos, self.features_per_crypto = self.data.shape # Use the total number of timesteps available in historical data

        # Action Space:
        # 0 = Hold
        # 1 to 100 = Buy symbol[i-1]
        # 101 = Sell
        self.action_space = spaces.Discrete(1 + self.num_cryptos + 1)

        # Observation Space: 
        # Recent OHLCV (Open, High, Low, Close & Volume) bars for 100 cryptos
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(self.num_cryptos * self.features_per_crypto,),
            dtype=np.float32
        )

    # Method preprocess_data: Preprocess the raw crypto data from csv into 3D tensor
    # Normalises the features (min/max scaling for price & log min/max scaling for volume/count) & Fills in Na's with forward/backward fill
    # Arguments: raw_df = The raw crypto data (csv format)
    # Updates: crypto_codes, num_cryptos, features_per_crypto, num_timesteps
    # Returns: 3D Tensor containing all data with shape [Timesteps, Cryptos, Features] and type np Float32
    def preprocess_data(self, raw_df):
        """
        Convert the stacked CSV format into a 3D tensor:
        [timesteps, symbols, features]
        """
        self.crypto_codes = sorted(raw_df['symbol'].unique()) # Sort the crypto codes in acsending order to ensure consistent ordering
        self.num_cryptos = len(self.crypto_codes) # Save the total number of codes
        self.features_per_crypto = 7 # We have 7 features - Open, High, Low, Close, VWap, Volume & Count
        self.num_timesteps = raw_df.groupby('symbol').size().min() # Ensures all cryptos have equal timesteps — truncates to the shortest to maintain uniform shape

        tensor = np.zeros((self.num_timesteps, self.num_cryptos, self.features_per_crypto), dtype=np.float32) # Preallocate tensor: [timesteps, cryptos, features]

        feature_cols = ['open', 'high', 'low', 'close', 'vwap', 'volume', 'count'] # Columns to extracts

        for i, symbol in enumerate(self.crypto_codes): # Loop through each crypto symbol
            df_symbol = raw_df[raw_df['symbol'] == symbol].head(self.num_timesteps) # Select the first N rows for this symbol
            df_symbol = df_symbol[feature_cols].astype(float) # Convert features to float
            
            # Fill prices with ffill then bfill
            df_symbol[['open', 'high', 'low', 'close', 'vwap']] = df_symbol[['open', 'high', 'low', 'close', 'vwap']].fillna(method='ffill')
            df_symbol[['open', 'high', 'low', 'close', 'vwap']] = df_symbol[['open', 'high', 'low', 'close', 'vwap']].fillna(method='bfill')

            # Fill volume and count with 0
            df_symbol[['volume', 'count']] = df_symbol[['volume', 'count']].fillna(0)
            
            # --- Min-Max scaling for price features ---
            for col in ['open', 'high', 'low', 'close', 'vwap']: # Loop through price columns
                min_val = df_symbol[col].min() # Extract the minimum value
                max_val = df_symbol[col].max() # Extract the maximum value
                range_val = max_val - min_val if max_val != min_val else 1.0  # Calculate the range - prevent divide-by-zero
                df_symbol[col] = (df_symbol[col] - min_val) / range_val # Normalise, feature = (value - minimum) / range

            # --- Log + Min-Max scaling for volume/count ---
            for col in ['volume', 'count']: # Loop through volume & count columns
                df_symbol[col] = np.log1p(df_symbol[col])  # Convert values to log(1 + x) to keep 0 valid
                min_val = df_symbol[col].min() # Extract the minimum value
                max_val = df_symbol[col].max() # Extract the maximum value
                range_val = max_val - min_val if max_val != min_val else 1.0 # Calculate the logarithmic range - prevent divide-by-zero
                df_symbol[col] = (df_symbol[col] - min_val) / range_val # Normalise, feature = (log(1 + value) - log_minimum) / log_range

            tensor[:, i, :] = df_symbol.to_numpy() # Fill tensor with normalized data for symbol[i]

        return tensor # Return tensor (3 Dimensions) containing all data with shape [Timesteps, Cryptos, Features] and type np Float32

    # Method reset: starts a new episode for training & initialises the environment state
    # Updates: start_index, current_step, active_crypto_index, buy_price, hold_duration, cash_balance, portfolio_value
    # Returns: Initial observation - observation at the current step
    def reset(self):
        self.start_index = np.random.randint(0, self.num_timesteps - self.timeout_steps - 1) # Randomly picks a starting timestep in the historical dataset
        self.current_step = self.start_index # Set the internal time pointer to the start index
        self.active_crypto_index = None # Reset the active crypto to none
        self.buy_price = 0.0 # Reset the buy price to 0
        self.hold_duration = 0 # Reset the hold duration to 0
        self.cash_balance = 1.0 # Reset the cash balance to 1
        self.portfolio_value = 1.0 # Reset the portfolio value to 1
        return self._get_observation() # Return the initial observation - observation at the current step

    # Method _get_observation: Returns the state vector that PPO will use as the observation at the current step - input for the neural network policy
    # Returns: Current observation (1D vector - Shape [num_cryptos*features_per_crypto])
    def _get_observation(self):
        """
        Return the observation for current step as a flat 1D vector.
        """
        obs = self.data[self.current_step] # Grab all crypto data at the current timestep - 2D Matrix Shape [num_cryptos, features_per_crypto]
        return obs.flatten() # Flatten the matrix into a single 1D vector expected by PPO - Shape [num_cryptos*features_per_crypto]

    # Method step: Defines how the agent interacts with the world at each time step
    # Arguments: action = current action
    # Returns: Current observation, current rewards (dense) & episode status
    # Updates: hold_duration, actibe_crypto_index, buy_price, portfolio_value, current_step
    def step(self, action):
        # Init the reward and done state to 0 & False respectively
        reward = 0
        done = False

        # If the action is Buy - Buy Action range is 1:num_cryptos (for #num_cryptos cryptos)
        if 1 <= action <= self.num_cryptos:  # Buy crypto[i]
            if self.active_crypto_index is None: # Check whether we already holding a crypto - prevents double buying
                self.active_crypto_index = action - 1 # Set active crypto index - Adjust index by -1 to match 0-based indexing
                self.buy_price = self.data[self.current_step, self.active_crypto_index, 3] # Capture buy price at the current step from the close feature (index 3)
                self.hold_duration = 0 # Reset the hold duration to 0
                reward = self._calculate_reward(0.0, dense=False, price_series=self._get_price_series(), trade_executed=True)
            else: # If we are already holding a crypto then give a small negative reward - teaches the agent the legal moves
                reward = -0.1  # Penalty: already holding
        
        # If the Action is Hold - 0 = Hold
        elif action == 0:  # Hold
            if self.active_crypto_index is not None: # If we are holding a crypto
                self.hold_duration += 1 # Increment up the hold duration by 1 (adds 1 per episode held)
                price_now = self.data[self.current_step, self.active_crypto_index, 3] # Save the current price
                profit = (price_now - self.buy_price) / self.buy_price # Calculate the current profit
                reward = self._calculate_reward(profit, dense=True, price_series=self._get_price_series()) # Calculate the reward at this time step (dense)

                if abs(profit) < 0.001: # If the profit is near 0
                    self.break_even_steps += 1 # Increment the break_even_steps up - assists discouraging break-even trades
                else: # If we have made positive/negative profit
                    self.break_even_steps = 0  # Reset the break_even_steps
            else: # If we are not holding a crypto then give a small negative reward - teaches the agent the legal moves
                reward = -0.1  # Penalty: holding nothing

        # If the Action if Sell - num_cryptos + 1 = Sell
        elif action == self.num_cryptos + 1:  # Sell
            if self.active_crypto_index is not None: # Check whether we are holding a crypto - prevents double selling
                sell_price = self.data[self.current_step, self.active_crypto_index, 3] # Take the price at the current step for the current crypto as the sell price
                profit = (sell_price - self.buy_price) / self.buy_price # Calculate the total profit
                self.portfolio_value *= (1 + profit) # Calculate the final portfolio value
                reward = self._calculate_reward(profit, dense=False, price_series=self._get_price_series(), trade_executed=True) # Calculate the final reward for the episode
                self._end_episode() # End the episode
                done = True # Reset done flag
            else: # If we are not holding a crypto then give a small negative reward - teaches the agent the legal moves
                reward = -0.1  # Penalty: nothing to sell

        self.current_step += 1 # Increment the step count

        # If we have exceeded the max steps for this episode or exceeded the max hold duration
        if self.current_step >= self.num_timesteps - 1 or self.hold_duration >= self.timeout_steps: # Check whether we have exceeded the max steps for this episode or the max hold duration
            done = True # Reset done flag
            if self.active_crypto_index is not None: # Check whether there is an active crypto
                final_price = self.data[self.current_step, self.active_crypto_index, 3] # Grab the final price
                profit = (final_price - self.buy_price) / self.buy_price # Calculate the final profit
                reward = self._calculate_reward(profit, dense=False, price_series=self._get_price_series(), trade_executed=True) # Calculate the total reward
                self._end_episode() # End the episode

        obs = self._get_observation() # Grab the current observation
        return obs, reward, done, {} # Return the current observation, current/total reward & done flag

    # Method _calculate_reward: Calculates the current/total rewards at the current/last step of the episode
    # Arguments: profit = the current/final profit, dense = bool to flag whether the reward calc will be dense or not (true = dense),
    # price_series = optional series of close prices, trade_executed = bool to flag whether a buy or sell has occured
    # Returns: Current/Total rewards
    def _calculate_reward(self, profit, dense=False, price_series=None, trade_executed=False):
        """
        Computes a risk-adjusted and trade-cost-aware reward signal.

        Parameters:
        - profit: % gain/loss from trade or current unrealized position
        - dense: whether this is an intermediate step or terminal sell
        - price_series: optional Series of close prices for volatility estimation
        - trade_executed: whether a Buy or Sell just occurred (for applying trading cost)
        """

        # --- Volatility-adjusted Sharpe-style bonus/penalty ---
        risk_penalty = 0 # Init risk penalty as 0
        if price_series is not None and len(price_series) >= 10: # If price series is set
            returns = price_series.pct_change().fillna(0) # Compute the percentage change from one price to the next and convert Na's to 0
            volatility = returns.std() # Calculate the volatility = standard deviation of returns
            if volatility > 0: # If the volatility if greater than 0
                sharpe_like = profit / volatility # Calculate the Sharpe-style penalty = profit / volatility
                risk_penalty = -abs(sharpe_like) * 0.5  # adjust strength here

        # --- Non-linear reward scaling ---
        if profit > 0: # If the profit is positive give positive rewards
            reward = profit ** 2 * 100 if not dense else profit * 10 # Calculate the reward = profit ^ 2 * 100 - Quadratic scaling to encourage big profits
        elif profit < 0: # If the profit is negative give negative rewards
            reward = -abs(profit) ** 2 * 100 if not dense else -abs(profit) * 10 # Calculate the rewards = -| profit | ^ 2 * 100 - Quadratic scaling to discourage big losses
        else: # If there is no profit (break-even) give negative rewards
            if dense: # If we are calculating dense rewards
                # Penalize holding a break-even position
                reward = -0.5 * self.break_even_steps # Calculate the reward = -0.5 * break_even_steps (decreases by -5 for every step we continuously break-even)
            else: # If we are not calculating dense rewards
                # Terminal break-even penalty includes duration-based cost
                base_penalty = -2 # Init base penalty
                time_penalty = -0.5 * self.break_even_steps # Calculate a time penalty for the total amount of steps we broke-even
                reward = base_penalty + time_penalty # Calculate the final reward

        # --- Penalty for trade execution (buy/sell only) ---
        if trade_executed:
            trade_cost = 0.002  # Hypthetical cost of making a trade = 0.2% of capital
            reward -= trade_cost * 100  # Scale it to the reward range and apply it to the reward

        # --- Combine with risk penalty (Sharpe-style) ---
        reward += risk_penalty

        return reward

    # Method _end_episode: Concludes the current episode
    # Updates: active_crypto_index, buy_price, hold_duration, cash_balance
    def _end_episode(self):
        self.active_crypto_index = None
        self.buy_price = 0.0
        self.hold_duration = 0
        self.cash_balance = self.portfolio_value

    # Method render: Prints the current step & current portfolio
    # Arguments: mode = rendering type ('human' for human interaction)
    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Portfolio: ${self.portfolio_value:.4f}") # Print current step & current portfolio

    # Method close: Used for when training is completed, resets/deletes the environment if necessary
    def close(self):
        pass # No logic required as yet
    
    # Method _get_price_series: Grabs the price series of the active crypto over a given window for volatility calculation
    # Arguments: window = the span of time-steps we want to calculate the volatility over
    # Returns: A panda series of close-prices - Length: [window]
    def _get_price_series(self, window=20):
        if self.active_crypto_index is None or self.current_step < window: # If there is an active crypto purchased or the current step is within the window
            return None # Return nohting
        return pd.Series(
            self.data[self.current_step - window:self.current_step, self.active_crypto_index, 3]
        ) # Return the close-prices of the active crypto over the given window