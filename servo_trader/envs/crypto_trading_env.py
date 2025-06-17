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

import gymnasium as gym
from gymnasium import spaces # Necessary for specifying valid actions & observations
import numpy as np
import pandas as pd
import os
from datetime import datetime
import json

class CryptoTradingEnv(gym.Env):
    """
    Custom Gym environment for PPO-based crypto trading with Buy, Sell, Hold actions.
    """
    metadata = {'render.modes': ['human']}

    
    def __init__(self, data, crypto_codes, episode_timeout=15, raw_csv_path=None):
        super(CryptoTradingEnv, self).__init__()
        """
        Constructor for CryptoTradingEnvironemnt.

        Args: 
            data: historical data 
            crypto_codes: 100 cryptos we are observing
            episode_timeout: max steps possible before each episode times out
        """

        self.data = data # Historical Data used for training 
        self.raw_df = data.copy() # Save a raw copy of the historical data for logging
        # Pre process raw data for logging
        self.raw_df["symbol"] = self.raw_df["symbol"].str.strip()
        self.raw_lookup = {
            sym: df.reset_index(drop=True)
            for sym, df in self.raw_df.groupby("symbol")
        }
        self.crypto_codes = crypto_codes # Codes for the 100 Cryptos we are observing
        self.timeout_steps = episode_timeout # The maximum length in minutes that an episode will be allowed
        self.current_step = 0 # Time step of the current episode
        self.start_index = 0 # The first step index of each episode
        self.active_crypto_index = None # Index of crypto purchased in the current episode
        self.buy_price = 0.0 # Buy price of the current episode
        self.cash_balance = 1.0  # Start with $1.00 virtual capital
        self.portfolio_value = self.cash_balance # Total portfolio value at each time step
        self.break_even_steps = 0 # Member to track how long we've been near break-even after a buy has been made
        self.history_window = 5  # Number of past timesteps to include in observation

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
        # Time remaining: Normalized scalar: how much time is left in the episode
        # Current profit: Normalized percentage profit (0 if not holding)
        # Held crypto: Index of held crypto, or a special value if none held
        obs_len = self.history_window * self.num_cryptos * self.features_per_crypto + 2 + self.num_cryptos + 1 # Calculate the length of the observation
        low = np.zeros(obs_len, dtype=np.float32) # Set the lower bound
        low[-(self.num_cryptos + 1 + 1)] = -1.0  # profit can be negative
        high = np.ones(obs_len, dtype=np.float32) # Set the upper bound
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Internal Logging
        self.log_dir = "/home/jarred/git/ServoTrader/logs" # Directory containing the log
        os.makedirs(self.log_dir, exist_ok=True) # Ensure the log exists
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") # Record the start datetime
        self.log_path = os.path.join(self.log_dir, f"env_log_{timestamp}.jsonl") # Create a jsonl file for the log
        self.episode_log = []  # Will hold step-level logs temporarily
        self.total_profit_percent = 0  # For global tracking of the profit
        self.episode_counter = 0 # Stores the episode count

    
    def preprocess_data(self, raw_df):
        """
        Converts the stacked CSV format into a 3D tensor: [timesteps, symbols, features]
        Normalises the features (min/max scaling for price & log min/max scaling for volume/count) 
        & Fills in Na's with forward/backward fill
        
        Args: 
            raw_df: The raw crypto data (csv format)

        Updates: 
            crypto_codes 
            num_cryptos
            features_per_crypto
            num_timesteps

        Returns:
            tensor (Float32): 3D Tensor containing all data with shape [Timesteps, Cryptos, Features] and type np Float32
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
            
            # Fill prices with forward-fill then back-fill
            df_symbol[['open', 'high', 'low', 'close', 'vwap']] = df_symbol[['open', 'high', 'low', 'close', 'vwap']].ffill()
            df_symbol[['open', 'high', 'low', 'close', 'vwap']] = df_symbol[['open', 'high', 'low', 'close', 'vwap']].bfill()

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

    def reset(self, *, seed=None, options=None):
        """
        Starts a new episode for training & initialises the environment state.

        Args:
            *
            seed
            options

        Updates: 
            start_index
            current_step
            active_crypto_index
            buy_price cash_balance
            portfolio_value

        Returns: 
            Initial observation (observation at the current step)
        """
        super().reset(seed=seed)  # sets self.np_random if using Gym >=0.25
        # Optional: set random seed manually if needed
        if seed is not None:
            self.seed(seed)
        self.start_index = np.random.randint(self.history_window, self.num_timesteps - self.timeout_steps - 1) # Randomly picks a starting timestep in the historical dataset
        self.current_step = self.start_index # Set the internal time pointer to the start index
        self.active_crypto_index = None # Reset the active crypto to none
        self.buy_price = 0.0 # Reset the buy price to 0
        self.cash_balance = 1.0 # Reset the cash balance to 1
        self.portfolio_value = 1.0 # Reset the portfolio value to 1

        # --- Log episode start step in structured JSON ---
        episode_record = {
            "type": "episode_start",
            "episode": int(self.episode_counter + 1),   # Convert to native int
            "start_step": int(self.start_index),        # Native int for step index
            "timestamp": datetime.now().isoformat()     # ISO timestamp string
        }
        self.episode_log.append(episode_record)        # Append to buffer for later writing

        return self._get_observation(), {} # Return the initial observation - observation at the current step

    def _get_observation(self):
        """
        Returns the observation for current step as a flat 1D vector.
        
        Returns:
            full_obs: Current observation (1D vector - Shape [num_cryptos*features_per_crypto + 3])
        """
        # Extract a rolling window of crypto features to allow trend awareness
        window_start = self.current_step - self.history_window + 1 # Compute the window start step
        window_end = self.current_step + 1 # Compute the window end step
        obs_window = self.data[window_start:window_end] # Grab all crypto data at the current timestep - Tensor Shape: [window, cryptos, features] 
        obs_window = np.nan_to_num(obs_window, nan=0.0, posinf=1e6, neginf=-1e6) # Replace any NaNs/Infs with safe numerical values
        obs_vector = obs_window.flatten() # Flatten the matrix into a single 1D vector expected by PPO - Shape [num_cryptos*features_per_crypto*obs_window]

        # --- Additional features
        # 1. Time remaining (normalized)
        time_remaining = (self.timeout_steps - (self.current_step - self.start_index)) / self.timeout_steps
        time_remaining = np.clip(time_remaining, 0.0, 1.0)

        # 2. Current profit (if holding)
        if self.active_crypto_index is not None and self.buy_price > 0:
            current_price = self.data[self.current_step, self.active_crypto_index, 3]
            current_profit = (current_price - self.buy_price) / self.buy_price
        else:
            current_profit = 0.0

        # 3. Held crypto (index or -1 if none) - using one-hot encoding
        held_one_hot = np.zeros(self.num_cryptos + 1, dtype=np.float32)
        index = self.active_crypto_index + 1 if self.active_crypto_index is not None else 0
        held_one_hot[index] = 1.0

        # Append new features
        extra_features = np.concatenate((
            np.array([time_remaining, current_profit], dtype=np.float32),
            held_one_hot
        ))

        full_obs = np.concatenate([obs_vector, extra_features])
        # Safety check: warn if unexpected values slipped through
        if np.any(np.isnan(full_obs)) or np.any(np.isinf(full_obs)):
            print(f"Warning: NaNs/Infs in observation at step {self.current_step}")
            full_obs = np.nan_to_num(full_obs)
        # Assert shape matches what PPO expects
        assert full_obs.shape == self.observation_space.shape, \
            f"Observation shape mismatch: expected {self.observation_space.shape}, got {full_obs.shape}"
        return full_obs # Return the observation
    
    def step(self, action):
        """
        Defines how the agent interacts with the world at each time step.

        Args: 
            action: current action
                    
        Updates: 
            active_crypto_index
            buy_price
            portfolio_value
            current_step

        Returns: 
            obs: Current observation
            reward: current rewards (dense)
            terminated: episode status
            truncated: flag for when episode exceeds its time limit
            info: NA
        """
        # Init the reward and done state to 0 & False respectively
        reward = 0
        done = False

        # If the action is Buy - Buy Action range is 1:num_cryptos (for #num_cryptos cryptos)
        if 1 <= action <= self.num_cryptos:  # Buy crypto[i]
            if self.active_crypto_index is None: # Check whether we already holding a crypto - prevents double buying
                self.active_crypto_index = action - 1 # Set active crypto index - Adjust index by -1 to match 0-based indexing
                self.buy_price = self.data[self.current_step, self.active_crypto_index, 3] # Capture buy price at the current step from the close feature (index 3)
                reward = self._calculate_reward(0.0, dense=False, price_series=self._get_price_series(), trade_executed=True)
            else: # If we are already holding a crypto then give a small negative reward - teaches the agent the legal moves
                reward = -0.1  # Penalty: already holding
        
        # If the Action is Hold - 0 = Hold
        elif action == 0:  # Hold
            if self.active_crypto_index is not None: # If we are holding a crypto
                price_now = self.data[self.current_step, self.active_crypto_index, 3] # Save the current price
                if self.buy_price is None or self.buy_price == 0 or np.isnan(self.buy_price): # Calculate the current profit
                    profit = 0  # or np.nan or some fallback strategy
                else:
                    profit = (price_now - self.buy_price) / self.buy_price
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
                if self.buy_price is None or self.buy_price == 0 or np.isnan(self.buy_price): # Calculate the total profit
                    profit = 0  # or handle however you prefer (e.g., skip trade)
                else:
                    profit = (sell_price - self.buy_price) / self.buy_price
                self.portfolio_value *= (1 + profit) # Calculate the final portfolio value
                reward = self._calculate_reward(profit, dense=False, price_series=self._get_price_series(), trade_executed=True) # Calculate the final reward for the episode
                self._end_episode() # End the episode
                done = True # Reset done flag
            else: # If we are not holding a crypto then give a small negative reward - teaches the agent the legal moves
                reward = -0.1  # Penalty: nothing to sell

        self.current_step += 1 # Increment the step count

        # If we have exceeded the timout steps for this episode
        if self.current_step - self.start_index >= self.timeout_steps: # Check whether we have exceeded the timeout steps for this episode
            done = True # Reset done flag
            if self.active_crypto_index is not None: # Check whether there is an active crypto
                final_price = self.data[self.current_step, self.active_crypto_index, 3] # Grab the final price
                # Calculate the final profit
                if self.buy_price is None or self.buy_price == 0 or np.isnan(self.buy_price):
                    profit = 0
                else:
                    profit = (final_price - self.buy_price) / self.buy_price
                reward = self._calculate_reward(profit, dense=False, price_series=self._get_price_series(), trade_executed=True) # Calculate the total reward
                self._end_episode() # End the episode

        obs = self._get_observation() # Grab the current observation

        # --- Sanitize reward ---
        if np.isnan(reward) or np.isinf(reward):
            print(f"[Warning] Invalid reward encountered at step {self.current_step}: {reward}")
            reward = 0.0

        # --- Sanitize observation ---
        if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
            print(f"[Warning] Invalid observation at step {self.current_step}")
            obs = np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)

        terminated = done # Terminated & done will be the same for our application
        truncated = (self.current_step - self.start_index) >= self.timeout_steps # Truncated is true if the total steps of the episode exceeds the timeout steps
        info = {}
        # Log the reason the episode ended
        # if terminated:
        #     if truncated:
        #         print(f"[Env] Episode ended due to timeout at step {self.current_step - self.start_index} (relative to episode start)")
        #     else:
        #         print(f"[Env] Episode terminated due to sell at step {self.current_step - self.start_index} (relative to episode start)")

        # --- Log the behavior at the current step in structured JSON ---
        step_offset = int(self.current_step - self.start_index)  # Ensure it's native int for JSON

        # Initialize placeholders for price/profit values
        current_price = None       # Normalized close price
        profit = None              # Profit percentage
        raw_price = None           # Raw close price
        raw_buy_price = None       # Buy price from raw data
        raw_sell_price = None      # Sell price from raw data

        # If currently holding a crypto, compute prices and profit
        if self.active_crypto_index is not None:
            current_price = float(self.data[self.current_step, self.active_crypto_index, 3])  # Normalized close
            profit = ((current_price - self.buy_price) / self.buy_price) * 100 if self.buy_price else 0.0
            symbol = self.crypto_codes[self.active_crypto_index]
            if symbol in self.raw_lookup:
                raw_price = float(self.raw_lookup[symbol].iloc[self.current_step]["close"])  # Raw close price

        # Create the base JSON record for this step
        step_event = {
            "type": "step",
            "offset": step_offset,
            "action": int(action),  # Ensure action is native int
            "reward": float(reward)
        }

        # Add price and symbol info if holding a position
        if self.active_crypto_index is not None:
            step_event["symbol"] = symbol
            step_event["price"] = raw_price if raw_price is not None else current_price
            step_event["profit_pct"] = float(profit)

        # --- BUY action logging ---
        if 1 <= action <= self.num_cryptos:
            buy_symbol = self.crypto_codes[action - 1]
            if buy_symbol in self.raw_lookup:
                raw_buy_price = float(self.raw_lookup[buy_symbol].iloc[self.current_step]["close"])
            step_event["action_type"] = "buy"
            step_event["symbol"] = buy_symbol
            step_event["price"] = raw_buy_price if raw_buy_price is not None else float(self.buy_price)

        # --- SELL action logging ---
        elif action == self.num_cryptos + 1 and hasattr(self, "_last_sell_context"):
            ctx = self._last_sell_context
            sell_symbol = ctx["symbol"]
            if sell_symbol in self.raw_lookup:
                raw_sell_price = float(self.raw_lookup[sell_symbol].iloc[self.current_step]["close"])
            step_event["action_type"] = "sell"
            step_event["symbol"] = sell_symbol
            step_event["buy_price"] = float(ctx["buy_price"])
            step_event["sell_price"] = raw_sell_price if raw_sell_price is not None else float(ctx["sell_price"])
            step_event["profit_pct"] = float(ctx["profit"])
            del self._last_sell_context  # Cleanup context after logging

        # --- HOLD or fallback logging ---
        else:
            step_event["action_type"] = "hold" if action == 0 else "unknown"

        # --- Add raw_index for test alignment ---
        if step_event.get("symbol") and step_event["symbol"] in self.raw_lookup:
            raw_idx = int(self.raw_lookup[step_event["symbol"]].index[self.current_step])
            step_event["raw_index"] = raw_idx

        # Append the structured step log
        self.episode_log.append(step_event)

        # --- At episode end, log summary ---
        if done:
            self.episode_counter += 1
            episode_steps = int(self.current_step - self.start_index)
            percent_return = float((self.portfolio_value - 1.0) * 100)
            self.total_profit_percent += percent_return

            summary = {
                "type": "episode_end",
                "episode": self.episode_counter,
                "steps": episode_steps,
                "final_value": float(self.portfolio_value),
                "episode_return_pct": percent_return,
                "termination": "timeout" if truncated else "sell",
                "timestamp": datetime.now().isoformat()
            }

            if self.raw_df is not None and self.active_crypto_index is not None:
                final_sym = self.crypto_codes[self.active_crypto_index]
                if final_sym in self.raw_lookup:
                    summary["final_raw_close"] = float(self.raw_lookup[final_sym].iloc[self.current_step]["close"])

            self.episode_log.append(summary)

            # Persist JSONL to disk
            with open(self.log_path, "a") as f:
                for record in self.episode_log:
                    f.write(json.dumps(record) + "\n")

            self.episode_log = []  # Clear buffer for next episode

        return obs, reward, terminated, truncated, info # Return the current observation, current/total reward & done flag

    def _calculate_reward(self, profit, dense=False, price_series=None, trade_executed=False):
        """
        Computes a risk-adjusted and trade-cost-aware reward signal.

        Parameters:
            profit: % gain/loss from trade or current unrealized position
            dense: whether this is an intermediate step or terminal sell
            price_series: optional Series of close prices for volatility estimation
            trade_executed: whether a Buy or Sell just occurred (for applying trading cost)
        
        Returns: 
            reward: Current/total rewards
        """

        # --- Volatility-adjusted Sharpe-style bonus/penalty ---
        risk_penalty = 0 # Init risk penalty as 0
        if price_series is not None and len(price_series) >= 10: # If price series is set
            returns = price_series.pct_change().fillna(0) # Compute the percentage change from one price to the next and convert Na's to 0
            if returns.dropna().shape[0] > 1: # Calculate the volatility = standard deviation of returns
                volatility = returns.std()
            else:
                volatility = 0 # or some small default like 1e-8 to avoid div-by-zero elsewhere
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

        # Handle possible Nan Rewards
        if not np.isfinite(reward):
            reward = 0 # Give zero rewards if the value is Nan or infinite

        return reward

    def _end_episode(self):
        """
        Concludes the current episode.

        Updates: 
            active crypto index
            buy price
            cash balance
        """
        # Save sell log data before resetting state
        if self.active_crypto_index is not None:
            self._last_sell_context = {
                "symbol": self.crypto_codes[self.active_crypto_index],
                "buy_price": self.buy_price,
                "sell_price": self.data[self.current_step, self.active_crypto_index, 3],
                "profit": (
                    (self.data[self.current_step, self.active_crypto_index, 3] - self.buy_price)
                    / self.buy_price * 100 if self.buy_price else 0
                )
            }

        self.active_crypto_index = None
        self.buy_price = 0.0
        self.cash_balance = self.portfolio_value

    def render(self, mode='human'):
        """
        Prints the current step & current portfolio.

        Args: 
            mode = rendering type ('human' for human interaction)
        """
        print(f"Step: {self.current_step}, Portfolio: ${self.portfolio_value:.4f}") # Print current step & current portfolio

    def close(self):
        """
        Used for when training is completed, resets/deletes the environment if necessary
        """
        pass # No logic required as yet
    
    def _get_price_series(self, window=20):
        """
        Grabs the price series of the active crypto over a given window for volatility calculation.

        Args: 
            window: the span of time-steps we want to calculate the volatility over

        Returns: 
            A panda series of close-prices with length: [window]
        """
        if self.active_crypto_index is None or self.current_step < window: # If there is an active crypto purchased or the current step is within the window
            return None # Return nothing
        return pd.Series(
            self.data[self.current_step - window:self.current_step, self.active_crypto_index, 3]
        ) # Return the close-prices of the active crypto over the given window