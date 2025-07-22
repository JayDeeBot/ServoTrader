"""
live_crypto_trading_env.py

A custom OpenAI Gym environment for reinforcement learning-based live cryptocurrency trading.

This environment connects to a real-world trading backend (e.g. Binance via ServoTraderBinance)
and feeds live OHLCV market data to a pre-trained agent (e.g. PPO). The agent learns and adapts
based on real market execution outcomes, managing a single asset position (long only).

Features:
- Discrete action space: {0: Hold, 1: Buy, 2: Sell}
- Real-time OHLCV data via callback function
- Reward shaping:
    - Realized profit after sells
    - Dense unrealized PnL during holds
    - Optional transaction penalties
- Compatible with any Gym-compliant RL agent
- Stateless observation access for continuous streaming

Dependencies:
- gymnasium
- numpy
- pandas

Author: Jarred Deluca
Created: 2025
License: MIT
"""

# envs/live_crypto_trading_env.py

def slope_func(x):
    """
    Computes the slope of a linear regression line fitted to the input array `x`.

    This function is intended to be used with pandas `.rolling().apply(...)`
    to calculate the short-term price trend (i.e., upward or downward movement)
    over a specified rolling window.

    Args:
        x (array-like): A 1D array of numeric values (e.g., closing prices)

    Returns:
        float: The slope of the best-fit line through the data points.
            Positive slope = upward trend, negative slope = downward trend.
    """
    return linregress(np.arange(len(x)), x).slope

import gymnasium as gym
from gymnasium import spaces # Necessary for specifying valid actions & observations
import numpy as np
import pandas as pd
import os
from datetime import datetime
import json
from scipy.stats import linregress
import time
from crypto_database_init import CryptoDatabaseInitialiser

class LiveCryptoTradingEnv(gym.Env):
    """
    A Gym environment for real-time cryptocurrency trading with a trained reinforcement learning agent.

    This environment interfaces with a live trading backend (e.g. Binance via ServoTraderBinance) to execute
    real market buy/sell/hold actions and receive updated observations through a user-supplied callback.
    
    Designed to support:
    - Online policy execution with PPO or similar agents
    - Reward calculation from realized and unrealized PnL
    - Real-world trading with a single crypto asset (e.g., BTCUSDT)

    Action Space:
        0: HOLD
        1: BUY
        2: SELL

    Observation:
        Live vector of features (e.g., OHLCV + indicators) from user callback

    Reward:
        - Realized profit on sells
        - Small dense PnL while holding
        - Optional transaction cost penalties

    This class assumes trading a single crypto at a time and supports continuous streaming inference,
    ideal for low-latency real-world deployments.
    """
    metadata = {'render.modes': ['human']}

    
    def __init__(self, crypto_codes, episode_timeout=30):
        super(LiveCryptoTradingEnv, self).__init__()
        """
        Constructor for LiveCryptoTradingEnvironemnt.

        Args: 
            crypto_codes: 100 cryptos we are observing
            episode_timeout: max steps possible before each episode times out
        """

        self.crypto_codes = crypto_codes # Codes for the 100 Cryptos we are observing
        self.timeout_steps = episode_timeout # The maximum length in minutes that an episode will be allowed
        self.current_step = 0 # Time step of the current episode
        self.start_index = 0 # The first step index of each episode
        self.active_crypto_index = None # Index of crypto purchased in the current episode
        self.buy_price = 0.0 # Buy price of the current episode
        self.cash_balance = 1.0  # Start with $1.00 virtual capital
        self.portfolio_value = self.cash_balance # Total portfolio value at each time step
        self.break_even_steps = 0 # Member to track how long we've been near break-even after a buy has been made
        self.history_window = 1  # Number of past timesteps to include in observation
        self.feature_window = 30 # Window used for computing additional features when pre-processing data

        # Store symbol list, preprocess raw dataframe
        self.crypto_codes = sorted(crypto_codes)

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
        obs_len = self.history_window * self.num_cryptos * self.features_per_crypto + 2 + self.num_cryptos # Calculate the length of the observation
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
        Computes and incorporates the following additional features:
            - Recent return
            - Volatility
            - Price position
            - Volume surge
            - Trend slope 
            - Moving avg
        
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
        self.num_timesteps = raw_df.groupby('symbol').size().min() # Ensures all cryptos have equal timesteps — truncates to the shortest to maintain uniform shape

        feature_cols = ['open', 'high', 'low', 'close', 'vwap', 'volume', 'count'] # Columns to extracts
        added_features = ['recent_return', 'volatility', 'price_position', 'volume_surge', 'trend_slope', 'moving_avg'] # Additional features to compute
        all_features = feature_cols + added_features # Total features (13)
        self.features_per_crypto = len(all_features) # We have 13 features - Open, High, Low, Close, VWap, Volume & Count (base) + recent return, volatility, price position, volume surge, trend slope, moving avg
        tensor = np.zeros((self.num_timesteps, self.num_cryptos, self.features_per_crypto), dtype=np.float32) # Preallocate tensor: [timesteps, cryptos, features]

        for i, symbol in enumerate(self.crypto_codes): # Loop through each crypto symbol
            df_symbol = raw_df[raw_df['symbol'] == symbol].head(self.num_timesteps) # Select the first N rows for this symbol
            df_symbol = df_symbol[feature_cols].astype(float) # Convert features to float
            
            # Fill prices with forward-fill then back-fill
            df_symbol[['open', 'high', 'low', 'close', 'vwap']] = df_symbol[['open', 'high', 'low', 'close', 'vwap']].ffill()
            df_symbol[['open', 'high', 'low', 'close', 'vwap']] = df_symbol[['open', 'high', 'low', 'close', 'vwap']].bfill()

            # Fill volume and count with 0
            df_symbol[['volume', 'count']] = df_symbol[['volume', 'count']].fillna(0)

            # --- Add engineered historical features ---
            window = self.feature_window

            # 1. Recent return (as percent change over window)
            df_symbol['recent_return'] = df_symbol['close'].pct_change(periods=window).fillna(0)

            # 2. Historical volatility (std dev over window)
            df_symbol['volatility'] = df_symbol['close'].rolling(window).std().fillna(0)

            # 3. Price position in recent range
            high = df_symbol['high'].rolling(window).max()
            low = df_symbol['low'].rolling(window).min()
            df_symbol['price_position'] = ((df_symbol['close'] - low) / (high - low + 1e-6)).fillna(0)

            # 4. Volume surge index
            avg_volume = df_symbol['volume'].rolling(window).mean()
            df_symbol['volume_surge'] = (df_symbol['volume'] / (avg_volume + 1e-6)).fillna(0)

            # 5. Trend slope (via linear regression)
            df_symbol['trend_slope'] = df_symbol['close'].rolling(window).apply(slope_func, raw=False).fillna(0)

            # 6. Moving average of close
            df_symbol['moving_avg'] = df_symbol['close'].rolling(window).mean().bfill()
            
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

            # Engineered features: min-max scale
            for col in added_features:
                min_val, max_val = df_symbol[col].min(), df_symbol[col].max()
                range_val = max_val - min_val
                df_symbol[col] = (df_symbol[col] - min_val) / (range_val if range_val != 0 else 1.0)


            # Populate the tensor slice for this crypto - Final shape [T, F]
            tensor[:, i, :] = df_symbol[all_features].to_numpy() # Fill tensor with normalized data and additional features for symbol[i]

        return tensor # Return tensor (3 Dimensions) containing all data with shape [Timesteps, Cryptos, Features] and type np Float32

    def reset(self):
        """
        Initializes the environment state for a new live trading session.

        Returns:
            tuple: (initial observation, info dict including action mask)
        """
        self.active_crypto_index = None
        self.buy_price = 0.0

        # Refresh balance and portfolio value from the live trader
        self.portfolio_value, self.cash_balance = self.trader.get_cash_balance() # Assume no position held yet

        self.current_step = int(time.time())  # Use Unix timestamp as a time marker (if needed)

        # Log the start of a live episode
        episode_record = {
            "type": "episode_start",
            "episode": int(self.episode_counter + 1),
            "start_timestamp": datetime.now().isoformat(),
            "cash_balance": self.cash_balance
        }
        self.episode_log.append(episode_record) # Append the episode log

        obs = self.get_observation() # Grab the initial observation
        return obs, {"action_mask": self._get_action_mask()} # Return the initial observation - use the action mask to mask out illegal moves


    def _get_observation(self):
        """
        Returns the observation for the current step as a flat 1D vector.
        Downloads live OHLCV data from Kraken, preprocesses it into a tensor,
        and appends features: time remaining, profit %, and held crypto one-hot.

        Returns:
            np.ndarray: Full flattened observation vector.
        """
        # Step 1: Pull live data from Kraken using CryptoDatabaseInitialiser logic
        fetcher = CryptoDatabaseInitialiser(
            csv_path=self.csv_path,
            json_path=self.json_path,
            yaml_path=self.yaml_path
        )

        interval = fetcher.params.get("loop_interval_minutes", 1)
        desired_lines = self.history_window
        raw_data = []

        for symbol in self.crypto_codes:
            df = fetcher.fetch_crypto_data(symbol, interval, desired_lines)
            df["symbol"] = symbol
            raw_data.append(df)

        df_full = pd.concat(raw_data, ignore_index=True)
        tensor = self.preprocess_data(df_full)  # Shape: [T, N, F]
        obs_window = tensor[-self.history_window:]  # Shape: [window, cryptos, features]
        self.data = tensor  # Cache for use in reward calc if needed

        obs_window = np.nan_to_num(obs_window, nan=0.0, posinf=1e6, neginf=-1e6)
        obs_vector = obs_window.flatten()

        # --- Additional features ---
        # 1. Current profit (if holding)
        if self.active_crypto_index is not None and self.buy_price > 0:
            current_price = tensor[-1, self.active_crypto_index, 3]  # Latest close price
            current_profit = (current_price - self.buy_price) / self.buy_price
        else:
            current_profit = 0.0

        # 2. One-hot encoding for currently held crypto
        held_one_hot = np.zeros(self.num_cryptos + 1, dtype=np.float32)
        index = self.active_crypto_index + 1 if self.active_crypto_index is not None else 0
        held_one_hot[index] = 1.0

        # Combine final observation
        extra_features = np.concatenate([
            np.array([current_profit], dtype=np.float32),
            held_one_hot
        ])
        full_obs = np.concatenate([obs_vector, extra_features])

        # Validate shape and clean unexpected values
        if np.any(np.isnan(full_obs)) or np.any(np.isinf(full_obs)):
            print(f"Warning: NaNs/Infs in observation at step {self.current_step}")
            full_obs = np.nan_to_num(full_obs)

        assert full_obs.shape == self.observation_space.shape, \
            f"Observation shape mismatch: expected {self.observation_space.shape}, got {full_obs.shape}"

        return full_obs
    
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
                # BUY CRYPTO with ServoTrader
                # Save BUY PRICE
                reward = self._calculate_reward(0.0, "BUY", price_series=self._get_price_series(), trade_executed=True)
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

                if abs(profit) < 0.001: # If the profit is near 0
                    self.break_even_steps += 1 # Increment the break_even_steps up - assists discouraging break-even trades
                else: # If we have made positive/negative profit
                    self.break_even_steps = 0  # Reset the break_even_steps

                reward = self._calculate_reward(profit, "HOLD") # Calculate the reward at this time step
            else: # If we are not holding a crypto then give a small negative reward - teaches the agent the legal moves
                reward = -0.1  # Penalty: holding nothing

        # If the Action if Sell - num_cryptos + 1 = Sell
        elif action == self.num_cryptos + 1:  # Sell
            if self.active_crypto_index is not None: # Check whether we are holding a crypto - prevents double selling
                # Sell with ServoTrader
                # Save the sell price
                sell_price = 0
                if self.buy_price is None or self.buy_price == 0 or np.isnan(self.buy_price): # Calculate the total profit
                    profit = 0  # or handle however you prefer (e.g., skip trade)
                else:
                    profit = (sell_price - self.buy_price) / self.buy_price
                self.portfolio_value *= (1 + profit) # Calculate the final portfolio value
                reward = self._calculate_reward(profit, "SELL") # Calculate the sell reward
            else: # If we are not holding a crypto then give a small negative reward - teaches the agent the legal moves
                reward = -0.1  # Penalty: nothing to sell

        self.current_step += 1 # Increment the step count

        # If we have exceeded the timout steps for this episode
        if self.current_step - self.start_index >= self.timeout_steps: # Check whether we have exceeded the timeout steps for this episode
            done = True # Reset done flag
            if self.active_crypto_index is not None: # Check whether there is an active crypto
                # Sell with ServoTrader
                # Save the sell price
                final_price = 0
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

        # Compute legal action mask
        action_mask = np.zeros(self.action_space.n, dtype=bool)
        if self.active_crypto_index is None:
            action_mask[1:self.num_cryptos + 1] = True  # Buy actions
        else:
            action_mask[0] = True  # Hold
            action_mask[self.num_cryptos + 1] = True  # Sell

        info["action_mask"] = action_mask

        return obs, reward, terminated, truncated, info # Return the current observation, current/total reward & done flag

    def _calculate_reward(self, profit, action, dense=False, price_series=None, trade_executed=False):
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

            if action == "BUY":
                # No reward for buy action
                reward = 0

            elif action == "HOLD":
                # --- Non-linear reward scaling ---
                if profit > 0: # If the profit is positive give positive rewards
                    # reward = profit - Linear for mid-trade rewards
                    profit
                elif profit < 0: # If the profit is negative give negative rewards
                    # reward = -abs(profit) - Linear for mid-trade rewards
                    reward = -abs(profit)
                else: # If there is no profit (break-even) give negative rewards
                    # Penalize holding a break-even position
                    reward = -0.5 * self.break_even_steps # Calculate the reward = -0.5 * break_even_steps (decreases by -5 for every step we continuously break-even)

            elif action == "Sell":
                # --- Non-linear reward scaling ---
                if profit > 0: # If the profit is positive give positive rewards
                    # reward = profit ** 2 # Calculate the reward = profit ^ 2 - Quadratic scaling to encourage big profits
                    reward = profit ** 2
                elif profit < 0: # If the profit is negative give negative rewards
                    # reward = -abs(profit) ** 2 # Calculate the rewards = -| profit | ^ 2 - Quadratic scaling to discourage big losses
                    reward = -abs(profit) ** 2
                else: # If there is no profit (break-even) give negative rewards
                    # Penalize holding a break-even position
                    reward = -0.5 * self.break_even_steps # Calculate the reward = -0.5 * break_even_steps (decreases by -5 for every step we continuously break-even)

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
    
    def _get_action_mask(self):
        """
        Generates a boolean mask indicating which actions are currently legal.

        Legal actions depend on the agent's current portfolio state:
            - If no crypto is held (active_crypto_index is None):
                → Only buy actions (indices 1 to num_cryptos) are legal.
            - If a crypto is currently held:
                → Only hold (index 0) and sell (index num_cryptos + 1) are legal.

        Returns:
            np.ndarray (bool): A 1D boolean array where True indicates a legal action.
        """
        # Initialize all actions as illegal
        mask = np.zeros(self.action_space.n, dtype=bool)

        if self.active_crypto_index is None:
            # No crypto held: enable only buy actions (1 to num_cryptos)
            mask[1:self.num_cryptos + 1] = True
        else:
            # Crypto held: enable only hold (0) and sell (num_cryptos + 1)
            mask[0] = True
            mask[self.num_cryptos + 1] = True

        return mask
