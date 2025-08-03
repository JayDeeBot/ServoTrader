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

class CryptoTradingEnv(gym.Env):
    """
    Custom Gym environment for PPO-based crypto trading with Buy, Sell, Hold actions.
    """
    metadata = {'render.modes': ['human']}

    
    def __init__(self, data, crypto_codes, episode_timeout=30):
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
        self.global_step = 0  # Tracks absolute timestep across chunks
        self.active_crypto_index = None # Index of current crypto
        self.buy_price = 0.0 # Buy price of the current crypo
        self.cash_balance = 1.0  # Start with $1.00 virtual capital
        self.portfolio_value = self.cash_balance # Total portfolio value at each time step
        self.break_even_steps = 0 # Member to track how long we've been near break-even after a buy has been made
        self.feature_window = episode_timeout # Window used for computing additional features when pre-processing data
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
        print("Preprocessing data...")
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
        self.start_index = self.current_step # Save the start index for the next episode
        self.active_crypto_index = None # Reset the active crypto to none
        self.buy_price = 0.0 # Reset the buy price to 0
        self.cash_balance = 1.0 # Reset the cash balance to 1
        self.portfolio_value = 1.0 # Reset the portfolio value to 1
        self.start_step = self.global_step  # Marks episode start

        # --- Log episode start step in structured JSON ---
        episode_record = {
            "type": "episode_start",
            "episode": int(self.episode_counter + 1), # Convert to native int
            "start_step": int(self.start_index), # Native int for step index
            "timestamp": datetime.now().isoformat() # ISO timestamp string
        }
        self.episode_log.append(episode_record) # Append to buffer for later writing

        return self._get_observation(), {"action_mask": self._get_action_mask()} # Return the initial observation - observation at the current step
        
    def _get_observation(self):
        """
        Returns the observation for current step as a flat 1D vector.
        
        Returns:
            full_obs: Current observation (1D vector - Shape [num_cryptos*features_per_crypto + 3])
        """
        # Extract a rolling window of crypto features to allow trend awareness
        # window_start = self.current_step - self.history_window + 1 # Compute the window start step
        # window_end = self.current_step + 1 # Compute the window end step
        # obs_window = self.data[window_start:window_end] # Grab all crypto data at the current timestep - Tensor Shape: [window, cryptos, features] 
        # Determine window boundaries
        window_start = self.current_step - self.history_window + 1
        window_end = self.current_step + 1

        # Clip window bounds to valid data range
        window_start = max(0, window_start)
        window_end = min(window_end, len(self.data))

        # Slice the available portion of the window
        obs_window = self.data[window_start:window_end]

        # Pad the beginning if the window is too short (start of dataset)
        pad_len = self.history_window - obs_window.shape[0]
        if pad_len > 0:
            pad_shape = (pad_len, self.num_cryptos, self.features_per_crypto)
            pad = np.zeros(pad_shape, dtype=np.float32)
            obs_window = np.concatenate((pad, obs_window), axis=0)

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
        profit = 0

        # If the action is Buy - Buy Action range is 1:num_cryptos (for #num_cryptos cryptos)
        if 1 <= action <= self.num_cryptos:  # Buy crypto[i]
            if self.active_crypto_index is None: # Check whether we already holding a crypto - prevents double buying
                self.active_crypto_index = action - 1 # Set active crypto index - Adjust index by -1 to match 0-based indexing
                self.buy_price = self.data[self.current_step, self.active_crypto_index, 3] # Capture buy price at the current step from the close feature (index 3)
                reward = self._calculate_reward(0.0, "BUY", price_series=self._get_price_series())
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
                reward = self._calculate_reward(profit, "HOLD") # Calculate the reward at this time step (dense)

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
                reward = self._calculate_reward(profit,"SELL") # Calculate the final reward for the episode
                self._end_episode() # End the episode
                done = True # Reset done flag
            else: # If we are not holding a crypto then give a small negative reward - teaches the agent the legal moves
                reward = -0.1  # Penalty: nothing to sell

        # If we have exceeded the timout steps for this episode
        if self.global_step - self.start_step >= self.timeout_steps: # Check whether we have exceeded the timeout steps for this episode
            done = True # Reset done flag
            if self.active_crypto_index is not None: # Check whether there is an active crypto
                symbol = self.crypto_codes[self.active_crypto_index] # Grab the active symbol we are selling
                final_price = self.data[self.current_step, self.active_crypto_index, 3] # Grab the final price
                # Calculate the final profit
                if self.buy_price is None or self.buy_price == 0 or np.isnan(self.buy_price):
                    profit = 0
                else:
                    profit = (final_price - self.buy_price) / self.buy_price
                self.portfolio_value *= (1 + profit) # Calculate the final portfolio value
                reward = self._calculate_reward(profit, "SELL") # Calculate the total reward
                 # --- ⬇ Add synthetic 'sell' step for timeout ---
                raw_price = None
                if symbol in self.raw_lookup:
                    try:
                        raw_price = float(self.raw_lookup[symbol].iloc[self.current_step]["close"])
                    except:
                        raw_price = None

                timeout_sell_step = {
                    "type": "step",
                    "offset": int(self.global_step - self.start_step),
                    "action": self.num_cryptos + 1,
                    "action_type": "sell",
                    "symbol": symbol,
                    "price": raw_price if raw_price is not None else float(final_price),
                    "profit_pct": profit * 100,
                    "reward": float(reward),
                    "raw_index": int(self.raw_lookup[symbol].index[self.current_step]) if symbol in self.raw_lookup else None
                }
                self.episode_log.append(timeout_sell_step)

                self._last_sell_context = {
                    "symbol": symbol,
                    "buy_price": self.buy_price,
                    "sell_price": final_price,
                    "profit": profit * 100,
                }
                self._end_episode() # End the episode

        # --- Periodically load a new 10k-row dataset every 10k steps ---
        if (self.global_step + 1) % 10000 == 0:
            chunk_index = (self.global_step // 10000) + 1
            chunk_path = f"/home/jarred/git/ServoTrader/data/split_10k_chunks_modern/{chunk_index:03}.csv"

            print(f"📦 Loading dataset chunk: {chunk_path}")
            if not os.path.exists(chunk_path):
                raise FileNotFoundError(f"❌ Chunk {chunk_index:03}.csv not found!")

            new_df = pd.read_csv(chunk_path)

            self.raw_df = new_df.copy()
            self.raw_df["symbol"] = self.raw_df["symbol"].str.strip()
            self.raw_lookup = {
                sym: df.reset_index(drop=True)
                for sym, df in self.raw_df.groupby("symbol")
            }

            self.data = self.preprocess_data(new_df)
            self.num_timesteps, self.num_cryptos, self.features_per_crypto = self.data.shape

            self.current_step = 0  # ✅ Reset local index
            print(f"✅ Chunk {chunk_index:03}.csv loaded and preprocessed")

        else:
            self.current_step += 1  # Continue stepping normally

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
        truncated = (self.global_step - self.start_step) >= self.timeout_steps # Truncated is true if the total steps of the episode exceeds the timeout steps
        info = {}
        # Log the reason the episode ended
        # if terminated:
        #     if truncated:
        #         print(f"[Env] Episode ended due to timeout at step {self.global_step - self.start_step} (relative to episode start)")
        #     else:
        #         print(f"[Env] Episode terminated due to sell at step {self.global_step - self.start_step} (relative to episode start)")

        # --- Log the behavior at the current step in structured JSON ---
        step_offset = int(self.global_step - self.start_step)  # Ensure it's native int for JSON

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
            episode_steps = int(self.global_step - self.start_step)
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

        self.global_step += 1  # Increment global step AFTER all logic

        return obs, reward, terminated, truncated, info # Return the current observation, current/total reward & done flag

    def _calculate_reward(self, profit, action, price_series=None):
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
            reward = 0 # Base reward
            # --- Volatility-adjusted Sharpe-style penalty ---
            if price_series is not None and len(price_series) >= 10:
                returns = price_series.pct_change().fillna(0)

                if returns.std() > 0:
                    volatility = returns.std()
                else:
                    volatility = 1e-6  # avoid div by zero

                # Penalize buys in high-volatility environments
                reward = -volatility * 0.1  # tune scaling factor

        elif action == "HOLD":
            # --- Non-linear reward scaling ---
            if profit > 0: # If the profit is positive give positive rewards
                # reward = profit - Linear for mid-trade rewards
                reward = profit
            elif profit < 0: # If the profit is negative give negative rewards
                # reward = -abs(profit) - Linear for mid-trade rewards
                reward = -abs(profit)
            else: # If there is no profit (break-even) give negative rewards
                # Penalize holding a break-even position
                reward = 0 # Give a zero reward for breaking even
                # reward = -0.5 * self.break_even_steps # Calculate the reward = -0.5 * break_even_steps (decreases by -5 for every step we continuously break-even)

        elif action == "SELL":
            # --- Non-linear reward scaling ---
            if profit > 0: # If the profit is positive give positive rewards
                # reward = profit ** 2 # Calculate the reward = profit ^ 2 - Quadratic scaling to encourage big profits
                reward = profit ** 2
            elif profit < 0: # If the profit is negative give negative rewards
                # reward = -abs(profit) ** 2 # Calculate the rewards = -| profit | ^ 2 - Quadratic scaling to discourage big losses
                reward = -abs(profit) ** 2
            else: # If there is no profit (break-even) give negative rewards
                # Penalize holding a break-even position
                reward = 0 # Give a zero reward for breaking even
                # reward = -0.5 * self.break_even_steps # Calculate the reward = -0.5 * break_even_steps (decreases by -5 for every step we continuously break-even)

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