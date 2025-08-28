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
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from servo_trader.servo_trader_binance import ServoTraderBinance
from servo_trader.crypto_database_init import CryptoDatabaseInitialiser
from threading import Lock

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

        # --- Define paths ---
        self.csv_path = "/home/jarred/git/ServoTrader/data/historical_crypto_data.csv"
        self.json_path = "/home/jarred/git/ServoTrader/servo_trader/config/crypto_codes.json"
        self.yaml_path = "/home/jarred/git/ServoTrader/servo_trader/config/params.yaml"

        # --- Create data output folder if needed ---
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)

        self.trader = ServoTraderBinance(self.yaml_path) # Init the trader object
        self.trader.liquify() # Completely liquify all assets before session begins

        # Declare data storage members for the observation space and logging - set all to none to begin with
        self.data = None
        self.raw_df = None
        self.raw_lookup = None

        self.crypto_codes = crypto_codes # Codes for the 100 Cryptos we are observing
        self.timeout_steps = episode_timeout # The maximum length in minutes that an episode will be allowed
        self.current_step = 0 # Time step of the current episode
        self.start_index = 0 # The first step index of each episode
        self.active_crypto_index = None # Index of crypto purchased in the current episode
        self.active_crypto_code = None # Code of crypto purchased in the current episode
        self.buy_order_id = None # The API buy order ID for the crypto purchased
        self.sell_order_id = None # The API sell order ID for the crypto sold
        self.buy_price = 0.0 # Buy price of the current episode
        self.sell_price = 0.0 # Sell price for the current episode
        self.cash_balance = self.trader.get_cash_balance()  # Fetch the available cash balance from Binance
        self.break_even_steps = 0 # Member to track how long we've been near break-even after a buy has been made
        self.feature_window = episode_timeout # Window used for computing additional features when pre-processing data
        self.history_window = 5  # Number of past timesteps to include in observation
        self.unusable_crypto_codes = []  # Track any codes that fail to fetch valid data
        self.avg_low_prices = {}  # Tracks average low prices for each code
        self.current_action = None # Tracks the current action for reporting back to the GUI

        # cadence / timing controls
        self.step_period_seconds = 60.0      # target period per step
        self.print_countdown = True          # print a live countdown while waiting
        self.skip_wait_on_terminate = False  # set True to skip waiting if episode just ended
        self.seconds_left = 0.0              # GUI can read this

        # Store symbol list, preprocess raw dataframe
        self.crypto_codes = sorted(crypto_codes)

        self.num_cryptos = len(self.crypto_codes) # Save the total number of cryptos in question
        # We are using 13 features for observation 
        # including open, high, low, close, vwap, volume, count (native to the database init class)
        # & recent_return, volatility, price_position, volume_surge, trend_slope, moving_avg (additional engineered features)
        self.features_per_crypto = 13 # Set total amount of features for calculating the size of the observation

        # Action Space:
        # 0              = Hold
        # 1 to num_cryptos = Buy symbol[i-1]
        # (1 + num_cryptos) = Sell
        # (2 + num_cryptos) = Not Buy (special skip option, only valid at episode start)
        self.action_space = spaces.Discrete(1 + self.num_cryptos + 2)

        # Observation Space: 
        # Recent 13 (see above) features for 100 cryptos
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
        self.total_profit_decimal = 0.0  # For global tracking of the profit
        self.episode_profit_decimal = 0.0 # For the current episode return
        self.episode_counter = 0 # Stores the episode count
        self.current_ep_profit_decimal = 0.0 # Stores the current profit of the episode at a hold step
        self.estimated_balance_usdt = 0.0 # Stores the current estimated balance

        # Manage log thread
        self._log_lock = Lock()
        os.makedirs(self.log_dir, exist_ok=True)
        # make sure the file exists
        open(self.log_path, "a").close()

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
        self.num_timesteps = self.feature_window # Set the number of steps to the feature window

        feature_cols = ['open', 'high', 'low', 'close', 'vwap', 'volume', 'count'] # Columns to extracts
        added_features = ['recent_return', 'volatility', 'price_position', 'volume_surge', 'trend_slope', 'moving_avg'] # Additional features to compute
        all_features = feature_cols + added_features # Total features (13)
        self.features_per_crypto = len(all_features) # We have 13 features - Open, High, Low, Close, VWap, Volume & Count (base) + recent return, volatility, price position, volume surge, trend slope, moving avg
        tensor = np.zeros((self.num_timesteps, self.num_cryptos, self.features_per_crypto), dtype=np.float32) # Preallocate tensor: [timesteps, cryptos, features]

        for i, symbol in enumerate(self.crypto_codes): # Loop through each crypto symbol
            df_symbol = raw_df[raw_df['symbol'] == symbol].tail(self.feature_window).reset_index(drop=True) # Grab all the rows for this symbol
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
        Initializes the environment state for a new live trading session.
        Returns: (obs, info)
        """
        try:
            super().reset(seed=seed)
        except Exception:
            pass

        # Clear position state
        self.active_crypto_index = None
        self.active_crypto_code = None
        self.buy_price = 0.0
        self.sell_price = 0.0
        self.episode_profit_decimal = 0.0
        self.break_even_steps = 0
        self.seconds_left = 0.0  # if you use the cadence timer

        # Refresh balance
        self.cash_balance = self.trader.get_cash_balance() # Assume no position held yet
        eq, _ = self.trader.get_estimated_balance_usdt()
        self.estimated_balance_usdt = float(eq)

        # Episode step counters
        self.current_step = 0
        self.start_index = 0

        # Log the start of a live episode
        episode_record = {
            "type": "episode_start",
            "episode": int(self.episode_counter + 1),
            "start_timestamp": datetime.now().isoformat(),
            "cash_balance": self.cash_balance
        }
        self._write_logline(episode_record) # Write to the log

        obs = self._get_observation() # Grab the initial observation
        self._refresh_crypto_codes() # Purge + replace invalid or risky crypto codes

        return obs, {"action_mask": self._get_action_mask()} # Return the initial observation - use the action mask to mask out illegal moves

    def _get_observation(self):
        """
        Returns the observation for the current step as a flat 1D vector.
        Downloads live OHLCV data from Binance, preprocesses it into a tensor,
        and appends features: time remaining, profit %, and held crypto one-hot.

        Returns:
            np.ndarray: Full flattened observation vector.
        """
        # Pull live data from Binance using CryptoDatabaseInitialiser logic
        database = CryptoDatabaseInitialiser(
            csv_path=self.csv_path,
            json_path=self.json_path,
            yaml_path=self.yaml_path
        )

        # Print any unusable (due to invalid data) crypto codes detected
        if database.unusable_crypto_codes:
            print(f"⚠️ Unusable crypto codes detected: {database.unusable_crypto_codes}")
            self.unusable_crypto_codes = database.unusable_crypto_codes # Save the unusable codes

        # Load the pulled data
        self.data = pd.read_csv('/home/jarred/git/ServoTrader/data/historical_crypto_data.csv')

        self.raw_df = self.data.copy() # Save a raw copy of the historical data for logging
        # Pre process raw data for logging
        self.raw_df["symbol"] = self.raw_df["symbol"].str.strip()
        self.raw_lookup = {
            sym: df.reset_index(drop=True)
            for sym, df in self.raw_df.groupby("symbol")
        }

        self.data = self.preprocess_data(self.data)  # Now self.data is [T, N, F] tensor

        # Extract a rolling window of crypto features to allow trend awareness
        window = self.history_window
        obs_window = self.data[-window:]  # Shape: [5, 100, 13]

        obs_window = np.nan_to_num(obs_window, nan=0.0, posinf=1e6, neginf=-1e6) # Replace any NaNs/Infs with safe numerical values
        obs_vector = obs_window.flatten() # Flatten the matrix into a single 1D vector expected by PPO - Shape [num_cryptos*features_per_crypto*obs_window]

        # --- Additional features
        # 1. Time remaining (normalized)
        time_remaining = (self.timeout_steps - (self.current_step - self.start_index)) / self.timeout_steps
        time_remaining = np.clip(time_remaining, 0.0, 1.0)

        # 2. Current profit (if holding)
        if self.active_crypto_index is not None and self.buy_price > 0:
            current_price = self.data[-1, self.active_crypto_index, 3]
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
        # --- Timing: start of step ---
        step_start_monotonic = time.monotonic()

        self.current_action = self._get_current_action(action) # Save the current action for GUI update

        # Init the reward and done state to 0 & False respectively
        reward = 0
        done = False

        ### ----- TIMEOUT EXCEEDED ----- ###
        if self.current_step - self.start_index >= self.timeout_steps: # Check whether timeout has been exceeded
            if self.active_crypto_index is not None: # Check if there is an active crypto
                self.sell_price = self._sell() # Attempt the sell and save the sell prince
                if self.sell_price is None: # Check if a sell proce was returned
                    # ⚠️ Sell failed — fallback to HOLD behavior
                    print("❌ Sell failed during timeout. Treating as HOLD.")
                    current_price = self.data[-1, self.active_crypto_index, 3] # Grab the current price of the active crypto
                    if self.buy_price is None or self.buy_price == 0 or np.isnan(self.buy_price): # Safety check for profit
                        profit = 0
                    else:
                        profit = (current_price - self.buy_price) / self.buy_price # Calculate the profit
                    self.current_ep_profit_decimal = profit # Save the current profit decimal for GUI
                    reward = self._calculate_reward(profit, "HOLD") # Calculate the reward (Hold)
                    # Log HOLD step (still holding after failed timeout sell)
                    self._log_step(
                        action=0,
                        action_type="hold",
                        reward=reward,
                        symbol=self.active_crypto_code,
                        price=current_price,
                        profit_pct=(profit * 100.0),
                    )
                else:
                    # ✅ Sell succeeded — handle as normal
                    if self.buy_price is None or self.buy_price == 0 or np.isnan(self.buy_price): # Safety check for profit
                        profit = 0
                    else:
                        profit = (self.sell_price - self.buy_price) / self.buy_price # Calculate the profit
                    reward = self._calculate_reward(profit, "SELL") # Calculate the reward (Sell)
                    # Log the sell info
                    self._log_step(
                        action=self.num_cryptos + 1,
                        action_type="sell",
                        reward=reward,
                        symbol=self.active_crypto_code,
                        price=self.sell_price,
                        profit_pct=(profit * 100.0),
                        buy_price=self.buy_price if self.buy_price else 0.0,
                        sell_price=self.sell_price,
                    )
                    done = True # Set the done flag to true - episode is overs

        ### ----- NOT BUY ACTION ----- ###
        # If the action is Not Buy - index = 2 + num_cryptos (special skip option)
        elif action == (2 + self.num_cryptos):  
            # End the episode immediately with a small fixed penalty
            reward = -0.01  # Penalty for skipping the episode
            done = True   # Episode ends due to Not Buy
            self._log_step(action=action, action_type="not_buy", reward=reward) # Log Not Buy action

        ### ----- BUY ACTION ----- ###
        # If the action is Buy - Buy Action range is 1:num_cryptos (for #num_cryptos cryptos)
        elif 1 <= action <= self.num_cryptos:  # Buy crypto[i]
            if self.active_crypto_index is None: # Check whether we already holding a crypto - prevents double buying
                self.active_crypto_index = action - 1 # Set active crypto index - Adjust index by -1 to match 0-based indexing
                self.active_crypto_code = self._get_buy_action_code(action) # Save the active crypto code
                self.buy_price = self._buy(self.active_crypto_code) # Execute the buy - BLOCKING CALL completes when buy goes through & save the buy price
                if self.buy_price is None:
                    print("❌ Buy failed. Treating as HOLD.")
                    self.active_crypto_index = None
                    self.active_crypto_code = None
                    reward = 0.0  # Nil penalty for error
                else:
                    reward = self._calculate_reward(0.0, "BUY", price_series=self._get_price_series())
                    # Log BUY (no profit_pct on buy; include step, action, symbol, price)
                    self._log_step(
                        action=action,
                        action_type="buy",
                        reward=reward,
                        symbol=self.active_crypto_code,
                        price=self.buy_price,
                    )
            else: # If we are already holding a crypto then give a small negative reward - teaches the agent the legal moves
                reward = -0.1  # Penalty: already holding
                print("ILLEGAL BUY MOVE")
                # self._log_step(action=action, action_type="buy", reward=reward) # Log illegal move
        
        ### ----- HOLD ACTION ----- ###
        # If the Action is Hold - 0 = Hold
        elif action == 0:  # Hold
            if self.active_crypto_index is not None: # If we are holding a crypto
                # current_price = self.data[-1, self.active_crypto_index, 3] # Save the current price
                current_price = float(self.raw_lookup[self.active_crypto_code].iloc[-1]["close"])  # Raw close price
                if self.buy_price is None or self.buy_price == 0 or np.isnan(self.buy_price): # Calculate the current profit
                    profit = 0  # or np.nan or some fallback strategy
                else:
                    profit = (current_price - self.buy_price) / self.buy_price
                self.current_ep_profit_decimal = profit # Save the current profit decimal for GUI
                reward = self._calculate_reward(profit, "HOLD") # Calculate the reward at this time step (dense)
                if abs(profit) < 0.001: # If the profit is near 0
                    self.break_even_steps += 1 # Increment the break_even_steps up - assists discouraging break-even trades
                else: # If we have made positive/negative profit
                    self.break_even_steps = 0  # Reset the break_even_steps
                # Log HOLD
                self._log_step(
                    action=0,
                    action_type="hold",
                    reward=reward,
                    symbol=self.active_crypto_code,
                    price=current_price,
                    profit_pct=(profit * 100.0),
                )
            else: # If we are not holding a crypto then give a small negative reward - teaches the agent the legal moves
                reward = -0.1  # Penalty: holding nothing
                print("ILLEGAL HOLD MOVE")
                # self._log_step(action=0, action_type="hold", reward=reward) # Log illegal action

        ### ----- SELL ACTION ----- ###
        # If the Action if Sell - num_cryptos + 1 = Sell
        elif action == self.num_cryptos + 1:  # Sell
            if self.active_crypto_index is not None: # Check whether we are holding a crypto - prevents double selling
                self.sell_price = self._sell() # Execute the sell - BLOCKING CALL completes when sell goes through & save the sell price
                if self.sell_price is None:
                     # ⚠️ Sell failed — fallback to HOLD behavior
                    print("❌ Sell failed during timeout. Treating as HOLD.")
                    # current_price = self.data[-1, self.active_crypto_index, 3]
                    current_price = float(self.raw_lookup[self.active_crypto_code].iloc[-1]["close"])  # Raw close price
                    if self.buy_price is None or self.buy_price == 0 or np.isnan(self.buy_price):
                        profit = 0
                    else:
                        profit = (current_price - self.buy_price) / self.buy_price
                    self.current_ep_profit_decimal = profit # Save the current profit decimal for GUI
                    reward = self._calculate_reward(profit, "HOLD")
                    # Log HOLD because we’re still in the position
                    self._log_step(
                        action=0,
                        action_type="hold",
                        reward=reward,
                        symbol=self.active_crypto_code,
                        price=current_price,
                        profit_pct=(profit * 100.0),
                    )
                    # Do NOT mark done or end episode
                else:
                    if self.buy_price is None or self.buy_price == 0 or np.isnan(self.buy_price): # Calculate the total profit
                        profit = 0  # or handle however you prefer (e.g., skip trade)
                    else:
                        profit = (self.sell_price - self.buy_price) / self.buy_price
                    reward = self._calculate_reward(profit,"SELL") # Calculate the final reward for the episode
                    # Log SELL with buy/sell price + profit_pct
                    self._log_step(
                        action=self.num_cryptos + 1,
                        action_type="sell",
                        reward=reward,
                        symbol=self.active_crypto_code,
                        profit_pct=(profit * 100.0),
                        buy_price=self.buy_price if self.buy_price else 0.0,
                        sell_price=self.sell_price,
                    )
                    done = True # Reset done flag
            else: # If we are not holding a crypto then give a small negative reward - teaches the agent the legal moves
                reward = -0.1  # Penalty: nothing to sell
                print("ILLEGAL SELL MOVE")
                # self._log_step(action=self.num_cryptos + 1, action_type="sell", reward=reward)

        terminated = done # Terminated & done will be the same for our application
        truncated = (self.current_step - self.start_index) >= self.timeout_steps # Truncated is true if the total steps of the episode exceeds the timeout steps
        info = {}
        # Log the reason the episode ended
        # if terminated:
        #     if truncated:
        #         print(f"[Env] Episode ended due to timeout at step {self.current_step - self.start_index} (relative to episode start)")
        #     else:
        #         print(f"[Env] Episode terminated due to sell at step {self.current_step - self.start_index} (relative to episode start)")


        # --- At episode end, log summary ---
        if done:
            self.episode_counter += 1 # Increment the episode counter
            # Calculate the episode return
            if self.buy_price is None or self.buy_price == 0 or np.isnan(self.buy_price):
                self.episode_profit_decimal = 0.0
            else:
                self.episode_profit_decimal = (self.sell_price - self.buy_price) / self.buy_price
            # Calculate the total profit percentage over all episodes in this session
            self.total_profit_decimal = (1.0 + self.total_profit_decimal) * (1.0 + self.episode_profit_decimal) - 1.0
            # Format the episode summary log
            summary = {
                "type": "episode_end",
                "episode": self.episode_counter,
                "episode_return_pct": 100 * self.episode_profit_decimal,
                "total_profit_pct": 100 * self.total_profit_decimal,
                "termination": "timeout" if truncated else "sell",
                "timestamp": datetime.now().isoformat()
            }
            self._write_logline(summary) # Write to the log

        # Compute legal action mask
        action_mask = np.zeros(self.action_space.n, dtype=bool)
        if self.active_crypto_index is None:
            action_mask[1:self.num_cryptos + 1] = True # Buy actions
            action_mask[2 + self.num_cryptos] = True # Not Buy (skip)
        else:
            action_mask[0] = True  # Hold
            action_mask[self.num_cryptos + 1] = True  # Sell

        # Fetch the estimated balance
        if (self.current_step % 3) == 0:  # update every ~3 minutes
            eq, _ = self.trader.get_estimated_balance_usdt()
            self.estimated_balance_usdt = float(eq)

        info["action_mask"] = action_mask

        # --- Enforce 1-minute cadence at the end of the step ---
        # If you want to skip waiting after a hard terminate, set skip_wait_on_terminate = True
        should_wait = True
        if self.skip_wait_on_terminate and (terminated or truncated):
            should_wait = False

        if should_wait:
            elapsed = time.monotonic() - step_start_monotonic
            remaining = self.step_period_seconds - elapsed

            if remaining > 0:
                # Live countdown (once per second) and GUI seconds_left updates
                # Use integer countdown for prints; keep self.seconds_left as precise float
                end_time = time.monotonic() + remaining
                while True:
                    now = time.monotonic()
                    self.seconds_left = max(0.0, end_time - now)

                    # Print countdown once per second if enabled
                    if self.print_countdown:
                        # ceiling for display (so "2.1s" shows as "3s" remaining)
                        display_seconds = int(self.seconds_left + 0.999)
                        print(f"⏳ Waiting for next minute tick: {display_seconds}s remaining", end="\r", flush=True)

                    if self.seconds_left <= 0.0:
                        break

                    # Sleep in short chunks so GUI can poll seconds_left smoothly
                    time.sleep(min(1.0, self.seconds_left))

                # Clear the countdown line (cosmetic)
                if self.print_countdown:
                    print(" " * 64, end="\r")

            else:
                # Overran the cadence; no waiting. Expose zero to GUI.
                self.seconds_left = 0.0
        else:
            # Skipped wait (typically on terminate/truncate)
            self.seconds_left = 0.0

        self.current_step += 1 # Increment the step count
        obs = self._get_observation() # Grab the current observation

        # --- Sanitize reward ---
        if np.isnan(reward) or np.isinf(reward):
            print(f"[Warning] Invalid reward encountered at step {self.current_step}: {reward}")
            reward = 0.0

        # --- Sanitize observation ---
        if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
            print(f"[Warning] Invalid observation at step {self.current_step}")
            obs = np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)

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
            # Not Buy (skip) action also enabled
            mask[2 + self.num_cryptos] = True                 # Not Buy (skip)
        else:
            # Crypto held: enable only hold (0) and sell (num_cryptos + 1)
            mask[0] = True
            mask[self.num_cryptos + 1] = True

        return mask
    
    def _get_buy_action_code(self, buy_action):
        """
        Converts a discrete buy action index into the corresponding crypto trading pair code (e.g., 'BTCUSDT').

        This function is used to translate the agent's action index into a string symbol
        that can be used to place real trades on the exchange.

        Args:
            buy_action (int): The discrete buy action index, ranging from 1 to num_cryptos inclusive.

        Returns:
            str: The trading pair symbol (e.g., 'BTCUSDT') corresponding to the action.

        Raises:
            ValueError: If the action is out of the valid buy action range.
        """
        if not (1 <= buy_action <= self.num_cryptos):
            raise ValueError(f"Invalid buy_action index: {buy_action}. Must be between 1 and {self.num_cryptos}.")

        crypto_index = buy_action - 1  # Map to 0-based index
        trading_pair = self.crypto_codes[crypto_index]  # e.g., 'BTC'
        return trading_pair
    
    def _buy(self, crypto_code):
        """
        Executes a blocking market buy for the given crypto and returns the confirmed average buy price.

        Args:
            crypto_code (str): The trading pair (e.g. 'BTCUSDT').

        Returns:
            float: The average buy price after confirmation.
        """
        self.buy_order_id = self.trader.execute_buy(crypto_code)  # Initiate buy order
        if self.buy_order_id is None:
            print("❌ Buy order failed to execute.")
            return None

        print("⌛ Waiting for buy order to be confirmed...")
        while True:
            order_info = self.trader.get_order_by_id(self.buy_order_id)
            if order_info and order_info['status'] == 'FILLED':
                print("✅ Buy order filled.")
                return float(order_info['avgPrice'])
            time.sleep(5)
    
    def _sell(self):
        """
        Executes a blocking market sell of the currently held crypto and returns the confirmed average sell price.

        Returns:
            float: The average sell price after confirmation.
        """
        self.sell_order_id = self.trader.execute_sell(self.active_crypto_code)
        if self.sell_order_id is None:
            print("❌ Sell order failed to execute.")
            return None

        print("⌛ Waiting for sell order to be confirmed...")
        while True:
            order_info = self.trader.get_order_by_id(self.sell_order_id)
            if order_info and order_info['status'] == 'FILLED':
                print("✅ Sell order filled.")
                return float(order_info['avgPrice'])
            time.sleep(5)

    def _refresh_crypto_codes(self):
        """
        Validate the symbol universe and replace any broken/unreliable ones with new,
        healthy symbols — strictly preserving the original universe size (num_cryptos).
        """
        # --- Step 0: determine current + target universe size (fixed) ---
        target_n = int(getattr(self, "num_cryptos", len(self.crypto_codes)))
        current = list(dict.fromkeys(self.crypto_codes))  # keep order + dedupe
        if len(current) != len(self.crypto_codes):
            # sanitize in memory if duplicates slipped in
            self.crypto_codes = current[:target_n]

        # --- Step 1: compute symbols to replace ---
        untradeable = self.trader.find_untradeable_codes(self.crypto_codes) or []
        unreliable = self._check_avg_low_price() or []
        # self.unusable_crypto_codes may have been set by _get_observation()
        known_unusable = getattr(self, "unusable_crypto_codes", []) or []

        to_replace = sorted(set(untradeable) | set(unreliable) | set(known_unusable))

        print(f"[DEBUG] Untradeable: {untradeable}")
        print(f"[DEBUG] Unreliable: {unreliable}")
        print(f"[DEBUG] Known unusable: {known_unusable}")
        print(f"[DEBUG] Final to_replace: {to_replace}")

        if to_replace:
            print(f"🔻 Replacement candidates ({len(to_replace)}): {to_replace}")

        if not to_replace:
            # nothing to do; still ensure size is exactly target_n
            if len(self.crypto_codes) != target_n:
                print(f"⚠️ Symbol count drift detected ({len(self.crypto_codes)} != {target_n}); trimming.")
                self.crypto_codes = sorted(self.crypto_codes)[:target_n]
            return  # ✅ Nothing to fix

        # --- Step 2: request viable replacements (may return many — we will filter/slice) ---
        print("🔄 Refreshing crypto codes...")
        current_set = set(self.crypto_codes)
        needed = len(to_replace)

        # We'll loop until we can validate a full, fixed-size list
        while True:
            # Ask trader for a big pool, then filter
            pool = self.trader.find_viable_replacement_codes(
                replace_codes=to_replace,
                current_codes=self.crypto_codes
            ) or []

            # Remove any codes that are already in the current set (avoid duplicates)
            pool = [c for c in pool if c not in current_set]

            if len(pool) < needed:
                print(f"⚠️ Only found {len(pool)} unique replacements, need {needed}. Retrying…")
                time.sleep(2)
                continue

            # Slice EXACTLY the number we need
            replacements = pool[:needed]

            # Build updated list: drop the to_replace, add replacements, keep fixed size
            kept = [c for c in self.crypto_codes if c not in to_replace]
            updated = kept + replacements

            # Enforce fixed size (target_n) — trim or pad (padding should never happen here)
            updated = sorted(list(dict.fromkeys(updated)))  # dedupe once more just in case
            if len(updated) > target_n:
                updated = updated[:target_n]
            elif len(updated) < target_n:
                # shouldn't happen; but if it does, try to top up from the remaining pool
                top_up_needed = target_n - len(updated)
                extra = [c for c in pool if c not in updated][:top_up_needed]
                updated += extra
                updated = updated[:target_n]

            # ✅ TEMPORARILY overwrite JSON with the proposed new codes
            with open(self.json_path, "w") as f:
                json.dump({"crypto_codes": updated}, f, indent=4)

            # Validate via a fresh database pull
            temp_database = CryptoDatabaseInitialiser(
                csv_path=self.csv_path,
                json_path=self.json_path,
                yaml_path=self.yaml_path
            )

            bad = temp_database.unusable_crypto_codes or []
            if not bad:
                # success — lock it in
                self.crypto_codes = updated
                self.unusable_crypto_codes = []  # reset after successful refresh
                print(f"✅ Updated crypto codes ({len(self.crypto_codes)}): {self.crypto_codes}")
                break
            else:
                print(f"❌ Invalid historical data for: {bad}. Retrying…")
                # remove bad from updated and try to refill only those slots
                to_replace = bad  # next loop will try to replace just these
                current_set = set([c for c in updated if c not in bad])
                # keep the good portion in self.crypto_codes while we search
                self.crypto_codes = sorted(list(current_set))[:target_n]
                time.sleep(2)

        # --- Step 3: rebuild observation inputs (same universe size, so model-compatible) ---
        # NOTE: observation_space shape remains constant because target_n is fixed.
        self._get_observation()

    def _check_avg_low_price(self) -> list[str]:
        """
        Identifies crypto symbols whose current close price has dropped more than 80% below
        their historical average low price.

        This function maintains a member dictionary `avg_low_prices` which stores the rolling
        average low price for each symbol seen. On each call:
            - Symbols not yet in the list are added using their historical data.
            - Symbols with valid price data are compared to their avg low.
            - Symbols with a close price < 20% of avg low are marked 'unreliable'.
            - Unreliable symbols are removed from `avg_low_prices`.

        Returns:
            list[str]: A list of symbols considered unreliable and flagged for removal.
        """
        unreliable = []

        for symbol, df in self.raw_lookup.items():
            try:
                # Ensure historical low price can be calculated
                if symbol not in self.avg_low_prices:
                    avg_low = df['low'].mean()
                    if pd.isna(avg_low) or avg_low <= 0:
                        continue  # Skip if low price is invalid
                    self.avg_low_prices[symbol] = avg_low

                avg_low = self.avg_low_prices[symbol]
                current_close = df.iloc[-1]['close']

                # Check if current close has dropped 80% below the historical avg low
                if current_close < 0.2 * avg_low:
                    print(f"⚠️ {symbol} dropped below crash threshold: close={current_close:.4f}, avg_low={avg_low:.4f}")
                    unreliable.append(symbol)

            except Exception as e:
                print(f"\033[91mError processing {symbol} in avg_low check: {e}\033[0m")
                continue

        # Remove unreliable symbols from the tracked average lows
        for symbol in unreliable:
            self.avg_low_prices.pop(symbol, None)

        return unreliable
    
    def _log_step(self, *, action: int, action_type: str, reward: float,
              symbol: str | None = None,
              price: float | None = None,
              profit_pct: float | None = None,
              buy_price: float | None = None,
              sell_price: float | None = None):
        """
        Append a single structured step log entry.
        Required fields: type, step, action, action_type, reward.
        Optional: symbol, price (raw if available), profit_pct, buy_price, sell_price.
        """
        step_no = int(self.current_step - self.start_index) + 1  # 1-based within episode
        rec = {
            "type": "step",
            "step": step_no,
            "action": int(action),
            "action_type": action_type,
            "reward": float(reward),
        }
        if symbol is not None:
            rec["symbol"] = symbol
        if price is not None:
            rec["price"] = float(price)
        if profit_pct is not None:
            rec["profit_pct"] = float(profit_pct)
        if buy_price is not None:
            rec["buy_price"] = float(buy_price)
        if sell_price is not None:
            rec["sell_price"] = float(sell_price)

        self._write_logline(rec) # Write to the log

    def _get_current_action(self, action: int) -> str:
        """
        Converts an action index into a human-readable description for GUI or logging.

        Args:
            action (int): The discrete action index from the RL agent.

        Returns:
            str: A string describing the action, e.g. 'HOLD', 'BUY: BTCUSDT', or 'SELL: BTCUSDT'.
        """
        if action == 0:
            return "HOLD"

        elif 1 <= action <= self.num_cryptos:
            symbol = self.crypto_codes[action - 1]  # Adjust for 0-based index
            return f"BUY: {symbol}"

        elif action == self.num_cryptos + 1:
            # Show the held symbol if there is one
            if self.active_crypto_index is not None:
                symbol = self.crypto_codes[self.active_crypto_index]
                return f"SELL: {symbol}"
            return "SELL"

        return "UNKNOWN"
    
    def _write_logline(self, rec: dict):
        """Append one JSONL line safely (thread-safe if GUI reads concurrently)."""
        try:
            with self._log_lock:
                with open(self.log_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
        except Exception as e:
            # Optional: soft-fail so trading never breaks on logging errors
            print(f"[Log warn] Failed to write log line: {e}")