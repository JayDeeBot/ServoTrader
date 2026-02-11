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
    x = np.asarray(x, dtype=np.float64)
    if np.any(np.isnan(x)) or np.all(x == x[0]):  # constant or NaN
        return 0.0
    return linregress(np.arange(len(x)), x).slope

import gymnasium as gym
from gymnasium import spaces # Necessary for specifying valid actions & observations
import numpy as np
import pandas as pd
import os
from datetime import datetime
import json
from scipy.stats import linregress
from typing import Dict, Tuple, Optional

BUY_INDEX = "close" # Buy at the close/high price
SELL_INDEX = "close" # Sell at the low/close price
HOLD_INDEX = "close" # Hold at the low/close price

class CryptoTradingEnv(gym.Env):
    """
    Custom Gym environment for PPO-based crypto trading with Buy, Sell, Hold actions.
    """
    metadata = {'render.modes': ['human']}

    
    def __init__(self, data, crypto_codes, episode_timeout=60):
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
        self.history_window = 1  # Number of past timesteps to include in observation

        # Store symbol list, preprocess raw dataframe
        self.crypto_codes = sorted(crypto_codes)
        self.data = self.preprocess_data(data)  # Now self.data is [T, N, F] tensor

        self.num_timesteps, self.num_cryptos, self.features_per_crypto = self.data.shape # Use the total number of timesteps available in historical data

        # Action Space:
        # 0              = Hold
        # 1 to num_cryptos = Buy symbol[i-1]
        # (1 + num_cryptos) = Sell
        self.action_space = spaces.Discrete(1 + self.num_cryptos + 1)

        # Observation Space: 
        # Engineered features including ['trend_ema_dev', 'momentum_rsi', 'vol_atr_norm', 'flow_cmf']
        # Current profit: Normalized percentage profit (0 if not holding)
        # Held crypto: Index of held crypto, or a special value if none held
        FEATURES_PER_CRYPTO = self.features_per_crypto  # should be 4
        H = int(self.history_window)
        C = int(self.num_cryptos)

        # --- lengths ---
        FEATURE_BLOCK_LEN = H * C * FEATURES_PER_CRYPTO          # history of lean features
        TIME_LEN          = 1                                    # normalized time remaining in episode [0,1]
        PROFIT_LEN        = 1                                    # normalized PnL or unrealized PnL
        ONE_HOT_LEN       = C + 1                                # +1 for "no position"

        obs_len = FEATURE_BLOCK_LEN + TIME_LEN + PROFIT_LEN + ONE_HOT_LEN

        # --- bounds ---
        # Lean features: keep broad but finite bounds (engineered features are roughly small)
        # - trend_ema_dev: typically ~[-0.2, +0.2], but allow +/-5
        # - momentum_rsi:  [0,1]
        # - vol_atr_norm:  small positive; allow [0,1]
        # - flow_cmf:      [-1,1]
        # For simplicity and safety across all features/history, we’ll set a broad default
        # and then tighten known-bounded parts (time_remaining, profit & one-hot).
        low  = np.full(obs_len, -5.0, dtype=np.float32)
        high = np.full(obs_len,  5.0, dtype=np.float32)

        # Indices:
        # time_remaining sits right after the feature block
        TIME_IDX   = FEATURE_BLOCK_LEN
        # profit comes right after time_remaining
        PROFIT_IDX = TIME_IDX + 1
        # one-hot starts after profit
        ONE_HOT_START = PROFIT_IDX + 1
        ONE_HOT_END   = ONE_HOT_START + ONE_HOT_LEN

        # time_remaining is strictly [0,1]
        low[TIME_IDX]  = 0.0
        high[TIME_IDX] = 1.0

        # Profit can swing more; allow +/-1000% (i.e., +/-10 if you encode as fraction)
        low[PROFIT_IDX]  = -10.0
        high[PROFIT_IDX] =  10.0

        # One-hot block at the end is strictly {0,1}
        low[ONE_HOT_START:ONE_HOT_END]  = 0.0
        high[ONE_HOT_START:ONE_HOT_END] = 1.0

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Internal Logging
        self.take_logs = True # Flag to decide where we should take logs of the training session
        self.log_dir = "/home/jarred/git/ServoTrader/logs" # Directory containing the log
        os.makedirs(self.log_dir, exist_ok=True) # Ensure the log exists
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") # Record the start datetime
        self.log_path = os.path.join(self.log_dir, f"env_log_{timestamp}.jsonl") # Create a jsonl file for the log
        self.episode_log = []  # Will hold step-level logs temporarily
        self.total_profit_percent = 0  # For global tracking of the profit
        self.episode_counter = 0 # Stores the episode count

    def preprocess_data(self, raw_df: pd.DataFrame) -> np.ndarray:
        """
        Build a 3D observation tensor per symbol that includes raw OHLC context **and**
        four engineered features. The engineered features are computed using the **LOW**
        price wherever that is statistically meaningful; CLOSE is only used where the
        classic definition would otherwise degenerate (notably, CMF).

        ----------------------
        OUTPUT (per time t, per symbol s)
        ----------------------
        Feature order (8 total):
        1) open_norm(t,s)    : Min–max normalized OPEN
        2) high_norm(t,s)    : Min–max normalized HIGH
        3) low_norm(t,s)     : Min–max normalized LOW
        4) close_norm(t,s)   : Min–max normalized CLOSE
        5) trend_ema_dev_low : LOW / EMA(LOW, span=30) - 1
        6) momentum_rsi_low  : RSI computed on LOW (Wilder smoothing, N=20), scaled to [0,1]
        7) vol_atr_norm_low  : ATR(High, Low, PrevClose) / |LOW|
        8) flow_cmf          : Chaikin Money Flow (N=20) using **standard definition with CLOSE**
                                (Using LOW here would collapse MFM to -1 when High!=Low.)

        ----------------------
        ENGINEERED FEATURES (details)
        ----------------------
        1) trend_ema_dev_low (span=30)
        EMA_t(low) = α*Low_t + (1-α)*EMA_{t-1}(low), α = 2/(N+1), N=30
        trend_ema_dev_low = Low_t / EMA_t(low) - 1

        2) momentum_rsi_low (period=20; scaled to [0,1])
        Δ = Low_t - Low_{t-1}
        AvgGain, AvgLoss via Wilder exponential smoothing (α=1/N, N=20)
        RS  = AvgGain / AvgLoss
        RSI = 100 - 100/(1+RS)  → divide by 100 to map to [0,1]

        NOTE: Using LOW instead of CLOSE is acceptable—RSI is a generic oscillator on any price series.

        3) vol_atr_norm_low (ATR period=20; normalized by |LOW|)
        True Range TR_t = max(High-Low, |High-PrevClose|, |Low-PrevClose|)
        ATR = EWM(TR, α=1/N) with N=20
        vol_atr_norm_low = ATR / |Low_t|
        Rationale: keep the standard TR/ATR definition (it depends on PrevClose),
        but normalize by LOW to follow your "use low where possible" requirement.

        4) flow_cmf (window=20; standard Chaikin Money Flow)
        Money Flow Multiplier: MFM = ((Close-Low) - (High-Close)) / (High-Low)
        Money Flow Volume:     MFV = MFM * Volume
        CMF_N = sum(MFV over N) / sum(Volume over N)

        IMPORTANT: If we replaced CLOSE with LOW, MFM would become -1 for High>Low,
        collapsing the indicator. So we **retain CLOSE** here.

        ----------------------
        WINDOWS & WARM-UP
        ----------------------
        - EMA span: 30
        - RSI period: 20
        - ATR period: 20
        - CMF window: 20
        - We require fully-formed windows for all features, so we drop the first `discard=30` rows.
        For each symbol series of length L, we output L-30 rows (time-aligned across symbols).

        ----------------------
        DATA HYGIENE
        ----------------------
        - Sort each symbol chronologically by 'timestamp' if present.
        - Replace ±inf → NaN; ffill/bfill price columns (open/high/low/close/vwap).
        - Fill volume/count NaNs with 0.
        - Guard divisions with small epsilon.
        - OHLC are min–max scaled per symbol to [0,1] before dropping warm-up so magnitudes
        are comparable. (If you prefer raw OHLC, comment out the min–max section.)

        ----------------------
        SIDE EFFECTS (class state)
        ----------------------
        - self.crypto_codes: alphabetical symbol list
        - self.num_cryptos: number of symbols
        - self.features_per_crypto: 8
        - self.num_timesteps: common truncated length across symbols (after warm-up drop)

        Returns
        -------
        tensor : np.ndarray, shape [timesteps, symbols, 8], dtype float32
        """
        print("Preprocessing data (OHLC + 4 engineered; LOW-centric)…")

        # ---------- Config ----------
        w_ema = 30   # EMA span for trend deviation
        w_rsi = 20   # RSI period
        w_atr = 20   # ATR period
        w_cmf = 20   # CMF rolling window
        discard = max(w_ema, w_rsi, w_atr, w_cmf)  # ensure all features are fully formed
        eps = 1e-9

        # ---------- Symbol setup & alignment ----------
        self.crypto_codes = sorted(raw_df['symbol'].unique().tolist())
        self.num_cryptos = len(self.crypto_codes)

        per_len = {}
        for sym in self.crypto_codes:
            df_sym = raw_df[raw_df['symbol'] == sym]
            if 'timestamp' in df_sym.columns:
                df_sym = df_sym.sort_values('timestamp')
            per_len[sym] = len(df_sym)

        min_len = min(per_len.values()) if per_len else 0
        if min_len <= discard:
            raise ValueError(
                f"Not enough bars per symbol to form features: min_len={min_len}, required>{discard}."
            )

        # Timesteps AFTER dropping warm-up
        self.num_timesteps = min_len - discard

        # ---------- Columns & output tensor ----------
        price_cols  = ['open', 'high', 'low', 'close', 'vwap']
        needed_cols = price_cols + ['volume', 'count']  # vwap for fill continuity; volume for CMF

        feature_names = [
            'open', 'high', 'low', 'close',
            'trend_ema_dev_low', 'momentum_rsi_low', 'vol_atr_norm_low', 'flow_cmf'
        ]
        self.features_per_crypto = len(feature_names)

        tensor = np.zeros(
            (self.num_timesteps, self.num_cryptos, self.features_per_crypto),
            dtype=np.float32
        )

        # ---------- Per-symbol processing ----------
        for i, symbol in enumerate(self.crypto_codes):
            # Slice this symbol and sort chronologically
            df_symbol = raw_df[raw_df['symbol'] == symbol]
            if 'timestamp' in df_symbol.columns:
                df_symbol = df_symbol.sort_values('timestamp')

            # Truncate to the common aligned length
            df_symbol = df_symbol.head(min_len)

            # Keep only needed columns and coerce numeric
            df_symbol = df_symbol[needed_cols].apply(pd.to_numeric, errors='coerce')

            # Replace infinities; fill prices (ffill/bfill) to maintain level continuity
            df_symbol = df_symbol.replace([np.inf, -np.inf], np.nan)
            df_symbol[price_cols] = df_symbol[price_cols].ffill().bfill()

            # Fill volume/count NaNs with a neutral baseline
            df_symbol[['volume', 'count']] = df_symbol[['volume', 'count']].fillna(0)

            # Last resort: ensure no NaNs remain in price columns
            df_symbol[price_cols] = df_symbol[price_cols].fillna(0.0)

            # Aliases
            open_  = df_symbol['open']
            high   = df_symbol['high']
            low    = df_symbol['low']
            close  = df_symbol['close']
            volume = df_symbol['volume']

            # ---------- OHLC min–max scaling (per symbol) ----------
            # Comment this block out if you prefer raw prices.
            for col in ['open', 'high', 'low', 'close']:
                col_min = df_symbol[col].min()
                col_max = df_symbol[col].max()
                rng = (col_max - col_min) if col_max != col_min else 1.0
                df_symbol[col + '_norm'] = ((df_symbol[col] - col_min) / (rng + eps)).astype(float)

            # ---------- (1) TREND — EMA deviation using LOW ----------
            ema_low = low.ewm(span=w_ema, adjust=False, min_periods=w_ema).mean()
            trend_ema_dev_low = (low / (ema_low + eps) - 1.0).astype(float)

            # ---------- (2) MOMENTUM — RSI on LOW (Wilder, N=20) ----------
            d_low = low.diff()
            gain = d_low.clip(lower=0)
            loss = (-d_low).clip(lower=0)
            avg_gain = gain.ewm(alpha=1.0 / w_rsi, min_periods=w_rsi, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1.0 / w_rsi, min_periods=w_rsi, adjust=False).mean()
            rs  = avg_gain / (avg_loss + eps)
            rsi = 100.0 - (100.0 / (1.0 + rs))
            momentum_rsi_low = (rsi / 100.0).astype(float)  # map to [0,1]

            # ---------- (3) VOLATILITY — ATR normalized by |LOW| ----------
            # Keep the canonical TR definition that uses PrevClose for gap risk.
            prev_close = close.shift(1)
            tr1 = (high - low).abs()
            tr2 = (high - prev_close).abs()
            tr3 = (low  - prev_close).abs()
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.ewm(alpha=1.0 / w_atr, min_periods=w_atr, adjust=False).mean()
            vol_atr_norm_low = (atr / (low.abs() + eps)).astype(float)

            # ---------- (4) FLOW — Chaikin Money Flow (N=20) using CLOSE ----------
            # Using LOW here breaks the indicator (MFM collapses to -1 when High>Low).
            hl_range = (high - low)
            mfm = ((close - low) - (high - close)) / (hl_range.replace(0, np.nan) + eps)
            mfv = mfm.fillna(0.0) * volume
            vol_sum = volume.rolling(window=w_cmf, min_periods=w_cmf).sum()
            mfv_sum = mfv.rolling(window=w_cmf, min_periods=w_cmf).sum()
            flow_cmf = (mfv_sum / (vol_sum + eps)).astype(float)

            # ---------- Align & drop warm-up rows ----------
            feats = pd.DataFrame({
                'open':          df_symbol['open'],
                'high':          df_symbol['high'],
                'low':           df_symbol['low'],
                'close':         df_symbol['close'],
                'trend_ema_dev_low':  trend_ema_dev_low,
                'momentum_rsi_low':   momentum_rsi_low,
                'vol_atr_norm_low':   vol_atr_norm_low,
                'flow_cmf':           flow_cmf
            }).iloc[discard:]  # length = min_len - discard

            # Final safety: kill any residual NaNs/Infs
            feats = feats.replace([np.inf, -np.inf], 0.0).fillna(0.0)

            # ---------- Write into tensor ----------
            tensor[:, i, 0] = feats['open'].to_numpy(dtype=np.float32, copy=False)
            tensor[:, i, 1] = feats['high'].to_numpy(dtype=np.float32, copy=False)
            tensor[:, i, 2] = feats['low'].to_numpy(dtype=np.float32, copy=False)
            tensor[:, i, 3] = feats['close'].to_numpy(dtype=np.float32, copy=False)
            tensor[:, i, 4] = feats['trend_ema_dev_low'].to_numpy(dtype=np.float32, copy=False)
            tensor[:, i, 5] = feats['momentum_rsi_low'].to_numpy(dtype=np.float32, copy=False)
            tensor[:, i, 6] = feats['vol_atr_norm_low'].to_numpy(dtype=np.float32, copy=False)
            tensor[:, i, 7] = feats['flow_cmf'].to_numpy(dtype=np.float32, copy=False)

        return tensor

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
        # self.active_crypto_index = None # Reset the active crypto to none
        self.buy_price = 0.0 # Reset the buy price to 0
        self.cash_balance = 1.0 # Reset the cash balance to 1
        self.portfolio_value = 1.0 # Reset the portfolio value to 1
        self.start_step = self.global_step  # Marks episode start

        # --- Log episode start step in structured JSON ---
        if self.take_logs:
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

        # 2. Current profit (if holding) (LOW for mark-to-exit):
        if self.active_crypto_index is not None and self.buy_price > 0:
            sym = self.crypto_codes[self.active_crypto_index]
            current_price = float(self.raw_lookup[sym].iloc[self.current_step][SELL_INDEX])
            current_profit = (current_price - self.buy_price) / max(self.buy_price, 1e-12)
        else:
            current_profit = 0.0

        current_profit = float(np.clip(current_profit, -10.0, 10.0)) - (0.3/100) # Clip profit to safe range - 0.3% transaction cost

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
        truncated = False
        not_buy_termination = False
        timeout_termination = False

        #### TIMEOUT HANDLING ####
        if self.global_step - self.start_step >= self.timeout_steps: # Check whether we have exceeded the timeout steps for this episode
            done = False # Reset done flag
            timeout_termination = True # Flag for timeout termination
            if self.active_crypto_index is not None: # Check whether there is an active crypto
                symbol = self.crypto_codes[self.active_crypto_index] # Grab the active symbol we are selling
                # final_price = self.data[self.current_step, self.active_crypto_index, LOW_IDX] # Grab the final price - low feature
                final_price = float(self.raw_lookup[symbol].iloc[self.current_step][SELL_INDEX]) # use raw low price for sell
                # Calculate the final profit
                if self.buy_price is None or self.buy_price == 0 or np.isnan(self.buy_price):
                    profit = 0
                else:
                    profit = (final_price - self.buy_price) / self.buy_price
                self.portfolio_value *= (1 + profit) # Calculate the final portfolio value
                reward = -1.0

                if self.take_logs:
                    # --- ⬇ Add synthetic 'sell' step for timeout ---
                    raw_price = None
                    if symbol in self.raw_lookup:
                        try:
                            raw_price = float(self.raw_lookup[symbol].iloc[self.current_step][SELL_INDEX]) # use low price for sell
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

        ### BUY ACTION ###
        # If the action is Buy - Buy Action range is 1:num_cryptos (for #num_cryptos cryptos)
        elif 1 <= action <= self.num_cryptos:  # Buy crypto[i]
            if self.active_crypto_index is None: # Check whether we already holding a crypto - prevents double buying
                self.active_crypto_index = action - 1 # Set active crypto index - Adjust index by -1 to match 0-based indexing
                symbol = self.crypto_codes[self.active_crypto_index] # Grab the symbol we are buying
                self.buy_price = float(self.raw_lookup[symbol].iloc[self.current_step][BUY_INDEX]) # use raw close price for buy
                reward = self._calculate_reward(0.0, "BUY")
            else: # If we are already holding a crypto then give a small negative reward - teaches the agent the legal moves
                reward = -0.1  # Penalty: already holding
        
        ### HOLD ACTION ###
        # If the Action is Hold - 0 = Hold
        elif action == 0:  # Hold
            if self.active_crypto_index is not None: # If we are holding a crypto
                symbol = self.crypto_codes[self.active_crypto_index] # Grab the active symbol we are holding
                price_now = float(self.raw_lookup[symbol].iloc[self.current_step][HOLD_INDEX]) # use raw low price for profit calc
                # price_now = float(self.raw_lookup[symbol].iloc[self.current_step]["close"]) # use raw close price for profit calc
                if self.buy_price is None or self.buy_price == 0 or np.isnan(self.buy_price): # Calculate the current profit
                    profit = 0  # or np.nan or some fallback strategy
                else:
                    profit = (price_now - self.buy_price) / self.buy_price - (0.3/100) # 0.3% transaction cost for buy & eventual sell
                reward = self._calculate_reward(profit, "HOLD") # Calculate the reward at this time step (dense)

                if abs(profit) < 0.001: # If the profit is near 0
                    self.break_even_steps += 1 # Increment the break_even_steps up - assists discouraging break-even trades
                else: # If we have made positive/negative profit
                    self.break_even_steps = 0  # Reset the break_even_steps
            else: # If we are not holding a crypto then give a small negative reward - teaches the agent the legal moves
                reward = -0.1  # Penalty: holding nothing

        ### SELL ACTION ###
        # If the Action if Sell - num_cryptos + 1 = Sell
        elif action == self.num_cryptos + 1:  # Sell
            if self.active_crypto_index is not None: # Check whether we are holding a crypto - prevents double selling
                symbol = self.crypto_codes[self.active_crypto_index] # Grab the active symbol we are selling
                sell_price = float(self.raw_lookup[symbol].iloc[self.current_step][SELL_INDEX]) # use raw low price for sell
                if self.buy_price is None or self.buy_price == 0 or np.isnan(self.buy_price): # Calculate the total profit
                    profit = 0  # or handle however you prefer (e.g., skip trade)
                else:
                    profit = (sell_price - self.buy_price) / self.buy_price - (0.3/100) # 0.3% transaction cost for buy & sell
                self.portfolio_value *= (1 + profit) # Calculate the final portfolio value
                reward = self._calculate_reward(profit, "SELL") # Calculate the final reward for the episode
                self._end_episode() # End the episode
                done = True # Reset done flag
            else: # If we are not holding a crypto then give a small negative reward - teaches the agent the legal moves
                reward = -0.1  # Penalty: nothing to sell

        terminated = done # Terminated & done will be the same for our application
        truncated = timeout_termination # Truncated is true if the total steps of the episode exceeds the timeout steps
        info = {}   
        # Log the reason the episode ended
        # if terminated:
        #     if truncated:
        #         print(f"[Env] Episode ended due to timeout at step {self.global_step - self.start_step} (relative to episode start)")
        #     else:
        #         print(f"[Env] Episode terminated due to sell at step {self.global_step - self.start_step} (relative to episode start)")

        if self.take_logs:
            # --- Log the behavior at the current step in structured JSON ---
            step_offset = int(self.global_step - self.start_step)  # Ensure it's native int for JSON

            # Initialize placeholders for price/profit values
            current_price = None       # Normalized price
            log_profit_pct = None              # Profit percentage
            raw_price = None           # Rawprice
            raw_buy_price = None       # Buy price from raw data
            raw_sell_price = None      # Sell price from raw data

            # If currently holding a crypto, compute prices and profit
            if self.active_crypto_index is not None:
                symbol = self.crypto_codes[self.active_crypto_index] # Grab the active symbol we are holding
                current_price = float(self.raw_lookup[symbol].iloc[self.current_step][HOLD_INDEX])  # Raw low price
                log_profit_pct = ((current_price - self.buy_price) / self.buy_price) * 100 - 0.3 if self.buy_price else 0.0
                if symbol in self.raw_lookup:
                    raw_price = float(self.raw_lookup[symbol].iloc[self.current_step][HOLD_INDEX])  # Raw low price

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
                step_event["profit_pct"] = float(log_profit_pct)

            # --- BUY action logging ---
            if 1 <= action <= self.num_cryptos:
                buy_symbol = self.crypto_codes[action - 1]
                if buy_symbol in self.raw_lookup:
                    raw_buy_price = float(self.raw_lookup[buy_symbol].iloc[self.current_step][BUY_INDEX])
                step_event["action_type"] = "buy"
                step_event["symbol"] = buy_symbol
                step_event["price"] = raw_buy_price if raw_buy_price is not None else float(self.buy_price)

            # --- SELL action logging ---
            elif action == self.num_cryptos + 1 and hasattr(self, "_last_sell_context"):
                ctx = self._last_sell_context
                sell_symbol = ctx["symbol"]
                if sell_symbol in self.raw_lookup:
                    raw_sell_price = float(self.raw_lookup[sell_symbol].iloc[self.current_step][SELL_INDEX])
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
                    "termination": "timeout" if timeout_termination else ("no_buy" if not_buy_termination else "sell"),
                    "timestamp": datetime.now().isoformat()
                }

                if self.raw_df is not None and self.active_crypto_index is not None:
                    final_sym = self.crypto_codes[self.active_crypto_index]
                    if final_sym in self.raw_lookup:
                        summary["final_raw_low"] = float(self.raw_lookup[final_sym].iloc[self.current_step][SELL_INDEX])

                self.episode_log.append(summary)

                # Persist JSONL to disk
                with open(self.log_path, "a") as f:
                    for record in self.episode_log:
                        f.write(json.dumps(record) + "\n")

                self.episode_log = []  # Clear buffer for next episode

        # --- Periodically load a new 10k-row dataset every 10k steps ---
        if (self.global_step + 1) % 10000 == 0:
            chunk_index = (self.global_step // 10000) + 1
            chunk_path = f"/home/jarred/git/ServoTrader/data/split_10k_chunks_ancient/{chunk_index:03}.csv"

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

        # Compute legal action mask
        action_mask = np.zeros(self.action_space.n, dtype=bool)
        if self.active_crypto_index is None:
            # No crypto currently held → can Buy any symbol or skip the episode (NOT_BUY)
            action_mask[1:self.num_cryptos + 1] = True               # Buy actions
            # action_mask[2 + self.num_cryptos] = True                 # Not Buy (skip)
        else:
            # Already holding → only Hold or Sell are valid
            action_mask[0] = True                                    # Hold
            action_mask[self.num_cryptos + 1] = True                 # Sell

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
            is_up, pct_change = self.lookahead_window_summary_buy()
            if is_up:
                reward = 0 # Reward scaled by expected rise
            else:
                reward = 0 # Penalize expected drops

        elif action == "HOLD":
            # if profit > 0: # If the profit is positive give positive rewards
            #     reward = profit # Apply a small penalty for trading costs
            # elif profit < 0: # If the profit is negative give negative rewards
            #     reward = -abs(profit)
            # else: # If there is no profit (break-even) give negative rewards
            #     reward = 0 # Give a zero reward for breaking even
            reward = 0 # Neutral base reward for holding

        elif action == "SELL":
            if profit > 0: # If the profit is positive give positive rewards
                reward = profit # Apply a small penalty for trading costs
            elif profit < 0: # If the profit is negative give negative rewards
                reward = -abs(profit)
            else: # If there is no profit (break-even) give negative rewards
                reward = 0 # Give a zero reward for breaking even

        # Handle possible Nan Rewards
        if not np.isfinite(reward):
            reward = 0 # Give zero rewards if the value is Nan or infinite

        return reward
    
    def lookahead_window_summary_not_buy(self) -> Tuple[bool, Dict[str, float]]:
        """
        Look ahead for each symbol from step t+1 to t+timeout_steps and compute:
        - The maximum future value (using BUY_INDEX column).
        - If it exceeds the current value -> record % increase (positive).
        - Otherwise -> record % change to the minimum future value (negative or zero).

        Returns:
        any_will_increase (bool),
        per_symbol (dict: {symbol -> pct_change})
        """

        any_will_increase = False
        per_symbol: Dict[str, float] = {}

        for sym, df in self.raw_lookup.items():
            n = len(df)
            if BUY_INDEX not in df.columns or self.global_step < 0 or self.global_step >= n:
                per_symbol[sym] = 0.0
                continue

            curr_val = df.loc[self.global_step, BUY_INDEX]

            # Future slice
            start = min(max(self.global_step + 1, 0), n)
            end_excl = min(self.global_step + 1 + self.timeout_steps, n)

            if start >= end_excl or not np.isfinite(curr_val):
                per_symbol[sym] = 0.0
                continue

            future_vals = df.loc[start:end_excl-1, BUY_INDEX].to_numpy(dtype=float)

            max_future = np.nanmax(future_vals)
            min_future = np.nanmin(future_vals)

            if max_future > curr_val:
                pct_change = (max_future / curr_val) - 1.0
                any_will_increase = True
            else:
                pct_change = (min_future / curr_val) - 1.0

            per_symbol[sym] = float(pct_change)

        return any_will_increase, per_symbol
    
    def lookahead_window_summary_buy(self) -> Tuple[bool, float]:
        """
        For the current crypto only (self.current_crypto), look ahead from
        self.global_step+1 up to the episode timeout boundary and determine:
        - If the future max(BUY_INDEX) exceeds the current BUY_INDEX -> will_rise=True and return that max.
        - Else -> will_rise=False and return the future min(BUY_INDEX).

        Window end is aligned to the episode timeout:
        remaining_steps = self.timeout_steps - (self.global_step - self.start_step)

        Returns:
        (will_rise: bool, extreme_value: float | None)
        """
        sym = getattr(self, "current_crypto", None)
        if sym is None or sym not in self.raw_lookup:
            return False, None

        df = self.raw_lookup[sym]
        if BUY_INDEX not in df.columns:
            return False, None

        n = len(df)
        t = self.global_step

        # bounds
        if t < 0 or t >= n:
            return False, None

        curr_val = df.loc[t, BUY_INDEX]
        if not np.isfinite(curr_val):
            return False, None

        # remaining steps until episode timeout
        elapsed = max(0, t - int(self.start_step))
        remaining_steps = max(0, int(self.timeout_steps) - elapsed)

        # future window [t+1, t+remaining_steps] (end exclusive)
        start = min(max(t + 1, 0), n)
        end_excl = min(t + 1 + remaining_steps, n)

        if start >= end_excl:
            return False, None

        future_vals = df.loc[start:end_excl - 1, BUY_INDEX].to_numpy(dtype=float)
        if future_vals.size == 0 or not np.isfinite(future_vals).any():
            return False, None

        max_future = float(np.nanmax(future_vals))
        min_future = float(np.nanmin(future_vals))

        if np.isfinite(max_future) and (max_future > curr_val):
            print("lookahead_window_summary_buy says the bought crypto is increasing")
            return True, max_future
        else:
            print("lookahead_window_summary_buy says the bought crypto is decreasing")
            return False, min_future
    
    def _end_episode(self):
        """
        Concludes the current episode.

        Updates: 
            active crypto index
            buy price
            cash balance
        """
        if self.take_logs:
            # Save sell log data before resetting state
            if self.active_crypto_index is not None:
                symbol = self.crypto_codes[self.active_crypto_index]
                raw_low = float(self.raw_lookup[symbol].iloc[self.current_step][SELL_INDEX])
                self._last_sell_context = {
                    "symbol": symbol,
                    "buy_price": self.buy_price,
                    "sell_price": raw_low,
                    "profit": (raw_low - self.buy_price) / max(self.buy_price, 1e-12) * 100.0
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
        if self.active_crypto_index is None or self.current_step < window:
            return None
        sym = self.crypto_codes[self.active_crypto_index]
        df = self.raw_lookup.get(sym)
        if df is None or len(df) == 0:
            return None
        start = self.current_step - window
        end = self.current_step
        start = max(0, start)
        end = min(end, len(df) - 1)
        # Use iloc slice to keep alignment with current_step indexing
        return pd.Series(df["close"].iloc[start:end+1].astype(float).values)

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
            # Not Buy (skip) action also enabled
            mask[1:self.num_cryptos + 1] = True
            # mask[self.active_crypto_index + 1] = False # Disable the buy action for the previously held crypto
            # mask[2 + self.num_cryptos] = True                 
        else:
            # Crypto held: enable only hold (0) and sell (num_cryptos + 1)
            mask[0] = True
            mask[self.num_cryptos + 1] = True

        return mask