#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
buy_training_env.py

BuyTrainingEnv — a single-decision, ranking-based PPO pretraining environment
for cryptocurrency selection.

Purpose
-------
This environment is designed to pretrain a policy to make *buy-only* selections,
analogous in spirit to the initial supervised/self-supervised pretraining phase
that precedes RL fine-tuning in LLMs. At each episode, the agent makes exactly
one decision: select which crypto to buy *right now*. The episode terminates
immediately after that choice.

Reward Shaping (Ranking)
------------------------
After the agent picks a crypto, we compute each symbol's **future return** over
a fixed look-ahead window H, then rank all symbols by that return (best to
worst). The agent's reward is a normalized rank-based score. By default:
    reward = rank / (N - 1)  in [0, 1]
where rank=0 is worst and rank=N-1 is best among N cryptos.

Key Properties
--------------
- **Action space** is unchanged from your main env:
    {0: Hold, 1..N: Buy crypto[i-1], N+1: Sell}
  But we **mask** all actions except the N BUY actions at every step.
- **Observation space** shape is unchanged. The "held-crypto" part of the
  observation is **always encoded as 'none held'**, i.e., the special index.
- **Single-step episodes**: one action → immediate reward → done=True.
- **No logging**: intentionally minimal to keep pretraining fast.

Configurable knobs
------------------
- lookahead_steps (int): horizon H for future return ranking (default: 30).
- history_window (int): how many past steps to include in the observation
  stack (default: 5), matching your PPO input format.
- reward_mode: "zero_to_one" (default) or "minus_one_to_one".

Assumptions
-----------
- Input data comes in the same stacked CSV style used in your main env
  (columns include: symbol, open, high, low, close, vwap, volume, count).
- We preserve a **raw close tensor** (unscaled) internally so that future
  returns are computed on real price levels (robust ranking across symbols).

Dependencies
------------
- gymnasium
- numpy
- pandas
- scipy (for linregress used in engineered features)

Author: Jarred Deluca
Created: 2025
License: MIT
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from scipy.stats import linregress

# -----------------------------
# Utility: slope for features
# -----------------------------
def slope_func(x: np.ndarray) -> float:
    """
    Fit a simple linear regression to a 1D array `x` and return the slope.

    Intended for use with pandas .rolling().apply(...) to estimate trend.

    Args:
        x (array-like): 1D numeric window.

    Returns:
        float: slope of best-fit line (positive=uptrend, negative=downtrend).
    """
    return linregress(np.arange(len(x)), x).slope if len(x) > 1 else 0.0


class BuyTrainingEnv(gym.Env):
    """
    A single-decision, buy-only pretraining environment for PPO.

    The agent observes the same feature vector used in the main trading env,
    but it always appears as if **no crypto is held**. At each episode start,
    the agent chooses exactly one BUY action (1..N). We then compute the
    future returns of all N cryptos over a fixed horizon, rank them, and give
    the agent a reward based on the rank of its chosen crypto. The episode
    then terminates immediately.

    Action space (unchanged shape):
        0: Hold         (masked as illegal)
        1..N: Buy i-1   (only these N actions are legal)
        N+1: Sell       (masked as illegal)

    Observation space (unchanged shape):
        Stacked [history_window] of [N, F] features + extra fields:
        - time_remaining (kept for shape consistency; set to 1 at reset)
        - current_profit (always 0.0 here)
        - held_one_hot (size N+1), always "none held" (index 0 = 1.0)
    """
    metadata = {"render.modes": ["human"]}

    # -------------
    # Constructor
    # -------------
    def __init__(
        self,
        data: pd.DataFrame,
        crypto_codes: list[str],
        episode_timeout: int = 30,
        history_window: int = 5,
        lookahead_steps: int = 30,
        reward_mode: str = "zero_to_one",
    ):
        """
        Args:
            data (pd.DataFrame): Stacked OHLCV(+count, vwap) dataframe with 'symbol' column.
            crypto_codes (list[str]): Symbols to include (e.g., 100 cryptos).
            episode_timeout (int): Unused for termination here (kept for shape parity).
            history_window (int): Number of past steps to include in observation.
            lookahead_steps (int): Horizon H for future return ranking.
            reward_mode (str): "zero_to_one" or "minus_one_to_one" scaling of rank.

        Notes:
            - We preprocess into a tensor [T, N, F] and also store a raw-close
              tensor for unbiased cross-symbol return comparisons.
            - Episodes are single-step; we advance an internal pointer so that
              each subsequent reset() slides forward by 1 timestep.
        """
        super().__init__()

        # Config
        self.crypto_codes = sorted(list(crypto_codes))
        self.timeout_steps = int(episode_timeout)      # kept for obs parity
        self.history_window = int(history_window)
        self.lookahead_steps = int(lookahead_steps)
        self.reward_mode = reward_mode  # ranking scale

        # Preprocess -> self.data (normalized features), self.raw_close (unscaled close)
        self.data, self.raw_close = self._preprocess_data(data)

        # Shapes
        self.num_timesteps, self.num_cryptos, self.features_per_crypto = self.data.shape

        # --- Action space (unchanged) ---
        # 0 = Hold, 1..N = Buy symbol[i-1], N+1 = Sell
        self.action_space = spaces.Discrete(1 + self.num_cryptos + 1)

        # --- Observation space (unchanged shape) ---
        # Flattened [history_window, N, F] + [time_remaining, current_profit] + held_one_hot (N+1)
        obs_len = self.history_window * self.num_cryptos * self.features_per_crypto + 2 + (self.num_cryptos + 1)
        low = np.zeros(obs_len, dtype=np.float32)
        # allow negative profit slot (even though we'll keep it 0 here)
        low[-(self.num_cryptos + 1 + 1)] = -1.0
        high = np.ones(obs_len, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Rolling pointer for dataset episodes
        # Start far enough in so we can build the history window without padding
        self.pointer = max(0, self.history_window - 1)
        self.current_step = self.pointer  # alias for clarity

        # Cached episode bookkeeping
        self._episode_started_at = 0

    # -------------------
    # Data preprocessing
    # -------------------
    def _preprocess_data(self, raw_df: pd.DataFrame):
        """
        Convert stacked dataframe into normalized tensor [T, N, F] and
        preserve a raw (unscaled) close-price tensor for fair cross-symbol returns.

        Engineered features replicate your existing pipeline so the PPO input
        shape and semantics remain consistent.

        Returns:
            data_tensor (np.ndarray): [T, N, F] float32 normalized features
            raw_close   (np.ndarray): [T, N]    float64 raw closes (filled)
        """
        # Ensure consistent symbol ordering & equal-length sequences
        codes = sorted(raw_df["symbol"].unique())
        codes = [c for c in codes if c in self.crypto_codes]
        N = len(codes)
        T = raw_df.groupby("symbol").size().min()

        feature_cols = ["open", "high", "low", "close", "vwap", "volume", "count"]
        added_features = ["recent_return", "volatility", "price_position", "volume_surge", "trend_slope", "moving_avg"]
        all_features = feature_cols + added_features
        F = len(all_features)

        data_tensor = np.zeros((T, N, F), dtype=np.float32)
        raw_close = np.zeros((T, N), dtype=np.float64)

        # Feature engineering window — use episode_timeout for consistency with your template
        window = max(2, int(self.timeout_steps))

        for j, sym in enumerate(codes):
            df = (
                raw_df.loc[raw_df["symbol"] == sym, feature_cols]
                .head(T)
                .astype(float)
                .copy()
            )

            # Fill prices forward/back
            for col in ["open", "high", "low", "close", "vwap"]:
                df[col] = df[col].ffill().bfill()

            # Fill volume/count with zeros
            df[["volume", "count"]] = df[["volume", "count"]].fillna(0.0)

            # Preserve raw close BEFORE normalization for unbiased returns
            raw_close[:, j] = df["close"].to_numpy()

            # --- Engineered features (on raw, then normalized later) ---
            df["recent_return"] = df["close"].pct_change(periods=window).fillna(0.0)
            df["volatility"] = df["close"].rolling(window).std().fillna(0.0)

            high_roll = df["high"].rolling(window).max()
            low_roll = df["low"].rolling(window).min()
            df["price_position"] = ((df["close"] - low_roll) / (high_roll - low_roll + 1e-6)).fillna(0.0)

            vol_avg = df["volume"].rolling(window).mean()
            df["volume_surge"] = (df["volume"] / (vol_avg + 1e-6)).fillna(0.0)

            df["trend_slope"] = df["close"].rolling(window).apply(slope_func, raw=False).fillna(0.0)
            df["moving_avg"] = df["close"].rolling(window).mean().bfill()

            # Normalize price-like columns by min-max
            for col in ["open", "high", "low", "close", "vwap"]:
                cmin, cmax = float(df[col].min()), float(df[col].max())
                rng = (cmax - cmin) if cmax != cmin else 1.0
                df[col] = (df[col] - cmin) / rng

            # Log + min-max for volume/count
            for col in ["volume", "count"]:
                df[col] = np.log1p(df[col])
                cmin, cmax = float(df[col].min()), float(df[col].max())
                rng = (cmax - cmin) if cmax != cmin else 1.0
                df[col] = (df[col] - cmin) / rng

            # Normalize engineered features by min-max
            for col in added_features:
                cmin, cmax = float(df[col].min()), float(df[col].max())
                rng = (cmax - cmin) if cmax != cmin else 1.0
                df[col] = (df[col] - cmin) / rng

            data_tensor[:, j, :] = df[all_features].to_numpy(dtype=np.float32)

        # Update canonical lists and sizes on the instance
        self.crypto_codes = codes
        return data_tensor, raw_close

    # -------------
    # Gym methods
    # -------------
    def reset(self, *, seed=None, options=None):
        """
        Start a new single-decision episode.

        Behavior:
            - Advances an internal pointer by 1 each call.
            - Ensures there's at least 1 future step to evaluate returns.
            - Returns observation at `current_step` and the (buy-only) action mask.
        """
        super().reset(seed=seed)

        # Slide the pointer forward by 1 to use "every point in the dataset".
        # Keep a margin of 1 step for lookahead; if near the end, wrap to a
        # safe index that still supports the history window.
        self.pointer += 1
        max_start = self.num_timesteps - 2  # need at least one future step
        if self.pointer > max_start:
            self.pointer = max(self.history_window - 1, 0)

        self.current_step = self.pointer
        self._episode_started_at = self.current_step

        obs = self._get_observation()
        info = {"action_mask": self._get_action_mask()}
        return obs, info

    def step(self, action: int):
        """
        Single-decision step.

        Args:
            action (int): Expected in [1..N] (BUY actions). Other actions are
                          masked out and treated as illegal (reward=0).

        Returns:
            obs (np.ndarray): Next observation (not used by PPO for single-step,
                              but returned for interface completeness).
            reward (float):   Rank-based reward for the chosen crypto.
            terminated (bool): Always True (single-decision episode).
            truncated (bool):  Always False (no time-based truncation).
            info (dict):       Contains the (buy-only) action mask.
        """
        # --- Enforce buy-only semantics via mask (safety) ---
        legal = self._get_action_mask()
        if action < 0 or action >= self.action_space.n or not legal[action]:
            # Illegal action: zero reward, immediate termination
            reward = 0.0
            terminated, truncated = True, False
            obs = self._get_observation()  # shape-consistent return
            return obs, float(reward), terminated, truncated, {"action_mask": legal}

        # Map action (1..N) -> symbol index [0..N-1]
        chosen_idx = action - 1

        # --- Compute future returns for all cryptos from current_step ---
        future_returns = self._compute_future_returns(self.current_step, self.lookahead_steps)
        # Rank-based reward
        reward = self._rank_to_reward(future_returns, chosen_idx, mode=self.reward_mode)

        # Single-step episode ends immediately
        terminated, truncated = True, False

        # Return a fresh observation (shape compatibility), though training won’t use it
        obs = self._get_observation()
        info = {"action_mask": self._get_action_mask()}
        return obs, float(reward), terminated, truncated, info

    # -------------------
    # Observation helper
    # -------------------
    def _get_observation(self) -> np.ndarray:
        """
        Build the flat observation vector with identical shape to the main env.

        Composition:
            - Flattened [history_window, N, F] slice ending at `current_step`
              (left-padded with zeros at dataset start if needed).
            - time_remaining: set to 1.0 for shape parity.
            - current_profit: always 0.0 in this pretraining env.
            - held_one_hot (N+1): always "none held" => index 0 is 1.0.
        """
        # Determine window bounds
        w_end = self.current_step + 1
        w_start = max(0, w_end - self.history_window)
        window = self.data[w_start:w_end]  # [w, N, F]

        # Left-pad if short at the dataset start
        if window.shape[0] < self.history_window:
            pad = np.zeros(
                (self.history_window - window.shape[0], self.num_cryptos, self.features_per_crypto),
                dtype=np.float32,
            )
            window = np.concatenate([pad, window], axis=0)

        obs_vec = window.flatten().astype(np.float32)

        # Extra features for shape compatibility
        time_remaining = np.array([1.0], dtype=np.float32)   # single-step ⇒ just 1.0
        current_profit = np.array([0.0], dtype=np.float32)   # always 0.0 here

        # Held crypto one-hot (N+1); encode "none held"
        held_one_hot = np.zeros(self.num_cryptos + 1, dtype=np.float32)
        held_one_hot[0] = 1.0

        full = np.concatenate([obs_vec, time_remaining, current_profit, held_one_hot]).astype(np.float32)

        # Final safety
        if np.any(~np.isfinite(full)):
            full = np.nan_to_num(full, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        # Shape assert (useful during integration)
        assert full.shape == self.observation_space.shape, \
            f"Obs shape mismatch: expected {self.observation_space.shape}, got {full.shape}"
        return full

    # -------------------
    # Reward components
    # -------------------
    def _compute_future_returns(self, t: int, H: int) -> np.ndarray:
        """
        Compute per-symbol future return from timestep t to t+H:
            r_i = (Close[t+H, i] - Close[t, i]) / Close[t, i]
        using the **raw** (unscaled) close tensor.

        If t+H exceeds the dataset, we clamp H to the last available index.

        Returns:
            np.ndarray shape [N]: future returns for each symbol at time t.
        """
        t0 = t
        t1 = min(self.num_timesteps - 1, t + max(1, H))  # need at least one step ahead
        base = self.raw_close[t0, :]      # [N]
        future = self.raw_close[t1, :]    # [N]

        # Avoid div-by-zero; if base==0, treat return as 0 for that symbol
        denom = np.where(base != 0.0, base, np.nan)
        ret = (future - base) / denom
        ret = np.nan_to_num(ret, nan=0.0, posinf=0.0, neginf=0.0)
        return ret.astype(np.float32)

    def _rank_to_reward(self, returns: np.ndarray, chosen_idx: int, mode: str = "zero_to_one") -> float:
        """
        Convert the chosen symbol's rank among all symbols into a scalar reward.

        Args:
            returns (np.ndarray): shape [N], future returns for each symbol.
            chosen_idx (int): index [0..N-1] of the chosen symbol.
            mode (str):
                - "zero_to_one"    : reward in [0, 1], worst=0, best=1.
                - "minus_one_to_one": reward in [-1, 1], worst=-1, best=+1.

        Returns:
            float: rank-based reward.
        """
        N = returns.shape[0]
        # argsort returns ascending order; highest return should get highest rank
        order = np.argsort(returns)  # ascending
        ranks = np.empty_like(order)
        ranks[order] = np.arange(N)  # 0..N-1
        r = float(ranks[chosen_idx])  # 0=worst, N-1=best

        if N <= 1:
            return 0.0

        if mode == "minus_one_to_one":
            # Map 0..(N-1) → [-1, 1]
            return (2.0 * (r / (N - 1))) - 1.0
        else:
            # Default: 0..(N-1) → [0, 1]
            return r / (N - 1)

    # -------------------
    # Action mask helper
    # -------------------
    def _get_action_mask(self) -> np.ndarray:
        """
        Generate a boolean mask of legal actions.

        In this pretraining env:
            - Only BUY actions (1..N) are legal.
            - HOLD (0) and SELL (N+1) are always illegal.
        """
        mask = np.zeros(self.action_space.n, dtype=bool)
        mask[1 : 1 + self.num_cryptos] = True  # enable only BUY actions
        return mask

    # -------------
    # Gym extras
    # -------------
    def render(self, mode="human"):
        """Optional: nothing to render for single-step ranking env."""
        pass

    def close(self):
        """No resources to clean up."""
        pass
