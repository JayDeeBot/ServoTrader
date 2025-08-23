#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sell_hold_training_env.py

SellHoldTrainingEnv — a single-decision, HOLD/SELL pretraining environment
for cryptocurrency trading policies.

Purpose
-------
Pretrain a policy to decide whether to HOLD or SELL an already-held crypto at
the current timestep. The episode ends immediately after the decision.

Reward
------
- Compute the **future return** over a lookahead horizon H for the *held* symbol:
      r = (Close[t+H] - Close[t]) / Close[t]
- If action == HOLD: reward =  r
- If action == SELL: reward = -r
(Linear, sign-correct; optionally bounded via tanh if desired.)

Key Properties
--------------
- Action space shape matches your main env:
    {0: Hold, 1..N: Buy[i-1], N+1: Sell}
  but we **mask** everything except HOLD (0) and SELL (N+1).
- Observation shape matches your main env:
  stacked [history_window, N, F] + [time_remaining, current_profit, held_one_hot(N+1)].
  - time_remaining fixed at 1.0 (single-step)
  - current_profit fixed at 0.0 at decision time
  - held_one_hot marks the selected held symbol (index>0)
- Single-step episodes (fast pretraining; dense signal).
- No disk logging.

Configurable
------------
- history_window (default 5)
- lookahead_steps H (default 30)
- held_selection: "random" or "round_robin" (default "random")
- reward_bounded: if True, applies tanh scale to keep rewards in [-1, 1]

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
from typing import Literal, Tuple, List


# -----------------------------
# Utility: slope for features
# -----------------------------
def slope_func(x: np.ndarray) -> float:
    """
    Fit a simple linear regression to a 1D array `x` and return the slope.

    Used with pandas .rolling().apply(...) to estimate trend.
    """
    return linregress(np.arange(len(x)), x).slope if len(x) > 1 else 0.0


class SellHoldTrainingEnv(gym.Env):
    """
    Single-decision SELL/HOLD pretraining environment.

    At reset():
        - Choose a timestep t and a *held* symbol i (random or round-robin).
    At step(action):
        - action ∈ {0: HOLD, N+1: SELL} (all others masked)
        - Compute look-ahead return r_i(t→t+H) on raw prices.
        - Reward =  r if HOLD, else -r if SELL.
        - done=True (episode terminates).
    """
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        data: pd.DataFrame,
        crypto_codes: List[str],
        episode_timeout: int = 30,
        history_window: int = 5,
        lookahead_steps: int = 30,
        held_selection: Literal["random", "round_robin"] = "random",
        reward_bounded: bool = False,
        seed: int | None = None,
    ):
        """
        Args:
            data: Stacked OHLCV(+count, vwap) dataframe with a 'symbol' column.
            crypto_codes: List of symbols to include (e.g., 100 cryptos).
            episode_timeout: Kept for feature/shape parity (not used to truncate).
            history_window: Number of past steps to include in observation.
            lookahead_steps: Horizon H for future return calculation.
            held_selection: How to choose the held symbol per episode.
            reward_bounded: If True, apply tanh scaling to the linear return.
            seed: Optional RNG seed for reproducibility (random selection).
        """
        super().__init__()
        self.rng = np.random.default_rng(seed)

        # Config
        self.crypto_codes = sorted(list(crypto_codes))
        self.timeout_steps = int(episode_timeout)
        self.history_window = int(history_window)
        self.lookahead_steps = int(lookahead_steps)
        self.held_selection = held_selection
        self.reward_bounded = reward_bounded

        # Preprocess -> self.data (normalized features), self.raw_close (unscaled)
        self.data, self.raw_close = self._preprocess_data(data)

        # Shapes
        self.num_timesteps, self.num_cryptos, self.features_per_crypto = self.data.shape

        # --- Action space (unchanged shape) ---
        # 0 = Hold, 1..N = Buy, N+1 = Sell
        self.action_space = spaces.Discrete(1 + self.num_cryptos + 1)

        # --- Observation space (unchanged shape) ---
        obs_len = (
            self.history_window * self.num_cryptos * self.features_per_crypto
            + 2  # time_remaining, current_profit
            + (self.num_cryptos + 1)  # held_one_hot
        )
        low = np.zeros(obs_len, dtype=np.float32)
        # allow negative for the current_profit slot (kept at 0, but shape-compatible)
        low[-(self.num_cryptos + 1 + 1)] = -1.0
        high = np.ones(obs_len, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Episode state
        self.pointer = max(0, self.history_window - 1)  # rolling time index
        self.current_step = self.pointer
        self.held_index: int | None = None

        # For round-robin symbol selection
        self._round_idx = 0

    # -------------------
    # Data preprocessing
    # -------------------
    def _preprocess_data(self, raw_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert stacked dataframe into normalized [T, N, F] tensor and preserve
        a raw (unscaled) close-price tensor [T, N] for unbiased returns.

        Engineered features mirror your main env to maintain PPO input semantics.
        """
        codes = sorted(raw_df["symbol"].unique())
        codes = [c for c in codes if c in self.crypto_codes]
        N = len(codes)
        T = raw_df.groupby("symbol").size().min()

        feature_cols = ["open", "high", "low", "close", "vwap", "volume", "count"]
        added = ["recent_return", "volatility", "price_position", "volume_surge", "trend_slope", "moving_avg"]
        all_cols = feature_cols + added
        F = len(all_cols)

        data_tensor = np.zeros((T, N, F), dtype=np.float32)
        raw_close = np.zeros((T, N), dtype=np.float64)

        window = max(2, int(self.timeout_steps))

        for j, sym in enumerate(codes):
            df = (
                raw_df.loc[raw_df["symbol"] == sym, feature_cols]
                .head(T)
                .astype(float)
                .copy()
            )

            # Fill prices forward/back
            for c in ["open", "high", "low", "close", "vwap"]:
                df[c] = df[c].ffill().bfill()

            # Fill volume/count with zeros
            df[["volume", "count"]] = df[["volume", "count"]].fillna(0.0)

            # Preserve raw close BEFORE normalization
            raw_close[:, j] = df["close"].to_numpy()

            # Engineered features
            df["recent_return"] = df["close"].pct_change(periods=window).fillna(0.0)
            df["volatility"] = df["close"].rolling(window).std().fillna(0.0)

            hroll = df["high"].rolling(window).max()
            lroll = df["low"].rolling(window).min()
            df["price_position"] = ((df["close"] - lroll) / (hroll - lroll + 1e-6)).fillna(0.0)

            vavg = df["volume"].rolling(window).mean()
            df["volume_surge"] = (df["volume"] / (vavg + 1e-6)).fillna(0.0)

            df["trend_slope"] = df["close"].rolling(window).apply(slope_func, raw=False).fillna(0.0)
            df["moving_avg"] = df["close"].rolling(window).mean().bfill()

            # Normalize prices with min-max
            for c in ["open", "high", "low", "close", "vwap"]:
                cmin, cmax = float(df[c].min()), float(df[c].max())
                rng = (cmax - cmin) if cmax != cmin else 1.0
                df[c] = (df[c] - cmin) / rng

            # Log + min-max for volume/count
            for c in ["volume", "count"]:
                df[c] = np.log1p(df[c])
                cmin, cmax = float(df[c].min()), float(df[c].max())
                rng = (cmax - cmin) if cmax != cmin else 1.0
                df[c] = (df[c] - cmin) / rng

            # Normalize engineered features
            for c in added:
                cmin, cmax = float(df[c].min()), float(df[c].max())
                rng = (cmax - cmin) if cmax != cmin else 1.0
                df[c] = (df[c] - cmin) / rng

            data_tensor[:, j, :] = df[all_cols].to_numpy(dtype=np.float32)

        self.crypto_codes = codes
        return data_tensor, raw_close

    # -------------
    # Gym methods
    # -------------
    def reset(self, *, seed=None, options=None):
        """
        Start a new episode:

        - Advance time pointer so we iterate over most timesteps.
        - Select a held symbol (random or round-robin).
        - Return the observation at current_step and the HOLD/SELL mask.
        """
        super().reset(seed=seed)

        # Advance pointer (use every point); keep one future step for returns
        self.pointer += 1
        max_start = self.num_timesteps - 2  # need at least one future step
        if self.pointer > max_start:
            self.pointer = max(self.history_window - 1, 0)
        self.current_step = self.pointer

        # Choose which symbol we are "holding" for this episode
        if self.held_selection == "round_robin":
            self.held_index = self._round_idx % self.num_cryptos
            self._round_idx += 1
        else:  # random
            self.held_index = int(self.rng.integers(low=0, high=self.num_cryptos))

        obs = self._get_observation()
        info = {"action_mask": self._get_action_mask()}
        return obs, info

    def step(self, action: int):
        """
        Single decision:
            - If HOLD (0): reward = +future_return(held)
            - If SELL (N+1): reward = -future_return(held)
            - Episode terminates immediately.
        """
        mask = self._get_action_mask()
        if action < 0 or action >= self.action_space.n or not mask[action]:
            # Illegal action → zero reward, terminate to keep training robust
            reward = 0.0
            return self._get_observation(), float(reward), True, False, {"action_mask": mask}

        r = self._future_return(self.current_step, self.held_index, self.lookahead_steps)

        if action == 0:  # HOLD
            reward = r
        else:            # SELL at index N+1
            reward = -r

        if self.reward_bounded:
            # Optional: squash via tanh to [-1, 1] for stability with large moves
            reward = float(np.tanh(reward))

        # Single-step episode ends
        terminated, truncated = True, False
        obs = self._get_observation()  # returned for interface completeness
        return obs, float(reward), terminated, truncated, {"action_mask": mask}

    # -------------------
    # Helpers
    # -------------------
    def _get_observation(self) -> np.ndarray:
        """
        Build the flat observation vector with the same shape as your main env.

        Composition:
          - Flattened [history_window, N, F] slice ending at current_step
            (left-padded with zeros if needed).
          - time_remaining = 1.0 (single-step)
          - current_profit = 0.0 (decision moment)
          - held_one_hot (N+1) with the held symbol marked (index>0)
        """
        w_end = self.current_step + 1
        w_start = max(0, w_end - self.history_window)
        window = self.data[w_start:w_end]  # [w, N, F]

        if window.shape[0] < self.history_window:
            pad = np.zeros(
                (self.history_window - window.shape[0], self.num_cryptos, self.features_per_crypto),
                dtype=np.float32,
            )
            window = np.concatenate([pad, window], axis=0)

        obs_vec = window.flatten().astype(np.float32)

        # Extra fields for shape parity
        time_remaining = np.array([1.0], dtype=np.float32)
        current_profit = np.array([0.0], dtype=np.float32)

        # Held crypto one-hot (N+1): index 0 means "none"; here we *do* hold one
        held_hot = np.zeros(self.num_cryptos + 1, dtype=np.float32)
        idx = int(self.held_index) + 1 if self.held_index is not None else 0
        held_hot[idx] = 1.0

        full = np.concatenate([obs_vec, time_remaining, current_profit, held_hot]).astype(np.float32)

        if np.any(~np.isfinite(full)):
            full = np.nan_to_num(full, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        # Shape check for integration debugging
        assert full.shape == self.observation_space.shape, \
            f"Obs shape mismatch: expected {self.observation_space.shape}, got {full.shape}"
        return full

    def _get_action_mask(self) -> np.ndarray:
        """
        Only HOLD (0) and SELL (N+1) are legal in this env.
        """
        mask = np.zeros(self.action_space.n, dtype=bool)
        mask[0] = True                          # HOLD
        mask[self.num_cryptos + 1] = True       # SELL
        return mask

    def _future_return(self, t: int, i: int, H: int) -> float:
        """
        Compute raw-price future return for symbol i from t to t+H:
            (C[t+H,i] - C[t,i]) / C[t,i]
        Clamps t+H to the dataset end; returns 0 if denominator is zero.
        """
        t0 = t
        t1 = min(self.num_timesteps - 1, t + max(1, H))
        base = float(self.raw_close[t0, i])
        futr = float(self.raw_close[t1, i])
        if base == 0.0 or not np.isfinite(base) or not np.isfinite(futr):
            return 0.0
        return (futr - base) / base

    # -------------
    # Gym extras
    # -------------
    def render(self, mode="human"):
        """Nothing to render for single-step pretraining."""
        pass

    def close(self):
        """No resources to clean up."""
        pass
