#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
buy_training_env.py  (robust, chunking-enabled, with Not-Buy action)

Single-decision PPO pretraining environment that chooses either:
- BUY one crypto now (rank-based reward from future returns), or
- NOT_BUY (reward based on whether *any* crypto would have risen).

Key features
------------
- Canonical symbol normalization on BOTH config codes & CSV chunks
- Tolerant detection of symbol column names (symbol/pair/ticker)
- Periodic dataset chunk loading (e.g., every 10k episodes)
- Rank-based reward from future returns for BUY actions
- Not-Buy reward that is positive when *all* returns ≤ 0, negative when *any* return > 0
- Thorough comments and docstrings for maintainability

Author
------
Jarred Deluca

License
-------
MIT License

Copyright (c) 2025 Jarred Deluca

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the “Software”), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from __future__ import annotations

import os
import re
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from scipy.stats import linregress


# -----------------------------
# Utilities
# -----------------------------
def slope_func(x: np.ndarray) -> float:
    """Linear slope over a 1D window (for engineered feature)."""
    return linregress(np.arange(len(x)), x).slope if len(x) > 1 else 0.0


def _norm_symbol(s: str) -> str:
    """
    Normalize symbols to canonical BASEQUOTE:
      - uppercase
      - remove '/', '-', '_'
      e.g., 'btc/usdt' -> 'BTCUSDT', 'XBT-USD' -> 'XBTUSD'
    """
    if not isinstance(s, str):
        return s
    s = s.strip().upper()
    return re.sub(r"[\/\-_]", "", s)


def _ensure_symbol_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the dataframe has a 'symbol' column; accept 'pair'/'ticker' aliases.
    Lower-cases columns to be tolerant of casing.

    Returns
    -------
    pd.DataFrame
        Copy of the input with columns normalized and 'symbol_norm' added.
    """
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    if "symbol" not in df.columns:
        for cand in ("pair", "ticker"):
            if cand in df.columns:
                df.rename(columns={cand: "symbol"}, inplace=True)
                break
    if "symbol" not in df.columns:
        raise ValueError(f"No symbol/pair/ticker column found. Columns={list(df.columns)}")
    df["symbol"] = df["symbol"].astype(str).str.strip()
    # Normalized column used for lookups & membership tests
    df["symbol_norm"] = df["symbol"].map(_norm_symbol)
    return df


class BuyTrainingEnv(gym.Env):
    """
    Single-decision pretraining environment for PPO with optional periodic CSV
    chunk reloading (e.g., /.../split_10k_chunks_ancient/000.csv, 001.csv, ...).

    Action space
    ------------
        0                 : Hold  (masked illegal)
        1..N              : Buy i (legal)
        N+1               : Sell  (masked illegal)
        N+2 (2+num_cryptos): Not Buy (legal; the "skip" option)

    Episode semantics
    -----------------
    One decision per episode → immediate reward → done.

    Rewards
    -------
    - BUY(i): rank-based reward derived from future returns over `lookahead_steps`.
    - NOT_BUY:
        * If ANY crypto's future return is > 0 → negative reward (you missed upside).
        * If ALL cryptos' future returns are ≤ 0 → positive reward (you avoided losses).
      By default, magnitudes scale with the “best missed” up-move or the average down-move.

    Notes
    -----
    - The environment is designed to be simple and stable for pretraining policies
      for entry selection; it complements separate HOLD/SELL pretraining.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        data: pd.DataFrame,
        crypto_codes: list[str],
        episode_timeout: int = 30,
        history_window: int = 5,
        lookahead_steps: int = 30,
        reward_mode: str = "zero_to_one",  # affects BUY rank mapping
        # --- Not-Buy reward mode: "magnitude" or "binary"
        not_buy_mode: str = "magnitude",
        not_buy_positive_cap: float = 1.0,
        not_buy_negative_cap: float = 1.0,
        # --- Chunking knobs ---
        chunk_dir: str | None = "/home/jarred/git/ServoTrader/data/split_10k_chunks_ancient",
        chunk_size: int = 10_000,
        start_chunk_index: int = 0,          # index for the *initial* `data`
        enable_chunking: bool | None = None, # None => auto True if dir exists
    ):
        """
        Parameters
        ----------
        data : pd.DataFrame
            Stacked OHLCV(+count) rows for multiple symbols with a 'symbol'/'pair'/'ticker' column.
        crypto_codes : list[str]
            Symbols to include (any of 'BTC/USDT', 'btc-usdt', 'BTCUSDT' accepted; normalized internally).
        episode_timeout : int
            Window for engineered features (affects rolling stats).
        history_window : int
            Number of past timesteps included in the observation.
        lookahead_steps : int
            Horizon used to compute future returns for rewards.
        reward_mode : {"zero_to_one","minus_one_to_one"}
            Mapping for BUY rank-based reward.
        not_buy_mode : {"magnitude","binary"}
            - "magnitude": +avg negative if all <= 0 else -max positive (clipped by caps).
            - "binary": +1 if all <= 0 else -1 (still clipped by caps).
        not_buy_positive_cap : float
            Upper bound for positive Not-Buy reward.
        not_buy_negative_cap : float
            Lower bound (absolute) for negative Not-Buy penalty.
        chunk_dir : str | None
            Directory holding sequential CSV chunks named 000.csv, 001.csv, ...
        chunk_size : int
            Number of *episodes* between chunk reloads when chunking is enabled.
        start_chunk_index : int
            The starting chunk index corresponding to the initially supplied `data`.
        enable_chunking : bool | None
            If None, enable automatically when `chunk_dir` exists.
        """
        super().__init__()

        # --- Config (normalize configured codes) ---
        self.crypto_codes = sorted({_norm_symbol(c) for c in crypto_codes})
        self.timeout_steps = int(episode_timeout)
        self.history_window = int(history_window)
        self.lookahead_steps = int(lookahead_steps)
        self.reward_mode = reward_mode

        # Not-Buy behavior toggles
        self.not_buy_mode = not_buy_mode
        self.not_buy_positive_cap = float(not_buy_positive_cap)
        self.not_buy_negative_cap = float(not_buy_negative_cap)

        # --- Chunking config/state ---
        self.chunk_dir = chunk_dir
        self.chunk_size = int(chunk_size)
        self.loaded_chunk_index = int(start_chunk_index)
        if enable_chunking is None:
            self.enable_chunking = bool(self.chunk_dir and os.path.isdir(self.chunk_dir))
        else:
            self.enable_chunking = bool(enable_chunking)

        # --- Initial data prep (robust symbol handling) ---
        self.raw_df = _ensure_symbol_column(data)
        self._rebuild_lookups_and_tensors(self.raw_df, first_time=True)

        # --- Action/Observation spaces ---
        # Action Space indices:
        # 0              = Hold (illegal/masked)
        # 1..N           = Buy symbol[i-1] (legal)
        # N+1            = Sell (illegal/masked)
        # N+2            = Not Buy (legal)
        self.NOT_BUY_INDEX = 2  # offset after 0(Hold) and 1..N(Buy) and one for Sell
        self.action_space = spaces.Discrete(1 + self.num_cryptos + 2)

        # Observation = flattened history + [time_remaining, current_profit] + held_one_hot (N+1)
        obs_len = self.history_window * self.num_cryptos * self.features_per_crypto + 2 + (self.num_cryptos + 1)
        low = np.zeros(obs_len, dtype=np.float32)
        # allow negative profit slot (even though we keep it 0 here)
        low[-(self.num_cryptos + 1 + 1)] = -1.0
        high = np.ones(obs_len, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Rolling pointer for dataset episodes
        self.pointer = max(0, self.history_window - 1)
        self.current_step = self.pointer

        # Episode bookkeeping
        self._episode_started_at = 0

        # Global step counter (episodes == steps for single-decision env)
        self.global_step = 0

        print(
            f"[BuyTrainingEnv] init: CSV syms(norm)={len(set(self.raw_df['symbol_norm']))} | "
            f"JSON codes(norm)={len(set(self.crypto_codes))} | Using={self.num_cryptos} symbols"
        )

    # -------------------------------------------------------
    # Internal: construct tensors from a (normalized) dataframe
    # -------------------------------------------------------
    def _rebuild_lookups_and_tensors(self, df: pd.DataFrame, first_time: bool = False) -> None:
        """
        Given a dataframe with 'symbol' and 'symbol_norm':
          - Intersect with configured codes (already normalized)
          - Build data tensor [T, N, F] and raw_close [T, N]
          - Update env shapes / bookkeeping
        """
        # Group lookups keyed by normalized symbol
        self.raw_lookup = {sym: g.reset_index(drop=True) for sym, g in df.groupby("symbol_norm")}

        # Build tensors
        data, raw_close, used_codes = self._preprocess_data(df)

        self.data = data
        self.raw_close = raw_close
        self.num_timesteps, self.num_cryptos, self.features_per_crypto = self.data.shape
        self.crypto_codes = used_codes  # normalized, ordered

        if first_time:
            # On first build, keep pointer where it is (set in __init__)
            pass
        else:
            # On chunk reload, reset local pointer to a safe start
            self.pointer = max(0, self.history_window - 1)
            self.current_step = self.pointer

    # -------------------
    # Data preprocessing
    # -------------------
    def _preprocess_data(self, raw_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """
        Convert stacked dataframe into:
          - data_tensor: [T, N, F] normalized features
          - raw_close  : [T, N]     raw close prices
        using NORMALIZED symbols for membership/grouping.
        """
        if "symbol_norm" not in raw_df.columns:
            raw_df = _ensure_symbol_column(raw_df)

        # Intersect CSV symbols with configured set (both normalized)
        csv_syms = sorted(set(raw_df["symbol_norm"].astype(str).unique()))
        overlap = [s for s in csv_syms if s in self.crypto_codes]
        N = len(overlap)
        if N == 0:
            sample_csv = csv_syms[:10]
            sample_cfg = sorted(self.crypto_codes)[:10]
            raise ValueError(
                "No overlapping symbols between dataset and crypto_codes after normalization.\n"
                f"Example CSV syms: {sample_csv}\nExample JSON codes: {sample_cfg}\n"
                "Check separators (/, -, _), case, and exchange naming (BTC vs XBT)."
            )

        # Ensure equal length across symbols by clamping to min available T
        T = raw_df.groupby("symbol_norm").size().min()

        feature_cols = ["open", "high", "low", "close", "vwap", "volume", "count"]
        added_features = ["recent_return", "volatility", "price_position", "volume_surge", "trend_slope", "moving_avg"]
        all_features = feature_cols + added_features
        F = len(all_features)

        data_tensor = np.zeros((T, N, F), dtype=np.float32)
        raw_close = np.zeros((T, N), dtype=np.float64)

        window = max(2, int(self.timeout_steps))

        for j, sym_norm in enumerate(overlap):
            df = (
                raw_df.loc[raw_df["symbol_norm"] == sym_norm, feature_cols]
                .head(T)
                .astype(float)
                .copy()
            )

            # Fill price-like fields forward/back
            for col in ["open", "high", "low", "close", "vwap"]:
                df[col] = df[col].ffill().bfill()
            # Volume/count zeros for missing
            df[["volume", "count"]] = df[["volume", "count"]].fillna(0.0)

            # Preserve raw close BEFORE normalization
            raw_close[:, j] = df["close"].to_numpy()

            # Engineered features
            df["recent_return"] = df["close"].pct_change(periods=window).fillna(0.0)
            df["volatility"] = df["close"].rolling(window).std().fillna(0.0)

            high_roll = df["high"].rolling(window).max()
            low_roll = df["low"].rolling(window).min()
            df["price_position"] = ((df["close"] - low_roll) / (high_roll - low_roll + 1e-6)).fillna(0.0)

            vol_avg = df["volume"].rolling(window).mean()
            df["volume_surge"] = (df["volume"] / (vol_avg + 1e-6)).fillna(0.0)

            df["trend_slope"] = df["close"].rolling(window).apply(slope_func, raw=False).fillna(0.0)
            df["moving_avg"] = df["close"].rolling(window).mean().bfill()

            # Normalize price-like by min-max
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

        return data_tensor, raw_close, overlap  # overlap is the normalized symbol order

    # -------------------
    # Chunk loader
    # -------------------
    def _load_next_chunk(self) -> None:
        """
        Load the next CSV chunk, rebuild lookups/tensors, and reset local indices.
        Raises FileNotFoundError if the file is missing.
        """
        if not self.enable_chunking:
            return

        self.loaded_chunk_index += 1
        path = os.path.join(self.chunk_dir, f"{self.loaded_chunk_index:03}.csv")
        print(f"📦 Loading dataset chunk: {path}")

        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ Chunk {self.loaded_chunk_index:03}.csv not found at {path}!")

        new_df = pd.read_csv(path)
        new_df = _ensure_symbol_column(new_df)

        self.raw_df = new_df
        self._rebuild_lookups_and_tensors(self.raw_df, first_time=False)

        print(f"✅ Chunk {self.loaded_chunk_index:03}.csv loaded and preprocessed")

    # -------------
    # Gym methods
    # -------------
    def reset(self, *, seed=None, options=None):
        """
        Start a new single-decision episode; advances the local pointer by 1,
        wrapping safely so the history window is valid.

        Returns
        -------
        obs : np.ndarray
            Flattened observation vector.
        info : dict
            Contains the boolean 'action_mask'.
        """
        super().reset(seed=seed)

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
        Single-decision step with periodic dataset rollover.

        Logic
        -----
        - If action is illegal (masked), return 0, done=True.
        - If action is BUY(i):
            compute rank-based reward from future returns and finish.
        - If action is NOT_BUY:
            reward depends on whether any symbol rises over the lookahead window.
        """
        legal = self._get_action_mask()
        if action < 0 or action >= self.action_space.n or not legal[action]:
            # Illegal action: zero reward, terminate
            reward = 0.0
            terminated, truncated = True, False
            obs = self._get_observation()

            # Count the episode and maybe roll the chunk
            self.global_step += 1
            if self.enable_chunking and self.chunk_size > 0 and (self.global_step % self.chunk_size == 0):
                self._load_next_chunk()

            return obs, float(reward), terminated, truncated, {"action_mask": legal}

        future_returns = self._compute_future_returns(self.current_step, self.lookahead_steps)

        if action == (2 + self.num_cryptos):  # NOT_BUY
            reward = self._not_buy_reward(future_returns, mode=self.not_buy_mode)
        else:
            # Map action (1..N) -> symbol index [0..N-1]
            chosen_idx = action - 1
            reward = self._rank_to_reward(future_returns, chosen_idx, mode=self.reward_mode)

        terminated, truncated = True, False
        obs = self._get_observation()
        info = {"action_mask": self._get_action_mask()}

        # Episode complete; increment global step and possibly roll chunk
        self.global_step += 1
        if self.enable_chunking and self.chunk_size > 0 and (self.global_step % self.chunk_size == 0):
            self._load_next_chunk()

        return obs, float(reward), terminated, truncated, info

    # -------------------
    # Observation helper
    # -------------------
    def _get_observation(self) -> np.ndarray:
        """
        Flattened [history_window, N, F] + [time_remaining, current_profit] + held_one_hot (N+1).
        Held is always 'none' for pretraining.
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
        time_remaining = np.array([1.0], dtype=np.float32)  # single-step ⇒ constant
        current_profit = np.array([0.0], dtype=np.float32)  # always 0 here

        held_one_hot = np.zeros(self.num_cryptos + 1, dtype=np.float32)
        held_one_hot[0] = 1.0  # 'none held'

        full = np.concatenate([obs_vec, time_remaining, current_profit, held_one_hot]).astype(np.float32)

        if np.any(~np.isfinite(full)):
            full = np.nan_to_num(full, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        assert full.shape == self.observation_space.shape, \
            f"Obs shape mismatch: expected {self.observation_space.shape}, got {full.shape}"
        return full

    # -------------------
    # Reward components
    # -------------------
    def _compute_future_returns(self, t: int, H: int) -> np.ndarray:
        """
        Per-symbol future return from t to t+H using RAW closes:

            r_i = (Close[t+H, i] - Close[t, i]) / Close[t, i]

        NaNs/Infs are safely converted to 0.
        """
        t0 = t
        t1 = min(self.num_timesteps - 1, t + max(1, H))
        base = self.raw_close[t0, :]
        future = self.raw_close[t1, :]

        denom = np.where(base != 0.0, base, np.nan)
        ret = (future - base) / denom
        ret = np.nan_to_num(ret, nan=0.0, posinf=0.0, neginf=0.0)
        return ret.astype(np.float32)

    def _rank_to_reward(self, returns: np.ndarray, chosen_idx: int, mode: str = "zero_to_one") -> float:
        """
        Convert rank among all symbols into a scalar reward.

        Parameters
        ----------
        returns : np.ndarray
            Vector of per-symbol future returns.
        chosen_idx : int
            Index of the chosen symbol (0..N-1).
        mode : {"zero_to_one","minus_one_to_one"}
            Mapping for rank → reward.

        Returns
        -------
        float
            Reward in [0,1] or [-1,1] depending on mode.
        """
        N = returns.shape[0]
        if N <= 1:
            return 0.0

        order = np.argsort(returns)  # ascending
        ranks = np.empty_like(order)
        ranks[order] = np.arange(N)  # 0..N-1
        r = float(ranks[chosen_idx]) # 0=worst, N-1=best

        if mode == "minus_one_to_one":
            return (2.0 * (r / (N - 1))) - 1.0
        return r / (N - 1)

    def _not_buy_reward(self, returns: np.ndarray, mode: str = "magnitude") -> float:
        """
        Reward for selecting NOT_BUY at the decision point.

        Intuition
        ---------
        - If ANY asset would have gone up (return > 0), then skipping was a mistake → negative reward.
        - If ALL assets would have gone down (returns ≤ 0), then skipping avoided losses → positive reward.

        Modes
        -----
        "magnitude" (default):
            - Positive case (all ≤ 0):  +mean(abs(negative returns))  (clipped to not_buy_positive_cap)
            - Negative case (any > 0): -max(positive returns)         (clipped to -not_buy_negative_cap)
        "binary":
            - Positive case (all ≤ 0): +1
            - Negative case (any > 0): -1
          (Then each is clipped by the corresponding cap.)

        Returns
        -------
        float
            Scalar reward; sign indicates correctness of the NOT_BUY decision, magnitude reflects confidence.
        """
        any_up = np.any(returns > 0.0)

        if mode == "binary":
            if any_up:
                return float(-min(self.not_buy_negative_cap, 1.0))
            else:
                return float(min(self.not_buy_positive_cap, 1.0))

        # magnitude mode
        pos = returns[returns > 0.0]
        neg = returns[returns <= 0.0]

        if any_up:
            # Missed upside → penalty proportional to the best missed up-move.
            penalty = float(np.max(pos)) if pos.size > 0 else 0.0
            penalty = min(penalty, self.not_buy_negative_cap)  # cap magnitude
            return -penalty
        else:
            # Correct skip → reward proportional to the average down-move avoided.
            gain = float(np.mean(np.abs(neg))) if neg.size > 0 else 0.0
            gain = min(gain, self.not_buy_positive_cap)  # cap magnitude
            return gain

    # -------------------
    # Action mask helper
    # -------------------
    def _get_action_mask(self) -> np.ndarray:
        """
        Legal at the single decision:
          - BUY actions (1..N)
          - NOT_BUY action (N+2)
        Illegal (masked):
          - HOLD (0)
          - SELL (N+1)
        """
        mask = np.zeros(self.action_space.n, dtype=bool)
        # BUY range
        mask[1 : 1 + self.num_cryptos] = True
        # NOT_BUY
        mask[1 + self.num_cryptos + 1] = True  # index = 2 + num_cryptos
        return mask

    # -------------
    # Gym extras
    # -------------
    def render(self, mode="human"):
        pass

    def close(self):
        pass
