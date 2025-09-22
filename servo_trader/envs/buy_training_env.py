#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
buy_training_env.py  (robust, chunking-enabled)

Single-decision, buy-only PPO pretraining environment with:
- Canonical symbol normalization on BOTH config codes & CSV chunks
- Tolerant detection of symbol column names (symbol/pair/ticker)
- Periodic dataset chunk loading (e.g., every 10k episodes)
- Rank-based reward from future returns
"""

from __future__ import annotations

import os
import re
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from scipy.stats import linregress
from collections import deque  # ← NEW: for forward max precompute


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
    Single-decision, buy-only pretraining environment for PPO, with optional
    periodic CSV chunk reloading (e.g., /.../split_10k_chunks_ancient/000.csv, 001.csv, ...).

    Action space:
        0        : Hold  (masked illegal)
        1..N     : Buy i (legal)
        N+1      : Sell  (masked illegal)

    Episode:
        One decision per episode → immediate rank-based reward → done.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        data: pd.DataFrame,
        crypto_codes: list[str],
        episode_timeout: int = 60,
        history_window: int = 5,
        lookahead_steps: int = 60,
        reward_mode: str = "zero_to_one",
        # --- NEW: reward behavior toggles ---
        use_peak_within_window: bool = False,   # if True, exit uses best pessimistic Low within [t+1..t+H]
        # --- Chunking knobs ---
        chunk_dir: str | None = "/home/jarred/git/ServoTrader/data/split_10k_chunks_ancient",
        chunk_size: int = 10_000,
        start_chunk_index: int = 0,          # index for the *initial* `data`
        enable_chunking: bool | None = None, # None => auto True if dir exists
    ):
        super().__init__()

        # --- Config (normalize configured codes) ---
        self.crypto_codes = sorted({_norm_symbol(c) for c in crypto_codes})
        self.timeout_steps = int(episode_timeout)
        self.history_window = int(history_window)
        self.lookahead_steps = int(lookahead_steps)
        self.reward_mode = reward_mode
        self.use_peak_within_window = bool(use_peak_within_window)

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

        # --- Action/Observation spaces (unchanged shape semantics) ---
        self.action_space = spaces.Discrete(1 + self.num_cryptos + 1)
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

        # --- NEW: precompute forward max(low) for peak-in-window mode ---
        self._precompute_forward_max_low(self.lookahead_steps)

        if first_time:
            # On first build, keep pointer where it is (set in __init__)
            pass
        else:
            # On chunk reload, reset local pointer to a safe start
            self.pointer = max(0, self.history_window - 1)
            # need at least one future step
            self.pointer = min(self.pointer, self.num_timesteps - 2)
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
        # --- NEW: raw high/low for stringent pricing ---
        raw_high = np.zeros((T, N), dtype=np.float64)
        raw_low  = np.zeros((T, N), dtype=np.float64)

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

            # Preserve raw close/high/low BEFORE normalization
            raw_close[:, j] = df["close"].to_numpy()
            raw_high[:, j]  = df["high"].to_numpy()
            raw_low[:, j]   = df["low"].to_numpy()

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

        # Attach raws for later use (stringent pricing & peak)
        self.raw_high = raw_high
        self.raw_low = raw_low

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

    # -------------------
    # Forward max cache (for peak-in-window)
    # -------------------
    def _precompute_forward_max_low(self, H: int) -> None:
        """
        Precompute forward window MAX of low prices for each t (exclude t itself):
        fwd_max_low[t, i] = max( raw_low[t+1 : t+H, i] ), clamped within bounds.
        Shape: [T, N]. Near tail, the window shrinks naturally.
        """
        T, N = getattr(self, "num_timesteps", 0), getattr(self, "num_cryptos", 0)
        if T == 0 or N == 0:
            self._fwd_max_low = None
            return

        H = max(1, int(H))
        fml = np.zeros((T, N), dtype=np.float64)
        low = self.raw_low  # [T, N]

        # For each column, compute sliding forward max over (t+1 .. t+H)
        # using a monotonic deque in O(T) time per column.
        last_idx = T - 1
        for j in range(N):
            arr = low[:, j]
            dq: deque[int] = deque()

            # Initialize deque with indices 1..min(H, last_idx)
            upper = min(H, last_idx)
            for idx in range(1, upper + 1):
                while dq and arr[dq[-1]] <= arr[idx]:
                    dq.pop()
                dq.append(idx)

            for t in range(T):
                # Maintain window [t+1, min(t+H, last_idx)]
                left_bound = t + 1
                while dq and dq[0] < left_bound:
                    dq.popleft()

                # Read max (if window empty, fallback to arr[min(t+1, last_idx)])
                if dq:
                    fml[t, j] = arr[dq[0]]
                else:
                    fml[t, j] = arr[left_bound if left_bound <= last_idx else last_idx]

                # Advance window: include new index for next t
                new_idx = t + H + 1
                if new_idx <= last_idx:
                    while dq and arr[dq[-1]] <= arr[new_idx]:
                        dq.pop()
                    dq.append(new_idx)

        self._fwd_max_low = fml  # [T, N]

    # -------------
    # Gym methods
    # -------------
    def reset(self, *, seed=None, options=None):
        """
        Start a new single-decision episode; advances the local pointer by 1,
        wrapping safely so the history window is valid.
        """
        super().reset(seed=seed)

        self.pointer += 1
        max_start = self.num_timesteps - 2  # need at least one future step
        if self.pointer > max_start:
            self.pointer = max(self.history_window - 1, 0)

        self.current_step = self.pointer
        self._episode_started_at = self.current_step

        obs = self._get_observation()
        info = {
            "action_mask": self._get_action_mask(),
            "peak_mode": self.use_peak_within_window,
        }
        return obs, info

    def step(self, action: int):
        """
        Single-decision step with periodic dataset rollover.
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

            return obs, float(reward), terminated, truncated, {"action_mask": legal, "peak_mode": self.use_peak_within_window}

        # Map action (1..N) -> symbol index [0..N-1]
        chosen_idx = action - 1

        # Rank reward from future returns (stringent pricing; peak mode optional)
        future_returns = self._compute_future_returns(self.current_step, self.lookahead_steps)
        reward = self._rank_to_reward(future_returns, chosen_idx, mode=self.reward_mode)

        terminated, truncated = True, False
        obs = self._get_observation()
        info = {
            "action_mask": self._get_action_mask(),
            "horizon_used": getattr(self, "_last_horizon_used", self.lookahead_steps),
            "peak_mode": self.use_peak_within_window,
        }

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
        Per-symbol future return using STRINGENT pricing:
          - Entry at time t uses RAW HIGH (worst-case buy fill).
          - Exit uses RAW LOW with either:
              * end-of-window: Low[t+H]
              * peak-in-window: max Low over (t+1 .. t+H] if use_peak_within_window=True
        Returns:
            r_i = (Exit_i - Entry_i) / Entry_i
        """
        t0 = int(t)
        # Effective horizon if near the tail (at least 1 step forward)
        H_eff = max(1, min(int(H), (self.num_timesteps - 1) - t0))
        t1 = t0 + H_eff

        # Entry = pessimistic buy fill at current bar
        entry = self.raw_high[t0, :]  # [N]

        if self.use_peak_within_window:
            # Best pessimistic exit within the window: forward MAX of Low(t+1..t+H)
            if getattr(self, "_fwd_max_low", None) is not None:
                future_pess = self._fwd_max_low[t0, :]  # [N], already excludes t itself
            else:
                # Fallback: use end-of-window Low if cache unavailable
                future_pess = self.raw_low[t1, :]
        else:
            # End-of-window pessimistic exit
            future_pess = self.raw_low[t1, :]

        denom = np.where(entry != 0.0, entry, np.nan)
        ret = (future_pess - entry) / denom
        ret = np.nan_to_num(ret, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        # for logging
        self._last_horizon_used = H_eff
        return ret

    def _rank_to_reward(self, returns: np.ndarray, chosen_idx: int, mode: str = "zero_to_one") -> float:
        """
        Convert rank among all symbols into a scalar reward.
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

    # -------------------
    # Action mask helper
    # -------------------
    def _get_action_mask(self) -> np.ndarray:
        """
        Only BUY actions (1..N) are legal.
        """
        mask = np.zeros(self.action_space.n, dtype=bool)
        mask[1 : 1 + self.num_cryptos] = True
        return mask

    # -------------
    # Gym extras
    # -------------
    def render(self, mode="human"):
        pass

    def close(self):
        pass