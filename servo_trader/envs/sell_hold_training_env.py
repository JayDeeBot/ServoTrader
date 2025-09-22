#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sell_hold_training_env.py  (robust, chunking-enabled)

Single-decision HOLD/SELL pretraining environment with:
- Canonical symbol normalization on BOTH config codes & CSV chunks
- Tolerant detection of symbol column names (symbol/pair/ticker)
- Periodic dataset chunk loading (e.g., every 10k episodes)
- Future-return reward: HOLD = +r, SELL = -r (optional tanh bounding)

Update (stringent pricing):
- The future-return r now uses a *pessimistic* exit at the horizon:
    r = ( Low[t+H] - Close[t] ) / Close[t]
  This keeps rewards in **percentage return space** (normalized across assets),
  while being stricter than Close→Close.
"""

from __future__ import annotations

import os
import re
from typing import Literal, Tuple, List

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
    Lower-cases columns to be tolerant of casing. Adds 'symbol_norm'.
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
    df["symbol_norm"] = df["symbol"].map(_norm_symbol)
    return df


class SellHoldTrainingEnv(gym.Env):
    """
    Single-decision SELL/HOLD pretraining environment.

    At reset():
        - Advance a rolling time pointer.
        - Choose a *held* symbol i (random or round-robin).
    At step(action):
        - Legal actions: HOLD (0), SELL (N+1). Others masked.
        - Compute look-ahead return r_i(t→t+H) on raw prices (percentage).
        - Reward =  +r if HOLD,  -r if SELL  (optionally tanh-bounded).
        - done=True (single-step episode).

    Stringent pricing (this version):
        r_i = (Low[t+H, i] - Close[t, i]) / Close[t, i]
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        data: pd.DataFrame,
        crypto_codes: List[str],
        episode_timeout: int = 60,
        history_window: int = 5,
        lookahead_steps: int = 1,
        held_selection: Literal["random", "round_robin"] = "random",
        reward_bounded: bool = False,
        seed: int | None = None,
        # --- Chunking knobs (match buy env) ---
        chunk_dir: str | None = "/home/jarred/git/ServoTrader/data/split_10k_chunks_ancient_2",
        chunk_size: int = 10_000,
        start_chunk_index: int = 0,          # index for the *initial* `data`
        enable_chunking: bool | None = None, # None => auto True if dir exists
    ):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.low_to_low_one_step = True  # Set reward calculation mode

        # --- Config (normalize configured codes) ---
        self.crypto_codes = sorted({_norm_symbol(c) for c in crypto_codes})
        self.timeout_steps = int(episode_timeout)
        self.history_window = int(history_window)
        self.lookahead_steps = int(lookahead_steps)
        self.held_selection = held_selection
        self.reward_bounded = reward_bounded

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
        obs_len = (
            self.history_window * self.num_cryptos * self.features_per_crypto
            + 2  # time_remaining, current_profit
            + (self.num_cryptos + 1)  # held_one_hot
        )
        low = np.zeros(obs_len, dtype=np.float32)
        # allow negative current_profit slot (even though we keep it 0 here)
        low[-(self.num_cryptos + 1 + 1)] = -1.0
        high = np.ones(obs_len, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Episode state
        self.pointer = max(0, self.history_window - 1)  # rolling time index
        self.current_step = self.pointer
        self.held_index: int | None = None

        # For round-robin symbol selection
        self._round_idx = 0

        # Global step counter (episodes == steps for single-decision env)
        self.global_step = 0

        print(
            f"[SellHoldEnv] init: CSV syms(norm)={len(set(self.raw_df['symbol_norm']))} | "
            f"JSON codes(norm)={len(set(self.crypto_codes))} | Using={self.num_cryptos} symbols"
        )

    # -------------------------------------------------------
    # Internal: construct tensors from a (normalized) dataframe
    # -------------------------------------------------------
    def _rebuild_lookups_and_tensors(self, df: pd.DataFrame, first_time: bool = False) -> None:
        """
        Given a dataframe with 'symbol' and 'symbol_norm':
          - Intersect with configured codes (already normalized)
          - Build data tensor [T, N, F] and raw price matrices
              * raw_close: [T, N]
              * raw_low  : [T, N]
          - Update env shapes / bookkeeping
        """
        # Group lookups keyed by normalized symbol
        self.raw_lookup = {sym: g.reset_index(drop=True) for sym, g in df.groupby("symbol_norm")}

        # Build tensors
        data, raw_close, used_codes = self._preprocess_data(df)

        self.data = data
        self.raw_close = raw_close
        # self.raw_low is attached inside _preprocess_data
        self.num_timesteps, self.num_cryptos, self.features_per_crypto = self.data.shape
        self.crypto_codes = used_codes  # normalized, ordered

        if first_time:
            pass
        else:
            # On chunk reload, reset local pointer to a safe start
            self.pointer = max(0, self.history_window - 1)
            self.current_step = self.pointer

    # -------------------
    # Data preprocessing
    # -------------------
    def _preprocess_data(self, raw_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Convert stacked dataframe into:
          - data_tensor: [T, N, F] normalized features
          - raw_close  : [T, N]     raw close prices  (decision-time mark)
        using NORMALIZED symbols for membership/grouping.

        Note:
          We also extract and attach `self.raw_low` for stringent reward modelling:
            r = (Low[t+H] - Close[t]) / Close[t]
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

        # Equal length across symbols (clamp to min available T)
        T = raw_df.groupby("symbol_norm").size().min()

        feature_cols = ["open", "high", "low", "close", "vwap", "volume", "count"]
        added = ["recent_return", "volatility", "price_position", "volume_surge", "trend_slope", "moving_avg"]
        all_cols = feature_cols + added
        F = len(all_cols)

        data_tensor = np.zeros((T, N, F), dtype=np.float32)
        raw_close   = np.zeros((T, N), dtype=np.float64)
        raw_low     = np.zeros((T, N), dtype=np.float64)  # ← store future exit baseline (stringent)

        window = max(2, int(self.timeout_steps))

        for j, sym_norm in enumerate(overlap):
            df = (
                raw_df.loc[raw_df["symbol_norm"] == sym_norm, feature_cols]
                .head(T)
                .astype(float)
                .copy()
            )

            # Fill prices forward/back to avoid small gaps
            for c in ["open", "high", "low", "close", "vwap"]:
                df[c] = df[c].ffill().bfill()

            # Fill volume/count with zeros
            df[["volume", "count"]] = df[["volume", "count"]].fillna(0.0)

            # Preserve raw prices BEFORE normalization for rewards
            raw_close[:, j] = df["close"].to_numpy()
            raw_low[:, j]   = df["low"].to_numpy()

            # Engineered features (computed on CLOSE for stability)
            df["recent_return"] = df["close"].pct_change(periods=window).fillna(0.0)
            df["volatility"] = df["close"].rolling(window).std().fillna(0.0)

            hroll = df["high"].rolling(window).max()
            lroll = df["low"].rolling(window).min()
            df["price_position"] = ((df["close"] - lroll) / (hroll - lroll + 1e-6)).fillna(0.0)

            vavg = df["volume"].rolling(window).mean()
            df["volume_surge"] = (df["volume"] / (vavg + 1e-6)).fillna(0.0)

            df["trend_slope"] = df["close"].rolling(window).apply(slope_func, raw=False).fillna(0.0)
            df["moving_avg"] = df["close"].rolling(window).mean().bfill()

            # Normalize prices with min-max per symbol (0..1)
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

        # Attach raws for reward computation
        self.raw_low = raw_low

        return data_tensor, raw_close, overlap  # overlap is normalized symbol order

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
        Start a new episode:
          - Advance time pointer (wrap safely with lookahead margin).
          - Select a held symbol (random or round-robin).
          - Return observation and HOLD/SELL mask.
        """
        super().reset(seed=seed)

        # Advance pointer; keep one future step for returns
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
            - HOLD (0):   reward = +future_return(held)
            - SELL (N+1): reward = -future_return(held)
            - Episode terminates immediately.

        Also rolls over dataset every `chunk_size` episodes if chunking is enabled.
        """
        mask = self._get_action_mask()
        if action < 0 or action >= self.action_space.n or not mask[action]:
            # Illegal action → zero reward, terminate to keep training robust
            reward = 0.0
            obs = self._get_observation()
            terminated, truncated = True, False

            # Count the episode and maybe roll the chunk
            self.global_step += 1
            if self.enable_chunking and self.chunk_size > 0 and (self.global_step % self.chunk_size == 0):
                self._load_next_chunk()

            return obs, float(reward), terminated, truncated, {"action_mask": mask}

        # Compute percentage future return on the held symbol (stringent: Low at horizon)
        r = self._future_return(self.current_step, int(self.held_index), self.lookahead_steps)

        # Symmetric reward design preserves credit assignment:
        reward = r if action == 0 else -r  # HOLD vs SELL
        if self.reward_bounded:
            # Optional squashing to [-1, 1] for stability on volatile series
            reward = float(np.tanh(reward))

        # Single-step episode ends
        terminated, truncated = True, False
        obs = self._get_observation()

        # Episode complete; increment global step and possibly roll chunk
        self.global_step += 1
        if self.enable_chunking and self.chunk_size > 0 and (self.global_step % self.chunk_size == 0):
            self._load_next_chunk()

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
        time_remaining = np.array([1.0], dtype=np.float32)
        current_profit = np.array([0.0], dtype=np.float32)

        # Held crypto one-hot (N+1): index 0 means "none"; here we *do* hold one
        held_hot = np.zeros(self.num_cryptos + 1, dtype=np.float32)
        idx = int(self.held_index) + 1 if self.held_index is not None else 0
        held_hot[idx] = 1.0

        full = np.concatenate([obs_vec, time_remaining, current_profit, held_hot]).astype(np.float32)

        if np.any(~np.isfinite(full)):
            full = np.nan_to_num(full, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

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
        Compute **percentage** future return for symbol i from t to t+H
        using a *stringent* exit at the horizon:

            If low_to_low_one_step: one-step % change in Low:
            r = (Low[t+1,i] - Low[t,i]) / Low[t,i]
            Else (default stringent horizon): (Low[t+H,i] - Close[t,i]) / Close[t,i]

        Notes:
        - We clamp t+H within dataset bounds; ensure at least 1 step of lookahead.
        - Returns 0 if denominator is zero/invalid.
        - This keeps reward magnitudes normalized across cryptos (percentage space),
          but is stricter than Close→Close because it marks exit at the worst
          price of the future bar.

        Args:
            t (int): current time index
            i (int): held symbol index
            H (int): lookahead horizon (in bars)

        Returns:
            float: percentage return (can be negative/positive)
        """
        if getattr(self, "low_to_low_one_step", False):
            t0 = t
            t1 = min(self.num_timesteps - 1, t + 1)
            base = float(self.raw_low[t0, i])
            futr = float(self.raw_low[t1, i])
            if base == 0.0 or not np.isfinite(base) or not np.isfinite(futr):
                return 0.0
            return (futr - base) / base

        # fallback: current stringent (Low at horizon vs Close now)
        t0 = t
        t1 = min(self.num_timesteps - 1, t + max(1, H))
        base = float(self.raw_close[t0, i])
        futr = float(self.raw_low[t1, i])
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