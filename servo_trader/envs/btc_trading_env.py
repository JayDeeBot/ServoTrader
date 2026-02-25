"""
btc_trading_env.py

Custom OpenAI Gymnasium environment for single-asset BTC trading using the 14
Boruta-confirmed hourly features. Designed for use with a custom PPO Transformer
training pipeline.

Episode Structure
-----------------
Each episode starts at a random timestep in the dataset. The agent cycles
through two phases:

  Phase 1 – Pre-trade (no position held):
      Legal actions: NOT_BUY (0), BUY (1)
      The agent may wait as many steps as it likes before entering a trade.

  Phase 2 – In-trade (position held):
      Legal actions: HOLD (2), SELL (3)
      The agent must eventually sell to end the episode.
      A max_hold_steps ceiling prevents indefinite holding from stalling training.

Action Space (Discrete, 4)
--------------------------
    0: NOT_BUY  – skip this candle (Phase 1 only)
    1: BUY      – enter a long position at close price (Phase 1 only)
    2: HOLD     – maintain the current position (Phase 2 only)
    3: SELL     – exit the position at close price (Phase 2 only)

Observation Space
-----------------
Flat vector of shape (24 × 14 + 3,) = (339,):
    • 24 history steps × 14 Boruta features  (sequence window)
    • unrealized_pnl         – float, mark-to-market P&L as fraction
    • position_flag          – 0.0 (no position) or 1.0 (in position)
    • steps_in_trade_norm    – steps held / max_hold_steps ∈ [0, 1]

Trading Cost
------------
A 0.3% round-trip cost (0.15% each side) is applied at SELL:
    realized_pnl = (sell_price / buy_price) - 1.0 - 0.003

Reward Shaping
--------------
    NOT_BUY : 0.0
    BUY     : 0.0
    HOLD    : Δ unrealized PnL since last step  (dense mark-to-market signal)
    SELL    : realized_pnl (includes -0.003 cost)

Logging
-------
Produces a JSONL log per training run in `log_dir` for post-training analysis.
Each episode_end record contains steps, final PnL, and trade details.

Author: Jarred Deluca
Project: ServoTrader
License: MIT
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import os
import json
from datetime import datetime
from typing import Optional, Tuple, Dict, Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRADE_COST = 0.003          # 0.3% round-trip cost, deducted at SELL
ACTION_NOT_BUY  = 0
ACTION_BUY      = 1
ACTION_HOLD     = 2
ACTION_SELL     = 3

# Boruta-confirmed feature columns (order must match the dataset CSV)
BORUTA_FEATURES = [
    "rsi_24",
    "adx_14",
    "stoch_k",
    "stoch_d",
    "bb_percent_b_48",
    "bb_percent_b_24",
    "volume_ratio_24",
    "vwap_deviation",
    "returns_6h",
    "bb_percent_b_12",
    "returns_1h",
    "volatility_6h",
    "rsi_6",
    "trends_bitcoin_zscore",
]

N_FEATURES       = len(BORUTA_FEATURES)   # 14
HISTORY_WINDOW   = 24                     # candles of context fed as a sequence (tokens)
N_CONTEXT        = 3                      # unrealized_pnl, position_flag, steps_in_trade_norm
OBS_DIM          = HISTORY_WINDOW * N_FEATURES + N_CONTEXT   # 24*14+3 = 339


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class BTCTradingEnv(gym.Env):
    """
    Single-asset BTC trading environment for PPO Transformer training.

    Parameters
    ----------
    df : pd.DataFrame
        Pre-processed dataset containing at minimum all columns in BORUTA_FEATURES
        plus a 'close' column for execution prices.
    max_hold_steps : int
        Maximum number of candles the agent may hold a position before a forced
        sell terminates the episode. Prevents degenerate hold-forever strategies
        during early training (default: 200).
    log_dir : str
        Directory for JSONL episode logs.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        max_hold_steps: int = 200,
        log_dir: str = "/home/jarred/git/ServoTrader/logs",
    ):
        super().__init__()

        # ---- Data ----------------------------------------------------------
        self._validate_columns(df)
        self.df          = df.reset_index(drop=True)
        self.n_rows      = len(self.df)
        self.close_prices = self.df["close"].to_numpy(dtype=np.float64)

        # Normalised feature matrix: shape [n_rows, 14]
        self.feature_matrix = self._build_feature_matrix()

        # ---- Config --------------------------------------------------------
        self.max_hold_steps = int(max_hold_steps)

        # ---- Spaces --------------------------------------------------------
        self.action_space = spaces.Discrete(4)   # NOT_BUY, BUY, HOLD, SELL

        low  = np.full(OBS_DIM, -10.0, dtype=np.float32)
        high = np.full(OBS_DIM,  10.0, dtype=np.float32)
        # Context slots have tighter bounds
        low[-N_CONTEXT:]  = np.array([-5.0, 0.0, 0.0], dtype=np.float32)
        high[-N_CONTEXT:] = np.array([ 5.0, 1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # ---- State ---------------------------------------------------------
        self.current_step    : int   = 0
        self.in_position     : bool  = False
        self.buy_price       : float = 0.0
        self.prev_price      : float = 0.0
        self.steps_in_trade  : int   = 0
        self.episode_start   : int   = 0

        # ---- Logging -------------------------------------------------------
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path      = os.path.join(self.log_dir, f"btc_env_{ts}.jsonl")
        self.episode_log   : list    = []
        self.episode_counter: int    = 0
        self.total_profit_pct: float = 0.0

        # Per-episode trade tracking
        self._buy_step  : Optional[int]   = None
        self._buy_price : float           = 0.0

        print(
            f"[BTCTradingEnv] Loaded {self.n_rows:,} rows | "
            f"OBS={OBS_DIM} | max_hold={self.max_hold_steps}"
        )

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        # Start at a random timestep that leaves enough history + future candles
        min_start = HISTORY_WINDOW
        max_start = self.n_rows - self.max_hold_steps - 2
        if max_start <= min_start:
            max_start = min_start + 1
        self.current_step = int(self.np_random.integers(min_start, max_start))
        self.episode_start = self.current_step

        # Reset state
        self.in_position    = False
        self.buy_price      = 0.0
        self.prev_price     = self.close_prices[self.current_step]
        self.steps_in_trade = 0
        self._buy_step      = None
        self._buy_price     = 0.0

        self.episode_log = []
        self.episode_log.append({
            "type":        "episode_start",
            "episode":     int(self.episode_counter + 1),
            "start_step":  int(self.current_step),
            "timestamp":   datetime.now().isoformat(),
        })

        obs    = self._get_observation()
        info   = {"action_mask": self._get_action_mask()}
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        reward      = 0.0
        terminated  = False
        truncated   = False
        profit_pct  = 0.0

        current_price = self.close_prices[self.current_step]

        # ------------------------------------------------------------------ #
        #  NOT_BUY                                                            #
        # ------------------------------------------------------------------ #
        if action == ACTION_NOT_BUY:
            if self.in_position:
                # Illegal move during Phase 2 — small penalty to teach legal moves
                reward = -0.01
            else:
                reward = 0.0   # Neutral: waiting is free but not rewarded

        # ------------------------------------------------------------------ #
        #  BUY                                                                #
        # ------------------------------------------------------------------ #
        elif action == ACTION_BUY:
            if self.in_position:
                reward = -0.01   # Illegal; already holding
            else:
                self.in_position    = True
                self.buy_price      = current_price
                self.steps_in_trade = 0
                self._buy_step      = int(self.current_step)
                self._buy_price     = current_price
                reward = 0.0   # Neutral at entry; cost reflected at SELL

                self._log_step(action, "buy", current_price, 0.0, reward)

        # ------------------------------------------------------------------ #
        #  HOLD                                                               #
        # ------------------------------------------------------------------ #
        elif action == ACTION_HOLD:
            if not self.in_position:
                reward = -0.01   # Illegal; nothing to hold
            else:
                # Dense mark-to-market reward: change in unrealized PnL this step
                prev_unrealized = (self.prev_price  - self.buy_price) / max(self.buy_price, 1e-12)
                curr_unrealized = (current_price    - self.buy_price) / max(self.buy_price, 1e-12)
                reward = float(curr_unrealized - prev_unrealized)
                self.steps_in_trade += 1

                self._log_step(action, "hold", current_price,
                               curr_unrealized * 100, reward)

                # Forced sell if max_hold_steps exceeded
                if self.steps_in_trade >= self.max_hold_steps:
                    profit_pct, reward_adj = self._compute_sell(current_price)
                    reward     += reward_adj   # Layer forced-exit sell reward on top
                    truncated   = True
                    self._log_step(ACTION_SELL, "sell_forced", current_price, profit_pct, reward_adj)
                    self._finalise_episode(profit_pct, terminated=False, truncated=True)
                    self._advance_step()
                    return self._get_observation(), float(reward), terminated, truncated, {}

        # ------------------------------------------------------------------ #
        #  SELL                                                               #
        # ------------------------------------------------------------------ #
        elif action == ACTION_SELL:
            if not self.in_position:
                reward = -0.01   # Illegal; nothing to sell
            else:
                profit_pct, reward = self._compute_sell(current_price)
                terminated = True
                self._log_step(action, "sell", current_price, profit_pct, reward)
                self._finalise_episode(profit_pct, terminated=True, truncated=False)
                self._advance_step()
                return self._get_observation(), float(reward), terminated, truncated, {}

        # Record previous price for delta calculation
        self.prev_price = current_price
        self._advance_step()

        obs  = self._get_observation()
        info = {"action_mask": self._get_action_mask()}
        return obs, float(reward), terminated, truncated, info

    def get_action_mask(self) -> np.ndarray:
        """Convenience accessor for external wrappers."""
        return self._get_action_mask()

    def render(self, mode: str = "human"):
        price = self.close_prices[self.current_step] if self.current_step < self.n_rows else 0.0
        status = "HOLDING" if self.in_position else "WAITING"
        print(f"[BTCEnv] step={self.current_step} | {status} | price={price:.2f}")

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _validate_columns(self, df: pd.DataFrame):
        missing = [c for c in BORUTA_FEATURES + ["close"] if c not in df.columns]
        if missing:
            raise ValueError(
                f"[BTCTradingEnv] Dataset missing required columns: {missing}\n"
                f"Available: {list(df.columns)}"
            )

    def _build_feature_matrix(self) -> np.ndarray:
        """
        Extracts and normalises the 14 Boruta features from the dataframe.
        Per-feature z-score normalisation (clip to ±5σ) for stable training.
        Returns float32 array of shape [n_rows, 14].
        """
        raw = self.df[BORUTA_FEATURES].to_numpy(dtype=np.float64)

        # Fill any NaNs introduced by rolling calculations
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)

        mu  = raw.mean(axis=0, keepdims=True)
        sig = raw.std(axis=0, keepdims=True) + 1e-8
        normalized = np.clip((raw - mu) / sig, -5.0, 5.0)
        return normalized.astype(np.float32)

    def _get_observation(self) -> np.ndarray:
        """
        Constructs the (339,) observation vector:
          [history_window × 14 features (flattened)] + [pnl, position_flag, steps_norm]
        """
        start = max(0, self.current_step - HISTORY_WINDOW)
        end   = self.current_step   # exclusive: rows [start, end)

        window = self.feature_matrix[start:end]   # shape [≤24, 14]

        # Pad at the front if we don't have a full window yet
        pad_len = HISTORY_WINDOW - window.shape[0]
        if pad_len > 0:
            window = np.concatenate(
                [np.zeros((pad_len, N_FEATURES), dtype=np.float32), window], axis=0
            )

        feature_vec = window.flatten()   # (336,)

        # Context features
        if self.in_position and self.buy_price > 0:
            cur_price       = self.close_prices[min(self.current_step, self.n_rows - 1)]
            unrealized_pnl  = float(np.clip(
                (cur_price - self.buy_price) / max(self.buy_price, 1e-12), -5.0, 5.0
            ))
        else:
            unrealized_pnl  = 0.0

        position_flag      = 1.0 if self.in_position else 0.0
        steps_in_trade_norm = float(
            np.clip(self.steps_in_trade / max(self.max_hold_steps, 1), 0.0, 1.0)
        )

        context = np.array(
            [unrealized_pnl, position_flag, steps_in_trade_norm], dtype=np.float32
        )

        obs = np.concatenate([feature_vec, context])
        obs = np.nan_to_num(obs, nan=0.0, posinf=5.0, neginf=-5.0)
        return obs

    def _get_action_mask(self) -> np.ndarray:
        """
        Boolean mask over the 4 actions.
        Phase 1 (no position): NOT_BUY and BUY are legal.
        Phase 2 (in position): HOLD and SELL are legal.
        """
        mask = np.zeros(4, dtype=bool)
        if not self.in_position:
            mask[ACTION_NOT_BUY] = True
            mask[ACTION_BUY]     = True
        else:
            mask[ACTION_HOLD]    = True
            mask[ACTION_SELL]    = True
        return mask

    def _compute_sell(self, sell_price: float) -> Tuple[float, float]:
        """
        Computes realised PnL and sell reward, applying 0.3% round-trip cost.

        Returns
        -------
        profit_pct : float   (percentage, e.g. 1.5 = 1.5%)
        reward     : float   (fractional PnL − cost, used as RL reward)
        """
        raw_return = (sell_price / max(self.buy_price, 1e-12)) - 1.0
        reward     = float(raw_return - TRADE_COST)
        profit_pct = float(reward * 100.0)

        # Reset position state
        self.in_position    = False
        self.steps_in_trade = 0
        self.buy_price      = 0.0

        return profit_pct, reward

    def _advance_step(self):
        """Move the timestep pointer forward, wrapping to avoid out-of-bounds."""
        self.current_step += 1
        if self.current_step >= self.n_rows:
            self.current_step = HISTORY_WINDOW   # Wrap to start of valid range

    def _finalise_episode(self, profit_pct: float, *, terminated: bool, truncated: bool):
        """Writes episode summary to the JSONL log."""
        self.episode_counter   += 1
        self.total_profit_pct  += profit_pct
        steps_taken = self.current_step - self.episode_start

        summary = {
            "type":            "episode_end",
            "episode":         int(self.episode_counter),
            "steps":           int(steps_taken),
            "buy_step":        self._buy_step,
            "buy_price":       float(self._buy_price) if self._buy_price else None,
            "profit_pct":      float(profit_pct),
            "total_profit_pct": float(self.total_profit_pct),
            "termination":     "sell" if terminated else ("forced_sell" if truncated else "unknown"),
            "timestamp":       datetime.now().isoformat(),
        }
        self.episode_log.append(summary)

        # Flush to disk
        with open(self.log_path, "a") as f:
            for record in self.episode_log:
                f.write(json.dumps(record) + "\n")
        self.episode_log = []

    def _log_step(
        self,
        action:     int,
        action_str: str,
        price:      float,
        profit_pct: float,
        reward:     float,
    ):
        self.episode_log.append({
            "type":       "step",
            "step":       int(self.current_step),
            "action":     int(action),
            "action_str": action_str,
            "price":      float(price),
            "profit_pct": float(profit_pct),
            "reward":     float(reward),
        })