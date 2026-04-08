"""
btc_trading_env_5m.py  —  v6

5-minute BTC single-trade environment for MLP PPO training.

Episode Structure
-----------------
  Each episode = one trade. A random start position is sampled from the
  dataset, the agent waits in Phase 1 (NOT_BUY / BUY) until it decides
  to enter, then manages the position in Phase 2 (HOLD / SELL) until it
  exits or hits the max_hold ceiling.

  Phase 1 (no position)  → legal: NOT_BUY (0), BUY (1)
  Phase 2 (in position)  → legal: HOLD (2), SELL (3)

Why single-trade episodes (not multi-trade)?
---------------------------------------------
  Multi-trade episodes (v5) caused the agent to learn NOT_BUY 56% of
  steps because the 0.5% stop-loss triggered on normal market noise 40%
  of the time, making trading systematically more costly than waiting.
  The agent's response was rational — the reward signal was broken.

  Single-trade episodes are simpler to debug, have cleaner credit
  assignment (one entry / one exit per episode), and v4 demonstrated
  the agent CAN learn profitable behaviour in this structure — it peaked
  at 58% win rate and +0.135% mean profit at step 1.94M before running
  out of training budget.

Stop-loss
---------
  No stop-loss in v6. The 0.5% threshold in v5 triggered on random BTC
  noise ~40% of the time (BTC 5-min vol ≈ 0.26%/candle; expected move
  over 12 candles = 0.90% 1-sigma). Removing it lets the agent learn
  exit timing purely from HOLD mark-to-market rewards and the max_hold
  ceiling. If a stop-loss is added in future, use ≥ 1.5% (1.7× expected
  random move) to avoid triggering on noise.

Reward Shaping
--------------
  NOT_BUY : −0.0001 per step (tiny wait cost; ≤ 0.003 over 30 steps)
  BUY     : entry quality penalty — proportional to expected loss over
            min_hold_steps, capped at −TRADE_COST. Zero for good entries.
  HOLD    : Δ unrealized PnL (mark-to-market per candle)
  SELL    : realized_pnl − TRADE_COST (0.003)

Key Parameters
--------------
  HISTORY_WINDOW : 48 candles  (4 hours of 5-min context)
  min_hold_steps : 6  candles  (30 min minimum hold)
  max_hold_steps : 72 candles  (6-hour ceiling, then forced SELL)

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
from typing import Optional, Tuple, Dict

# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------

TRADE_COST     = 0.003
ACTION_NOT_BUY = 0
ACTION_BUY     = 1
ACTION_HOLD    = 2
ACTION_SELL    = 3

# 38 Boruta-confirmed 5-minute features (order must match the dataset CSV)
BORUTA_FEATURES = [
    "bb_bandwidth_48", "bb_percent_b_48", "returns_24p", "returns_12p",
    "bb_bandwidth_72", "bb_percent_b_72", "returns_6p",  "returns_3p",
    "returns_1p",      "vwap_deviation",  "stoch_k",     "obv_ma_48",
    "returns_mean_24p","volume_ma_48",    "volume_ma_24","volume_ma_12",
    "returns_std_144p","returns_mean_144p","atr_12",     "atr_24",
    "atr_48",          "obv",             "returns_48p", "stoch_d",
    "bb_percent_b_24", "rsi_12",          "rsi_24",      "returns_72p",
    "rsi_48",          "rsi_72",          "macd",        "macd_signal",
    "macd_histogram",  "bb_percent_b_12", "returns_mean_48p",
    "bb_bandwidth_24", "rsi_6",           "count",
]

N_FEATURES     = len(BORUTA_FEATURES)               # 38
HISTORY_WINDOW = 48                                  # 4 hours of 5-min candles
N_CONTEXT      = 3                                   # unrealized_pnl, position_flag, hold_progress
OBS_DIM        = HISTORY_WINDOW * N_FEATURES + N_CONTEXT   # 1,827


# ---------------------------------------------------------------------------
#  Environment
# ---------------------------------------------------------------------------

class BTCTradingEnv5m(gym.Env):
    """
    5-minute BTC single-trade environment for MLP PPO (v6).

    Parameters
    ----------
    df             : Pre-processed DataFrame with BORUTA_FEATURES + 'close'.
    max_hold_steps : Candles before forced exit (default 72 → 6 hours).
    min_hold_steps : Candles before SELL becomes legal (default 6 → 30 min).
    log_dir        : Directory for JSONL episode logs.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df:             pd.DataFrame,
        max_hold_steps: int = 72,
        min_hold_steps: int = 6,
        log_dir:        str = "/home/jarred/git/ServoTrader/logs",
    ):
        super().__init__()

        self._validate_columns(df)
        self.df           = df.reset_index(drop=True)
        self.n_rows       = len(self.df)
        self.close_prices = self.df["close"].to_numpy(dtype=np.float64)
        self.feature_matrix  = self._build_feature_matrix()

        self.max_hold_steps = int(max_hold_steps)
        self.min_hold_steps = int(min_hold_steps)

        # Action / observation spaces
        self.action_space = spaces.Discrete(4)
        low  = np.full(OBS_DIM, -10.0, dtype=np.float32)
        high = np.full(OBS_DIM,  10.0, dtype=np.float32)
        low[-N_CONTEXT:]  = np.array([-5.0, 0.0, 0.0], dtype=np.float32)
        high[-N_CONTEXT:] = np.array([ 5.0, 1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Episode state
        self.current_step     : int   = 0
        self.in_position      : bool  = False
        self.buy_price        : float = 0.0
        self.prev_price       : float = 0.0
        self.steps_in_trade   : int   = 0
        self.episode_steps    : int   = 0   # explicit counter, avoids wrap-around bug
        self.episode_not_buys : int   = 0   # NOT_BUY decisions this episode

        # Per-trade logging state
        self._buy_step  : Optional[int] = None
        self._buy_price : float         = 0.0

        # JSONL logging
        self.log_dir          = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path         = os.path.join(self.log_dir, f"btc5m_env_{ts}.jsonl")
        self.episode_log      : list  = []
        self.episode_counter  : int   = 0
        self.total_profit_pct : float = 0.0

        print(
            f"[BTCTradingEnv5m v6] {self.n_rows:,} rows | OBS={OBS_DIM} | "
            f"min_hold={self.min_hold_steps} ({self.min_hold_steps*5}min) | "
            f"max_hold={self.max_hold_steps} ({self.max_hold_steps*5//60}h) | "
            f"no stop-loss"
        )

    # -----------------------------------------------------------------------
    #  Public API
    # -----------------------------------------------------------------------

    def reset(self, *, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        min_start = HISTORY_WINDOW
        max_start = self.n_rows - self.max_hold_steps - 2
        if max_start <= min_start:
            max_start = min_start + 1

        self.current_step     = int(self.np_random.integers(min_start, max_start))
        self.in_position      = False
        self.buy_price        = 0.0
        self.prev_price       = self.close_prices[self.current_step]
        self.steps_in_trade   = 0
        self.episode_steps    = 0
        self.episode_not_buys = 0
        self._buy_step        = None
        self._buy_price       = 0.0

        self.episode_log = [{
            "type":       "episode_start",
            "episode":    int(self.episode_counter + 1),
            "start_step": int(self.current_step),
            "timestamp":  datetime.now().isoformat(),
        }]

        obs  = self._get_observation()
        info = {"action_mask": self._get_action_mask()}
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        reward     = 0.0
        terminated = False
        truncated  = False
        profit_pct = 0.0

        current_price = self.close_prices[self.current_step]

        # ── Phase 1: no position ──────────────────────────────────────────
        if action == ACTION_NOT_BUY:
            if self.in_position:
                reward = -0.01   # illegal in Phase 2
            else:
                reward = -0.0001
                self.episode_not_buys += 1
                self._log_step(action, "not_buy", current_price, 0.0, reward)

        elif action == ACTION_BUY:
            if self.in_position:
                reward = -0.01
            else:
                # Entry quality shaping: look ahead min_hold_steps candles.
                # Penalise buying into a declining market, proportional to
                # expected loss, capped at one trading cost.
                future_idx    = min(self.current_step + self.min_hold_steps, self.n_rows - 1)
                future_price  = self.close_prices[future_idx]
                hypothetical  = (future_price / max(current_price, 1e-12)) - 1.0
                entry_penalty = max(hypothetical, -TRADE_COST) if hypothetical < 0 else 0.0

                self.in_position    = True
                self.buy_price      = current_price
                self.steps_in_trade = 0
                self._buy_step      = int(self.current_step)
                self._buy_price     = current_price
                reward              = entry_penalty
                self._log_step(action, "buy", current_price, 0.0, reward)

        # ── Phase 2: in position ──────────────────────────────────────────
        elif action == ACTION_HOLD:
            if not self.in_position:
                reward = -0.01
            else:
                prev_unr = (self.prev_price  - self.buy_price) / max(self.buy_price, 1e-12)
                curr_unr = (current_price    - self.buy_price) / max(self.buy_price, 1e-12)
                reward   = float(curr_unr - prev_unr)
                self.steps_in_trade += 1
                self._log_step(action, "hold", current_price, curr_unr * 100, reward)

                # Forced exit at max_hold ceiling
                if self.steps_in_trade >= self.max_hold_steps:
                    profit_pct, sell_r = self._compute_sell(current_price)
                    reward    += sell_r
                    truncated  = True
                    self._log_step(ACTION_SELL, "sell_forced", current_price, profit_pct, sell_r)
                    self._finalise_episode(profit_pct, terminated=False, truncated=True)
                    self._advance_step()
                    return self._get_observation(), float(reward), terminated, truncated, {}

        elif action == ACTION_SELL:
            if not self.in_position:
                reward = -0.01
            else:
                profit_pct, reward = self._compute_sell(current_price)
                terminated         = True
                self._log_step(action, "sell", current_price, profit_pct, reward)
                self._finalise_episode(profit_pct, terminated=True, truncated=False)
                self._advance_step()
                return self._get_observation(), float(reward), terminated, truncated, {}

        self.prev_price = current_price
        self._advance_step()
        obs  = self._get_observation()
        info = {"action_mask": self._get_action_mask()}
        return obs, float(reward), terminated, truncated, info

    def get_action_mask(self) -> np.ndarray:
        return self._get_action_mask()

    def render(self, mode="human"):
        price  = self.close_prices[min(self.current_step, self.n_rows - 1)]
        status = "HOLDING" if self.in_position else "WAITING"
        print(f"[BTCEnv5m v6] step={self.current_step} | {status} | price={price:.2f}")

    # -----------------------------------------------------------------------
    #  Internals
    # -----------------------------------------------------------------------

    def _validate_columns(self, df: pd.DataFrame):
        missing = [c for c in BORUTA_FEATURES + ["close"] if c not in df.columns]
        if missing:
            raise ValueError(
                f"[BTCTradingEnv5m] Missing columns: {missing}\n"
                f"Run prepare_btc_5min_dataset.py first."
            )

    def _build_feature_matrix(self) -> np.ndarray:
        """Z-score normalise all 38 features, clip to ±5σ."""
        raw = self.df[BORUTA_FEATURES].to_numpy(dtype=np.float64)
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        mu  = raw.mean(axis=0, keepdims=True)
        sig = raw.std(axis=0,  keepdims=True) + 1e-8
        return np.clip((raw - mu) / sig, -5.0, 5.0).astype(np.float32)

    def _get_observation(self) -> np.ndarray:
        start  = max(0, self.current_step - HISTORY_WINDOW)
        window = self.feature_matrix[start:self.current_step]
        pad    = HISTORY_WINDOW - window.shape[0]
        if pad > 0:
            window = np.concatenate(
                [np.zeros((pad, N_FEATURES), dtype=np.float32), window], axis=0
            )
        feature_vec = window.flatten()   # (1,824,)

        if self.in_position and self.buy_price > 0:
            cp  = self.close_prices[min(self.current_step, self.n_rows - 1)]
            unr = float(np.clip((cp - self.buy_price) / max(self.buy_price, 1e-12), -5.0, 5.0))
        else:
            unr = 0.0

        context = np.array([
            unr,
            1.0 if self.in_position else 0.0,
            float(np.clip(self.steps_in_trade / max(self.max_hold_steps, 1), 0.0, 1.0)),
        ], dtype=np.float32)

        obs = np.concatenate([feature_vec, context])
        return np.nan_to_num(obs, nan=0.0, posinf=5.0, neginf=-5.0)

    def _get_action_mask(self) -> np.ndarray:
        mask = np.zeros(4, dtype=bool)
        if not self.in_position:
            mask[ACTION_NOT_BUY] = True
            mask[ACTION_BUY]     = True
        else:
            mask[ACTION_HOLD] = True
            # SELL only legal after min_hold_steps — prevents BUY→SELL-immediately
            if self.steps_in_trade >= self.min_hold_steps:
                mask[ACTION_SELL] = True
        return mask

    def _compute_sell(self, sell_price: float) -> Tuple[float, float]:
        raw_return  = (sell_price / max(self.buy_price, 1e-12)) - 1.0
        reward      = float(raw_return - TRADE_COST)
        profit_pct  = float(reward * 100.0)
        self.in_position    = False
        self.steps_in_trade = 0
        self.buy_price      = 0.0
        return profit_pct, reward

    def _advance_step(self):
        self.episode_steps += 1
        self.current_step  += 1
        # Wrap to keep within dataset bounds (single-trade; wrap is fine)
        if self.current_step >= self.n_rows:
            self.current_step = HISTORY_WINDOW

    def _finalise_episode(self, profit_pct: float, *, terminated: bool, truncated: bool):
        self.episode_counter  += 1
        self.total_profit_pct += profit_pct
        summary = {
            "type":             "episode_end",
            "episode":          int(self.episode_counter),
            "steps":            int(self.episode_steps),
            "not_buy_steps":    int(self.episode_not_buys),
            "buy_step":         self._buy_step,
            "buy_price":        float(self._buy_price) if self._buy_price else None,
            "profit_pct":       float(profit_pct),
            "total_profit_pct": float(self.total_profit_pct),
            "termination":      "sell" if terminated else (
                                "forced_sell" if truncated else "unknown"),
            "timestamp":        datetime.now().isoformat(),
        }
        self.episode_log.append(summary)
        with open(self.log_path, "a") as f:
            for record in self.episode_log:
                f.write(json.dumps(record) + "\n")
        self.episode_log = []

    def _log_step(self, action: int, action_str: str,
                  price: float, profit_pct: float, reward: float):
        self.episode_log.append({
            "type":       "step",
            "step":       int(self.current_step),
            "action":     int(action),
            "action_str": action_str,
            "price":      float(price),
            "profit_pct": float(profit_pct),
            "reward":     float(reward),
        })