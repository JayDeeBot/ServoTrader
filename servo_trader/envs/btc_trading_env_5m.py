"""
btc_trading_env_5m.py

5-minute version of the BTC trading environment, updated for the 38
Boruta-confirmed features from the 5-minute Boruta analysis.

Key changes from the hourly version
-------------------------------------
  HISTORY_WINDOW : 48 candles  (was 24)  → 4 hours of 5-min context
  max_hold_steps : 72 candles  (default) → 6-hour hold ceiling
  N_FEATURES     : 38          (was 14)
  OBS_DIM        : 48×38 + 3 = 1,827    (was 339)

Everything else — action space, reward shaping, action masking, JSONL
logging — is unchanged from the hourly environment.

Episode Structure
-----------------
  Phase 1 (no position)  → legal: NOT_BUY (0), BUY (1)
  Phase 2 (in position)  → legal: HOLD (2), SELL (3)

Reward Shaping
--------------
  NOT_BUY : retrospective shaping — +0.001 if avoided a losing entry
            (hypothetical return over next min_hold_steps < 0), else -0.0001
  BUY     : 0.0
  HOLD    : Δ unrealized PnL (mark-to-market per step)
  SELL    : realized_pnl − 0.003

Trading Cost
------------
  0.3% round-trip deducted at SELL.

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

TRADE_COST      = 0.003
ACTION_NOT_BUY  = 0
ACTION_BUY      = 1
ACTION_HOLD     = 2
ACTION_SELL     = 3

# 38 Boruta-confirmed 5-minute features (order must match the dataset CSV)
BORUTA_FEATURES = [
    "bb_bandwidth_48",
    "bb_percent_b_48",
    "returns_24p",
    "returns_12p",
    "bb_bandwidth_72",
    "bb_percent_b_72",
    "returns_6p",
    "returns_3p",
    "returns_1p",
    "vwap_deviation",
    "stoch_k",
    "obv_ma_48",
    "returns_mean_24p",
    "volume_ma_48",
    "volume_ma_24",
    "volume_ma_12",
    "returns_std_144p",
    "returns_mean_144p",
    "atr_12",
    "atr_24",
    "atr_48",
    "obv",
    "returns_48p",
    "stoch_d",
    "bb_percent_b_24",
    "rsi_12",
    "rsi_24",
    "returns_72p",
    "rsi_48",
    "rsi_72",
    "macd",
    "macd_signal",
    "macd_histogram",
    "bb_percent_b_12",
    "returns_mean_48p",
    "bb_bandwidth_24",
    "rsi_6",
    "count",
]

N_FEATURES     = len(BORUTA_FEATURES)   # 38
HISTORY_WINDOW = 48                     # 4 hours of 5-min candles
N_CONTEXT      = 3                      # unrealized_pnl, position_flag, steps_in_trade_norm
OBS_DIM        = HISTORY_WINDOW * N_FEATURES + N_CONTEXT   # 48*38+3 = 1,827


# ---------------------------------------------------------------------------
#  Environment
# ---------------------------------------------------------------------------

class BTCTradingEnv5m(gym.Env):
    """
    5-minute BTC trading environment for MLP PPO training.

    Parameters
    ----------
    df : pd.DataFrame
        Pre-processed 5-minute dataset containing all BORUTA_FEATURES + 'close'.
    max_hold_steps : int
        Max candles the agent may hold before forced exit (default=72 → 6 hours).
    log_dir : str
        Directory for JSONL episode logs.
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
        self.max_hold_steps  = int(max_hold_steps)
        self.min_hold_steps  = int(min_hold_steps)   # SELL not legal until this many steps held

        # Spaces
        self.action_space = spaces.Discrete(4)

        low  = np.full(OBS_DIM, -10.0, dtype=np.float32)
        high = np.full(OBS_DIM,  10.0, dtype=np.float32)
        low[-N_CONTEXT:]  = np.array([-5.0, 0.0, 0.0], dtype=np.float32)
        high[-N_CONTEXT:] = np.array([ 5.0, 1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # State
        self.current_step     : int   = 0
        self.in_position      : bool  = False
        self.buy_price        : float = 0.0
        self.prev_price       : float = 0.0
        self.steps_in_trade   : int   = 0
        self.episode_start    : int   = 0
        self._buy_step        : Optional[int]   = None
        self._buy_price       : float = 0.0

        # Logging
        self.log_dir          = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path         = os.path.join(self.log_dir, f"btc5m_env_{ts}.jsonl")
        self.episode_log      : list  = []
        self.episode_counter  : int   = 0
        self.total_profit_pct : float = 0.0

        print(
            f"[BTCTradingEnv5m] Loaded {self.n_rows:,} rows | "
            f"OBS={OBS_DIM} | HISTORY={HISTORY_WINDOW} candles (4h) | "
            f"min_hold={self.min_hold_steps} candles ({self.min_hold_steps*5}min) | "
            f"max_hold={self.max_hold_steps} candles ({self.max_hold_steps*5//60}h)"
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

        self.current_step  = int(self.np_random.integers(min_start, max_start))
        self.episode_start = self.current_step
        self.in_position   = False
        self.buy_price     = 0.0
        self.prev_price    = self.close_prices[self.current_step]
        self.steps_in_trade = 0
        self._buy_step     = None
        self._buy_price    = 0.0

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

        if action == ACTION_NOT_BUY:
            if self.in_position:
                reward = -0.01   # illegal in Phase 2
            else:
                # Retrospective shaping: peek at what would have happened if
                # the agent had bought right now and held for min_hold_steps.
                # - If that hypothetical trade would have lost → small positive
                #   reward for correctly staying out.
                # - If it would have gained → tiny cost for missing the entry.
                #
                # Asymmetric magnitudes are intentional: reward avoiding bad
                # entries clearly (+0.001) but barely penalise missing good
                # ones (-0.0001). We don't want the agent to feel pressured to
                # buy on every candle — just to learn that some entries are
                # clearly worth skipping.
                future_idx    = min(self.current_step + self.min_hold_steps, self.n_rows - 1)
                future_price  = self.close_prices[future_idx]
                hypothetical  = (future_price / max(current_price, 1e-12)) - 1.0
                if hypothetical < 0:
                    reward = 0.001    # correctly avoided a losing entry
                else:
                    reward = -0.0001  # missed a winning entry (small cost only)

        elif action == ACTION_BUY:
            if self.in_position:
                reward = -0.01
            else:
                self.in_position    = True
                self.buy_price      = current_price
                self.steps_in_trade = 0
                self._buy_step      = int(self.current_step)
                self._buy_price     = current_price
                reward              = 0.0
                self._log_step(action, "buy", current_price, 0.0, 0.0)

        elif action == ACTION_HOLD:
            if not self.in_position:
                reward = -0.01
            else:
                prev_unr  = (self.prev_price  - self.buy_price) / max(self.buy_price, 1e-12)
                curr_unr  = (current_price    - self.buy_price) / max(self.buy_price, 1e-12)
                reward    = float(curr_unr - prev_unr)
                self.steps_in_trade += 1
                self._log_step(action, "hold", current_price, curr_unr * 100, reward)

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
        print(f"[BTCEnv5m] step={self.current_step} | {status} | price={price:.2f}")

    # -----------------------------------------------------------------------
    #  Internals
    # -----------------------------------------------------------------------

    def _validate_columns(self, df):
        missing = [c for c in BORUTA_FEATURES + ["close"] if c not in df.columns]
        if missing:
            raise ValueError(
                f"[BTCTradingEnv5m] Dataset missing required columns: {missing}\n"
                f"Run prepare_btc_5min_dataset.py first."
            )

    def _build_feature_matrix(self) -> np.ndarray:
        """
        Z-score normalise the 38 features across the full dataset.
        Clipped to ±5σ for stable training.
        OBV can have very large absolute values — z-scoring handles that.
        """
        raw = self.df[BORUTA_FEATURES].to_numpy(dtype=np.float64)
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        mu  = raw.mean(axis=0, keepdims=True)
        sig = raw.std(axis=0, keepdims=True) + 1e-8
        return np.clip((raw - mu) / sig, -5.0, 5.0).astype(np.float32)

    def _get_observation(self) -> np.ndarray:
        start  = max(0, self.current_step - HISTORY_WINDOW)
        window = self.feature_matrix[start:self.current_step]
        pad    = HISTORY_WINDOW - window.shape[0]
        if pad > 0:
            window = np.concatenate(
                [np.zeros((pad, N_FEATURES), dtype=np.float32), window], axis=0
            )
        feature_vec = window.flatten()   # (1824,)

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
            mask[ACTION_HOLD]    = True
            # SELL only becomes legal after min_hold_steps have elapsed.
            # This prevents the degenerate BUY→SELL-immediately policy where
            # the agent always loses the 0.3% trading cost without learning
            # anything about when to actually exit a position.
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
        self.current_step += 1
        if self.current_step >= self.n_rows:
            self.current_step = HISTORY_WINDOW

    def _finalise_episode(self, profit_pct, *, terminated, truncated):
        self.episode_counter  += 1
        self.total_profit_pct += profit_pct
        summary = {
            "type":             "episode_end",
            "episode":          int(self.episode_counter),
            "steps":            int(self.current_step - self.episode_start),
            "buy_step":         self._buy_step,
            "buy_price":        float(self._buy_price) if self._buy_price else None,
            "profit_pct":       float(profit_pct),
            "total_profit_pct": float(self.total_profit_pct),
            "termination":      "sell" if terminated else ("forced_sell" if truncated else "unknown"),
            "timestamp":        datetime.now().isoformat(),
        }
        self.episode_log.append(summary)
        with open(self.log_path, "a") as f:
            for record in self.episode_log:
                f.write(json.dumps(record) + "\n")
        self.episode_log = []

    def _log_step(self, action, action_str, price, profit_pct, reward):
        self.episode_log.append({
            "type":       "step",
            "step":       int(self.current_step),
            "action":     int(action),
            "action_str": action_str,
            "price":      float(price),
            "profit_pct": float(profit_pct),
            "reward":     float(reward),
        })