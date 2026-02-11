#!/usr/bin/env python3
"""
rise_predict_seq_env.py

A single-symbol, regression-style environment that returns a rolling window
of length `seq_len` with 11 features per timestep: shape (T, F) = (seq_len, 11).

Action: continuous scalar in [-clip_target, +clip_target]
Target: future % change over `horizon` bars (precomputed upstream), clipped
Reward: negative Huber loss (default). An optional direction bonus can be enabled
         at init-time, but is OFF by default (recommended to keep the env clean).

Notes
-----
- We intentionally keep this env "dumb" (no tau weighting or sign loss here).
  Directional-accuracy improvements are handled in the *model-side* via a small
  supervised "sign head" that we train after PPO in the training script.

Author: Jarred Deluca (ServoCrypto/ServoTrader)
License: MIT
"""
from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Dict


class RisePredictSeqEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        features: np.ndarray,      # [T, 11] float32
        targets: np.ndarray,       # [T] float32 (same T, already clipped)
        seq_len: int = 32,         # lookback window length T
        clip_target: float = 10.0,
        huber_delta: float = 1.0,
        direction_bonus: float = 0.0,  # keep 0.0 to disable (clean regression reward)
        seed: Optional[int] = None,
    ):
        super().__init__()
        if features.ndim != 2 or features.shape[1] != 11:
            raise ValueError(f"[RisePredictSeqEnv] features must be [T, 11], got {features.shape}")
        if targets.ndim != 1 or targets.shape[0] != features.shape[0]:
            raise ValueError(
                f"[RisePredictSeqEnv] targets must be [T] with same T as features; got {targets.shape} vs {features.shape}"
            )
        if features.shape[0] < seq_len + 2:
            raise ValueError("Dataset too short to form at least one window.")

        self.X = np.asarray(features, dtype=np.float32)
        self.y = np.asarray(targets, dtype=np.float32)

        self.seq_len     = int(seq_len)
        self.clip_target = float(clip_target)
        self.huber_delta = float(huber_delta)
        self.dir_bonus   = float(direction_bonus)

        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        # Obs: (T, F) window
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.seq_len, 11), dtype=np.float32
        )
        # Action: scalar prediction
        self.action_space = spaces.Box(
            low=np.array([-self.clip_target], dtype=np.float32),
            high=np.array([+self.clip_target], dtype=np.float32),
            shape=(1,), dtype=np.float32,
        )

        self._T = self.X.shape[0]
        self._t = self.seq_len  # index of current label; obs window is [t-seq_len, t)

    # ------------------ helpers ------------------

    def _huber(self, err: float) -> float:
        abs_e = abs(err)
        d = self.huber_delta
        if abs_e <= d:
            return 0.5 * err * err
        return d * (abs_e - 0.5 * d)

    def _make_obs(self, t: int) -> np.ndarray:
        # window spans indices [t-seq_len, t)
        return self.X[(t - self.seq_len):t, :]

    # ------------------ Gym API ------------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        # choose a window whose label exists (t in [seq_len, T-1])
        low  = self.seq_len
        high = self._T - 1
        self._t = int(self.np_random.integers(low, high + 1)) if hasattr(self, "np_random") else int(np.random.randint(low, high + 1))
        obs = self._make_obs(self._t)
        return obs.astype(np.float32, copy=False), {}

    def step(self, action):
        pred = float(action[0]) if isinstance(action, (np.ndarray, list, tuple)) else float(action)
        pred = np.clip(pred, -self.clip_target, self.clip_target)

        target = float(self.y[self._t])
        err = pred - target
        reward = -self._huber(err)

        # # Optional: small bonus for correct sign. KEEP DISABLED by default.
        # if self.dir_bonus != 0.0:
        #     if (pred > 0 and target > 0) or (pred < 0 and target < 0):
        #         reward += self.dir_bonus

        # advance one step; keep a streaming feel
        self._t += 1
        done = (self._t >= self._T - 1)
        obs = self._make_obs(self._t if not done else self._T - 1)

        info = {}
        if done:
            info["terminal_observation"] = obs.astype(np.float32, copy=False)

        return obs.astype(np.float32, copy=False), float(reward), done, False, info

    def render(self):
        print(f"[RisePredictSeqEnv] t={self._t}/{self._T}")

    def close(self):
        pass
