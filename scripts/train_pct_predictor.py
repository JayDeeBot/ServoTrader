#!/usr/bin/env python3
"""
train_pct_predictor.py  (GPU-ready, SubprocVecEnv-friendly, NaN-safe)

Single-asset percentage-rise predictor for BTC (continuous action).

- Observation (8): [timestamp_unix, open, high, low, close, vwap, volume, count]
- Action: Box(1) predicting % change close[t] -> close[t+1]  (e.g., +0.5 = +0.5%)
- Reward: -|y_true - y_hat|   (or - (err^2)/L2_SCALE if REWARD_MODE="l2")
- Episode: ONE STEP (reset -> predict -> done)

Training:
- Stable-Baselines3 PPO (Gaussian policy)
- Continued training supported
- Multi-env (Dummy/SubprocVecEnv) — workers load CSV themselves (no big pickles)
- TensorBoard logging
- GPU auto-detect or force via --device {auto,cuda,cpu}
"""

from __future__ import annotations

import os
import argparse
from typing import Tuple, Optional, Callable, Dict

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

# (optional) quiet the Axes3D warning
import warnings
warnings.filterwarnings("ignore", message="Unable to import Axes3D")

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

# --------------------------- Defaults ---------------------------

CSV_PATH = "/home/jarred/git/ServoTrader/data/individual_crypto_csvs_ancient/BTCUSDT.csv"

LOG_DIR = "./logs_pct_predictor/"
CHECKPOINT_DIR = "./checkpoints_pct_predictor/"
MODEL_NAME = "ppo_pct_predictor.zip"

TOTAL_STEPS_DEFAULT = 1_000_000
N_ENVS_DEFAULT = 8
USE_SUBPROC_DEFAULT = True
CONTINUE_TRAINING_DEFAULT = False
SEED_DEFAULT = 123

MAX_PCT_DEFAULT = 10.0
REWARD_MODE_DEFAULT = "l1"  # 'l1' or 'l2'
L2_SCALE_DEFAULT = 4.0
TEST_FRAC_DEFAULT = 0.2

SANITIZE_CLIP = 1e6  # safety clip after standardization

# ---------------------------------------------------------------

def ensure_dirs():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def _coerce_numeric(df: pd.DataFrame, cols) -> pd.DataFrame:
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def load_btc_csv_to_arrays(path: str) -> Dict[str, np.ndarray]:
    """
    Load CSV and return dict of numpy arrays (features X and targets y).
    - Coerces numeric cols
    - Drops NaNs after coercion
    - Removes ±inf
    """
    df = pd.read_csv(path)

    required = ["timestamp", "open", "high", "low", "close", "vwap", "volume", "count"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    # Parse timestamp to unix seconds
    ts = pd.to_datetime(df["timestamp"], utc=True)
    df["timestamp_unix"] = ts.astype("int64") / 1e9  # seconds

    # Coerce numerics & clean
    num_cols = ["timestamp_unix", "open", "high", "low", "close", "vwap", "volume", "count"]
    df = _coerce_numeric(df, num_cols)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=num_cols, inplace=True)

    # Sort by time
    df = df.sort_values("timestamp_unix").reset_index(drop=True)

    # Build X
    X = df[num_cols].to_numpy(dtype=np.float32)

    # Build y (next-step % change of close)
    close = df["close"].to_numpy(dtype=np.float32)
    if len(close) < 2:
        raise ValueError("Not enough rows after cleaning to build targets (need >= 2).")
    denom = np.where(close[:-1] == 0.0, 1e-12, close[:-1])
    y = (close[1:] - close[:-1]) / denom * 100.0  # length N-1

    # valid indices where next exists
    valid_idx = np.arange(0, len(df) - 1, dtype=np.int64)

    return {"X": X, "y": y, "valid_idx": valid_idx}

# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------

class OneStepPctEnv(gym.Env):
    """
    One-step regression-like env (continuous prediction).
    Workers load CSV locally; parent only passes tiny stats & config.
    """
    metadata = {"render_modes": ["human"]}

    _cache: Dict[str, Dict[str, np.ndarray]] = {}  # per-process cache {csv_path: {"X","y","valid_idx"}}

    def __init__(
        self,
        csv_path: str,
        feature_mean: Optional[np.ndarray],
        feature_std: Optional[np.ndarray],
        max_pct: float = 10.0,
        reward_mode: str = "l1",
        l2_scale: float = 4.0,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.csv_path = csv_path
        self.max_pct = float(max_pct)
        self.reward_mode = reward_mode
        self.l2_scale = float(l2_scale)

        # Load or fetch from per-process cache
        if csv_path not in OneStepPctEnv._cache:
            OneStepPctEnv._cache[csv_path] = load_btc_csv_to_arrays(csv_path)
        arrs = OneStepPctEnv._cache[csv_path]

        X = arrs["X"].copy()         # shape (N, 8)
        y = arrs["y"].copy()         # shape (N-1,)
        valid_idx = arrs["valid_idx"]

        # Standardize using provided stats (from train split)
        if feature_mean is not None and feature_std is not None:
            eps = 1e-8
            X = (X - feature_mean) / (feature_std + eps)

        # Sanitize: replace non-finite, clip extreme values
        X[~np.isfinite(X)] = 0.0
        X = np.clip(X, -SANITIZE_CLIP, SANITIZE_CLIP)
        y[~np.isfinite(y)] = 0.0
        y = np.clip(y, -SANITIZE_CLIP, SANITIZE_CLIP)

        self.X = X.astype(np.float32)
        self.targets = y.astype(np.float32)
        self.valid_idx = valid_idx

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([-self.max_pct], dtype=np.float32),
                                       high=np.array([+self.max_pct], dtype=np.float32),
                                       dtype=np.float32, shape=(1,))

        self._t = None
        self._done = True
        self._rng = np.random.default_rng(seed)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._t = int(self._rng.choice(self.valid_idx))
        self._done = False
        obs = self.X[self._t]
        # extra defensive sanitize
        if not np.isfinite(obs).all():
            obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        return obs, {}

    def step(self, action):
        if self._done:
            raise RuntimeError("Call reset() before step().")
        a = float(np.clip(action[0], -self.max_pct, +self.max_pct))
        y_true = float(self.targets[self._t])
        err = y_true - a
        reward = -abs(err) if self.reward_mode == "l1" else - (err * err) / self.l2_scale

        info = {
            "t_index": int(self._t),
            "y_true_pct": y_true,
            "y_pred_pct": a,
            "abs_error": abs(err),
            "squared_error": err * err,
        }
        terminated, truncated = True, False
        self._done = True
        obs = self.X[self._t]
        return obs, float(reward), terminated, truncated, info

    def render(self):
        if self._t is None:
            print("Env not reset.")
        else:
            print(f"[t={self._t}] y_true={self.targets[self._t]:.4f}%")

# -----------------------------------------------------------------------------
# Vec env factory
# -----------------------------------------------------------------------------

def make_env(csv_path: str, feat_mean: np.ndarray, feat_std: np.ndarray, seed_base: int, seed_offset: int,
             **env_kwargs) -> Callable[[], gym.Env]:
    def _thunk():
        env = OneStepPctEnv(
            csv_path=csv_path,
            feature_mean=feat_mean,
            feature_std=feat_std,
            **env_kwargs
        )
        env = Monitor(env)
        env.reset(seed=seed_base + seed_offset)
        return env
    return _thunk

# -----------------------------------------------------------------------------
# CLI + main
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="GPU-ready PPO trainer for BTC % rise predictor (subproc-safe, NaN-safe)")
    p.add_argument("--csv", default=CSV_PATH, type=str, help="Path to BTC CSV")
    p.add_argument("--total-steps", default=TOTAL_STEPS_DEFAULT, type=int)
    p.add_argument("--n-envs", default=N_ENVS_DEFAULT, type=int)
    p.add_argument("--subproc", default=USE_SUBPROC_DEFAULT, action=argparse.BooleanOptionalAction,
                   help="Use SubprocVecEnv (default true)")
    p.add_argument("--continue", dest="cont", default=CONTINUE_TRAINING_DEFAULT, action=argparse.BooleanOptionalAction)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Torch device")
    p.add_argument("--seed", default=SEED_DEFAULT, type=int)

    p.add_argument("--max-pct", default=MAX_PCT_DEFAULT, type=float)
    p.add_argument("--reward-mode", default=REWARD_MODE_DEFAULT, choices=["l1", "l2"])
    p.add_argument("--l2-scale", default=L2_SCALE_DEFAULT, type=float)
    p.add_argument("--test-frac", default=TEST_FRAC_DEFAULT, type=float)

    return p.parse_args()

def main(args):
    ensure_dirs()

    # ---------- Device selection & CUDA info ----------
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    torch.backends.cudnn.benchmark = True
    print(f"[Device] Requested={args.device}  Using={device}")
    if device == "cuda":
        print(f"[CUDA] is_available={torch.cuda.is_available()}  device_count={torch.cuda.device_count()}")
        try:
            print(f"[CUDA] name={torch.cuda.get_device_name(0)}  capability={torch.cuda.get_device_capability(0)}")
        except Exception:
            pass

    print(f"[Config] CSV={args.csv}")
    print(f"[Config] LOG_DIR={LOG_DIR}  CHECKPOINT_DIR={CHECKPOINT_DIR}")
    print(f"[Config] steps={args.total_steps:,}  n_envs={args.n_envs}  subproc={args.subproc}  continue={args.cont}")
    print(f"[TensorBoard] Logs -> {LOG_DIR}")

    # ---------- Load once in parent for stats ----------
    arrs_all = load_btc_csv_to_arrays(args.csv)
    X_all = arrs_all["X"]
    N = X_all.shape[0]
    split = max(2, min(int(round(N * (1.0 - args.test_frac))), N - 1))

    # Train split stats (standardize on train)
    X_train = X_all[:split].astype(np.float32)
    feat_mean = X_train.mean(axis=0, keepdims=True).astype(np.float32)
    feat_std = X_train.std(axis=0, keepdims=True).astype(np.float32)
    # avoid zero std
    feat_std = np.where(feat_std == 0.0, 1.0, feat_std)

    # ---------- Vec envs ----------
    env_kwargs = dict(max_pct=args.max_pct, reward_mode=args.reward_mode, l2_scale=args.l2_scale)

    if args.n_envs == 1:
        vec_env = DummyVecEnv([make_env(args.csv, feat_mean, feat_std, seed_base=args.seed, seed_offset=0, **env_kwargs)])
    else:
        fns = [make_env(args.csv, feat_mean, feat_std, seed_base=args.seed, seed_offset=i, **env_kwargs)
               for i in range(args.n_envs)]
        vec_env = SubprocVecEnv(fns) if args.subproc else DummyVecEnv(fns)

    # Eval env: match type to training env type to silence warning
    if isinstance(vec_env, SubprocVecEnv):
        eval_env = SubprocVecEnv([make_env(args.csv, feat_mean, feat_std, seed_base=9999, seed_offset=0, **env_kwargs)])
    else:
        eval_env = DummyVecEnv([make_env(args.csv, feat_mean, feat_std, seed_base=9999, seed_offset=0, **env_kwargs)])

    # ---------- Model path ----------
    model_path = os.path.join(CHECKPOINT_DIR, MODEL_NAME)

    # ---------- Create or load model ----------
    if args.cont and os.path.exists(model_path):
        print(f"[Continue] Loading model from {model_path}")
        model = PPO.load(model_path, env=vec_env, tensorboard_log=LOG_DIR, print_system_info=True, device=device)
    else:
        print("[Init] Creating new PPO model")
        model = PPO(
            policy="MlpPolicy",
            env=vec_env,
            verbose=1,
            seed=args.seed,
            tensorboard_log=LOG_DIR,
            device=device,                 # <<< GPU/CPU here
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.0,                    # one-step episodes
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(
                net_arch=[128, 128],
                # activation_fn=torch.nn.ReLU,
            ),
        )

    # ---------- Eval callback ----------
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=CHECKPOINT_DIR,
        log_path=CHECKPOINT_DIR,
        eval_freq=10_000,
        deterministic=True,
        render=False,
    )

    # ---------- Train ----------
    model.learn(total_timesteps=args.total_steps, callback=eval_cb)
    model.save(model_path)
    print(f"[Save] Model saved to {model_path}")

    # ---------- Quick eval ----------
    from stable_baselines3.common.evaluation import evaluate_policy
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=500, deterministic=True)
    print(f"[Eval] mean_reward={mean_reward:.6f}  std_reward={std_reward:.6f}")

if __name__ == "__main__":
    args = parse_args()
    main(args)
