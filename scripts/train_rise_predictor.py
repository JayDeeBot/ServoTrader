"""
train_rise_predictor.py

Train an LSTM policy (RecurrentPPO) to predict future price change
magnitude/direction for many symbols in parallel — one dataset per env.

What this script does:
- ✅ Discovers CSVs in your per-symbol folder
- ✅ Uses multiprocessing (8 workers) to preprocess each CSV into
     (features[ T, 11 ], targets[ T ]) — OUTSIDE the env
- ✅ Builds N parallel envs (SubprocVecEnv), each bound to a distinct dataset
- ✅ Trains a RecurrentPPO LSTM policy on continuous actions in [-10, 10]
- ✅ Saves the trained model

Adjust N_ENVS to control how many datasets/envs you want at once.
Point DATA_DIR to your CSV directory.

Author: Jarred Deluca
License: MIT
"""

from __future__ import annotations
import os
import glob
from typing import Tuple, List
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from sb3_contrib import RecurrentPPO

# Local import path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from servo_trader.envs.rise_predict_env import RisePredictEnv


# ========= Config =========
DATA_DIR = "/home/jarred/git/ServoTrader/data/individual_crypto_csvs_ancient"
GLOB_PATTERN = "*.csv"

# Labeling / episode params
HORIZON = 1            # bars ahead for target
EPISODE_LEN = 1       # steps per episode (set 1 for one-step episodes)
CLIP_TARGET = 10.0     # actions & labels clipped to [-10, +10]
HUBER_DELTA = 1.0

# Training params
N_ENVS = 8                 # number of parallel environments (and datasets)
TOTAL_TIMESTEPS = 1_000_000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TBOARD_LOG = "/home/jarred/git/ServoTrader/logs"
MODEL_OUT = "/home/jarred/git/ServoTrader/models/rise_predictor_ppo_lstm"

# SB3 PPO (recurrent) params — keep batch_size multiple of N_ENVS
N_STEPS = 512
BATCH_SIZE = 128           # 128 is 16 * N_ENVS if N_ENVS=8 → good practice
N_EPOCHS = 10


# ========= Feature Engineering (outside env) =========

def compute_features_and_targets(df: pd.DataFrame, horizon: int, clip_target: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the 11-feature matrix and aligned target vector for one symbol dataframe.

    Returns
    -------
    X : float32 np.ndarray [T, 11]
    y : float32 np.ndarray [T]
        Final rows lacking future labels are removed so X and y align index-wise.
    """
    # Clean / typing
    df = df.copy()
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp")
    # Coerce numerics
    for col in ["open", "high", "low", "close", "vwap", "volume", "count"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["open", "high", "low", "close", "vwap"])
    df["volume"] = df["volume"].fillna(0.0)
    df["count"] = df["count"].fillna(0.0)

    # Engineered features
    eps = 1e-9
    # (1) trend_ema_dev (LOW-based)
    ema_low = df["low"].ewm(span=30, adjust=False, min_periods=30).mean()
    trend_ema_dev = (df["low"] / (ema_low + eps) - 1.0)

    # (2) momentum_rsi (LOW-based, Wilder N=20 → [0,1])
    d_low = df["low"].diff()
    gain = d_low.clip(lower=0.0)
    loss = (-d_low).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / 20, min_periods=20, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 20, min_periods=20, adjust=False).mean()
    rs = avg_gain / (avg_loss + eps)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    momentum_rsi = (rsi / 100.0)

    # (3) vol_atr_norm (ATR / |LOW|, N=20)
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 20, min_periods=20, adjust=False).mean()
    vol_atr_norm = atr / (df["low"].abs() + eps)

    # (4) flow_cmf (standard CMF, N=20, CLOSE-based MFM)
    hl_range = (df["high"] - df["low"])
    mfm = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (hl_range.replace(0, np.nan) + eps)
    mfv = mfm.fillna(0.0) * df["volume"]
    vol_sum = df["volume"].rolling(20, min_periods=20).sum()
    mfv_sum = mfv.rolling(20, min_periods=20).sum()
    flow_cmf = (mfv_sum / (vol_sum + eps))

    # Warmup drop to ensure features formed
    discard = 30
    if len(df) <= discard + horizon:
        raise ValueError("Not enough rows after warmup to form labels.")
    df = df.iloc[discard:].reset_index(drop=True)
    trend_ema_dev = trend_ema_dev.iloc[discard:].reset_index(drop=True)
    momentum_rsi = momentum_rsi.iloc[discard:].reset_index(drop=True)
    vol_atr_norm = vol_atr_norm.iloc[discard:].reset_index(drop=True)
    flow_cmf = flow_cmf.iloc[discard:].reset_index(drop=True)

    # Label: future change over `horizon` bars in units of 100%
    close = df["close"].astype(float)
    future = close.shift(-horizon)
    ret_frac = (future - close) / (close + eps)      # e.g., +0.05 for +5%
    y = ret_frac.astype(float)

    # Align lengths: drop last `horizon` rows that don't have labels
    valid_len = len(df) - horizon
    df = df.iloc[:valid_len]
    trend_ema_dev = trend_ema_dev.iloc[:valid_len]
    momentum_rsi = momentum_rsi.iloc[:valid_len]
    vol_atr_norm = vol_atr_norm.iloc[:valid_len]
    flow_cmf = flow_cmf.iloc[:valid_len]
    y = y.iloc[:valid_len]

    # Features matrix
    X = np.stack([
        df["open"].to_numpy(float),
        df["high"].to_numpy(float),
        df["low"].to_numpy(float),
        df["close"].to_numpy(float),
        df["vwap"].to_numpy(float),
        df["volume"].to_numpy(float),
        df["count"].to_numpy(float),
        trend_ema_dev.fillna(0.0).to_numpy(float),
        momentum_rsi.fillna(0.0).to_numpy(float),
        vol_atr_norm.fillna(0.0).to_numpy(float),
        flow_cmf.fillna(0.0).to_numpy(float),
    ], axis=1).astype(np.float32)

    y = np.clip(y.to_numpy(np.float32), -clip_target, +clip_target)
    return X, y


def preprocess_file(path: str, horizon: int, clip_target: float) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    return compute_features_and_targets(df, horizon, clip_target)


# ========= Utilities =========

def list_symbol_files(data_dir: str, pattern: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(data_dir, pattern)))
    if not files:
        raise FileNotFoundError(f"No CSVs found under {data_dir} matching {pattern}")
    return files


def make_env_fn(X: np.ndarray, y: np.ndarray, seed: int) -> callable:
    # Use default-arg capture to avoid late-binding issues in lambdas
    return lambda: RisePredictEnv(
        features=X,
        targets=y,
        episode_len=EPISODE_LEN,
        clip_target=CLIP_TARGET,
        huber_delta=HUBER_DELTA,
        seed=seed,
    )


def main():
    print(f"[Device] Using {DEVICE}")
    files = list_symbol_files(DATA_DIR, GLOB_PATTERN)

    # Pick N_ENVS files (or all if fewer available)
    picked = files[:N_ENVS]
    if len(picked) < N_ENVS:
        print(f"[Warn] Only found {len(picked)} files; creating {len(picked)} envs.")

    # --- Multiprocessing preprocess (8 workers) ---
    print(f"[Preprocess] Starting with {len(picked)} files using 8 workers…")
    datasets: List[Tuple[np.ndarray, np.ndarray]] = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = [
            ex.submit(preprocess_file, f, HORIZON, CLIP_TARGET)
            for f in picked
        ]
        for fut in as_completed(futs):
            X, y = fut.result()
            datasets.append((X, y))

    # Keep order consistent with 'picked' (as_completed may scramble); rebuild with map
    # (In practice, SB3 doesn’t require name order, but it’s cleaner.)
    datasets_ordered: List[Tuple[np.ndarray, np.ndarray]] = []
    for f in picked:
        X, y = preprocess_file(f, HORIZON, CLIP_TARGET)
        datasets_ordered.append((X, y))

    # --- Build parallel envs (one dataset per env) ---
    env_fns = []
    for i, (X, y) in enumerate(datasets_ordered):
        # Basic sanity
        if X.shape[0] < max(EPISODE_LEN + 1, 64):
            print(f"[Warn] {os.path.basename(picked[i])}: very short dataset (T={X.shape[0]}). Training may be noisy.")
        env_fns.append(make_env_fn(X, y, seed=1000 + i))

    venv = SubprocVecEnv(env_fns, start_method="spawn")
    venv = VecMonitor(venv)

    # --- Build model ---
    policy_kwargs = dict(
        lstm_hidden_size=128,
        n_lstm_layers=1,
        shared_lstm=False,
        enable_critic_lstm=False,
        ortho_init=False,
        activation_fn=nn.SiLU,
        net_arch=dict(pi=[256, 128], vf=[256, 128]),
    )

    # Ensure batch_size is a multiple of n_envs
    batch_size = max(BATCH_SIZE, N_ENVS * 16)
    if batch_size % N_ENVS != 0:
        # round up to nearest multiple of N_ENVS
        batch_size = int(np.ceil(batch_size / N_ENVS) * N_ENVS)

    model = RecurrentPPO(
        policy="MlpLstmPolicy",
        env=venv,
        learning_rate=3e-4,
        ent_coef=1e-3,
        n_steps=N_STEPS,          # rollout length per env
        batch_size=batch_size,    # must be divisible by n_envs
        n_epochs=N_EPOCHS,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=TBOARD_LOG,
        device=DEVICE,
        policy_kwargs=policy_kwargs,
        verbose=1,
    )

    # --- Train ---
    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=False)

    # --- Save ---
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    model.save(MODEL_OUT)
    print(f"[Save] Model saved to {MODEL_OUT}")

    venv.close()


if __name__ == "__main__":
    main()
