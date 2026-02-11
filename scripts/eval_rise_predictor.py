#!/usr/bin/env python3
"""
eval_rise_predictor.py

Evaluate a trained ServoTrader predictor on a single-symbol CSV and report:
  - Average Absolute Percent Error (AAPE)
  - Directional Accuracy with a single tau threshold (default 0.05%)

Supports:
  1) PPO + Transformer features extractor: observation is a window (seq_len, 11)
  2) RecurrentPPO (LSTM): observation is a single vector (11,)

Assumptions:
  - Horizon is fixed to 1.
  - Evaluate as many steps as possible from the dataset.

Usage (Transformer example):
  /bin/python3.11 eval_rise_predictor.py \
    --model "/home/jarred/git/ServoTrader/models/rise_predictor_ppo_transformer_5.zip" \
    --csv "/home/jarred/git/ServoTrader/data/individual_crypto_csvs_ancient_15_min_2/ACHUSDT.csv" \
    --seq-len 64 \
    --tau 0.00025

Usage (LSTM example):
  /bin/python3.11 eval_rise_predictor.py \
    --model "/home/jarred/git/ServoTrader/models/rise_predictor_ppo_lstm.zip" \
    --csv "/home/jarred/git/ServoTrader/data/individual_crypto_csvs_ancient_2/ACHUSDT.csv" \
    --seq-len 1
"""
from __future__ import annotations
import argparse
import os
import numpy as np
import pandas as pd
import torch

from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO

from typing import Tuple

# Make the repo root importable so `models/...` resolves
import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --- Compat shim for old saved paths ----------------------------------------
# Ensures "models.transformer_extractor.TransformerSeqExtractor" is importable,
# matching the path stored in the saved PPO zip.
import types

def ensure_models_transformer_path():
    # 1) Best case: it already exists (project has models/transformer_extractor.py)
    try:
        from models.transformer_extractor import TransformerSeqExtractor  # noqa:F401
        return  # nothing to do
    except Exception:
        pass

    # 2) Fallbacks: try where the class actually lives now
    TSE = None
    try:
        from models.transformer_extractor import TransformerSeqExtractor as TSE  # noqa:F401
    except Exception:
        try:
            from models.transformer_extractor import TransformerSeqExtractor as TSE  # noqa:F401
        except Exception:
            TSE = None

    if TSE is None:
        # No known location; leave as-is (LSTM loads won't need it)
        return

    # 3) Create a fake 'models' package and 'models.transformer_extractor' submodule
    if "models" not in sys.modules:
        sys.modules["models"] = types.ModuleType("models")
    mod = types.ModuleType("models.transformer_extractor")
    setattr(mod, "TransformerSeqExtractor", TSE)
    sys.modules["models.transformer_extractor"] = mod
    setattr(sys.modules["models"], "transformer_extractor", mod)

# Call after inserting PROJECT_ROOT into sys.path
ensure_models_transformer_path()
# ---------------------------------------------------------------------------

# ---------- Feature engineering (must mirror training) ----------

# def compute_features_and_targets(df: pd.DataFrame, clip_target: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
#     """
#     Build features [T, 11] and targets [T] with horizon=1 (fixed).
#     Targets are in '1 = 100%' units and clipped to [-clip_target, +clip_target].
#     """
#     df = df.copy()
#     if "timestamp" in df.columns:
#         df = df.sort_values("timestamp")

#     for col in ["open", "high", "low", "close", "vwap", "volume", "count"]:
#         df[col] = pd.to_numeric(df[col], errors="coerce")

#     df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["open", "high", "low", "close", "vwap"])
#     df["volume"] = df["volume"].fillna(0.0)
#     df["count"]  = df["count"].fillna(0.0)

#     eps = 1e-9

#     # (1) trend_ema_dev (LOW-based)
#     ema_low = df["low"].ewm(span=30, adjust=False, min_periods=30).mean()
#     trend_ema_dev = (df["low"] / (ema_low + eps) - 1.0)

#     # (2) momentum_rsi (LOW-based, Wilder N=20 → [0,1])
#     d_low = df["low"].diff()
#     gain = d_low.clip(lower=0.0)
#     loss = (-d_low).clip(lower=0.0)
#     avg_gain = gain.ewm(alpha=1/20, min_periods=20, adjust=False).mean()
#     avg_loss = loss.ewm(alpha=1/20, min_periods=20, adjust=False).mean()
#     rs  = avg_gain / (avg_loss + eps)
#     rsi = 100.0 - (100.0 / (1.0 + rs))
#     momentum_rsi = (rsi / 100.0)

#     # (3) vol_atr_norm (ATR / |LOW|, N=20)
#     prev_close = df["close"].shift(1)
#     tr = pd.concat([
#         (df["high"] - df["low"]).abs(),
#         (df["high"] - prev_close).abs(),
#         (df["low"]  - prev_close).abs()
#     ], axis=1).max(axis=1)
#     atr = tr.ewm(alpha=1/20, min_periods=20, adjust=False).mean()
#     vol_atr_norm = atr / (df["low"].abs() + eps)

#     # (4) flow_cmf (CLOSE-based MFM, N=20)
#     hl_range = (df["high"] - df["low"])
#     mfm = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (hl_range.replace(0, np.nan) + eps)
#     mfv = mfm.fillna(0.0) * df["volume"]
#     vol_sum = df["volume"].rolling(20, min_periods=20).sum()
#     mfv_sum = mfv.rolling(20, min_periods=20).sum()
#     flow_cmf = (mfv_sum / (vol_sum + eps))

#     # Warmup discard (ensure indicators are formed)
#     discard = 30
#     horizon = 1
#     if len(df) <= discard + horizon:
#         raise ValueError("Not enough rows after warmup to form labels.")

#     df = df.iloc[discard:].reset_index(drop=True)
#     trend_ema_dev = trend_ema_dev.iloc[discard:].reset_index(drop=True)
#     momentum_rsi  = momentum_rsi.iloc[discard:].reset_index(drop=True)
#     vol_atr_norm  = vol_atr_norm.iloc[discard:].reset_index(drop=True)
#     flow_cmf      = flow_cmf.iloc[discard:].reset_index(drop=True)

#     # Label: 1-bar ahead % change (units of 1=100%)
#     close  = df["close"].astype(float)
#     future = close.shift(-horizon)
#     ret_frac = (future - close) / (close + eps)

#     valid_len = len(df) - horizon
#     df            = df.iloc[:valid_len]
#     trend_ema_dev = trend_ema_dev.iloc[:valid_len]
#     momentum_rsi  = momentum_rsi.iloc[:valid_len]
#     vol_atr_norm  = vol_atr_norm.iloc[:valid_len]
#     flow_cmf      = flow_cmf.iloc[:valid_len]
#     y             = ret_frac.iloc[:valid_len]

#     X = np.stack([
#         df["open"].to_numpy(float),
#         df["high"].to_numpy(float),
#         df["low"].to_numpy(float),
#         df["close"].to_numpy(float),
#         df["vwap"].to_numpy(float),
#         df["volume"].to_numpy(float),
#         df["count"].to_numpy(float),
#         trend_ema_dev.fillna(0.0).to_numpy(float),
#         momentum_rsi.fillna(0.0).to_numpy(float),
#         vol_atr_norm.fillna(0.0).to_numpy(float),
#         flow_cmf.fillna(0.0).to_numpy(float),
#     ], axis=1).astype(np.float32)

#     y = np.clip(y.to_numpy(np.float32), -clip_target, +clip_target)
#     return X, y

def compute_features_and_targets(df: pd.DataFrame, horizon: int, clip_target: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      X: [T, 11] float32 features
      y: [T]     float32 future % return (fraction), clipped to [-clip_target, +clip_target]
    """
    df = df.copy()
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp")
    for col in ["open", "high", "low", "close", "vwap", "volume", "count"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["open", "high", "low", "close", "vwap"])
    df["volume"] = df["volume"].fillna(0.0)
    df["count"]  = df["count"].fillna(0.0)

    eps = 1e-9
    # LOW-based engineered features
    ema_low = df["low"].ewm(span=30, adjust=False, min_periods=30).mean()
    trend_ema_dev = (df["low"] / (ema_low + eps) - 1.0)

    d_low = df["low"].diff()
    gain = d_low.clip(lower=0.0)
    loss = (-d_low).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1/20, min_periods=20, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/20, min_periods=20, adjust=False).mean()
    rs  = avg_gain / (avg_loss + eps)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    momentum_rsi = (rsi / 100.0)

    prev_close = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/20, min_periods=20, adjust=False).mean()
    vol_atr_norm = atr / (df["low"].abs() + eps)

    hl = (df["high"] - df["low"])
    mfm = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (hl.replace(0, np.nan) + eps)
    mfv = mfm.fillna(0.0) * df["volume"]
    vol_sum = df["volume"].rolling(20, min_periods=20).sum()
    mfv_sum = mfv.rolling(20, min_periods=20).sum()
    flow_cmf = (mfv_sum / (vol_sum + eps))

    # Discard warmup and build labels
    discard = 30
    if len(df) <= discard + horizon:
        raise ValueError("Not enough rows after warmup to form labels.")
    df = df.iloc[discard:].reset_index(drop=True)
    trend_ema_dev = trend_ema_dev.iloc[discard:].reset_index(drop=True)
    momentum_rsi  = momentum_rsi.iloc[discard:].reset_index(drop=True)
    vol_atr_norm  = vol_atr_norm.iloc[discard:].reset_index(drop=True)
    flow_cmf      = flow_cmf.iloc[discard:].reset_index(drop=True)

    close  = df["close"].astype(float)
    future = close.shift(-horizon)
    ret_frac = (future - close) / (close + eps)
    y = ret_frac.astype(float)

    valid_len = len(df) - horizon
    df            = df.iloc[:valid_len]
    trend_ema_dev = trend_ema_dev.iloc[:valid_len]
    momentum_rsi  = momentum_rsi.iloc[:valid_len]
    vol_atr_norm  = vol_atr_norm.iloc[:valid_len]
    flow_cmf      = flow_cmf.iloc[:valid_len]
    y             = y.iloc[:valid_len]

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

# ---------- Metrics ----------

def directional_accuracy_with_tau(preds: np.ndarray, targs: np.ndarray, tau: float) -> float:
    """
    Directional accuracy when both |pred| and |true| exceed tau.
    Returns accuracy in [0,1]; NaN if no samples pass the threshold.
    """
    preds = preds.astype(np.float32)
    targs = targs.astype(np.float32)
    mask = (np.abs(preds) >= tau) & (np.abs(targs) >= tau)
    if not np.any(mask):
        return float("nan")
    return float((np.sign(preds[mask]) == np.sign(targs[mask])).mean())


# ---------- Evaluation ----------

def evaluate(model_path: str,
             csv_path: str,
             seq_len: int,
             tau: float = 0.0005,
             clip_target: float = 10.0,
             save_preds_csv: str | None = None,
             device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Eval] Device: {device}")
    print(f"[Eval] Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    X, y = compute_features_and_targets(df, horizon=1, clip_target=clip_target)

    # Load model: try PPO (Transformer) first, then RecurrentPPO
    try:
        print(f"[Eval] Loading PPO model: {model_path}")
        model = PPO.load(model_path, device=device)
        model_kind = "ppo"
    except Exception:
        print(f"[Eval] PPO load failed; trying RecurrentPPO.")
        model = RecurrentPPO.load(model_path, device=device)
        model_kind = "recurrent_ppo"

    # Build evaluation stream (use ALL possible steps)
    if seq_len > 1:
        # Transformer/windowed path
        T = len(X)
        start = seq_len - 1           # first timestep with a full window
        end   = T - 1                 # last usable index
        n     = max(0, end - start + 1)
        print(f"[Eval] Transformer windows: seq_len={seq_len} usable_steps={n} (t={start}..{end})")

        preds = np.zeros(n, dtype=np.float32)
        targs = y[start:start + n].astype(np.float32)

        for i, t in enumerate(range(start, end + 1)):
            obs = X[(t - seq_len + 1):(t + 1), :].astype(np.float32)  # (seq_len, 11)
            action, _ = model.predict(obs, deterministic=True)
            preds[i] = float(np.clip(action[0], -clip_target, +clip_target))
    else:
        # LSTM/per-step path
        T = len(X)
        n = T
        print(f"[Eval] LSTM steps: usable_steps={n}")

        preds = np.zeros(n, dtype=np.float32)
        targs = y[:n].astype(np.float32)

        if model_kind == "recurrent_ppo":
            lstm_state = None
            episode_start = np.array([True], dtype=bool)
            for t in range(n):
                obs = X[t].astype(np.float32)
                action, lstm_state = model.predict(
                    observation=obs,
                    state=lstm_state,
                    episode_start=episode_start,
                    deterministic=True
                )
                episode_start[...] = False
                preds[t] = float(np.clip(action[0], -clip_target, +clip_target))
        else:
            # Non-recurrent model receiving single vectors
            for t in range(n):
                obs = X[t].astype(np.float32)
                action, _ = model.predict(obs, deterministic=True)
                preds[t] = float(np.clip(action[0], -clip_target, +clip_target))

    # ----- Metrics -----
    err = preds - targs
    aape_percent = float(np.mean(np.abs(err)) * 100.0)  # average abs error in %
    dir_acc = directional_accuracy_with_tau(preds, targs, tau=tau)  # [0,1] or NaN

    print("\n========== Evaluation Results ==========")
    print(f"Model kind               : {model_kind}")
    print(f"Steps evaluated          : {len(preds):,}")
    print(f"Avg Abs Percent Error    : {aape_percent:.4f} %")
    if np.isfinite(dir_acc):
        print(f"Directional Accuracy (τ={tau*100:.3f}%) : {dir_acc*100:.2f} %")
    else:
        print(f"Directional Accuracy (τ={tau*100:.3f}%) : NaN (no samples passed threshold)")
    print("========================================\n")

    if save_preds_csv:
        out_dir = os.path.dirname(save_preds_csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        pd.DataFrame({"pred": preds, "target": targs, "error": err}).to_csv(save_preds_csv, index=False)
        print(f"[Eval] Saved predictions to: {save_preds_csv}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path to trained model .zip")
    p.add_argument("--csv", required=True, help="Path to evaluation CSV file")
    p.add_argument("--seq-len", type=int, default=32,
                   help="Window length for Transformer models. Use 1 for LSTM/per-step evaluation.")
    p.add_argument("--tau", type=float, default=0.0005,
                   help="Single threshold (in '1=100%%' units) for directional accuracy; e.g., 0.0005 = 0.05%%")
    p.add_argument("--save-preds-csv", type=str, default=None,
                   help="Optional path to save a CSV with columns [pred, target, error]")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        model_path=args.model,
        csv_path=args.csv,
        seq_len=args.seq_len,
        tau=args.tau,
        clip_target=10.0,
        save_preds_csv=args.save_preds_csv,
    )
