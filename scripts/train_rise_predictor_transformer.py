#!/usr/bin/env python3
"""
train_rise_predictor_transformer.py

Train a PPO agent that outputs a scalar price-change prediction for a single
symbol dataset per env. Observations are (T,11) windows; a Transformer-based
features extractor encodes them to a vector for the PPO heads.

After PPO finishes, we run a lightweight SUPERVISED sidecar training that:
  - Freezes the Transformer features (from the trained PPO policy)
  - Trains a tiny SignHead (linear layer) on volatility-normalized log-returns
  - Uses τ-aware sample weights to downweight near-zero moves
  - Tunes a probability threshold to maximize Directional Accuracy at τ

Outputs:
  - PPO policy at MODEL_OUT (unchanged)
  - SignHead state_dict at SIGN_HEAD_OUT
  - Sign threshold at SIGN_THRESH_OUT (JSON)

Author: Jarred Deluca
License: MIT
"""
from __future__ import annotations
import os, glob, sys, json
from typing import Tuple, List, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from gymnasium.wrappers import TimeLimit, RecordEpisodeStatistics

# Local imports (fix paths)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from servo_trader.envs.rise_predict_env import RisePredictSeqEnv
from models.transformer_extractor import TransformerSeqExtractor

# ===== Paths & data =====
DATA_DIR   = "/home/jarred/git/ServoTrader/data/individual_crypto_csvs_ancient_15_min"
GLOB       = "*.csv"
MODEL_OUT  = "/home/jarred/git/ServoTrader/models/rise_predictor_ppo_transformer_5"
SIGN_HEAD_OUT   = "/home/jarred/git/ServoTrader/models/sign_head.pt"
SIGN_THRESH_OUT = "/home/jarred/git/ServoTrader/models/sign_threshold.json"
TBOARD_LOG = "/home/jarred/git/ServoTrader/logs"

# ===== Labels & env =====
HORIZON     = 1         # bars ahead
CLIP_TARGET = 10.0
SEQ_LEN     = 64        # ↑ slightly longer context helps DA (try 48 or 64)
HUBER_DELTA = 1.0
DIR_BONUS   = 0.0       # keep 0 for clean regression reward

# Episode length (keep micro-episodes so VecMonitor logs regularly)
EPISODE_STEPS = 1

# ===== Training =====
N_ENVS         = 8
TOTAL_STEPS    = 1_000_000
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
N_STEPS        = 512
BATCH_SIZE     = 128
N_EPOCHS       = 10

# ===== Transformer hyperparams =====
D_MODEL  = 128
N_HEADS  = 4
N_LAYERS = 2
DROPOUT  = 0.05
USE_CLS  = True

# Additional extractor upgrades (must be supported by your extractor)
ATTN_DROPOUT = 0.05
MLP_RATIO    = 4.0
USE_ROPE     = True
DROPPATH     = 0.05

# ===== SignHead & tau-weighting =====
TAU = 0.0005  # 0.05% band for DA metric
THRESH_GRID = np.linspace(0.35, 0.65, 61)  # sweep for best DA threshold
SIGN_LR = 1e-3
SIGN_WD = 1e-2
SIGN_EPOCHS = 3        # short fine-tune; bump to 5-10 if you want more
BATCH_SIGN = 8192      # batch entire dataset in chunks (feature-extractor is frozen)


# ---------------- Feature engineering & targets ----------------

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


def preprocess_file(path: str, horizon: int, clip_target: float) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    return compute_features_and_targets(df, horizon, clip_target)


def list_symbol_files(data_dir: str, pattern: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(data_dir, pattern)))
    if not files:
        raise FileNotFoundError(f"No CSVs found under {data_dir} matching {pattern}")
    return files


# ---------------- VecEnv builder ----------------

def make_env_fn(X: np.ndarray, y: np.ndarray, seed: int) -> callable:
    def _thunk():
        env = RisePredictSeqEnv(
            features=X, targets=y,
            seq_len=SEQ_LEN,
            clip_target=CLIP_TARGET,
            huber_delta=HUBER_DELTA,
            direction_bonus=DIR_BONUS,
            seed=seed,
        )
        # Episode/stat wrappers
        env = TimeLimit(env, max_episode_steps=EPISODE_STEPS)
        env = RecordEpisodeStatistics(env)
        return env
    return _thunk


# ---------------- Sign head (supervised) ----------------

class SignHead(nn.Module):
    """
    Minimal classifier on top of frozen Transformer features.
    Input:  (N, d_model)
    Output: logits (N,)
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.fc = nn.Linear(d_model, 1)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x).squeeze(-1)


def build_windows(X: np.ndarray, y_raw: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Turn (T, F) & (T,) into (N, seq_len, F) windows and labels aligned to the last index.
    Drops the first `seq_len` rows because they can't form a full window.
    """
    T = X.shape[0]
    if T <= seq_len:
        return np.empty((0, seq_len, X.shape[1]), dtype=np.float32), np.empty((0,), dtype=np.float32)
    # windows end at t where t in [seq_len, T-1] -> index range [seq_len, T-1]
    idx_end = np.arange(seq_len, T, dtype=np.int64)
    # Stack windows efficiently
    windows = np.stack([X[t-seq_len:t, :] for t in idx_end], axis=0).astype(np.float32)  # (N, seq_len, F)
    labels  = y_raw[idx_end].astype(np.float32)                                          # (N,)
    return windows, labels


def directional_accuracy(pred_sign: np.ndarray, y_raw: np.ndarray, tau: float) -> float:
    """
    Compute DA at tau: only count samples where |y_raw| >= tau.
    pred_sign: bool array (True=up)
    y_raw:     float array (fraction)
    """
    mask = np.abs(y_raw) >= tau
    if not mask.any():
        return 0.0
    truth = (y_raw > 0)
    correct = (pred_sign[mask] == truth[mask]).mean()
    return float(correct)


def train_sign_head_on_features(
    feats: torch.Tensor,  # (N, d)
    y_raw: torch.Tensor,  # (N,)
    device: torch.device,
    epochs: int = SIGN_EPOCHS,
    lr: float = SIGN_LR,
    wd: float = SIGN_WD,
) -> SignHead:
    """
    Train a small classifier with τ-aware weights:
      w = sigmoid(sharpness * (abs(y_raw) - tau))
    Target is the sign of volatility-normalized log-returns.
    """
    # Build normalized log-return target (classification label)
    # Here we derive from y_raw (fraction) for convenience:
    # y_log ≈ log(1 + y_raw), stabilize small values; then normalize by rolling std.
    y_log = torch.log1p(y_raw.clamp(min=-0.95))  # avoid log of negative close to -1
    # approximate rolling std across the dataset with a global std (simple & stable)
    vol = torch.std(y_log).clamp(min=1e-6)
    y_norm = y_log / vol
    y_sign = (y_norm > 0).float()

    # τ-aware weights from *raw* return magnitude
    sharpness = 400.0
    w = torch.sigmoid((torch.abs(y_raw) - TAU) * sharpness).detach()

    model = SignHead(d_model=feats.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    N = feats.shape[0]
    bs = min(BATCH_SIGN, N)

    for _ in range(epochs):
        perm = torch.randperm(N, device=device)
        for i in range(0, N, bs):
            idx = perm[i:i+bs]
            logits = model(feats[idx])
            bce = F.binary_cross_entropy_with_logits(logits, y_sign[idx], weight=w[idx])

            opt.zero_grad(set_to_none=True)
            bce.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    return model


def extract_features_from_policy(
    policy, datasets_map: Dict[str, Tuple[np.ndarray, np.ndarray]], seq_len: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    For all picked symbols:
      - build windows (N, seq_len, F)
      - run frozen Transformer extractor to get (N, d_model)
    Returns:
      feats: (N, d_model) tensor
      y_raw: (N,) raw % returns (fraction)
    """
    policy.eval()
    extractor = policy.features_extractor  # our TransformerSeqExtractor
    feats_list = []
    y_list = []

    with torch.no_grad():
        for _, (X, y) in datasets_map.items():
            win, lab = build_windows(X, y, seq_len)
            if win.shape[0] == 0:
                continue
            obs = torch.from_numpy(win).to(device)
            f = extractor(obs)  # (N, d_model)
            feats_list.append(f)
            y_list.append(torch.from_numpy(lab).to(device))

    if not feats_list:
        raise RuntimeError("No windows built for sign-head training (datasets too short?).")

    feats = torch.cat(feats_list, dim=0)
    y_raw = torch.cat(y_list, dim=0)
    return feats, y_raw


def tune_threshold(model: SignHead, feats: torch.Tensor, y_raw: torch.Tensor) -> float:
    """
    Sweep probability thresholds and pick the one maximizing DA@TAU.
    Uses a simple 80/20 split for threshold tuning vs report set.
    """
    N = feats.shape[0]
    cut = int(0.8 * N)
    with torch.no_grad():
        probs = torch.sigmoid(model(feats)).cpu().numpy()

    probs_val = probs[:cut]
    y_val     = y_raw[:cut].cpu().numpy()
    best_p = max(THRESH_GRID, key=lambda p: directional_accuracy(probs_val > p, y_val, TAU))
    return float(best_p)


# ---------------- Main ----------------

def main():
    print(f"[Device] Using {DEVICE}")
    files = list_symbol_files(DATA_DIR, GLOB)
    picked = files[:N_ENVS] if len(files) >= N_ENVS else files
    if len(picked) < N_ENVS:
        print(f"[Warn] Only {len(picked)} files available; using {len(picked)} envs.")

    # Multiprocess preprocess
    print(f"[Preprocess] {len(picked)} files with 8 workers…")
    datasets_map: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(preprocess_file, f, HORIZON, CLIP_TARGET): f for f in picked}
        for fut in as_completed(futs):
            X, y = fut.result()
            datasets_map[futs[fut]] = (X, y)

    # Vec envs
    env_fns = []
    for i, f in enumerate(picked):
        X, y = datasets_map[f]
        if X.shape[0] < (SEQ_LEN + 64):
            print(f"[Warn] {os.path.basename(f)} short (T={X.shape[0]}).")
        env_fns.append(make_env_fn(X, y, seed=1000 + i))

    venv = SubprocVecEnv(env_fns, start_method="spawn")
    venv = VecMonitor(venv)  # keep last so it sees terminations

    # Ensure PPO batch-size multiple of n_envs
    batch_size = BATCH_SIZE
    if batch_size % len(env_fns) != 0:
        batch_size = int(np.ceil(batch_size / len(env_fns)) * len(env_fns))

    # Policy (Transformer extractor)
    policy_kwargs = dict(
        features_extractor_class=TransformerSeqExtractor,
        features_extractor_kwargs=dict(
            seq_len=SEQ_LEN,
            in_features=11,
            d_model=D_MODEL,
            n_heads=N_HEADS,
            n_layers=N_LAYERS,
            dropout=DROPOUT,
            use_cls_token=USE_CLS,
            # New goodies (must exist in your extractor)
            attn_dropout=ATTN_DROPOUT,
            mlp_ratio=MLP_RATIO,
            use_rope=USE_ROPE,
            droppath=DROPPATH,
        ),
        net_arch=dict(pi=[256, 128], vf=[256, 128]),
        activation_fn=nn.SiLU,
        ortho_init=False,
    )

    model = PPO(
        policy="MlpPolicy",
        env=venv,
        learning_rate=2.5e-4,
        n_steps=N_STEPS,
        batch_size=batch_size,
        n_epochs=N_EPOCHS,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        vf_coef=0.5,
        ent_coef=1e-3,
        max_grad_norm=0.5,
        tensorboard_log=TBOARD_LOG,
        device=DEVICE,
        policy_kwargs=policy_kwargs,
        verbose=1,
    )

    # --- PPO training ---
    model.learn(total_timesteps=TOTAL_STEPS, tb_log_name="ppo_transformer_2", log_interval=1, progress_bar=False)
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    model.save(MODEL_OUT)
    print(f"[Save] PPO model saved to {MODEL_OUT}")

    # # --- Supervised sidecar: train sign head for DA@τ ---
    # print("[SignHead] Extracting features from frozen Transformer...")
    # with torch.no_grad():
    #     feats, y_raw = extract_features_from_policy(model.policy, datasets_map, SEQ_LEN, torch.device(DEVICE))

    # print(f"[SignHead] Training on N={feats.shape[0]} windows…")
    # sign_model = train_sign_head_on_features(feats, y_raw, torch.device(DEVICE))
    # sign_model.eval()

    # # Threshold tuning (simple 80/20 split)
    # best_p = tune_threshold(sign_model, feats, y_raw)
    # print(f"[SignHead] Best probability threshold for DA@tau={TAU:.4f} is p*={best_p:.3f}")

    # # Save sign head + threshold
    # torch.save(sign_model.state_dict(), SIGN_HEAD_OUT)
    # os.makedirs(os.path.dirname(SIGN_THRESH_OUT), exist_ok=True)
    # with open(SIGN_THRESH_OUT, "w") as f:
    #     json.dump({"tau": TAU, "p_star": best_p}, f, indent=2)
    # print(f"[Save] SignHead -> {SIGN_HEAD_OUT}")
    # print(f"[Save] Threshold -> {SIGN_THRESH_OUT}")

    venv.close()


if __name__ == "__main__":
    main()
