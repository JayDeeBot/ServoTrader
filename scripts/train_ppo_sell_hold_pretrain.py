#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_ppo_sell_hold_pretrain.py

PPO pretraining script for SellHoldTrainingEnv (single-decision HOLD/SELL env).
"""

# scripts/train_ppo_sell_hold_pretrain.py

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import re
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd

from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

# 👇 Import your Sell/Hold pretraining environment
from servo_trader.envs.sell_hold_training_env import SellHoldTrainingEnv


# --------------------------
# Configurable paths/flags
# --------------------------
CODES_PATH = "/home/jarred/git/ServoTrader/servo_trader/config/crypto_codes_ancient.json"
DATA_PATH  = "/home/jarred/git/ServoTrader/data/split_10k_chunks_ancient/000.csv"

MODEL_DIR  = "/home/jarred/git/ServoTrader/models"
LOG_DIR    = "/home/jarred/git/ServoTrader/logs"
MODEL_NAME = "ppo_bulbasaur"

TOTAL_TIMESTEPS   = 1_000_000          # adjust as needed
CHECKPOINT_EVERY  = 100_000            # steps per checkpoint
CONTINUE_TRAINING = True               # set True to resume from MODEL_DIR/MODEL_NAME.zip

# SellHoldTrainingEnv knobs
HISTORY_WINDOW   = 5
LOOKAHEAD_STEPS  = 30                  # future-return horizon for reward
HELD_SELECTION   = "round_robin"       # "random" or "round_robin"
REWARD_BOUNDED   = False               # if True, tanh-squash reward to [-1, 1]
EPISODE_TIMEOUT  = 30                  # shape parity only; not used to truncate

# --------------------------
# Helpers (robust + normalized)
# --------------------------
def _norm_symbol(s: str) -> str:
    """Normalize to canonical BASEQUOTE (uppercase, strip / - _)."""
    s = (s or "").strip().upper()
    return re.sub(r"[\/\-_]", "", s)

def _looks_like_symbol(s: str) -> bool:
    """Heuristic to avoid mistaking config keys for symbols."""
    return bool(re.fullmatch(r"[A-Z0-9]{5,15}", s or ""))

def load_crypto_codes(path: str) -> list[str]:
    """
    Robustly load codes from JSON supporting shapes like:
      - ["BTCUSDT", "ETHUSDT", ...]
      - {"codes": [...]}, {"crypto_codes": [...]}, {"symbols": [...]}, {"tickers": [...]}
      - { "CRYPTOCODES": ["BTCUSDT", ...] } (values win)
      - { "BTCUSDT": {...}, "ETHUSDT": {...} }  (dict-of-meta; uses KEYS as last resort)
    """
    with open(path, "r") as f:
        blob = json.load(f)

    codes = None
    if isinstance(blob, list):
        codes = blob
    elif isinstance(blob, dict):
        # 1) Common keys
        for k in ("codes", "crypto_codes", "symbols", "tickers"):
            v = blob.get(k)
            if isinstance(v, list) and all(isinstance(x, str) for x in v):
                codes = v
                break
        # 2) Any dict value that is a list[str]
        if codes is None:
            for v in blob.values():
                if isinstance(v, list) and all(isinstance(x, str) for x in v):
                    codes = v
                    break
        # 3) Dict-of-meta: use KEYS if they look like symbols (last resort)
        if codes is None:
            keys = [k for k in blob.keys() if isinstance(k, str)]
            if keys and all(_looks_like_symbol(_norm_symbol(k)) for k in keys):
                codes = keys

    if not codes:
        raise ValueError(
            f"Could not find a list of crypto codes in {path}. "
            "Expected a list[str] or a dict with a list under a known key."
        )

    # Normalize & de-dup
    codes = sorted({_norm_symbol(c) for c in codes if isinstance(c, str) and c.strip()})
    if not codes:
        raise ValueError(f"Parsed codes from {path}, but they’re empty after normalization.")
    return codes

def find_symbol_column(df: pd.DataFrame) -> str:
    """Find a symbol-like column, tolerant of casing & aliases."""
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in ("symbol", "pair", "ticker"):
        if cand in cols_lower:
            return cols_lower[cand]
    raise ValueError(f"No symbol/pair/ticker column in CSV. Columns: {list(df.columns)}")


# --------------------------
# Env factory
# --------------------------
def make_env(crypto_codes, historical_df):
    """
    Factory for one SellHoldTrainingEnv instance wrapped with an action masker.
    """
    def _env_fn():
        env = SellHoldTrainingEnv(
            data=historical_df,
            crypto_codes=crypto_codes,
            episode_timeout=EPISODE_TIMEOUT,
            history_window=HISTORY_WINDOW,
            lookahead_steps=LOOKAHEAD_STEPS,
            held_selection=HELD_SELECTION,
            reward_bounded=REWARD_BOUNDED,
            # You can pass chunking knobs here if desired (uses env defaults otherwise)
            # chunk_dir="/home/jarred/git/ServoTrader/data/split_10k_chunks_ancient",
            # chunk_size=10_000,
            # start_chunk_index=0,
            # enable_chunking=None,
        )

        # Mask to only HOLD (0) and SELL (N+1)
        def mask_fn(e):
            if hasattr(e, "_get_action_mask"):
                return e._get_action_mask()
            mask = np.zeros(e.action_space.n, dtype=bool)
            mask[0] = True                      # HOLD
            mask[e.num_cryptos + 1] = True      # SELL
            return mask

        return ActionMasker(env, mask_fn)
    return _env_fn


# --------------------------
# Main
# --------------------------
def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    print("[SellHoldPretrain] Loading crypto codes…")
    crypto_codes = load_crypto_codes(CODES_PATH)
    print(f"[SellHoldPretrain] Loaded {len(crypto_codes)} codes. Sample: {crypto_codes[:10]}")

    print("[SellHoldPretrain] Loading dataset…")
    historical_df = pd.read_csv(DATA_PATH)

    # ---- Fast overlap sanity check before building the env ----
    sym_col = find_symbol_column(historical_df)
    csv_syms = sorted({_norm_symbol(s) for s in historical_df[sym_col].astype(str)})
    overlap = sorted(set(crypto_codes) & set(csv_syms))
    print(f"[SellHoldPretrain] CSV symbols: {len(csv_syms)} | JSON codes: {len(crypto_codes)} | Overlap: {len(overlap)}")
    if not overlap:
        print(f"[SellHoldPretrain] Sample CSV: {csv_syms[:10]}")
        print(f"[SellHoldPretrain] Sample JSON: {crypto_codes[:10]}")
        raise SystemExit(
            "No overlap after normalization — check separators (/, -, _), case, and exchange naming (BTC vs XBT), "
            "or ensure you paired the right JSON with the right CSV (ancient vs modern)."
        )

    # Narrow to the overlapping set so the env always initializes cleanly
    crypto_codes = overlap

    print("[SellHoldPretrain] Building environment…")
    env = DummyVecEnv([make_env(crypto_codes, historical_df)])
    env = VecMonitor(env)

    # PPO hyperparameters: suited for single-step episodes
    ppo_config = dict(
        learning_rate=5e-5,
        n_steps=256,                 # rollout length
        batch_size=64,
        gamma=0.97,                  # minor effect for single-step, but fine for consistency
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,               # encourage exploration across HOLD/SELL contexts
        vf_coef=0.5,
        max_grad_norm=0.5,
        normalize_advantage=True,
        device="cpu",
        verbose=1,
        tensorboard_log=LOG_DIR,
    )

    if CONTINUE_TRAINING:
        model_path = os.path.join(MODEL_DIR, MODEL_NAME)
        print(f"[SellHoldPretrain] Resuming from {model_path}.zip")
        model = MaskablePPO.load(
            model_path,
            env=env,
            tensorboard_log=LOG_DIR,
            device=ppo_config["device"],
        )
    else:
        print("[SellHoldPretrain] Creating new model…")
        model = MaskablePPO("MlpPolicy", env, **ppo_config)

    # Checkpointing
    checkpoint_cb = CheckpointCallback(
        save_freq=CHECKPOINT_EVERY,
        save_path=MODEL_DIR,
        name_prefix=MODEL_NAME
    )

    print(f"[SellHoldPretrain] Starting training for {TOTAL_TIMESTEPS:,} steps…")
    start_time = time.time()
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_cb)
    elapsed = time.time() - start_time
    print(f"[SellHoldPretrain] Training complete in {elapsed/60:.1f} minutes.")

    # Save final model
    final_path = os.path.join(MODEL_DIR, MODEL_NAME)
    model.save(final_path)
    print(f"[SellHoldPretrain] Saved final model to: {final_path}.zip")
    print(f"[TensorBoard] Logs: {LOG_DIR}")


if __name__ == "__main__":
    main()
