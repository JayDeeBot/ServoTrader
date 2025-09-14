#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_ppo_sell_hold_pretrain.py

PPO pretraining script for SellHoldTrainingEnv (single-decision HOLD/SELL env).

Now supports **either**:
- MaskablePPO (with action masking via ActionMasker), or
- RecurrentPPO (LSTM) using a lightweight LSTMActionMaskWrapper (same pattern as your other script).

What this script does
---------------------
• Loads crypto codes and a CSV dataset (robust normalization + overlap checks)
• Builds the SellHoldTrainingEnv (stringent reward uses Low[t+H] vs Close[t])
• Trains with MaskablePPO *or* RecurrentPPO (MlpLstmPolicy)
• Saves periodic checkpoints and the final model
• Logs training to TensorBoard (LOG_DIR)

Usage
-----
• Toggle `USE_LSTM` (True = RecurrentPPO; False = MaskablePPO).
• Toggle `CONTINUE_TRAINING` to resume from MODEL_DIR/MODEL_NAME.zip.
• Run TensorBoard: `tensorboard --logdir /home/jarred/git/ServoTrader/logs --port 6006`
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

# Algorithms
from sb3_contrib import MaskablePPO, RecurrentPPO
from sb3_contrib.common.wrappers import ActionMasker

# Optional LSTM action-mask wrapper (mirrors your other script)
# This wrapper should consult env._get_action_mask() to filter logits/action sampling.
# If your project already has this, import it; otherwise switch USE_LSTM=False.
from wrappers.action_mask_wrapper import LSTMActionMaskWrapper

# 👇 Import your Sell/Hold pretraining environment
from servo_trader.envs.sell_hold_training_env import SellHoldTrainingEnv


# --------------------------
# Configurable paths/flags
# --------------------------
CODES_PATH = "/home/jarred/git/ServoTrader/servo_trader/config/crypto_codes_ancient_2.json"
DATA_PATH  = "/home/jarred/git/ServoTrader/data/split_10k_chunks_ancient_2/000.csv"

MODEL_DIR  = "/home/jarred/git/ServoTrader/models"
LOG_DIR    = "/home/jarred/git/ServoTrader/logs"
MODEL_NAME = "ppo_goku"                 # name prefix for checkpoints/final model

TOTAL_TIMESTEPS   = 1_000_000           # adjust as needed
CHECKPOINT_EVERY  = 100_000             # steps per checkpoint
CONTINUE_TRAINING = True                # resume from MODEL_DIR/MODEL_NAME.zip if present

# Algorithm switch: True = LSTM (RecurrentPPO), False = MaskablePPO
USE_LSTM = False

# SellHoldTrainingEnv knobs
HISTORY_WINDOW   = 1
LOOKAHEAD_STEPS  = 45                   # future-return horizon for reward
HELD_SELECTION   = "round_robin"        # "random" or "round_robin"
REWARD_BOUNDED   = False                # if True, tanh-squash reward to [-1, 1]
EPISODE_TIMEOUT  = 60                   # shape parity only; not used to truncate


# --------------------------
# Helpers (robust + normalized)
# --------------------------
def _norm_symbol(s: str) -> str:
    """
    Normalize a symbol string to canonical BASEQUOTE (uppercase, strip / - _).

    Args:
        s (str): Input symbol (e.g., 'btc/usdt', 'ETH-USD').

    Returns:
        str: Normalized symbol (e.g., 'BTCUSDT', 'ETHUSD').
    """
    s = (s or "").strip().upper()
    return re.sub(r"[\/\-_]", "", s)


def _looks_like_symbol(s: str) -> bool:
    """
    Heuristic: determine if a string looks like an exchange symbol (A-Z0-9, len 5..15).

    Args:
        s (str): Candidate string.

    Returns:
        bool: True if it looks like a symbol.
    """
    return bool(re.fullmatch(r"[A-Z0-9]{5,15}", s or ""))


def load_crypto_codes(path: str) -> list[str]:
    """
    Robustly load symbols from a JSON file.
    Supports:
      - ["BTCUSDT", "ETHUSDT", ...]
      - {"codes": [...]}, {"crypto_codes": [...]}, {"symbols": [...]}, {"tickers": [...]}
      - { "CRYPTOCODES": ["BTCUSDT", ...] }             (values win)
      - { "BTCUSDT": {...}, "ETHUSDT": {...} }          (dict-of-meta; uses KEYS as last resort)

    Args:
        path (str): JSON file path.

    Returns:
        list[str]: Sorted, deduplicated, normalized symbols.

    Raises:
        ValueError: If no valid list can be found/parsed.
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
        # 2) Any dict value that is list[str]
        if codes is None:
            for v in blob.values():
                if isinstance(v, list) and all(isinstance(x, str) for x in v):
                    codes = v
                    break
        # 3) Dict-of-meta: use KEYS if they look like symbols
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
    """
    Find the symbol-like column in a CSV (tolerant of casing & aliases).

    Args:
        df (pd.DataFrame): DataFrame loaded from CSV.

    Returns:
        str: The original column name in the DF that holds the symbol/pair/ticker.

    Raises:
        ValueError: If no plausible column is found.
    """
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in ("symbol", "pair", "ticker"):
        if cand in cols_lower:
            return cols_lower[cand]
    raise ValueError(f"No symbol/pair/ticker column in CSV. Columns: {list(df.columns)}")


# --------------------------
# Environment builders
# --------------------------
def make_env(crypto_codes: list[str], historical_df: pd.DataFrame) -> callable:
    """
    Factory that returns a thunk creating a single SellHoldTrainingEnv instance.
    For MaskablePPO, we'll wrap with ActionMasker in build_vec_env().
    For LSTM (RecurrentPPO), we will wrap the underlying env with LSTMActionMaskWrapper after VecMonitor.

    Args:
        crypto_codes (list[str]): Normalized list of symbols to train on.
        historical_df (pd.DataFrame): CSV data as DataFrame.

    Returns:
        callable: A no-arg function that builds and returns the environment instance.
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
        return env
    return _env_fn


def build_vec_env(crypto_codes: list[str], historical_df: pd.DataFrame, *, use_lstm: bool):
    """
    Build and wrap the vectorized environment according to the algorithm selection.

    - For MaskablePPO: wraps with ActionMasker to supply a legal action mask (HOLD and SELL only).
    - For RecurrentPPO: wraps with LSTMActionMaskWrapper (your project wrapper) after VecMonitor.

    Args:
        crypto_codes (list[str]): Training symbols.
        historical_df (pd.DataFrame): CSV data.
        use_lstm (bool): Whether to build the env stack for LSTM training.

    Returns:
        DummyVecEnv: A VecEnv with proper wrappers applied.
    """
    if not use_lstm:
        # Maskable path (MaskablePPO expects a mask via ActionMasker)
        def mask_fn(e):
            """Return boolean mask for legal actions (only HOLD and SELL are legal)."""
            if hasattr(e, "_get_action_mask"):
                return e._get_action_mask()
            mask = np.zeros(e.action_space.n, dtype=bool)
            mask[0] = True
            mask[e.num_cryptos + 1] = True
            return mask

        env = DummyVecEnv([lambda: ActionMasker(make_env(crypto_codes, historical_df)(), mask_fn)])
        env = VecMonitor(env)
        return env

    # LSTM path (RecurrentPPO). Build the raw env and inject LSTMActionMaskWrapper after VecMonitor.
    env = DummyVecEnv([make_env(crypto_codes, historical_df)])
    env = VecMonitor(env)
    # Replace the single underlying env with the LSTM action-mask wrapper.
    # The wrapper will receive the model later (after we construct it).
    env.envs[0] = LSTMActionMaskWrapper(env.envs[0], model=None)
    return env


# --------------------------
# Algorithm / Model builders
# --------------------------
def ppo_hyperparams(use_lstm: bool) -> dict:
    """
    Default PPO hyperparameters tuned for single-step episodes.

    Args:
        use_lstm (bool): If True, returns defaults suitable for RecurrentPPO (MlpLstmPolicy).

    Returns:
        dict: Keyword arguments for the PPO/RecurrentPPO constructors.
    """
    base = dict(
        learning_rate=5e-5,
        n_steps=512 if use_lstm else 256,  # larger rollout can stabilize LSTM
        batch_size=64,
        gamma=0.97,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01 if use_lstm else 0.02,  # slightly lower entropy for LSTM by default
        vf_coef=0.5,
        max_grad_norm=0.5,
        normalize_advantage=True,
        device="cpu",
        verbose=1,
        tensorboard_log=LOG_DIR,
    )

    # Optional LSTM policy kwargs (uncomment to tune)
    # if use_lstm:
    #     base["policy_kwargs"] = dict(
    #         lstm_hidden_size=256,
    #         n_lstm_layers=1,
    #         shared_lstm=False,
    #     )

    return base


def build_model(env, *, use_lstm: bool, continue_training: bool):
    """
    Create or load the PPO model based on the selected algorithm.

    Args:
        env (VecEnv): Vectorized environment already wrapped.
        use_lstm (bool): If True, uses RecurrentPPO("MlpLstmPolicy"); else MaskablePPO("MlpPolicy").
        continue_training (bool): If True, loads existing model weights from MODEL_DIR/MODEL_NAME.zip.

    Returns:
        (BaseAlgorithm, str): The model instance and the resolved model path prefix for saves/checkpoints.
    """
    cfg = ppo_hyperparams(use_lstm)
    model_path = os.path.join(MODEL_DIR, MODEL_NAME)

    # Resume if requested and a valid checkpoint is present
    if continue_training and os.path.exists(model_path + ".zip"):
        print(f"[SellHoldPretrain] Resuming from {model_path}.zip")
        model_cls = RecurrentPPO if use_lstm else MaskablePPO
        model = model_cls.load(
            model_path,
            env=env,
            tensorboard_log=cfg["tensorboard_log"],
            device=cfg["device"],
        )
        # Inject the model into the LSTM wrapper after load (if needed)
        if use_lstm and hasattr(env, "envs") and len(env.envs) == 1 and isinstance(env.envs[0], LSTMActionMaskWrapper):
            env.envs[0].model = model
        return model, model_path

    # Fresh model
    print("[SellHoldPretrain] Creating new model…")
    if use_lstm:
        model = RecurrentPPO("MlpLstmPolicy", env, **cfg)
        # Inject the model into the LSTM wrapper so it can mask logits/actions
        if hasattr(env, "envs") and len(env.envs) == 1 and isinstance(env.envs[0], LSTMActionMaskWrapper):
            env.envs[0].model = model
    else:
        model = MaskablePPO("MlpPolicy", env, **cfg)

    return model, model_path


# --------------------------
# Main
# --------------------------
def main():
    """
    Entrypoint:
      - Loads codes & CSV
      - Validates overlap
      - Builds env (with Maskable or LSTM support)
      - Builds/loads model
      - Trains with checkpointing
      - Saves final model
    """
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
    env = build_vec_env(crypto_codes, historical_df, use_lstm=USE_LSTM)

    # Build or load model
    model, model_path = build_model(env, use_lstm=USE_LSTM, continue_training=CONTINUE_TRAINING)

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
    model.save(model_path)
    print(f"[SellHoldPretrain] Saved final model to: {model_path}.zip")
    print(f"[TensorBoard] Logs: {LOG_DIR}")


if __name__ == "__main__":
    main()
