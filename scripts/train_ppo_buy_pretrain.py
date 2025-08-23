#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_ppo_buy_pretrain.py

PPO pretraining script for BuyTrainingEnv (single-decision, buy-only ranking env).

Overview
--------
This script pretrains a PPO policy to *rank/select* cryptocurrencies using the
BuyTrainingEnv. Each episode is exactly one decision: pick the best crypto to
buy at the current timestep. Reward is a rank-based score derived from future
returns over a configurable lookahead horizon.

Why this pretraining?
---------------------
By training on a pure selection task with dense, well-shaped rewards (ranking),
the policy learns feature representations that correlate with future outperformance.
You can later fine-tune this policy on the full trading environment (hold/sell,
transaction costs, risk penalties, etc.).

Features
--------
- Loads historical data and crypto codes
- Wraps BuyTrainingEnv with action masking (MaskablePPO)
- Fresh training or resume from checkpoint
- TensorBoard logging
- Periodic model checkpoints

Usage
-----
- Set CONTINUE_TRAINING below as needed.
- Run:
    /bin/python3.11 /home/jarred/git/ServoTrader/scripts/train_ppo_buy_pretrain.py
- TensorBoard:
    tensorboard --logdir /home/jarred/git/ServoTrader/logs --port 6006
  then open http://localhost:6006

Dependencies
------------
- stable-baselines3
- sb3-contrib  (for MaskablePPO and ActionMasker)
- gymnasium, numpy, pandas

Author: Jarred Deluca
Project: ServoTrader
License: MIT
"""

# scripts/train_ppo_buy_pretrain.py

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import time
from datetime import datetime

import numpy as np
import pandas as pd

from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

# 👇 Import your BuyTrainingEnv (make sure the path/package matches your repo)
from servo_trader.envs.buy_training_env import BuyTrainingEnv


# --------------------------
# Configurable paths/flags
# --------------------------
CODES_PATH = "/home/jarred/git/ServoTrader/servo_trader/config/crypto_codes.json"
DATA_PATH  = "/home/jarred/git/ServoTrader/data/split_10k_chunks_modern/000.csv"

MODEL_DIR  = "/home/jarred/git/ServoTrader/models"
LOG_DIR    = "/home/jarred/git/ServoTrader/logs"
MODEL_NAME = "ppo_buy_pretrain_bulbasaur"

TOTAL_TIMESTEPS   = 300_000         # adjust as needed
CHECKPOINT_EVERY  = 100_000         # steps per checkpoint
CONTINUE_TRAINING = False           # set True to resume from MODEL_DIR/MODEL_NAME.zip

# BuyTrainingEnv knobs
HISTORY_WINDOW   = 5
LOOKAHEAD_STEPS  = 30               # future-return horizon for ranking
REWARD_MODE      = "zero_to_one"    # or "minus_one_to_one"
EPISODE_TIMEOUT  = 30               # not used for termination here; kept for parity


def make_env(crypto_codes, historical_df):
    """
    Factory for a single BuyTrainingEnv instance wrapped with an action masker.
    """
    def _env_fn():
        env = BuyTrainingEnv(
            data=historical_df,
            crypto_codes=crypto_codes,
            episode_timeout=EPISODE_TIMEOUT,
            history_window=HISTORY_WINDOW,
            lookahead_steps=LOOKAHEAD_STEPS,
            reward_mode=REWARD_MODE,
        )
        # Mask all actions except BUY 1..N at every step
        def mask_fn(e):
            # Try to use the env's own mask if available (future-proof)
            if hasattr(e, "_get_action_mask"):
                return e._get_action_mask()
            mask = np.zeros(e.action_space.n, dtype=bool)
            # 1..N enabled
            mask[1:1 + e.num_cryptos] = True
            return mask

        return ActionMasker(env, mask_fn)
    return _env_fn


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    print("[BuyPretrain] Loading crypto codes…")
    with open(CODES_PATH, "r") as f:
        crypto_codes = json.load(f)

    print("[BuyPretrain] Loading dataset…")
    historical_df = pd.read_csv(DATA_PATH)

    print("[BuyPretrain] Building environment…")
    # Vec stack: DummyVecEnv -> VecMonitor
    env = DummyVecEnv([make_env(crypto_codes, historical_df)])
    env = VecMonitor(env)

    # PPO hyperparameters tuned for single-step episodes:
    # - Slightly higher entropy to encourage broad exploration across symbols
    # - Shorter n_steps since episodes terminate each step anyway
    ppo_config = dict(
        learning_rate=5e-5,
        n_steps=256,                 # rollout length; fine even though episodes are single-step
        batch_size=64,
        gamma=0.97,                  # modest discount; has minor effect with single-step episodes
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,               # a bit higher to promote exploration across many BUYs
        vf_coef=0.5,
        max_grad_norm=0.5,
        normalize_advantage=True,
        device="cpu",
        verbose=1,
        tensorboard_log=LOG_DIR,
    )

    if CONTINUE_TRAINING:
        model_path = os.path.join(MODEL_DIR, MODEL_NAME)
        print(f"[BuyPretrain] Resuming from {model_path}.zip")
        model = MaskablePPO.load(
            model_path,
            env=env,
            tensorboard_log=LOG_DIR,
            device=ppo_config["device"],
        )
    else:
        print("[BuyPretrain] Creating new model…")
        model = MaskablePPO("MlpPolicy", env, **ppo_config)

    # Checkpointing
    checkpoint_cb = CheckpointCallback(
        save_freq=CHECKPOINT_EVERY,
        save_path=MODEL_DIR,
        name_prefix=MODEL_NAME
    )

    print(f"[BuyPretrain] Starting training for {TOTAL_TIMESTEPS:,} steps…")
    start_time = time.time()
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_cb)
    elapsed = time.time() - start_time
    print(f"[BuyPretrain] Training complete in {elapsed/60:.1f} minutes.")

    # Save final model
    final_path = os.path.join(MODEL_DIR, MODEL_NAME)
    model.save(final_path)
    print(f"[BuyPretrain] Saved final model to: {final_path}.zip")
    print(f"[TensorBoard] Logs: {LOG_DIR}")


if __name__ == "__main__":
    main()
