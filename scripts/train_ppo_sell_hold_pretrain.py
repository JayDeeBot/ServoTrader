#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_ppo_sell_hold_pretrain.py

PPO pretraining script for SellHoldTrainingEnv (single-decision HOLD/SELL env).

Overview
--------
This script pretrains a policy to decide whether to HOLD or SELL an already-held
crypto at the current timestep. Each episode is exactly one decision:
    - HOLD  → reward =  +future_return over horizon H
    - SELL  → reward =  -future_return over horizon H
The episode terminates immediately.

Why this pretraining?
---------------------
The policy learns a representation for exit timing, complementary to the
BuyTrainingEnv (entry selection). You can later fine-tune the same model on
your full trading environment.

Features
--------
- Loads historical data and crypto codes
- Wraps SellHoldTrainingEnv with MaskablePPO + ActionMasker
- Fresh training or resume from checkpoint
- TensorBoard logging
- Periodic model checkpoints

Usage
-----
- Set CONTINUE_TRAINING below as needed.
- Run:
    /bin/python3.11 /home/jarred/git/ServoTrader/scripts/train_ppo_sell_hold_pretrain.py
- TensorBoard:
    tensorboard --logdir /home/jarred/git/ServoTrader/logs --port 6006
  then open http://localhost:6006

Dependencies
------------
- stable-baselines3
- sb3-contrib   (MaskablePPO, ActionMasker)
- gymnasium, numpy, pandas

Author: Jarred Deluca
Project: ServoTrader
License: MIT
"""

# scripts/train_ppo_sell_hold_pretrain.py

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

# 👇 Import your Sell/Hold pretraining environment
from servo_trader.envs.sell_hold_training_env import SellHoldTrainingEnv


# --------------------------
# Configurable paths/flags
# --------------------------
CODES_PATH = "/home/jarred/git/ServoTrader/servo_trader/config/crypto_codes.json"
DATA_PATH  = "/home/jarred/git/ServoTrader/data/split_10k_chunks_modern/000.csv"

MODEL_DIR  = "/home/jarred/git/ServoTrader/models"
LOG_DIR    = "/home/jarred/git/ServoTrader/logs"
MODEL_NAME = "ppo_sell_hold_pretrain_ivysaur"

TOTAL_TIMESTEPS   = 300_000          # adjust as needed
CHECKPOINT_EVERY  = 100_000          # steps per checkpoint
CONTINUE_TRAINING = False            # set True to resume from MODEL_DIR/MODEL_NAME.zip

# SellHoldTrainingEnv knobs
HISTORY_WINDOW   = 5
LOOKAHEAD_STEPS  = 30                # future-return horizon for reward
HELD_SELECTION   = "random"          # "random" or "round_robin"
REWARD_BOUNDED   = False             # if True, tanh-squash reward to [-1, 1]
EPISODE_TIMEOUT  = 30                # shape parity only; not used to truncate


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
        )

        # Mask to only HOLD (0) and SELL (N+1)
        def mask_fn(e):
            mask = np.zeros(e.action_space.n, dtype=bool)
            mask[0] = True                      # HOLD
            mask[e.num_cryptos + 1] = True      # SELL
            return mask

        return ActionMasker(env, mask_fn)
    return _env_fn


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    print("[SellHoldPretrain] Loading crypto codes…")
    with open(CODES_PATH, "r") as f:
        crypto_codes = json.load(f)

    print("[SellHoldPretrain] Loading dataset…")
    historical_df = pd.read_csv(DATA_PATH)

    print("[SellHoldPretrain] Building environment…")
    # Vec stack: DummyVecEnv -> VecMonitor
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