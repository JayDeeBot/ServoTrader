#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_ppo_agent.py

Evaluate a pre-trained PPO agent on the historical CryptoTradingEnv for a fixed
number of timesteps (default: 100,000). Produces an end-of-run evaluation summary
appended to the environment JSONL log, mirroring the 'training_summary' style.

Features:
- Loads trained PPO weights (MaskablePPO by default; optional RecurrentPPO path)
- Runs exactly EVAL_STEPS timesteps across episodes (auto-resets on done)
- Uses the same action-masking logic as training/live scripts
- Writes a final JSON summary: total_episodes, total_return_pct, total_timesteps,
  simulated_days, compounded_daily_return_pct, avg_episode_return_pct, win_rate

Usage:
  /bin/python3.11 /home/jarred/git/ServoTrader/scripts/eval_ppo_agent.py

Author: Jarred Deluca
Project: ServoTrader (Historical Evaluation)
License: MIT
"""

import os
import sys
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd

# Make repo imports work when called from scripts/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- SB3 Imports ---
from sb3_contrib import MaskablePPO, RecurrentPPO
from sb3_contrib.common.wrappers import ActionMasker

# --- Env & optional wrapper ---
from servo_trader.envs.crypto_trading_env import CryptoTradingEnv
# If you ever evaluate an LSTM model, set USE_LSTM=True and ensure this wrapper exists:
# from wrappers.action_mask_wrapper import LSTMActionMaskWrapper  # noqa: F401

# ==========================
# Config
# ==========================
MODEL_PATH = "/home/jarred/git/ServoTrader/models/ppo_servo_trader_squirtle"   # .zip optional
CRYPTO_CODES_PATH = "/home/jarred/git/ServoTrader/servo_trader/config/crypto_codes_modern.json"
DATA_DIR = "/home/jarred/git/ServoTrader/data/split_10k_chunks_modern"        # env will load 001.csv, 002.csv, ... automatically
FIRST_CHUNK = "000.csv"                                                         # initial dataset for env ctor

EVAL_STEPS = 100_000          # total environment steps to evaluate
EPISODE_TIMEOUT = 30          # minutes per episode (match training if needed)
USE_LSTM = False              # set True ONLY if the saved model is RecurrentPPO
DEBUG_MASK = False            # print mask diagnostics occasionally


def _load_crypto_codes(path: str):
    """
    Load crypto codes as a list from JSON.
    Supports either:
      - {"crypto_codes": [...]} or
      - [...] (list directly)
    """
    with open(path, "r") as f:
        data = json.load(f)
    return data["crypto_codes"] if isinstance(data, dict) and "crypto_codes" in data else data


# --- Action masks (same semantics as your training/live scripts) ---
def action_masks(env):
    """
    Legal moves:
        - No Crypto Held: Buy actions only (1 .. num_cryptos)
        - Crypto Held: Hold (0) and Sell (num_cryptos + 1)
    Returns:
        mask (np.ndarray[bool]): True = legal
    """
    core = env.unwrapped
    mask = np.zeros(core.action_space.n, dtype=bool)

    if core.active_crypto_index is None:
        mask[1:core.num_cryptos + 1] = True  # buys
    else:
        mask[0] = True                         # hold
        mask[core.num_cryptos + 1] = True      # sell

    return mask


def main():
    # ------------------------
    # Build env from first chunk
    # ------------------------
    codes = _load_crypto_codes(CRYPTO_CODES_PATH)
    first_chunk_path = os.path.join(DATA_DIR, FIRST_CHUNK)
    if not os.path.exists(first_chunk_path):
        raise FileNotFoundError(f"Initial dataset chunk missing: {first_chunk_path}")

    initial_df = pd.read_csv(first_chunk_path)

    # Base env (historical backtest env)
    env_core = CryptoTradingEnv(
        data=initial_df,
        crypto_codes=codes,
        episode_timeout=EPISODE_TIMEOUT
    )

    # Optional: LSTM wrapper (only if you trained a RecurrentPPO)
    # if USE_LSTM:
    #     env_core = LSTMActionMaskWrapper(env_core, model=None)  # model injected after load

    # Wrap with ActionMasker so the model sees the same env signature as during training
    env_wrapped = ActionMasker(env_core, action_masks)

    # ------------------------
    # Load model
    # ------------------------
    if USE_LSTM:
        model = RecurrentPPO.load(MODEL_PATH, env=env_wrapped)
        print(f"✅ Loaded RecurrentPPO model from {MODEL_PATH}")
        # If using LSTMActionMaskWrapper, inject model for mask-aware logic (if your wrapper expects it):
        # if isinstance(env_core, LSTMActionMaskWrapper):
        #     env_core.model = model
    else:
        model = MaskablePPO.load(MODEL_PATH, env=env_wrapped)
        print(f"✅ Loaded MaskablePPO model from {MODEL_PATH}")

    # ------------------------
    # Evaluate
    # ------------------------
    total_steps = 0
    episode_returns = []   # per-episode % return
    episode_lengths = []   # per-episode steps

    obs, info = env_wrapped.reset()
    # LSTM state mgmt (only used if USE_LSTM)
    lstm_state = None
    episode_start = True

    start_ts = time.time()
    last_print = start_ts

    while total_steps < EVAL_STEPS:
        # Build legal action mask
        mask = action_masks(env_wrapped)

        if DEBUG_MASK and (total_steps % 2000 == 0):
            legal = np.where(mask)[0]
            print(f"[mask] step={total_steps} legal_actions={legal[:10]}{'...' if len(legal) > 10 else ''}")

        # Predict action
        if USE_LSTM:
            action, lstm_state = model.predict(
                obs,
                state=lstm_state,
                episode_start=episode_start,
                deterministic=True
            )
        else:
            action, _ = model.predict(obs, deterministic=True, action_masks=mask)

        # Step
        obs, reward, terminated, truncated, info = env_wrapped.step(int(action))
        done = bool(terminated or truncated)
        total_steps += 1
        episode_start = False

        # Episode end → record stats and reset
        if done:
            # Compute episode % return from env state
            ep_return_pct = float((env_core.portfolio_value - 1.0) * 100.0)
            ep_len = int(env_core.global_step - env_core.start_step)
            episode_returns.append(ep_return_pct)
            episode_lengths.append(ep_len)

            if USE_LSTM:
                # Reset LSTM state at episode boundaries
                lstm_state = None
                episode_start = True

            obs, info = env_wrapped.reset()

        # Light progress heartbeat
        now = time.time()
        if now - last_print > 5.0:
            eps = env_core.episode_counter
            print(f"[eval] steps={total_steps}/{EVAL_STEPS} | episodes={eps} | last_portfolio=${env_core.portfolio_value:.4f}")
            last_print = now

    # ------------------------
    # Build evaluation summary (mirrors training summary structure)
    # ------------------------
    simulated_minutes = total_steps
    simulated_days = simulated_minutes / (60 * 24)

    total_return_pct = float(env_core.total_profit_percent)  # cumulative across episodes

    if simulated_days > 0 and total_return_pct > -100:
        total_growth = 1.0 + (total_return_pct / 100.0)
        daily_growth_factor = total_growth ** (1.0 / simulated_days)
        compounded_daily_return_pct = (daily_growth_factor - 1.0) * 100.0
    else:
        compounded_daily_return_pct = 0.0

    # Extras: average episode return & win rate
    avg_episode_return_pct = float(np.mean(episode_returns)) if episode_returns else 0.0
    wins = sum(1 for r in episode_returns if r > 0.0)
    win_rate = (wins / len(episode_returns)) * 100.0 if episode_returns else 0.0

    evaluation_summary = {
        "type": "evaluation_complete",
        "total_episodes": int(env_core.episode_counter),
        "total_return_pct": total_return_pct,
        "total_timesteps": int(total_steps),
        "simulated_days": simulated_days,
        "compounded_daily_return_pct": compounded_daily_return_pct,
        "avg_episode_return_pct": avg_episode_return_pct,
        "win_rate_pct": win_rate,
        "timestamp": datetime.now().isoformat()
    }

    # Append to the same JSONL log the env is writing to
    with open(env_core.log_path, "a") as f:
        f.write(json.dumps(evaluation_summary) + "\n\n")

    # Console summary
    print("\n====== Evaluation Complete ======")
    print(f" Timesteps: {total_steps}")
    print(f" Episodes:  {env_core.episode_counter}")
    print(f" Total Return (%):            {total_return_pct:.4f}")
    print(f" Avg Episode Return (%):      {avg_episode_return_pct:.4f}")
    print(f" Win Rate (% episodes > 0):   {win_rate:.2f}")
    print(f" Simulated Days:              {simulated_days:.2f}")
    print(f" Compounded Daily Return (%): {compounded_daily_return_pct:.6f}")
    print(f" Log appended to:             {env_core.log_path}")
    print("================================\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Evaluation interrupted by user.")
