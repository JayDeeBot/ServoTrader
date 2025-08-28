"""
train_live_ppo_agent.py

Continue training a MaskablePPO (MlpPolicy) agent directly on the *live* trading environment.

Features:
- Loads the *live* trading environment (LiveCryptoTradingEnv)
- Uses MaskablePPO (MlpPolicy) with action masking (same logic as training)
- ALWAYS continues training from an existing saved model (no "new training" path)
- Saves periodic checkpoints and a final model on completion
- Shows the Tkinter GUI (LiveEnvRenderer) while training runs in the background
- Logs a final training summary to the environment's JSONL log

Usage:
- Ensure your Binance credentials / params.yaml and crypto_codes.json are configured
- Ensure a base model already exists at MODEL_PATH (zip)
- Run: /bin/python3.11 /home/jarred/git/ServoTrader/scripts/train_live_ppo_agent.py

Author: Jarred Deluca
Project: ServoTrader (Live Training Mode)
License: MIT
"""

import os
import sys
import time
import json
from datetime import datetime
import threading

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CheckpointCallback

# Make repo root importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Custom modules ---
from servo_trader.envs.live_crypto_trading_env import LiveCryptoTradingEnv
from gui.live_renderer import LiveEnvRenderer  # GUI should run on main thread


# ----------------------
# Configurable paths
# ----------------------
MODEL_PATH = "/home/jarred/git/ServoTrader/models/ppo_servo_trader_squirtle.zip"  # MUST exist
CRYPTO_CODES_PATH = "/home/jarred/git/ServoTrader/servo_trader/config/crypto_codes.json"
EPISODE_TIMEOUT = 30            # minutes per episode in the live env
TENSORBOARD_LOGDIR = "/home/jarred/git/ServoTrader/logs"
CHECKPOINT_DIR = "/home/jarred/git/ServoTrader/models/"
CHECKPOINT_PREFIX = "ppo_servo_trader_squirtle_live_ft"
TOTAL_TIMESTEPS = 100_000       # ≈ 69 days at real-time 1-min cadence


# ----------------------
# Helper: load crypto codes
# ----------------------
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


# ----------------------
# Action masking (same as your training script)
# ----------------------
def action_masks(env):
    """
    Masks the action space to ensure only legal moves are possible.

    Legal moves:
        - No Crypto Held: Buy actions only (1..num_cryptos)
        - Crypto Held: Hold (0) and Sell (num_cryptos + 1)

    Returns:
        mask (np.ndarray[bool]): True = legal action
    """
    core = env.unwrapped
    mask = np.zeros(core.action_space.n, dtype=bool)

    if core.active_crypto_index is None:
        mask[1:core.num_cryptos + 1] = True     # BUY actions
        mask[2 + core.num_cryptos] = True                 # Not Buy (skip)
        # Optional: also allow HOLD while flat:
        # mask[0] = True
    else:
        mask[0] = True                           # HOLD
        mask[core.num_cryptos + 1] = True        # SELL

    return mask


# ----------------------
# PPO hyperparameters (conservative defaults for fine-tuning)
# ----------------------
ppo_config = {
    "learning_rate": 5e-5,
    "n_steps": 512,
    "batch_size": 64,
    "gamma": 0.97,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "normalize_advantage": True,
    "device": "cpu",
    "verbose": 1,
    "tensorboard_log": TENSORBOARD_LOGDIR,
}


def _assert_model_exists(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Base model not found at {path}. This script only supports continued training."
        )


def _write_training_summary(env_core, total_timesteps: int):
    """
    Append a final JSON record summarizing the live fine-tuning session.
    Mirrors your earlier log style.
    """
    # Real-time minutes ~= timesteps in your env by design
    simulated_minutes = int(total_timesteps)
    simulated_days = simulated_minutes / (60 * 24)

    # In your live env, you track decimal returns; convert to pct
    total_profit_pct = float(getattr(env_core, "total_profit_decimal", 0.0) * 100.0)

    # Geometric (compounded) daily return estimate (best-effort, optional)
    if simulated_days > 0 and total_profit_pct > -100:
        total_growth = 1.0 + (total_profit_pct / 100.0)
        daily_growth_factor = total_growth ** (1.0 / simulated_days)
        compounded_daily_return_pct = (daily_growth_factor - 1.0) * 100.0
    else:
        compounded_daily_return_pct = 0.0

    training_summary = {
        "type": "training_complete_live_ft",
        "total_episodes": int(getattr(env_core, "episode_counter", 0)),
        "total_return_pct": total_profit_pct,         # cumulative return in %
        "total_timesteps": int(total_timesteps),      # total real-time steps requested
        "simulated_days": simulated_days,             # ~days at 1 step/min
        "compounded_daily_return_pct": compounded_daily_return_pct,
        "timestamp": datetime.now().isoformat(),
    }

    try:
        with open(env_core.log_path, "a") as f:
            f.write(json.dumps(training_summary) + "\n\n")
    except Exception as e:
        print(f"[train_live_ppo_agent] Could not write summary to {env_core.log_path}: {e}")


def _training_worker(env_wrapped, model, total_timesteps: int):
    """
    Background thread: runs MaskablePPO.learn() on the *wrapped* env.
    """
    # Periodic checkpointing
    checkpoint = CheckpointCallback(
        save_freq=10_000,
        save_path=CHECKPOINT_DIR,
        name_prefix=CHECKPOINT_PREFIX
    )

    try:
        model.learn(total_timesteps=total_timesteps, callback=checkpoint)
    finally:
        # Always save a final snapshot
        final_path = os.path.join(CHECKPOINT_DIR, f"{CHECKPOINT_PREFIX}_final")
        try:
            model.save(final_path)
            print(f"💾 Saved final fine-tuned model to: {final_path}")
        except Exception as e:
            print(f"[train_live_ppo_agent] Failed to save final model: {e}")

        # Write the summary using the *core* env (unwrapped)
        _write_training_summary(env_wrapped.unwrapped, total_timesteps)
        print("✅ Live fine-tuning session finished (summary appended).")


def main():
    # 1) Build the core env (GUI reads from this object)
    crypto_codes = _load_crypto_codes(CRYPTO_CODES_PATH)
    env_core = LiveCryptoTradingEnv(crypto_codes=crypto_codes, episode_timeout=EPISODE_TIMEOUT)

    # 2) Wrap the env for MaskablePPO with action masking (same logic as training)
    env_wrapped = ActionMasker(env_core, action_masks)

    # 3) Load the existing model (ALWAYS continue training)
    _assert_model_exists(MODEL_PATH)
    print(f"📦 Loading base model from: {MODEL_PATH}")
    model = MaskablePPO.load(
        MODEL_PATH,
        env=env_wrapped,
        tensorboard_log=ppo_config["tensorboard_log"],
        device=ppo_config["device"]
    )

    # 4) Apply/override hyperparameters (optional; keeps optimizer config aligned)
    #    This is a safe way to ensure the learner uses your desired fine-tuning config.
    model.learning_rate = ppo_config["learning_rate"]

    # 5) Start training in the background (Tk must be on main thread)
    worker = threading.Thread(
        target=_training_worker,
        args=(env_wrapped, model, TOTAL_TIMESTEPS),
        daemon=True
    )
    worker.start()

    # 6) Launch the GUI on the main thread, pointing at the *core* env
    try:
        gui = LiveEnvRenderer(env_core)
        gui.run()
    except KeyboardInterrupt:
        print("\n🛑 Exiting live fine-tuning session.")
    finally:
        # Optional: signal a graceful stop if you added a flag in the env
        try:
            setattr(env_core, "request_stop", True)
        except Exception:
            pass


if __name__ == "__main__":
    main()