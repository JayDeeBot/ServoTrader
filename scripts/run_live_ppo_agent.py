"""
run_live_ppo_agent.py

This script loads a pre-trained PPO reinforcement learning agent and runs it on the
LiveCryptoTradingEnv, which performs real cryptocurrency trades on Binance using live OHLCV data.

Features:
- Loads a trained PPO model (MaskablePPO with action masking)
- Executes one episode at a time in live mode (Buy/Hold/Sell)
- Integrates with a Tkinter GUI (LiveEnvRenderer) for real-time feedback
- Automatically resets and continues episodes until stopped
- Designed to run indefinitely or during market hours

Usage:
- Ensure your Binance credentials and crypto_codes.json are properly configured
- Run the script: /bin/python3.11 /home/jarred/git/ServoTrader/scripts/run_live_ppo_agent.py

Author: Jarred Deluca  
Project: ServoTrader (Live Trading Mode)  
License: MIT  
"""

import os
import sys
import time
import json
import threading
from datetime import datetime

import numpy as np  # for action masking helper

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- SB3 Imports ---
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

# --- Custom Modules ---
from servo_trader.envs.live_crypto_trading_env import LiveCryptoTradingEnv
from gui.live_renderer import LiveEnvRenderer  # Ensure this path matches where you saved the GUI class

# --- Configurable Paths ---
MODEL_PATH = "/home/jarred/git/ServoTrader/models/ppo_servo_trader_squirtle.zip"
CRYPTO_CODES_PATH = "/home/jarred/git/ServoTrader/servo_trader/config/crypto_codes.json"
EPISODE_TIMEOUT = 30  # minutes per episode

# Toggle for quick console debugging of mask/state each step
DEBUG_MASK = False


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


# --- Action masks (mirrors training script logic) ---
def action_masks(env):
    """
    Masks the action space to ensure only legal moves are possible.

    Legal moves:
        - No Crypto Held: Buy actions only (1 to num_cryptos)
        - Crypto Held: Hold (0) and Sell (num_cryptos + 1)

    Returns:
        mask (np.ndarray[bool]): True = legal
    """
    core = env.unwrapped  # always inspect the real env (not the wrapper)
    mask = np.zeros(core.action_space.n, dtype=bool)

    if core.active_crypto_index is None:
        # No crypto held → only Buy actions are legal
        mask[1:core.num_cryptos + 1] = True
        # Optional: allow HOLD while flat (uncomment if desired)
        # mask[0] = True
    else:
        # Crypto held → only Hold and Sell are legal
        mask[0] = True  # Hold
        mask[core.num_cryptos + 1] = True  # Sell

    return mask


def run_live_loop(env_wrapped, model):
    print("🚀 Starting live trading loop...")
    obs, info = env_wrapped.reset()
    while True:
        # 1) Compute the mask each step (from the wrapped env)
        mask = action_masks(env_wrapped)  # returns np.bool_ array shape (n_actions,)

        # 2) Pass it to predict
        action, _ = model.predict(obs, deterministic=True, action_masks=mask)

        # (Optional safety debug)
        # if not mask[action]:
        #     print(f"⚠️ Policy selected ILLEGAL action {action}; mask says False — check masking.")

        # 3) Step the env
        obs, reward, terminated, truncated, info = env_wrapped.step(action)

        if terminated or truncated:
            print(f"[END] Episode {env_wrapped.unwrapped.episode_counter} completed.")
            obs, info = env_wrapped.reset()

        time.sleep(1)

# --- Bootstrap ---
if __name__ == "__main__":
    # 1) Build the core env (GUI will read from this object)
    crypto_codes = _load_crypto_codes(CRYPTO_CODES_PATH)
    env_core = LiveCryptoTradingEnv(crypto_codes=crypto_codes, episode_timeout=EPISODE_TIMEOUT)

    # 2) Wrap with ActionMasker for the agent (use training-style action_masks)
    env_wrapped = ActionMasker(env_core, action_masks)

    # 3) Load the trained MaskablePPO model against the wrapped env
    model = MaskablePPO.load(MODEL_PATH, env=env_wrapped)
    print(f"✅ Loaded MaskablePPO model from {MODEL_PATH}")

    # 4) Start the live loop in a background thread (Tk must stay on main thread)
    worker = threading.Thread(target=run_live_loop, args=(env_wrapped, model), daemon=True)
    worker.start()

    # 5) Run the GUI on the main thread, pointing it at the *core* env
    try:
        gui = LiveEnvRenderer(env_core)
        gui.run()
    except KeyboardInterrupt:
        print("\n🛑 Exiting live trading session.")