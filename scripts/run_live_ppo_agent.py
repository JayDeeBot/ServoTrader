"""
run_live_ppo_agent.py

This script loads a pre-trained PPO reinforcement learning agent and runs it on the
LiveCryptoTradingEnv, which performs real cryptocurrency trades on Binance using live OHLCV data.

Features:
- Loads a trained PPO model (RecurrentPPO with LSTM policy by default)
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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- SB3 Imports ---
from sb3_contrib import RecurrentPPO

# --- Custom Modules ---
from servo_trader.envs.live_crypto_trading_env import LiveCryptoTradingEnv
from gui.live_renderer import LiveEnvRenderer  # Ensure this path matches where you saved the GUI class

# --- Configurable Paths ---
MODEL_PATH = "/home/jarred/git/ServoTrader/models/ppo_servo_trader_squirtle.zip"
CRYPTO_CODES_PATH = "/home/jarred/git/ServoTrader/servo_trader/config/crypto_codes.json"
EPISODE_TIMEOUT = 30  # minutes per episode

# --- Load Crypto Codes ---
with open(CRYPTO_CODES_PATH, "r") as f:
    crypto_codes = json.load(f)

# --- Create the Live Trading Environment ---
env = LiveCryptoTradingEnv(crypto_codes=crypto_codes, episode_timeout=EPISODE_TIMEOUT)

# --- Load the trained PPO model ---
model = RecurrentPPO.load(MODEL_PATH, env=env)
print(f"✅ Loaded model from {MODEL_PATH}")

# --- Launch the GUI in a separate thread ---
gui = LiveEnvRenderer(env)
threading.Thread(target=gui.run, daemon=True).start()


def run_live_loop():
    """
    Main loop to continuously run episodes using the PPO agent in live mode.
    Observations are passed to the model to determine actions, which are executed live.
    Each episode terminates when the SELL action is taken or timeout is reached.
    """
    print("🚀 Starting live trading loop...")
    obs, info = env.reset()
    state = None  # RNN state (for RecurrentPPO)
    done = False

    while True:
        # --- Predict next action using the trained agent ---
        action, state = model.predict(obs, state=state, episode_start=[done], deterministic=True)

        # --- Take the action in the environment ---
        obs, reward, terminated, truncated, info = env.step(action)

        # --- Reset if episode ends (due to SELL or timeout) ---
        done = terminated or truncated
        if done:
            print(f"[END] Episode {env.episode_counter} completed.")
            obs, info = env.reset()
            state = None  # Reset LSTM state

        # Optional: sleep briefly to reduce CPU usage
        time.sleep(1)


# --- Run the main live loop ---
if __name__ == "__main__":
    try:
        run_live_loop()
    except KeyboardInterrupt:
        print("\n🛑 Exiting live trading session.")
