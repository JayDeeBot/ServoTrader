#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
docker_live_ppo_agent.py

Run a pre-trained MaskablePPO policy on LiveCryptoTradingEnv for *live* Binance trading,
packaged for Docker.

Key features
------------
- Paths taken from environment variables (Docker-friendly).
- Reads API keys and runtime settings from a mounted params.yaml.
- Uses a mounted crypto_codes.json.
- Writes data/logs to mounted host directories.
- Optional Tk GUI via LiveEnvRenderer (set DISABLE_GUI=0 to enable).
- Action masking mirrors training logic.

Environment variables (with defaults)
------------------------------------
MODEL_PATH=/app/models/ppo_servo_trader_squirtle.zip
CRYPTO_CODES_PATH=/app/config/crypto_codes.json
PARAMS_PATH=/app/config/params.yaml   # If a directory is mounted here, we auto-append /params.yaml
DATA_DIR=/app/data
LOG_DIR=/app/logs
EPISODE_TIMEOUT=120
DISABLE_GUI=0
DEBUG_MASK=0

Author: Jarred Deluca
Project: ServoTrader (Live Trading Mode via Docker)
License: MIT
"""

import os
import sys
import time
import json
import yaml
import threading
from datetime import datetime
from typing import Any, Dict

import numpy as np  # for action masking

# Ensure project root is on path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- SB3 Imports ---
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

# --- Custom Modules ---
from servo_trader.envs.live_crypto_trading_env import LiveCryptoTradingEnv
from gui.live_renderer import LiveEnvRenderer  # Optional GUI

# -----------------------------
# Config via environment
# -----------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/ppo_servo_trader_squirtle.zip")
CRYPTO_CODES_PATH = os.getenv("CRYPTO_CODES_PATH", "/app/config/crypto_codes.json")
PARAMS_PATH = os.getenv("PARAMS_PATH", "/app/config/params.yaml")
DATA_DIR = os.getenv("DATA_DIR", "/app/data")
LOG_DIR = os.getenv("LOG_DIR", "/app/logs")
EPISODE_TIMEOUT = int(os.getenv("EPISODE_TIMEOUT", "120"))
DISABLE_GUI = os.getenv("DISABLE_GUI", "0") == "1"
DEBUG_MASK = os.getenv("DEBUG_MASK", "0") == "1"

# If PARAMS_PATH is a directory (bind-mount mistake), assume params.yaml inside it
if os.path.isdir(PARAMS_PATH):
    PARAMS_PATH = os.path.join(PARAMS_PATH, "params.yaml")

# -----------------------------
# Utilities
# -----------------------------
def _healthcheck() -> None:
    """Basic presence checks for required files/dirs; create data/logs if missing."""
    missing = []
    for p, name in [
        (MODEL_PATH, "MODEL_PATH"),
        (CRYPTO_CODES_PATH, "CRYPTO_CODES_PATH"),
        (PARAMS_PATH, "PARAMS_PATH"),
    ]:
        if not os.path.exists(p):
            missing.append(f"{name} not found at: {p}")
    for d, name in [
        (DATA_DIR, "DATA_DIR"),
        (LOG_DIR, "LOG_DIR"),
    ]:
        if not os.path.isdir(d):
            try:
                os.makedirs(d, exist_ok=True)
            except Exception as e:
                missing.append(f"Could not create {name} at {d}: {e}")
    if missing:
        err = "\n".join(missing)
        raise FileNotFoundError(f"Healthcheck failed:\n{err}")

def _load_params(path: str) -> Dict[str, Any]:
    """Load YAML config (contains API keys and runtime config)."""
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}

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

    if DEBUG_MASK:
        print(f"[MASK] legal actions: {np.where(mask)[0].tolist()}")
    return mask

def run_live_loop(env_wrapped, model):
    """Main headless runner loop (no GUI)."""
    print("🚀 Starting live trading loop (headless)…")
    obs, info = env_wrapped.reset()
    while True:
        mask = action_masks(env_wrapped)  # np.bool_ array shape (n_actions,)
        action, _ = model.predict(obs, deterministic=True, action_masks=mask)
        obs, reward, terminated, truncated, info = env_wrapped.step(action)

        if terminated or truncated:
            print(f"[END] Episode {env_wrapped.unwrapped.episode_counter} completed.")
            obs, info = env_wrapped.reset()

        time.sleep(1)

def run_live_loop_bg(env_wrapped, model):
    """Background loop used when GUI is enabled (keep Tk on main thread)."""
    print("🚀 Starting live trading loop (background)…")
    obs, info = env_wrapped.reset()
    while True:
        mask = action_masks(env_wrapped)
        action, _ = model.predict(obs, deterministic=True, action_masks=mask)
        obs, reward, terminated, truncated, info = env_wrapped.step(action)

        if terminated or truncated:
            print(f"[END] Episode {env_wrapped.unwrapped.episode_counter} completed.")
            obs, info = env_wrapped.reset()

        time.sleep(1)

# -----------------------------
# Bootstrap
# -----------------------------
if __name__ == "__main__":
    try:
        _healthcheck()
        print("✅ Healthcheck passed.")

        # Load params.yaml and export keys (if env expects them)
        PARAMS = _load_params(PARAMS_PATH)
        # Support either 'binance' or 'api' sections for keys
        binance_cfg = PARAMS.get("binance", {}) or PARAMS.get("api", {})
        if binance_cfg:
            os.environ["BINANCE_API_KEY"] = str(binance_cfg.get("api_key", ""))
            os.environ["BINANCE_API_SECRET"] = str(binance_cfg.get("api_secret", ""))

        # Build core env (GUI reads from this object)
        crypto_codes = _load_crypto_codes(CRYPTO_CODES_PATH)

        # Pass only kwargs that LiveCryptoTradingEnv supports
        from inspect import signature
        allowed = set(signature(LiveCryptoTradingEnv.__init__).parameters.keys())
        extra_kwargs = {}
        if "data_dir" in allowed:
            extra_kwargs["data_dir"] = DATA_DIR
        if "log_dir" in allowed:
            extra_kwargs["log_dir"] = LOG_DIR
        if "params" in allowed:
            extra_kwargs["params"] = PARAMS

        env_core = LiveCryptoTradingEnv(
            crypto_codes=crypto_codes,
            episode_timeout=EPISODE_TIMEOUT,
            **extra_kwargs,
        )

        # Wrap with ActionMasker
        env_wrapped = ActionMasker(env_core, action_masks)

        # Load model
        print(f"🧠 Loading model: {MODEL_PATH}")
        model = MaskablePPO.load(MODEL_PATH, env=env_wrapped)
        print(f"✅ Loaded MaskablePPO model from {MODEL_PATH}")

        if DISABLE_GUI:
            # Headless mode
            run_live_loop(env_wrapped, model)
        else:
            # GUI mode: start runner in background thread; keep Tk on main thread
            worker = threading.Thread(target=run_live_loop_bg, args=(env_wrapped, model), daemon=True)
            worker.start()

            try:
                gui = LiveEnvRenderer(env_core)
                gui.run()
            except KeyboardInterrupt:
                print("\n🛑 Exiting live trading session (GUI).")

    except Exception as e:
        print(f"❌ Fatal error in live agent: {e}")
        sys.exit(1)
