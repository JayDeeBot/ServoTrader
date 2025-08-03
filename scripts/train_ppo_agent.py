"""
train_ppo_agent.py

This script trains a Proximal Policy Optimization (PPO) reinforcement learning agent
on a custom Gym environment designed for high-frequency crypto trading using historical
OHLCV data. The environment simulates buy, hold, and sell actions across 100 cryptocurrencies,
with dense and risk-adjusted reward strategies.

Features:
- Loads historical crypto data and symbol codes
- Preprocesses and wraps the trading environment
- Supports fresh training or continued training from a saved model
- Saves periodic checkpoints and final model to disk
- Logs progress to TensorBoard for visualization

Usage:
- Set CONTINUE_TRAINING to True to resume from a saved model
- Run the script '/bin/python3.11 /home/jarred/git/ServoTrader/scripts/train_ppo_agent.py'
- Run `tensorboard --logdir /home/jarred/git/ServoTrader/logs --port 6006` to visualize training
- To view logs go to '[http://localhost:6006](http://localhost:6006)'

Author: Jarred Deluca  
Project: ServoTrader  
License: MIT  
"""

# scripts/train_ppo_agent.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import pandas as pd
import json
from datetime import datetime
import numpy as np
from stable_baselines3 import PPO # Import the PPO (Proximal Policy Optimization algorithm) Agent
from stable_baselines3.common.vec_env import DummyVecEnv 
from servo_trader.envs.crypto_trading_env import CryptoTradingEnv # Import the custom training environment
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor # Necessary for monitoring rewards
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from sb3_contrib import MaskablePPO, RecurrentPPO
from sb3_contrib.common.wrappers import ActionMasker
from wrappers.action_mask_wrapper import LSTMActionMaskWrapper
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta

print("Loading crypto codes...")

# --- Load crypto codes ---
with open('/home/jarred/git/ServoTrader/servo_trader/config/crypto_codes.json') as f:
    crypto_codes = json.load(f)

print("Loading initial dataset...")
# --- Load historical data ---
historical_df = pd.read_csv("/home/jarred/git/ServoTrader/data/split_10k_chunks_modern/000.csv")

# --- Create environment ---
USE_LSTM = False  # Set to False to use MaskablePPO instead

if USE_LSTM:
    # --- New method: RecurrentPPO with LSTMActionMaskWrapper ---
    env = DummyVecEnv([
        lambda: CryptoTradingEnv(data=historical_df, crypto_codes=crypto_codes, episode_timeout=15)
    ])
    env = VecMonitor(env)
    env.envs[0] = LSTMActionMaskWrapper(env.envs[0], model=None)  # Model injected later

else:
    # --- Original method: MaskablePPO with ActionMasker ---
    # Define action masks
    def action_masks(env):
        """
        Masks the action space to ensure only legal moves are possible.

        Legal moves:
            - No Crypto Held: Buy actions only (1 to num_cryptos)
            - Crypto Held: Hold (0) and Sell (num_cryptos + 1)

        Returns:
            mask (np array [bools]): an array of bools for each possible action, where each false index makes the action not possible.
        """
        mask = np.zeros(env.action_space.n, dtype=bool)  # Start all as False

        if env.active_crypto_index is None:
            # No crypto held → only Buy actions are legal
            mask[1:env.num_cryptos + 1] = True
        else:
            # Crypto held → only Hold and Sell are legal
            mask[0] = True  # Hold
            mask[env.num_cryptos + 1] = True  # Sell

        return mask

    env = DummyVecEnv([
        lambda: ActionMasker(
            CryptoTradingEnv(data=historical_df, crypto_codes=crypto_codes, episode_timeout=15),
            action_masks
        )
    ])
    env = VecMonitor(env)

# Define PPO hyperparameters
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
    "tensorboard_log": "/home/jarred/git/ServoTrader/logs"
}

# --- Set this flag to True if continuing training ---
CONTINUE_TRAINING = True

if CONTINUE_TRAINING:
    # --- Load existing model ---
    model_cls = RecurrentPPO if USE_LSTM else MaskablePPO
    model = model_cls.load(
        "/home/jarred/git/ServoTrader/models/ppo_servo_trader_squirtle",
        env=env,
        tensorboard_log=ppo_config["tensorboard_log"],
        device=ppo_config["device"]
    )
else:
    # --- Train new model ---
    if USE_LSTM:
        # MlpLstmPolicy = Multilayer Perceptron (i.e., a fully connected feedforward neural network)
        # verbose=1: This turns on basic logging output, which prints training information (episode rewards, losses, etc.) to the console during training.
        # tensorboard_log="./ppo_logs": This logs training metrics (e.g., rewards, losses, episode lengths) to a directory called ppo_logs/ for use with TensorBoard — a tool for visualizing training progress over time.
        model = RecurrentPPO("MlpLstmPolicy", env, **ppo_config)
        env.envs[0].model = model  # Inject model into wrapper
    else:
        model = MaskablePPO("MlpPolicy", env, **ppo_config)

# --- Set up checkpointing ---
checkpoint = CheckpointCallback(
    save_freq=100_000, save_path="/home/jarred/git/ServoTrader/models/", name_prefix="ppo_servo_trader_squirtle"
)

# --- Train model ---
model.learn(total_timesteps=100_000, callback=checkpoint)

# Save the total number of timesteps completed during training
actual_timesteps = model.num_timesteps  # Real number of steps — could be > 5000 due to n_steps batch rounding
simulated_minutes = actual_timesteps  # Since each step = 1 minute in your env
simulated_days = simulated_minutes / (60 * 24) # Compute how many trading days were simulated

print(f"[TensorBoard] Logs saved to: {ppo_config['tensorboard_log']}") # Check where the logs are going

# --- Log final training results as JSON ---
env_instance = env.envs[0].env  # Unwrap the inner CryptoTradingEnv from the VecEnv stack

total_return_pct = float(env_instance.total_profit_percent) # grab the total return

# Geometric (compounded) daily return
if simulated_days > 0 and total_return_pct > -100:
    total_growth = 1 + (total_return_pct / 100)
    daily_growth_factor = total_growth ** (1 / simulated_days)
    compounded_daily_return_pct = (daily_growth_factor - 1) * 100
else:
    compounded_daily_return_pct = 0.0

# Build a structured summary object
training_summary = {
    "type": "training_complete",                              # event type
    "total_episodes": env_instance.episode_counter,           # number of completed episodes
    "total_return_pct": float(env_instance.total_profit_percent),  # cumulative return (float)
    "total_timesteps": int(actual_timesteps), # total simulated minutes
    "simulated_days": simulated_days, # total simulated days
    "compounded_daily_return_pct": compounded_daily_return_pct,  # compound daily return
    "timestamp": datetime.now().isoformat()                  # timestamp of training end
}

# Append this record at the end of the JSONL log
with open(env_instance.log_path, "a") as f:
    f.write(json.dumps(training_summary) + "\n\n")

# --- Save final model ---
model.save("/home/jarred/git/ServoTrader/models/ppo_servo_trader_squirtle")