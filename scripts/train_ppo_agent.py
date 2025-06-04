"""
train_agent.py

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
- Run `tensorboard --logdir ../ppo_logs` to visualize training

Author: Jarred Deluca  
Project: ServoTrader  
License: MIT  
"""

# scripts/train_agent.py
# python scripts/train_ppo_agent.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import json
from stable_baselines3 import PPO # Import the PPO (Proximal Policy Optimization algorithm) Agent
from stable_baselines3.common.vec_env import DummyVecEnv 
from servo_trader.envs.crypto_trading_env import CryptoTradingEnv # Import the custom training environment
from stable_baselines3.common.callbacks import CheckpointCallback

# --- Load crypto codes ---
with open('/home/jarred/git/ServoTrader/servo_trader/config/crypto_codes.json') as f:
    crypto_codes = json.load(f)

# --- Load historical data ---
historical_df = pd.read_csv('/home/jarred/git/ServoTrader/data/historical_crypto_data.csv')

# --- Create environment ---
def make_env():
    return CryptoTradingEnv(data=historical_df, crypto_codes=crypto_codes, episode_timeout=15)

env = DummyVecEnv([make_env]) # Create the environment the agent will interact with

# --- Set this flag to True if continuing training ---
CONTINUE_TRAINING = False

if CONTINUE_TRAINING:
    # --- Load existing model ---
    model = PPO.load(
        "/home/jarred/git/ServoTrader/models/ppo_servo_trader",
        env=env,
        tensorboard_log="../ppo_logs",
        device="cpu",
        learning_rate=1e-5,
        policy_kwargs={"max_grad_norm": 0.5}
    )
else:
    # --- Train new model ---
    # MlpPolicy = Multilayer Perceptron (i.e., a fully connected feedforward neural network)
    # verbose=1: This turns on basic logging output, which prints training information (episode rewards, losses, etc.) to the console during training.
    # tensorboard_log="./ppo_logs": This logs training metrics (e.g., rewards, losses, episode lengths) to a directory called ppo_logs/ for use with TensorBoard — a tool for visualizing training progress over time.
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="../ppo_logs",
        device="cpu",
        learning_rate=1e-5,
        policy_kwargs={"max_grad_norm": 0.5}
    )

# --- Set up checkpointing ---
checkpoint = CheckpointCallback(
    save_freq=10_000, save_path="/home/jarred/git/ServoTrader/models/", name_prefix="ppo_servo_trader"
)

# --- Train model ---
model.learn(total_timesteps=1_000_000, callback=checkpoint)

# --- Save final model ---
model.save("/home/jarred/git/ServoTrader/models/ppo_servo_trader")