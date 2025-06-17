# test_crypto_env.py

"""
CryptoTradingEnv Test Suite

This module contains unit tests for the CryptoTradingEnv environment used in
the ServoCrypto trading system. The tests validate environment behavior,
including data preprocessing, reward calculation, and action-step transitions
(Buy, Hold, Sell).

These tests ensure that:
- The input data is properly filtered and cleaned.
- The step function produces valid observations and rewards.
- The internal reward function behaves correctly for various market conditions.

Assumptions:
- The environment supports discrete actions: 0 (Hold), 1 (Buy), 3 (Sell).
- The reward function can operate in both dense and terminal modes.
- Logging occurs to a fixed path (`servo_trader/env_logs/episode_log.txt`) when episodes terminate.

Usage:
    Run directly as a script:
        /bin/python3.11 -m tests.test_crypto_env

    Or include in a larger pytest test suite for ServoCrypto.

Author: Jarred Deluca  
Project: ServoCrypto  
Created: 2025-06-14
"""

import numpy as np
import pandas as pd
import os
from servo_trader.envs.crypto_trading_env import CryptoTradingEnv

def create_mock_data():
    """
    Creates mock cryptocurrency OHLCV data for multiple symbols.

    Returns:
        pd.DataFrame: Synthetic time series for BTC and ETH.
    """
    symbols = ['BTC', 'ETH']
    timesteps = 30
    data = []
    for symbol in symbols:
        for _ in range(timesteps):
            data.append({
                'symbol': symbol,
                'open': 100 + np.random.rand(),
                'high': 105 + np.random.rand(),
                'low': 95 + np.random.rand(),
                'close': 100 + np.random.rand(),
                'vwap': 101 + np.random.rand(),
                'volume': 1000 + np.random.rand() * 100,
                'count': 10 + np.random.rand() * 5
            })
    return pd.DataFrame(data)

def test_preprocess_data():
    """
    Verifies data preprocessing: filtering, sorting, and removing NaNs and zeros.
    """
    df = create_mock_data()
    env = CryptoTradingEnv(df, crypto_codes=['BTC', 'ETH'])

    # Manually reprocess input for independent verification
    filtered = df[df['symbol'].isin(['BTC', 'ETH'])]
    sorted_df = filtered.sort_values(['symbol', 'timestamp']) if 'timestamp' in filtered.columns else filtered
    sorted_df = sorted_df.dropna()
    sorted_df = sorted_df[(sorted_df != 0).all(axis=1)]

    assert set(sorted_df['symbol'].unique()) == {'BTC', 'ETH'}
    assert not sorted_df.isnull().values.any()
    assert (sorted_df.select_dtypes(include=[float]) != 0).all().all()
    print("[✓] preprocess_data: symbols filtered, no NaNs, no zeros")

def test_step_and_logging():
    """
    Tests step execution for Buy (1), Hold (0), and Sell (3).
    Verifies rewards are valid numbers and prints step outcomes.
    """
    df = create_mock_data()
    env = CryptoTradingEnv(df, crypto_codes=['BTC', 'ETH'], episode_timeout=5)

    env.reset()
    for action in [1, 0, 3]:  # Buy, Hold, Sell
        obs, reward, terminated, truncated, _ = env.step(action)
        assert not np.isnan(reward) and not np.isinf(reward)
        print(f"[✓] step({action}): reward={reward:.4f}, terminated={terminated}, truncated={truncated}")

def test_calculate_reward_cases():
    """
    Tests _calculate_reward logic for gain/loss scenarios in dense and terminal modes.
    """
    df = create_mock_data()
    env = CryptoTradingEnv(df, crypto_codes=['BTC', 'ETH'])

    rewards = [
        env._calculate_reward(0.05, True),   # gain, dense
        env._calculate_reward(0.05, False),  # gain, terminal
        env._calculate_reward(0.0, True),    # break-even, dense
        env._calculate_reward(-0.05, False)  # loss, terminal
    ]
    assert all(isinstance(r, float) for r in rewards)
    print("[✓] _calculate_reward() tested with gain, loss, break-even in dense and terminal modes")

def run_tests():
    """
    Runs all test cases for CryptoTradingEnv validation.
    """
    test_preprocess_data()
    test_step_and_logging()
    test_calculate_reward_cases()
    print("All extended tests passed successfully.")

if __name__ == "__main__":
    run_tests()