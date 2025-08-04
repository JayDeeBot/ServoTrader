#!/usr/bin/env python3
"""
test_live_crypto_trading_env.py

Unit test suite for the LiveCryptoTradingEnv class, specifically focusing on
the `_get_observation()` function.

This test checks that:
1. The crypto CSV generated contains data for all specified cryptos.
2. The internal `self.data` tensor has the correct shape and features.
3. The returned observation vector has the correct shape per the observation space.

Author: Jarred Deluca
Created: 2025
License: MIT
"""

import json
import unittest
import numpy as np
import pandas as pd
import sys
import os

# Add project root to PYTHONPATH so we can import from servo_trader
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from servo_trader.envs.live_crypto_trading_env import LiveCryptoTradingEnv

class TestLiveCryptoTradingEnv(unittest.TestCase):
    """
    Test case for validating core functionality of the LiveCryptoTradingEnv class.
    """

    @classmethod
    def setUpClass(cls):
        """
        Initializes the test environment and loads required config data.
        This runs once before all tests.
        """
        # Path to the crypto codes JSON
        crypto_code_path = "/home/jarred/git/ServoTrader/servo_trader/config/crypto_codes.json"
        with open(crypto_code_path, "r") as f:
            cls.crypto_codes = sorted(json.load(f)["crypto_codes"])

        # Instantiate the environment
        cls.env = LiveCryptoTradingEnv(crypto_codes=cls.crypto_codes, episode_timeout=30)

    def test_get_observation_correctness(self):
        """
        Test that _get_observation() performs the following correctly:
        1. Generates a CSV file with 30 rows per crypto and valid columns.
        2. Loads a 3D tensor into `self.data` of shape [30, 100, 13].
        3. Returns a flat observation vector matching the observation space.
        """
        # --- Step 1: Call _get_observation to trigger data generation ---
        obs = self.env._get_observation()

        # --- Step 2: Validate CSV output format ---
        csv_df = pd.read_csv(self.env.csv_path)

        # Check number of unique cryptos
        unique_symbols = csv_df['symbol'].nunique()
        self.assertEqual(unique_symbols, 100, f"Expected 100 unique cryptos, got {unique_symbols}")

        # Check total rows
        total_expected_rows = 100 * 30
        self.assertEqual(len(csv_df), total_expected_rows,
                         f"Expected {total_expected_rows} rows in CSV, got {len(csv_df)}")

        # Check required columns
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count', 'symbol']
        for col in required_cols:
            self.assertIn(col, csv_df.columns, f"Missing column in CSV: {col}")

        # --- Step 3: Validate internal data tensor ---
        data_tensor = self.env.data
        self.assertIsInstance(data_tensor, np.ndarray, "self.data is not a NumPy array.")
        self.assertEqual(data_tensor.shape, (30, 100, 13), f"Expected tensor shape [30, 100, 13], got {data_tensor.shape}")

        # Confirm that the processed data tensor includes the correct number of features per crypto
        expected_feature_count = 13
        actual_feature_count = self.env.data.shape[2]
        self.assertEqual(actual_feature_count, expected_feature_count,
                        f"Expected {expected_feature_count} engineered features per crypto, got {actual_feature_count}")

        # --- Step 4: Validate observation shape matches observation space ---
        self.assertEqual(obs.shape, self.env.observation_space.shape,
                         f"Observation shape mismatch. Expected: {self.env.observation_space.shape}, got: {obs.shape}")


if __name__ == "__main__":
    unittest.main()
