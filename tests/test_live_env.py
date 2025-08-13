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

    # def test_get_observation_correctness(self):
    #     """
    #     Test that _get_observation() performs the following correctly:
    #     1. Generates a CSV file with 30 rows per crypto and valid columns.
    #     2. Loads a 3D tensor into `self.data` of shape [30, 100, 13].
    #     3. Returns a flat observation vector matching the observation space.
    #     """
    #     # --- Step 1: Call _get_observation to trigger data generation ---
    #     obs = self.env._get_observation()

    #     # --- Step 2: Validate CSV output format ---
    #     csv_df = pd.read_csv(self.env.csv_path)

    #     # Check number of unique cryptos
    #     unique_symbols = csv_df['symbol'].nunique()
    #     self.assertEqual(unique_symbols, 100, f"Expected 100 unique cryptos, got {unique_symbols}")

    #     # Check total rows
    #     total_expected_rows = 100 * 30
    #     self.assertEqual(len(csv_df), total_expected_rows,
    #                      f"Expected {total_expected_rows} rows in CSV, got {len(csv_df)}")

    #     # Check required columns
    #     required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count', 'symbol']
    #     for col in required_cols:
    #         self.assertIn(col, csv_df.columns, f"Missing column in CSV: {col}")

    #     # --- Step 3: Validate internal data tensor ---
    #     data_tensor = self.env.data
    #     self.assertIsInstance(data_tensor, np.ndarray, "self.data is not a NumPy array.")
    #     self.assertEqual(data_tensor.shape, (30, 100, 13), f"Expected tensor shape [30, 100, 13], got {data_tensor.shape}")

    #     # Confirm that the processed data tensor includes the correct number of features per crypto
    #     expected_feature_count = 13
    #     actual_feature_count = self.env.data.shape[2]
    #     self.assertEqual(actual_feature_count, expected_feature_count,
    #                     f"Expected {expected_feature_count} engineered features per crypto, got {actual_feature_count}")

    #     # --- Step 4: Validate observation shape matches observation space ---
    #     self.assertEqual(obs.shape, self.env.observation_space.shape,
    #                      f"Observation shape mismatch. Expected: {self.env.observation_space.shape}, got: {obs.shape}")
        
    # def test_reset_function(self):
    #     """
    #     Test the reset() method of LiveCryptoTradingEnv.

    #     This function verifies that:
    #     - The reset operation returns a valid observation and info dict.
    #     - The crypto codes may be updated by _refresh_crypto_codes().
    #     - Key internal state variables are properly initialized/reset.
    #     - The reset duration is printed for performance tracking.
    #     - All relevant variables are printed for visual inspection.
    #     """
    #     import time

    #     # --- Step 1: Load crypto codes BEFORE reset ---
    #     json_path = "/home/jarred/git/ServoTrader/servo_trader/config/crypto_codes.json"
    #     with open(json_path, "r") as f:
    #         crypto_codes_before = sorted(json.load(f)["crypto_codes"])

    #     print("\n🔁 Crypto codes before reset():")
    #     print(crypto_codes_before)

    #     # --- Step 2: Time the reset function ---
    #     start_time = time.perf_counter()
    #     obs, info = self.env.reset()
    #     end_time = time.perf_counter()
    #     duration = end_time - start_time
    #     print(f"\n⏱️ reset() executed in {duration:.4f} seconds")

    #     # --- Step 3: Load crypto codes AFTER reset ---
    #     with open(json_path, "r") as f:
    #         crypto_codes_after = sorted(json.load(f)["crypto_codes"])

    #     print("\n🔁 Crypto codes after reset():")
    #     print(crypto_codes_after)

    #     # --- Step 4: Print relevant environment state variables ---
    #     print("\n📋 Environment state after reset:")
    #     print(f"active_crypto_index: {self.env.active_crypto_index}")
    #     print(f"active_crypto_code : {self.env.active_crypto_code}")
    #     print(f"buy_price          : {self.env.buy_price}")
    #     print(f"cash_balance       : {self.env.cash_balance}")
    #     print(f"portfolio_value    : {self.env.portfolio_value}")
    #     print(f"current_step       : {self.env.current_step}")
    #     print(f"start_index        : {self.env.start_index}")

    #     # --- Step 5: Basic sanity checks ---
    #     self.assertIsInstance(obs, np.ndarray, "Observation is not a NumPy array.")
    #     self.assertEqual(obs.shape, self.env.observation_space.shape,
    #                     "Observation shape does not match observation space.")

    #     self.assertIn("action_mask", info, "Info dict missing 'action_mask' key.")
    #     self.assertIsInstance(info["action_mask"], np.ndarray, "'action_mask' is not a NumPy array.")

    # def test_calculate_reward_cases(self):
    #     """
    #     Tests the _calculate_reward function for BUY, HOLD, and SELL actions
    #     using various profit scenarios and volatility.
    #     """
    #     env = self.env  # For brevity

    #     # --- BUY case: test with sample price series ---
    #     price_series = pd.Series([100, 101, 99, 98, 100, 102, 101, 99, 100, 100])
    #     reward_buy = env._calculate_reward(profit=0, action="BUY", price_series=price_series)
    #     self.assertTrue(np.isfinite(reward_buy), "BUY reward should be a finite value")
    #     self.assertLessEqual(reward_buy, 0, "BUY reward should be ≤ 0 due to volatility penalty")

    #     # --- HOLD cases ---
    #     reward_hold_positive = env._calculate_reward(profit=0.05, action="HOLD")
    #     reward_hold_negative = env._calculate_reward(profit=-0.05, action="HOLD")
    #     reward_hold_zero = env._calculate_reward(profit=0.0, action="HOLD")

    #     self.assertGreater(reward_hold_positive, 0, "HOLD with profit should yield > 0 reward")
    #     self.assertLess(reward_hold_negative, 0, "HOLD with loss should yield < 0 reward")
    #     self.assertEqual(reward_hold_zero, 0, "HOLD with 0 profit should yield 0 reward")

    #     # --- SELL cases ---
    #     reward_sell_positive = env._calculate_reward(profit=0.1, action="SELL")
    #     reward_sell_negative = env._calculate_reward(profit=-0.1, action="SELL")
    #     reward_sell_zero = env._calculate_reward(profit=0.0, action="SELL")

    #     self.assertAlmostEqual(reward_sell_positive, 0.01, places=4,
    #         msg="SELL with 0.1 profit should return 0.1^2 = 0.01")
    #     self.assertAlmostEqual(reward_sell_negative, -0.01, places=4,
    #         msg="SELL with -0.1 profit should return -0.1^2 = -0.01")
    #     self.assertEqual(reward_sell_zero, 0, "SELL with 0 profit should return 0")


    # def test_get_buy_action_code(self):
    #     """
    #     Tests the _get_buy_action_code() method to ensure:
    #     - It returns the correct crypto code for valid action indices.
    #     - It raises a ValueError for invalid indices.
    #     """
    #     env = self.env  # Shortcut
    #     num_cryptos = env.num_cryptos

    #     # --- Step 1: Check valid actions ---
    #     for action in range(1, num_cryptos + 1):
    #         expected_code = env.crypto_codes[action - 1]
    #         result_code = env._get_buy_action_code(action)
    #         self.assertEqual(result_code, expected_code,
    #                         f"Action {action} returned '{result_code}' instead of expected '{expected_code}'")

    #     # --- Step 2: Check invalid actions ---
    #     with self.assertRaises(ValueError):
    #         env._get_buy_action_code(0)  # Below valid range

    #     with self.assertRaises(ValueError):
    #         env._get_buy_action_code(num_cryptos + 1)  # Above valid range

    def test_step_function_with_logging(self):
        """
        Extended test for the step() function of LiveCryptoTradingEnv.
        Verifies:
        1. Correct behaviour for Buy → Hold → Sell workflow.
        2. Timeout termination behaviour.
        3. Correct structured logging for each step.
        """
        env = self.env
        env.reset()

        # --- Step 1: Buy ---
        buy_action = 1  # Buy first crypto
        obs, reward, terminated, truncated, info = env.step(buy_action)

        # Core behaviour checks
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertIsNotNone(env.active_crypto_index)
        self.assertEqual(env.active_crypto_code, env.crypto_codes[buy_action - 1])

        # Logging checks
        last_log = env.episode_log[-1]
        self.assertEqual(last_log["action_type"], "buy")
        self.assertEqual(last_log["symbol"], env.crypto_codes[buy_action - 1])
        self.assertIn("price", last_log)
        self.assertIsInstance(last_log["price"], float)

        # --- Step 2: Hold ---
        obs, reward, terminated, truncated, info = env.step(0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)

        last_log = env.episode_log[-1]
        self.assertEqual(last_log["action_type"], "hold")
        self.assertIn("symbol", last_log)
        self.assertIn("price", last_log)
        self.assertIn("profit_pct", last_log)

        # --- Step 3: Sell ---
        obs, reward, terminated, truncated, info = env.step(env.num_cryptos + 1)
        self.assertTrue(terminated)
        self.assertFalse(truncated)

        # After episode end, in-memory logs are flushed and cleared.
        # Verify by reading the JSONL at env.log_path and checking
        # for the most recent 'sell' step followed by 'episode_end'.
        import json
        sell_log = None
        summary_record = None

        with open(env.log_path, "r") as f:
            lines = f.readlines()

        # Walk backwards to find the last episode_end, then the sell prior to it.
        for i in range(len(lines) - 1, -1, -1):
            rec = json.loads(lines[i])
            if rec.get("type") == "episode_end":
                summary_record = rec
                # Search upward for the preceding sell step in the same episode window
                for j in range(i - 1, max(-1, i - 200), -1):  # limit search window
                    step_rec = json.loads(lines[j])
                    if step_rec.get("type") == "step" and step_rec.get("action_type") == "sell":
                        sell_log = step_rec
                        break
                break

        self.assertIsNotNone(summary_record, "No episode_end summary found in log file after SELL.")
        self.assertIsNotNone(sell_log, "SELL action not logged to file.")

        # Field checks on sell_log
        self.assertIn("symbol", sell_log)
        self.assertIn("profit_pct", sell_log)
        self.assertIn("reward", sell_log)
        # Optional fields if you included them in the sell log:
        # self.assertIn("buy_price", sell_log)
        # self.assertIn("sell_price", sell_log)

        # Basic checks on summary
        self.assertIn("episode_return_pct", summary_record)
        self.assertIn(summary_record["termination"], ["sell", "timeout"])

        # --- Step 4: Timeout (force on next step, no loops) ---
        env.reset()

        # Make tests fast & silent
        env.print_countdown = False
        env.skip_wait_on_terminate = True
        env.step_period_seconds = 0.0

        # Force timeout to trigger on the very next step:
        # The timeout check happens BEFORE current_step is incremented, so make it true now.
        # i.e., (current_step - start_index) >= timeout_steps  --> set start_index accordingly.
        env.start_index = env.current_step - env.timeout_steps

        # (Optional) If you want to test "timeout while holding", uncomment:
        # env.active_crypto_index = 0
        # env.active_crypto_code = env.crypto_codes[0]
        # env.buy_price = float(env.data[-1, env.active_crypto_index, 3])  # or any sane value

        # Single step triggers timeout path immediately
        obs, reward, terminated, truncated, info = env.step(0)

        self.assertTrue(truncated or terminated, "Expected timeout or termination at episode limit.")

        # Verify from file since in-memory logs are flushed/cleared on episode end
        import json
        summary_record = None
        with open(env.log_path, "r") as f:
            for line in reversed(f.readlines()):
                rec = json.loads(line)
                if rec.get("type") == "episode_end":
                    summary_record = rec
                    break

        self.assertIsNotNone(summary_record, "No episode_end summary found after forced timeout.")
        self.assertIn(summary_record["termination"], ["timeout", "sell"])

    # def test_sell_function(self):
    #     """
    #     Test the SELL functionality within the LiveCryptoTradingEnv class.

    #     This test specifically verifies that:
    #     1. The environment correctly handles a SELL action via the _sell() method.
    #     2. The _sell() method returns a valid average sell price (not None).
    #     3. The returned price is a float and greater than zero.

    #     Steps:
    #     - Manually set an active crypto code to simulate a held position.
    #     - Call the _sell() method directly.
    #     - Assert that the returned price is valid.
    #     - Print a confirmation message on success.
    #     """
    #     env = self.env

    #     # Manually set a crypto code to simulate that the agent is already holding a position.
    #     env.active_crypto_code = "1INCHUSDT"

    #     # Attempt to sell the currently held crypto.
    #     sell_price = env._sell()

    #     # Check that a valid sell price was returned.
    #     self.assertIsNotNone(sell_price, "Sell operation failed: no price returned.")
    #     self.assertIsInstance(sell_price, float, "Sell price should be a float.")
    #     self.assertGreater(sell_price, 0.0, "Sell price should be positive.")

    #     # Print confirmation message for debugging purposes.
    #     print(f"✅ Successfully sold {env.active_crypto_code} for {sell_price}")

    # def test_buy_function(self):
    #     """
    #     Test the BUY functionality via the internal _buy() method.

    #     This test specifically verifies that:
    #     1. Calling _buy(symbol) returns a valid executed average buy price (not None).
    #     2. The returned price is a float and strictly positive.

    #     Notes:
    #     - This test uses the live trader backend configured in the environment.
    #     - It assumes there is sufficient USDT balance and that the symbol is tradable.
    #     - We call _buy() directly (not through env.step) to isolate the method itself.
    #     """
    #     env = self.env

    #     # Choose a tradable symbol to buy. We set this explicitly for clarity,
    #     # though _buy() only relies on the symbol passed as an argument.
    #     env.active_crypto_code = "1INCHUSDT"

    #     # Execute a live market BUY for the specified symbol.
    #     # The _buy() method should block until exchange confirms the order
    #     # and then return the confirmed average fill price as a float.
    #     buy_price = env._buy(env.active_crypto_code)

    #     # Validate that we received a price back from the exchange.
    #     self.assertIsNotNone(buy_price, "Buy operation failed: no price returned.")
    #     self.assertIsInstance(buy_price, float, "Buy price should be a float.")
    #     self.assertGreater(buy_price, 0.0, "Buy price should be positive.")

    #     # Optional: If your _buy() does NOT set env.buy_price internally,
    #     # you may want to mirror production step() behavior here for later tests:
    #     # env.buy_price = buy_price

    #     # Print confirmation to aid debugging when running tests verbosely.
    #     print(f"✅ Successfully bought {env.active_crypto_code} for {buy_price}")

if __name__ == "__main__":
    unittest.main()
