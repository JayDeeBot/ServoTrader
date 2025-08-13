#!/usr/bin/env python3
"""
test_servo_trader_binance.py

Unit test suite for the ServoTraderBinance class.

This test case verifies the correctness of:
1. Setting maximum buy limits.
2. Setting daily cashout percentages.
3. Executing the cash_out_to_bank logic.
4. Identifying untradeable symbols.
5. Finding viable replacement codes.

Mocks are used for Binance client methods to avoid real API calls.

Author: Jarred Deluca
Created: 2025
License: MIT
"""

import sys
import os
import unittest
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from servo_trader.servo_trader_binance import ServoTraderBinance
import json
import time

class TestServoTraderBinance(unittest.TestCase):
    """
    Integration test for checking real trading logic using live crypto_codes.json.
    """

    def setUp(self):
        """
        Instantiates the ServoTraderBinance object using real credentials and config.
        This version allows real Binance queries to test symbol status.
        """
        import yaml
        with open("/home/jarred/git/ServoTrader/servo_trader/config/params.yaml", "r") as f:
            params = yaml.safe_load(f)

        # Create trader (will connect using API keys from params.yaml)
        self.trader = ServoTraderBinance()

    def test_set_max_buy_limit(self):
        """
        Test that set_max_buy_limit correctly assigns a valid limit,
        and rejects invalid values (≤ 0).
        """
        # Valid value
        self.trader.set_max_buy_limit(500)
        self.assertEqual(self.trader.max_buy_limit, 500)

        # Invalid value (zero)
        self.trader.set_max_buy_limit(0)
        self.assertEqual(self.trader.max_buy_limit, 500)  # Should remain unchanged

        # Invalid value (negative)
        self.trader.set_max_buy_limit(-100)
        self.assertEqual(self.trader.max_buy_limit, 500)  # Should remain unchanged

    def test_set_daily_cashout_percent(self):
        """
        Test that set_daily_cashout_percent correctly updates internal state
        and enforces 0 < percent ≤ 100.
        """
        # Valid value
        self.trader.set_daily_cashout_percent(25)
        self.assertEqual(self.trader.daily_cashout_percent, 25)

        # Invalid values
        self.trader.set_daily_cashout_percent(0)
        self.assertEqual(self.trader.daily_cashout_percent, 25)  # No change

        self.trader.set_daily_cashout_percent(150)
        self.assertEqual(self.trader.daily_cashout_percent, 25)  # No change

    # def test_cash_out_to_bank_with_percent(self):
    #     """
    #     Test the cash_out_to_bank logic when daily_cashout_percent is set
    #     and cash is available. Verifies transfer_to_bank is called with correct amount.
    #     """
    #     self.trader.set_max_buy_limit(1000)         # Cap
    #     self.trader.set_daily_cashout_percent(10)   # 10% cashout
    #     self.trader.get_cash_balance = MagicMock(return_value=1500)
    #     self.trader.transfer_to_bank = MagicMock()

    #     self.trader.cash_out_to_bank()

    #     expected_cashout = max(150, 500)  # 10% = 150, excess = 500
    #     self.trader.transfer_to_bank.assert_called_with(expected_cashout)

    # def test_cash_out_to_bank_with_zero_percent(self):
    #     """
    #     Test the cash_out_to_bank logic when cashout_percent is zero,
    #     so only excess over max limit is withdrawn.
    #     """
    #     self.trader.set_max_buy_limit(1200)
    #     self.trader.set_daily_cashout_percent(0)  # explicitly set to 0
    #     self.trader.get_cash_balance = MagicMock(return_value=1500)
    #     self.trader.transfer_to_bank = MagicMock()

    #     self.trader.cash_out_to_bank()
    #     self.trader.transfer_to_bank.assert_called_with(300)  # 1500 - 1200

    # def test_find_untradeable_codes(self):
    #     """
    #     Test that find_untradeable_codes correctly identifies any untradeable codes
    #     from the actual crypto_codes.json configuration used by the system.

    #     This test will:
    #     - Load the real JSON file containing configured crypto symbols.
    #     - Time the execution of find_untradeable_codes().
    #     - Print results and runtime.
    #     - Assert that the result is a valid list.
    #     """

    #     # Load codes from the actual config path
    #     json_path = "/home/jarred/git/ServoTrader/servo_trader/config/crypto_codes.json"
    #     with open(json_path, "r") as f:
    #         crypto_codes = json.load(f)["crypto_codes"]

    #     print(f"\n🔍 Checking {len(crypto_codes)} crypto codes for tradeability...")

    #     # --- Time the function ---
    #     start_time = time.perf_counter()
    #     untradeable = self.trader.find_untradeable_codes(crypto_codes)
    #     end_time = time.perf_counter()
    #     elapsed = end_time - start_time

    #     # --- Print Results ---
    #     print(f"\n⏱️ Runtime: {elapsed:.2f} seconds")
    #     print(f"\n❌ Untradeable codes found ({len(untradeable)}):")
    #     for code in untradeable:
    #         print(f" - {code}")

    #     # --- Assertions ---
    #     self.assertIsInstance(untradeable, list)
    #     # self.assertEqual(len(untradeable), 0, "Some symbols in your list are not tradeable.")

    # def test_invalid_symbols(self):
    #     """
    #     Test that find_untradeable_codes correctly flags known invalid or fake crypto symbols.

    #     This test will:
    #     - Pass in a small list of clearly invalid crypto codes.
    #     - Verify that all of them are reported as untradeable.
    #     - Time the operation and print results.
    #     """

    #     # --- Step 1: Define intentionally invalid symbols ---
    #     delisted_symbols = [
    #         "BCHSVUSDT",     # Bitcoin SV (delisted)
    #         "BCCUSDT",       # Bitcoin Cash Classic (old fork)
    #         "BTCDOWNUSDT",   # Leveraged token
    #         "USDSUSDT",      # Paxos USD Stablecoin (defunct)
    #         "SUSDUSDT",      # sUSD (delisted synthetic)
    #         "YOYOUSDT",      # Very old token
    #         "VENUSDT",       # VeChain before VEN->VET swap
    #         "HSRUSDT",       # Hshare (now HyperCash)
    #         "BUSDUSDT",      # Being phased out by Binance
    #     ]

    #     print(f"\n🔍 Testing detection of {len(delisted_symbols)} fake/untradeable symbols...")

    #     # --- Step 2: Run the test with timing ---
    #     start_time = time.perf_counter()
    #     untradeable = self.trader.find_untradeable_codes(delisted_symbols)
    #     end_time = time.perf_counter()
    #     elapsed = end_time - start_time

    #     # --- Step 3: Print results ---
    #     print(f"\n⏱️ Runtime: {elapsed:.2f} seconds")
    #     print(f"\n❌ Detected untradeable mock symbols ({len(untradeable)}):")
    #     for code in untradeable:
    #         print(f" - {code}")

    #     # --- Step 4: Assert all are detected as untradeable ---
    #     self.assertEqual(set(untradeable), set(delisted_symbols),
    #                     "Not all delisted/break symbols were detected as untradeable.")

    # def test_find_viable_replacement_codes(self):
    #     """
    #     Test that find_viable_replacement_codes:
    #     - Returns a replacement symbol for each delisted one.
    #     - Each replacement is actually tradeable on Binance.

    #     This test simulates a real replacement scenario where:
    #     - A list of known delisted symbols is passed as `replace_codes`.
    #     - A dummy list of existing codes is passed as `current_codes`.
    #     - The function is expected to return new, tradeable symbols.
    #     - The tradeability of each returned symbol is then verified.

    #     The test will fail if:
    #     - The number of replacements is less than expected.
    #     - Any of the replacement symbols are not tradeable.
    #     """

    #     # --- Step 1: Define known delisted symbols ---
    #     delisted_symbols = [
    #         "BCHSVUSDT", "BCCUSDT", "BTCDOWNUSDT", "USDSUSDT", "SUSDUSDT",
    #         "YOYOUSDT", "VENUSDT", "HSRUSDT", "BUSDUSDT"
    #     ]

    #     dummy_current_codes = ["BTCUSDT", "ETHUSDT", "XRPUSDT"]  # Pretend these are already in use

    #     print(f"\n🔁 Finding replacements for {len(delisted_symbols)} delisted symbols...")

    #     # --- Step 2: Run replacement search ---
    #     start_time = time.perf_counter()
    #     replacement_codes = self.trader.find_viable_replacement_codes(delisted_symbols, dummy_current_codes)
    #     end_time = time.perf_counter()
    #     elapsed = end_time - start_time

    #     print(f"\n⏱️ Replacement selection completed in {elapsed:.2f} seconds")
    #     print(f"✅ Suggested replacements: {replacement_codes}")

    #     # --- Step 3: Assert expected number of replacements ---
    #     self.assertEqual(len(replacement_codes), len(delisted_symbols),
    #                     "Did not receive the expected number of replacement codes.")

    #     # --- Step 4: Validate replacements are all tradeable ---
    #     print(f"\n🔍 Validating tradeability of suggested replacements...")
    #     untradeable = self.trader.find_untradeable_codes(replacement_codes)

    #     print(f"\n❌ Untradeable replacements found ({len(untradeable)}):")
    #     for code in untradeable:
    #         print(f" - {code}")

    #     # --- Step 5: Final assertion ---
    #     self.assertEqual(len(untradeable), 0,
    #                     "One or more replacement symbols were not tradeable on Binance.")

if __name__ == "__main__":
    unittest.main()
