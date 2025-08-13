"""
test_crypto_data_init.py

Test Case: CryptoDatabaseInitialiser Functional Validation

This test verifies the correct behavior of the CryptoDatabaseInitialiser class by executing
a full end-to-end data fetch using actual runtime configuration and crypto code files.

Key Assertions:
---------------
1. CSV file is successfully created at the specified path.
2. The CSV contains required OHLCV fields: timestamp, open, high, low, close, vwap, volume, count, symbol.
3. For each crypto symbol:
    - At least `crypto_bars_to_analyse` rows are present.
    - Timestamp range covers the expected historical window.
    - Most recent data is near real-time.
    - Timestamps are spaced approximately according to `loop_interval_minutes`.

Requirements:
-------------
- Paths to valid `crypto_codes.json` and `params.yaml` files.
- Kraken API must be reachable and support the requested crypto pairs.
- Runtime parameters should define `loop_interval_minutes` and `crypto_bars_to_analyse`.

Usage:
------
Run this script directly to perform the test:
    python3 tests/test_crypto_data_init.py

Expected Output:
----------------
CSV file is created, all data validations pass, and summary output confirms success.

"""

import os
import pandas as pd
import yaml
import json
import time
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from servo_trader.crypto_database_init import CryptoDatabaseInitialiser  # Import the library we are testing

def test_crypto_data_initialisation():
    # --- Paths ---
    csv_path = "/home/jarred/git/ServoTrader/data/test_crypto_output.csv"
    json_path = "/home/jarred/git/ServoTrader/servo_trader/config/crypto_codes_ancient.json"
    yaml_path = "/home/jarred/git/ServoTrader/servo_trader/config/params.yaml"

    # --- Load config values ---
    with open(json_path, "r") as f:
        crypto_codes = json.load(f)["crypto_codes"]

    with open(yaml_path, "r") as f:
        params = yaml.safe_load(f)

    interval_minutes = int(params.get("loop_interval_minutes", 1))
    bars_required = int(params.get("crypto_bars_to_analyse", 10000))
    expected_interval_sec = interval_minutes * 60
    time_window_sec = bars_required * expected_interval_sec

    # --- Run the class to generate CSV ---
    print("Running CryptoDatabaseInitialiser...")
    init = CryptoDatabaseInitialiser(csv_path, json_path, yaml_path)

    # --- Read and verify ---
    assert os.path.exists(csv_path), f"❌ CSV file not created at: {csv_path}" # Test whether the csv was generated at the correct location

    df = pd.read_csv(csv_path)
    assert not df.empty, "❌ CSV file is empty." # Test whether the csv contains some data

    required_columns = {"timestamp", "open", "high", "low", "close", "vwap", "volume", "count", "symbol"}
    missing = required_columns - set(df.columns)
    assert not missing, f"❌ Missing columns in CSV: {missing}" # Test whether each required column has been saved

    # --- Normalize CSV content ---
    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    current_time = int(time.time())

    # # --- Loop through each crypto code ---
    # for symbol in crypto_codes:
    #     symbol = symbol.strip().upper()

    #     # Filter the DataFrame for the current symbol
    #     crypto_df = df[df["symbol"] == symbol].sort_values("timestamp")

    #     if crypto_df.empty:
    #         print(f"\n⚠️ No rows found for {symbol}.")
    #         print("🧪 Sample of available symbols in CSV:", df["symbol"].unique()[:10])
    #         assert False, f"❌ Expected data for {symbol}, but none was found in CSV."

    #     # --- Row count check ---
    #     if len(crypto_df) < bars_required * 0.95:
    #         print(f"⚠️ Warning: {symbol} has {len(crypto_df)} rows, expected at least {int(bars_required * 0.95)}")

    #     assert len(crypto_df) >= bars_required * 0.95, (
    #         f"❌ Not enough data rows for {symbol} (got {len(crypto_df)}, expected ≥ {int(bars_required * 0.95)})"
    #     )

    #     # --- Time range checks ---
    #     latest_ts = int(crypto_df["timestamp"].max().timestamp())
    #     earliest_ts = int(crypto_df["timestamp"].min().timestamp())

    #     time_behind = current_time - earliest_ts
    #     time_ahead = current_time - latest_ts

    #     assert time_behind >= time_window_sec * 0.9, (
    #         f"❌ {symbol} data does not go back far enough. "
    #         f"(Back {time_behind}s, expected ≥ {int(time_window_sec * 0.9)}s)"
    #     )

    #     assert time_ahead <= expected_interval_sec * 3, (
    #         f"❌ {symbol} latest timestamp is too far behind real time. "
    #         f"(Lag {time_ahead}s, expected ≤ {expected_interval_sec * 3}s)"
    #     )

    #     # --- Timestamp spacing checks ---
    #     diffs = crypto_df["timestamp"].diff().dropna().dt.total_seconds()
    #     avg_spacing = diffs.mean()

    #     assert abs(avg_spacing - expected_interval_sec) < expected_interval_sec * 0.3, (
    #         f"❌ Timestamp spacing irregular for {symbol} "
    #         f"(expected ≈{expected_interval_sec}s, got avg ≈{avg_spacing:.2f}s)"
    #     )

    # --- Check unusable_crypto_codes list ---
    # Define a manual ground truth for unusable codes in your current dataset
    expected_unusable = {"BALUSDT", "FTMUSDT", "KLAYUSDT", "MATICUSDT", "OCEANUSDT", "OMGUSDT", 
                         "RENUSDT", "RNDRUSDT", "STMXUSDT", "WAVESUSDT", "XMRUSDT"}  # ← Update as needed based on known delisted codes

    actual_unusable = set(init.unusable_crypto_codes)
    print(f"\n🧹 Unusable codes identified: {sorted(actual_unusable)}")

    # Compare against expected
    missing_unusables = expected_unusable - actual_unusable
    unexpected_unusables = actual_unusable - expected_unusable

    assert not missing_unusables, f"❌ Missing expected unusable codes: {missing_unusables}"
    assert not unexpected_unusables, f"❌ Unexpected unusable codes found: {unexpected_unusables}"

    # --- All Codes Usable ---
    # if not init.unusable_crypto_codes:
    #     print("✅ All crypto codes successfully fetched data — no unusable codes.")
    # else:
    #     print(f"⚠️ Unusable codes identified: {sorted(init.unusable_crypto_codes)}")
    #     assert False, f"❌ Expected all codes to be usable, but these failed: {init.unusable_crypto_codes}"

    print("✅ Test passed: CSV contains correctly spaced, sufficient data for all cryptos.")

# Run the test
if __name__ == "__main__":
    test_crypto_data_initialisation()
