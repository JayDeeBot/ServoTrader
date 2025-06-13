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
    json_path = "/home/jarred/git/ServoTrader/servo_trader/config/crypto_codes.json"
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

    # Convert timestamp to numeric
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df.dropna(subset=["timestamp"], inplace=True)

    current_time = int(time.time())

    # Loop through all crypto codes, test whether there is enough data for each and whether the time range is accurate 
    for symbol in crypto_codes:
        crypto_df = df[df["symbol"] == symbol].sort_values("timestamp")
        assert len(crypto_df) >= bars_required, f"❌ Not enough data rows for {symbol} (got {len(crypto_df)}, expected ≥ {bars_required})" # Test whether enough data has been retrieved for each crypto

        # Check time coverage is recent and complete
        latest = int(crypto_df["timestamp"].max())
        earliest = int(crypto_df["timestamp"].min())

        time_behind = current_time - earliest
        time_ahead = current_time - latest

        assert time_behind >= time_window_sec * 0.9, ( # Test whether we go back the correct amount of time into the past
            f"❌ {symbol} data does not go back far enough. "
            f"(Back {time_behind}s, expected ≥ {int(time_window_sec * 0.9)}s)"
        )

        assert time_ahead <= expected_interval_sec * 3, ( # Test whether the data goes up to the current time (approximately)
            f"❌ {symbol} latest timestamp is too far behind real time. "
            f"(Lag {time_ahead}s, expected ≤ {expected_interval_sec * 3}s)"
        )

        # Check spacing between timestamps
        diffs = crypto_df["timestamp"].diff().dropna().astype(int)
        avg_spacing = diffs.mean()
        assert abs(avg_spacing - expected_interval_sec) < expected_interval_sec * 0.3, ( # Test the time stamp spacing
            f"❌ Timestamp spacing irregular for {symbol} (expected ≈{expected_interval_sec}s, got avg ≈{avg_spacing:.2f}s)"
        )

    print("✅ Test passed: CSV contains correctly spaced, sufficient data for all cryptos.")

# Run the test
if __name__ == "__main__":
    test_crypto_data_initialisation()
