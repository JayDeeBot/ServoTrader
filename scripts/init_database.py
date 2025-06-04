"""
Initialize historical cryptocurrency dataset using Kraken API.

This script fetches OHLCV data for all configured crypto codes and stores them
in a single CSV file for use in reinforcement learning environments.

Author: Jarred Deluca
License: MIT
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from servo_trader.crypto_database_init import CryptoDatabaseInitialiser

# --- Define paths ---
CSV_OUTPUT_PATH = "/home/jarred/git/ServoTrader/data/historical_crypto_data.csv"
CRYPTO_CODES_JSON = "/home/jarred/git/ServoTrader/servo_trader/config/crypto_codes.json"
PARAMS_YAML_PATH = "/home/jarred/git/ServoTrader/servo_trader/config/params.yaml"

# --- Create data output folder if needed ---
os.makedirs(os.path.dirname(CSV_OUTPUT_PATH), exist_ok=True)

# --- Initialize the database ---
if __name__ == "__main__":
    print("Starting historical data initialization...")
    db_init = CryptoDatabaseInitialiser(
        csv_path=CSV_OUTPUT_PATH,
        json_path=CRYPTO_CODES_JSON,
        yaml_path=PARAMS_YAML_PATH
    )