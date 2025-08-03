"""
crypto_database_init.py

This module defines the CryptoDatabaseInitialiser class, which uses the Binance API
(via the ccxt library) to download historical OHLCV data for a list of cryptocurrency
symbols. The data is fetched using multithreading for speed, and stored in a standard
format to a CSV file for use in backtesting or training reinforcement learning agents.

Key Features:
- Multithreaded fetching of OHLCV data from Binance
- Configurable symbol list and runtime parameters via JSON and YAML
- Automatic retry on API errors with colored terminal output
- Final dataset is stored in a unified format across all symbols

Author: Jarred Deluca
Created: 2025
License: MIT
"""

import pandas as pd
import json
import yaml
import time
import os
import ccxt
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta

class CryptoDatabaseInitialiser:
    """
    Generates the necessary database for each step in the ServoTrader main.
    Sources runtime specific variables:
        - loop_interval_minutes
        - crypto_bars_to_analyse
    ... and generates the raw data csv accessed by the PPO agent using Binance API.

    Features:
        - Threaded data collection for fast batch downloading
        - Retry logic for fault tolerance in API calls
        - Configurable interval and data volume via YAML
        - Color-coded terminal feedback for errors and warnings

    Args:
        csv_path (str): Path to output CSV file for storing raw OHLCV data.
        json_path (str): Path to JSON file containing the list of crypto symbols.
        yaml_path (str): Path to YAML file specifying runtime parameters.
    """

    RED_COLOR = "\033[91m"
    RESET_COLOR = "\033[0m"
    MAX_RETRIES = 3
    RETRY_DELAY = 1
    THREAD_COUNT = 8
    LIMIT = 1000
    SLEEP_BETWEEN_REQUESTS = 0.5

    def __init__(self, csv_path, json_path, yaml_path):
        """Initialises the class, loads configs, and starts data fetching."""
        self.csv_path = csv_path
        self.crypto_codes = self.load_crypto_codes(json_path)
        self.params = self.load_params(yaml_path)
        self.binance = ccxt.binance({
            'apiKey': '6BZOFxkzIau3dqljZu8tbbKY5tZxnptRJkOfHq6Nx5jZDbvogxseqFkaQ3RnuaBE',  # Optional: Insert Binance API key
            'secret': '6vguZphjUuW9t6SdPZUxImWNzB2anPr91jAWHw9dwIASLeFAVnbQQMZi0iZVBdru',  # Optional: Insert Binance secret key
            'enableRateLimit': True,
        })
        self.create_csv()

    def print_error(self, message):
        """Prints an error message to the console in red."""
        print(f"{self.RED_COLOR}Error: {message}{self.RESET_COLOR}")

    def load_crypto_codes(self, json_path):
        """Loads crypto symbols from a JSON file."""
        try:
            with open(json_path, 'r') as file:
                data = json.load(file)
            crypto_codes = data.get('crypto_codes', [])
            if not crypto_codes:
                raise ValueError("No crypto codes found in JSON file.")
            return crypto_codes
        except Exception as e:
            self.print_error(f"Failed to load crypto codes: {e}")
            return []

    def load_params(self, yaml_path):
        """Loads runtime parameters from a YAML file."""
        try:
            with open(yaml_path, 'r') as file:
                params = yaml.safe_load(file)
            if not params:
                raise ValueError("No parameters found in YAML file.")
            return params
        except Exception as e:
            self.print_error(f"Failed to load parameters: {e}")
            return {}

    def fetch_crypto_data(self, symbol, interval, desired_lines):
        """
        Fetches OHLCV data for a single crypto symbol using Binance API.

        Args:
            symbol (str): The crypto pair (e.g., 'BTCUSDT')
            interval (int): Time interval in minutes (e.g., 1)
            desired_lines (int): Number of candles to fetch

        Returns:
            pd.DataFrame: DataFrame of OHLCV data in ServoTrader format
        """
        retries = 0
        full_data = []
        # Convert seconds to milliseconds for Binance API
        start_time = int(time.time()) - (interval * 60 * desired_lines)
        start_time *= 1000

        while len(full_data) < desired_lines and retries < self.MAX_RETRIES:
            try:
                raw = self.binance.public_get_klines({
                    'symbol': symbol,
                    'interval': self.binance.timeframes[str(interval) + 'm'],
                    'startTime': start_time,
                    'limit': self.LIMIT
                })

                if not raw:
                    print(f"{self.RED_COLOR}No data returned for {symbol}. Retrying.{self.RESET_COLOR}")
                    retries += 1
                    time.sleep(self.RETRY_DELAY)
                    continue

                full_data.extend(raw)
                start_time = raw[-1][0] + 60_000  # Advance start time by 1 candle
                time.sleep(self.SLEEP_BETWEEN_REQUESTS)

            except Exception as e:
                self.print_error(f"Error fetching data for {symbol} on attempt {retries + 1}: {e}")
                retries += 1
                time.sleep(self.RETRY_DELAY)

        # Create a DataFrame with meaningful column names
        df = pd.DataFrame(full_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'count',
            'taker_base_vol', 'taker_quote_vol', 'ignore']
        )

        # Convert to numeric types where relevant
        for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Compute VWAP and clean up infinite values
        df['vwap'] = df['quote_volume'] / df['volume']
        df['vwap'] = df['vwap'].replace([float('inf'), -float('inf')], pd.NA)

        # Format timestamps and tag with symbol
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype('int64'), unit='ms', utc=True)
        df['symbol'] = symbol

        # Return final formatted DataFrame
        return df[['timestamp', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count', 'symbol']].head(desired_lines)

    def create_csv(self):
        """
        Fetches data for all crypto codes using threads and saves to a single CSV.
        """
        interval = self.params.get("loop_interval_minutes", 1)
        desired_lines = self.params.get("crypto_bars_to_analyse", 10000)

        all_data = []
        total_assets = len(self.crypto_codes)
        completed_tasks = 0
        start_time = time.time()

        # Start a thread for each symbol
        with ThreadPoolExecutor(max_workers=self.THREAD_COUNT) as executor:
            futures = {
                executor.submit(self.fetch_crypto_data, symbol, interval, desired_lines): symbol
                for symbol in self.crypto_codes
            }

            # As threads complete, collect the results
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    crypto_data = future.result()
                    if crypto_data.empty:
                        print(f"{self.RED_COLOR}Warning: No data for {symbol}. Skipping.{self.RESET_COLOR}")
                        continue
                    all_data.append(crypto_data)
                except Exception as e:
                    self.print_error(f"Error processing {symbol}: {e}")

                # Progress tracking
                completed_tasks += 1
                elapsed_time = time.time() - start_time
                estimated_total_time = (elapsed_time / completed_tasks) * total_assets
                estimated_time_left = estimated_total_time - elapsed_time
                progress = (completed_tasks / total_assets) * 100
                print(f"\rProgress: {progress:.2f}% - Estimated time left: {estimated_time_left:.2f} seconds", end="")

        # Final write to CSV
        if all_data:
            final_df = pd.concat(all_data, ignore_index=True)
            final_df.to_csv(self.csv_path, index=False)
            print(f"\nData saved to {self.csv_path}")
        else:
            print(f"\n{self.RED_COLOR}No data retrieved. CSV not created.{self.RESET_COLOR}")
