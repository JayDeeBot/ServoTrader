#!/usr/bin/env python3
"""
crypto_database_init.py

A multi-threaded crypto dataset initializer for Binance OHLCV data.

This script downloads the most recent minute-level OHLCV (Open, High, Low, Close, Volume)
data for a list of crypto symbols specified in a JSON file. The number of candles to retrieve
is controlled via a YAML configuration file. Data is fetched using the `ccxt` Binance API wrapper
and stored in a unified CSV for downstream use (e.g., reinforcement learning environments).

Key Features:
- Multi-threaded data collection using ThreadPoolExecutor (default: 8 workers)
- VWAP and count values included in the final dataset
- ETA and progress tracking for large symbol lists
- Handles API retries and data gaps gracefully
- Returns the most recent N minutes of data (configurable via YAML)

Intended for use with:
- Real-time or recent history crypto trading environments
- Feature engineering and ML-based strategies
- Environments requiring consistent, structured OHLCV input data

Author: Jarred Deluca  
Created: 2025  
License: MIT  
"""

import os
import json
import yaml
import time
import ccxt
import pandas as pd
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

class CryptoDatabaseInitialiser:
    """
    A class to initialize a live crypto dataset by downloading the most recent OHLCV data
    for multiple symbols from Binance, using multi-threading for fast collection.

    Features:
    - Reads crypto symbols from JSON
    - Reads parameters (e.g. number of candles to collect) from YAML
    - Uses ThreadPoolExecutor to speed up concurrent downloads
    - Appends results to a final CSV file
    - Supports VWAP calculation and proper feature ordering
    """

    RED_COLOR = "\033[91m"
    RESET_COLOR = "\033[0m"
    THREAD_COUNT = 8
    LIMIT = 1000
    SLEEP_BETWEEN_REQUESTS = 0.5
    INTERVAL = '1m'

    def __init__(self, csv_path, json_path, yaml_path):
        """
        Initializes the class and starts the CSV creation process.

        Args:
            csv_path (str): Output CSV file path.
            json_path (str): Path to crypto_codes.json file.
            yaml_path (str): Path to params.yaml file.
        """
        self.csv_path = csv_path
        self.crypto_codes = self.load_crypto_codes(json_path)
        self.params = self.load_params(yaml_path)
        self.max_candles = self.params.get("crypto_bars_to_analyse", 10000)
        self.binance = ccxt.binance({
            'apiKey': '',
            'secret': '',
            'enableRateLimit': True
        })
        self.create_csv()

    def print_error(self, message):
        """
        Prints an error message in red for visibility.

        Args:
            message (str): Error message to print.
        """
        print(f"{self.RED_COLOR}Error: {message}{self.RESET_COLOR}")

    def load_crypto_codes(self, json_path):
        """
        Loads crypto symbols from the provided JSON file.

        Args:
            json_path (str): Path to the JSON file.

        Returns:
            list[str]: List of crypto symbol strings.
        """
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            return data.get("crypto_codes", [])
        except Exception as e:
            self.print_error(f"Failed to load crypto codes: {e}")
            return []

    def load_params(self, yaml_path):
        """
        Loads parameters from the provided YAML file.

        Args:
            yaml_path (str): Path to the YAML file.

        Returns:
            dict: Dictionary of configuration parameters.
        """
        try:
            with open(yaml_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.print_error(f"Failed to load params: {e}")
            return {}

    def fetch_crypto_data(self, symbol):
        """
        Downloads the most recent OHLCV candles for a given crypto symbol.

        Args:
            symbol (str): The crypto trading pair symbol (e.g., BTCUSDT).

        Returns:
            pd.DataFrame | None: DataFrame with processed OHLCV rows, or None on failure.
        """
        collected = []
        candles_needed = self.max_candles
        end_time = None  # Start from the most recent and go backward

        while len(collected) < candles_needed:
            fetch_limit = min(self.LIMIT, candles_needed - len(collected))
            try:
                params = {
                    'symbol': symbol,
                    'interval': self.INTERVAL,
                    'limit': fetch_limit
                }
                if end_time:
                    params['endTime'] = end_time

                # Fetch data
                raw = self.binance.publicGetKlines(params)
                if not raw:
                    break

                collected = raw + collected  # Prepend new rows to maintain chronological order
                end_time = int(raw[0][0]) - 60_000  # Go backward 1 minute
                time.sleep(self.SLEEP_BETWEEN_REQUESTS)

            except Exception as e:
                self.print_error(f"{symbol} fetch error: {e}")
                time.sleep(5)

        if not collected:
            return None

        # Format and clean up
        df = pd.DataFrame(collected, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'count',
            'taker_base_vol', 'taker_quote_vol', 'ignore'
        ])
        for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Calculate VWAP and format
        df['vwap'] = df['quote_volume'] / df['volume']
        df['vwap'] = df['vwap'].replace([float('inf'), -float('inf')], pd.NA)
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype('int64'), unit='ms', utc=True)
        df['symbol'] = symbol

        # Return final structured DataFrame
        return df[['timestamp', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count', 'symbol']]

    def create_csv(self):
        """
        Coordinates parallel downloads of all crypto symbols and writes the result to a CSV.
        """
        all_data = []
        total = len(self.crypto_codes)
        print(f"\n📡 Downloading {self.max_candles} bars for {total} symbols...\n")

        start_time = time.time()
        completed = 0

        # Use multithreading to download data concurrently
        with ThreadPoolExecutor(max_workers=self.THREAD_COUNT) as executor:
            futures = {
                executor.submit(self.fetch_crypto_data, symbol): symbol
                for symbol in self.crypto_codes
            }

            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    df = future.result()
                    if df is not None and not df.empty:
                        all_data.append(df)
                    else:
                        print(f"⚠️ {symbol}: No data fetched.")
                except Exception as e:
                    self.print_error(f"Error processing {symbol}: {e}")

                # Progress bar and ETA
                completed += 1
                elapsed = time.time() - start_time
                pct = (completed / total) * 100
                eta = (elapsed / completed) * (total - completed) if completed > 0 else 0
                print(f"\r🔁 {completed}/{total} symbols | {pct:.1f}% complete | ETA: {timedelta(seconds=int(eta))}", end="")

        print("\n\n💾 Finalizing CSV export...")
        if all_data:
            final_df = pd.concat(all_data, ignore_index=True)
            final_df.to_csv(self.csv_path, index=False)
            print(f"✅ Saved {len(final_df):,} rows to {self.csv_path}")
        else:
            self.print_error("No data retrieved. CSV not created.")