import pandas as pd
import json
import yaml
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

class CryptoDatabaseInitialiser:
    RED_COLOR = "\033[91m"
    RESET_COLOR = "\033[0m"
    MAX_RETRIES = 3  # Maximum retries for each crypto asset
    RETRY_DELAY = 1  # Delay (in seconds) between retries
    THREAD_COUNT = 8  # Number of threads for concurrent data fetching
    KRAKEN_API_URL = "https://api.kraken.com/0/public/OHLC"

    def __init__(self, csv_path, json_path, yaml_path):
        self.csv_path = csv_path
        self.crypto_codes = self.load_crypto_codes(json_path)
        self.params = self.load_params(yaml_path)
        self.create_csv()

    def print_error(self, message):
        print(f"{self.RED_COLOR}Error: {message}{self.RESET_COLOR}")

    def load_crypto_codes(self, json_path):
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
        retries = 0
        full_data = pd.DataFrame()
        since = int(time.time()) - (86400 * 7)  # Start from the last 7 days

        while len(full_data) < desired_lines and retries < self.MAX_RETRIES:
            try:
                params = {
                    "pair": symbol,
                    "interval": interval,
                    "since": since
                }
                response = requests.get(self.KRAKEN_API_URL, params=params)
                data = response.json()

                if "error" in data and data["error"]:
                    print(f"{self.RED_COLOR}Error fetching {symbol}: {data['error']}{self.RESET_COLOR}")
                    retries += 1
                    time.sleep(self.RETRY_DELAY)
                    continue

                ohlc_data = data["result"].get(symbol, [])
                if not ohlc_data:
                    print(f"{self.RED_COLOR}No data returned for {symbol}. Retrying.{self.RESET_COLOR}")
                    retries += 1
                    time.sleep(self.RETRY_DELAY)
                    continue

                df = pd.DataFrame(ohlc_data, columns=[
                    "timestamp", "open", "high", "low", "close", "vwap", "volume", "count"
                ])
                full_data = pd.concat([full_data, df], ignore_index=True)

                # Update `since` to fetch older data in the next iteration
                if not df.empty:
                    since = int(df["timestamp"].min()) - (interval * 60)

            except Exception as e:
                self.print_error(f"Error fetching data for {symbol} on attempt {retries + 1}: {e}")
                retries += 1
                time.sleep(self.RETRY_DELAY)

        return full_data.head(desired_lines)

    def process_crypto(self, symbol, interval, desired_lines):
        return self.fetch_crypto_data(symbol, interval, desired_lines)

    def create_csv(self):
        interval = self.params.get("loop_interval_minutes", 60)  # Default to 1-hour candles
        desired_lines = self.params.get("desired_lines", 5000)  # Default to 5000 lines

        all_data = []
        total_assets = len(self.crypto_codes)
        completed_tasks = 0
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=self.THREAD_COUNT) as executor:
            futures = {
                executor.submit(self.process_crypto, symbol, interval, desired_lines): symbol
                for symbol in self.crypto_codes
            }

            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    crypto_data = future.result()
                    if crypto_data.empty:
                        print(f"{self.RED_COLOR}Warning: No data for {symbol}. Skipping.{self.RESET_COLOR}")
                        continue

                    crypto_data["symbol"] = symbol
                    all_data.append(crypto_data)
                except Exception as e:
                    self.print_error(f"Error processing {symbol}: {e}")

                completed_tasks += 1
                elapsed_time = time.time() - start_time
                estimated_total_time = (elapsed_time / completed_tasks) * total_assets
                estimated_time_left = estimated_total_time - elapsed_time
                progress = (completed_tasks / total_assets) * 100
                print(f"\rProgress: {progress:.2f}% - Estimated time left: {estimated_time_left:.2f} seconds", end="")

        if all_data:
            final_df = pd.concat(all_data, ignore_index=True)
            final_df.to_csv(self.csv_path, index=False)
            print(f"\nData saved to {self.csv_path}")
        else:
            print(f"\n{self.RED_COLOR}No data retrieved. CSV not created.{self.RESET_COLOR}")