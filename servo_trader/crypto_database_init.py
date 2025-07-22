"""
    CryptoDatabaseInitialiser is responsible for creating a unified dataset of historical 
    cryptocurrency data by fetching OHLCV data from the Kraken public API. 

    This class loads the list of target cryptocurrencies and runtime parameters from 
    JSON and YAML configuration files, respectively. It uses multithreading to 
    efficiently download historical price data for multiple assets in parallel and 
    saves the compiled dataset to a CSV file for use in training or analysis.

    Features:
        - Threaded data collection for fast batch downloading
        - Retry logic for fault tolerance in API calls
        - Configurable interval and data volume via YAML
        - Color-coded terminal feedback for errors and warnings

    Args:
        csv_path (str): Path to output CSV file for storing raw OHLCV data.
        json_path (str): Path to JSON file containing the list of crypto symbols.
        yaml_path (str): Path to YAML file specifying runtime parameters.

    Example usage:
        init = CryptoDatabaseInitialiser(
            csv_path="data/crypto_raw.csv",
            json_path="config/crypto_codes.json",
            yaml_path="config/params.yaml"
        )
"""

# crypto_database_init.py

import pandas as pd
import json
import yaml
import time
import requests # Required for HTTP requests to Kraken
from concurrent.futures import ThreadPoolExecutor, as_completed # Required for multithreaded data downloading

class CryptoDatabaseInitialiser:
    """
    Generates the necessary database for each step in the ServoTrader main. 
    Sources runtime specific variables:
        - loop_interval_minutes
        - crypto_bars_to_analyse
    ... and generates the raw data csv accessed by the PPO agent. 
    """
    RED_COLOR = "\033[91m" #  Set red colour for error reporting
    RESET_COLOR = "\033[0m" 
    MAX_RETRIES = 3  # Maximum retries for each crypto asset
    RETRY_DELAY = 1  # Delay (in seconds) between retries
    THREAD_COUNT = 8  # Number of threads for concurrent data fetching
    KRAKEN_API_URL = "https://api.kraken.com/0/public/OHLC" # Set the Kraken URL for access cypto data

    def __init__(self, csv_path, json_path, yaml_path, start_offset_minutes=None):
        """
        Constructor inits all member variables, loads the cryto codes we are using, loads the runtime params and creates the raw data csv.

        Args:
            csv_path (str): Path to the raw data csv.
            json_path (str): Path to the json containing the crypto codes in play.
            yaml_path (str): Path to the yaml containing our runtime parameters.
        """
        self.csv_path = csv_path # Save the csv path to member
        self.crypto_codes = self.load_crypto_codes(json_path) # Load the crypto codes in play
        self.params = self.load_params(yaml_path) # Load the runtime params
        self.start_offset_minutes = start_offset_minutes # Save the start offset
        self.create_csv() # Create/update the raw crypto data csv

    def print_error(self, message):
        """
        Prints a runtime error message in red.

        Args:
            message (str): The message contained in the thrown exception.
        """
        print(f"{self.RED_COLOR}Error: {message}{self.RESET_COLOR}") # Print error

    def load_crypto_codes(self, json_path):
        """
        Loads the crypto codes in play and saves them to a member variable. 

        Args:
            json_path (str): Path to the json containing the crypto codes in play.

        Returns:
            list: List of all crypto codes in play.
        """
        try:
            with open(json_path, 'r') as file: # Open the json file containing the crypto codes
                data = json.load(file) # Load the json data
            crypto_codes = data.get('crypto_codes', []) # Grab all data under the 'crypto_codes' label
            if not crypto_codes: # If there are no codes on file then report the issue
                raise ValueError("No crypto codes found in JSON file.")
            return crypto_codes # Return the list of all loaded crypto codes
        except Exception as e:
            self.print_error(f"Failed to load crypto codes: {e}") # Print error
            return [] # If there is an error return nothing

    def load_params(self, yaml_path):
        """
        Loads the runtime parameters in play and saves them to a member variable. 

        Args:
            yaml_path (str): Path to the yaml containing our runtime parameters.

        Returns:
            dictionary: Dictionary of all parameters and values used during runtime. 
        """
        try:
            with open(yaml_path, 'r') as file: # Open the yaml containing the runtime params
                params = yaml.safe_load(file) # Load all the data in the yaml
            if not params: # If there are not params on file report the issue
                raise ValueError("No parameters found in YAML file.")
            return params # Return the params found
        except Exception as e:
            self.print_error(f"Failed to load parameters: {e}") # Report error
            return {} # If there is an error return an empty list

    def fetch_crypto_data(self, symbol, interval, desired_lines):
        """
        Fetches the OHLCV data of a given crypto over the provided interval with the desired amount of bars.

        Args:
            symbol (str): The symbol of the crypto we are fetching data for.
            interval (str): The time interval the data will range. 
            desired_lines (int): The total amount of bars to be retrieved within the interval.

        Returns:
            list: List containing all the OHLCV features for the crypto over the provided interval with the desired amount of bars.
        """
        retries = 0 # Init retries, set to zero to begin, this will store the amount of retries performed to fetch data for the given cryto
        full_data = pd.DataFrame() # Init the data structure to store all the data for the given crypto
        # Start 'since' based on start_offset_minutes
        if self.start_offset_minutes > 0:
            since = int(time.time()) - (self.start_offset_minutes * 60)
        else:
            # Fallback to default "look-back" from current time
            since = int(time.time()) - (interval * 60 * desired_lines)

        while len(full_data) < desired_lines and retries < self.MAX_RETRIES: # Loop ends when when we have all desired data and the retries are within limit
            try:
                params = { # Organise the parameters for the data we are fetching
                    "pair": symbol, # Set the symbol we want to download
                    "interval": interval, # Set the time interval we want data for
                    "since": since # Set the data in the past we want to begin retrieving from
                }
                response = requests.get(self.KRAKEN_API_URL, params=params) # Send a request to Kraken for the data and save the response
                data = response.json() # Save the json-encoded content of a response

                if "error" in data and data["error"]: # If there is an error in the data saved handle gracefully
                    print(f"{self.RED_COLOR}Error fetching {symbol}: {data['error']}{self.RESET_COLOR}")
                    retries += 1 # Increment the retries
                    time.sleep(self.RETRY_DELAY) # Take a short sleep between retries
                    continue # Exit the loop here if there is an error in data

                ohlc_data = data["result"].get(symbol, []) # Extract the OHLCVC data from the json-encoded response for the given symbol
                if not ohlc_data: # If there is no data for this symbol report an error and try again (same retry logic as above)
                    print(f"{self.RED_COLOR}No data returned for {symbol}. Retrying.{self.RESET_COLOR}")
                    retries += 1
                    time.sleep(self.RETRY_DELAY)
                    continue

                df = pd.DataFrame(ohlc_data, columns=[ # Save the data to a labelled dictionary
                    "timestamp", "open", "high", "low", "close", "vwap", "volume", "count"
                ])
                full_data = pd.concat([full_data, df], ignore_index=True) # Concatenate full_data & df

                # Update `since` to fetch older data in the next iteration
                if not df.empty: # If the data retrieval was successful
                    since = int(df["timestamp"].min()) - (interval * 60) # Update since relative to the most recent timestamp in the retrieved data

            except Exception as e:
                self.print_error(f"Error fetching data for {symbol} on attempt {retries + 1}: {e}") # Report error
                retries += 1 # Increment the retries
                time.sleep(self.RETRY_DELAY) # Take a short sleep before retrying

        return full_data.head(desired_lines) # Once successful return the full data retrieved

    def create_csv(self):
        """
        Fetches historical OHLCV data for all crypto codes in parallel and saves to CSV.

        Uses a thread pool to download data concurrently from Kraken. Each crypto is fetched
        according to the loop interval and desired bar count defined in the runtime YAML.

        The final combined dataset is written to `self.csv_path`.

        Raises:
            No exceptions are raised directly — all errors are caught and printed.
        """
        interval = self.params.get("loop_interval_minutes", 1)  # Default to 1-minute candles
        desired_lines = self.params.get("crypto_bars_to_analyse", 10000)  # Default to 10000 lines

        all_data = [] # Init variable to store the data for all cryptos in play
        total_assets = len(self.crypto_codes) # Compute the total amount of cryptos in play
        completed_tasks = 0 # Init variable to keep track of how many cryptos we have successfully retrieved the data for
        start_time = time.time() # Set a start time for ETA tracking

        with ThreadPoolExecutor(max_workers=self.THREAD_COUNT) as executor: # Setup a thread pool with a maximum of self.THREAD_COUNT threads
            futures = { # Submit one thread per crypto code to fetch its data
                executor.submit(self.fetch_crypto_data, symbol, interval, desired_lines): symbol # Returns a Future object that represents the asynchronous execution of the function fetch_crypto_data for the given args
                for symbol in self.crypto_codes # A dictionary is built: {future: symbol} to keep track of which future corresponds to which symbol
            }

            for future in as_completed(futures): # Yield futures in the order they finish this lets us process results immediately as they return
                symbol = futures[future] # Save the current symbol
                try:
                    crypto_data = future.result() # Returns the result of the thread
                    if crypto_data.empty: # If the data is empty handle gracefully
                        print(f"{self.RED_COLOR}Warning: No data for {symbol}. Skipping.{self.RESET_COLOR}")
                        continue # Break loop and try again for this symbol

                    crypto_data["symbol"] = symbol # Label the current data under the current symbol
                    all_data.append(crypto_data) # Add the data to all data at the end
                except Exception as e:
                    self.print_error(f"Error processing {symbol}: {e}") # Report any errors

                completed_tasks += 1 # Increment up for each completed task
                # Compute the ETC, percentage completed and report
                elapsed_time = time.time() - start_time
                estimated_total_time = (elapsed_time / completed_tasks) * total_assets
                estimated_time_left = estimated_total_time - elapsed_time
                progress = (completed_tasks / total_assets) * 100
                print(f"\rProgress: {progress:.2f}% - Estimated time left: {estimated_time_left:.2f} seconds", end="")

        if all_data: # If all the data has been fetched concatenate it all and save it to our csv
            final_df = pd.concat(all_data, ignore_index=True) # Concatenate all data
            final_df.to_csv(self.csv_path, index=False) # Save to csv
            print(f"\nData saved to {self.csv_path}")
        else: # If there is no data found then report the issue
            print(f"\n{self.RED_COLOR}No data retrieved. CSV not created.{self.RESET_COLOR}")