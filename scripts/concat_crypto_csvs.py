import os
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
import time

# Config
FOLDER_PATH = "/home/jarred/git/ServoTrader/data/individual_crypto_csvs_ancient_2"
OUTPUT_FILE = "/home/jarred/git/ServoTrader/data/ancient_historical_crypto_data_2.csv"
NUM_WORKERS = 8

def read_csv_stream(file_path):
    symbol = os.path.basename(file_path).replace(".csv", "")
    try:
        df = pd.read_csv(file_path)
        df["symbol"] = symbol
        return df
    except Exception as e:
        return f"ERROR: {symbol}: {e}"

def main():
    csv_files = [os.path.join(FOLDER_PATH, f) for f in os.listdir(FOLDER_PATH) if f.endswith(".csv")]
    total_files = len(csv_files)
    print(f"🔢 Found {total_files} CSV files.")

    start_time = time.time()
    first = True
    expected_columns = None
    error_count = 0

    with Pool(NUM_WORKERS) as pool, open(OUTPUT_FILE, "w") as out_file:
        with tqdm(total=total_files, desc="🔄 Writing to CSV", unit="file") as pbar:
            for result in pool.imap(read_csv_stream, csv_files):
                if isinstance(result, str) and result.startswith("ERROR"):
                    print(f"⚠️ {result}")
                    error_count += 1
                    pbar.update(1)
                    continue

                # Schema check
                if expected_columns is None:
                    expected_columns = list(result.columns)
                elif list(result.columns) != expected_columns:
                    print(f"❌ Column mismatch in {result['symbol'].iloc[0]}. Skipping.")
                    error_count += 1
                    pbar.update(1)
                    continue

                # Stream write
                result.to_csv(out_file, index=False, header=first, mode='a')
                first = False
                pbar.update(1)

    print(f"\n✅ Combined CSV saved to: {OUTPUT_FILE}")
    print(f"🕒 Time elapsed: {time.time() - start_time:.2f} seconds")
    print(f"⚠️ Files skipped due to errors or schema mismatch: {error_count}")

if __name__ == "__main__":
    main()
