import ccxt
import pandas as pd
import time
import json
import os
from datetime import datetime, timezone, timedelta

# --- CONFIG ---
BINANCE_API_KEY = "6BZOFxkzIau3dqljZu8tbbKY5tZxnptRJkOfHq6Nx5jZDbvogxseqFkaQ3RnuaBE"
BINANCE_SECRET = "6vguZphjUuW9t6SdPZUxImWNzB2anPr91jAWHw9dwIASLeFAVnbQQMZi0iZVBdru"
CODES_JSON_PATH = "/home/jarred/git/ServoTrader/servo_trader/config/crypto_code_sundries.json"
FULL_DATASET_CSV = "/home/jarred/git/ServoTrader/data/large_historical_dataset.csv"
INDIVIDUAL_FOLDER = "/home/jarred/git/ServoTrader/data/individual_crypto_csvs"
INTERVAL = '1m'
START_DATE = '2021-01-01T00:00:00Z'
MAX_CANDLES = 1_000_000
LIMIT = 1000
SLEEP_BETWEEN_REQUESTS = 0.5

# --- INIT BINANCE ---
binance = ccxt.binance({
    'apiKey': BINANCE_API_KEY,
    'secret': BINANCE_SECRET,
    'enableRateLimit': True,
})

# --- LOAD SYMBOLS ---
with open(CODES_JSON_PATH, 'r') as f:
    crypto_codes = json.load(f)["crypto_codes"]

# --- ENSURE FOLDER EXISTS ---
os.makedirs(INDIVIDUAL_FOLDER, exist_ok=True)

# --- MAIN LOOP ---
all_data = []
total_codes = len(crypto_codes)
start_time = time.time()

for i, symbol in enumerate(crypto_codes):
    print(f"\n🚀 [{i+1}/{total_codes}] Fetching {MAX_CANDLES} 1m candles for {symbol}...")
    since = binance.parse8601(START_DATE)
    collected = []

    while len(collected) < MAX_CANDLES:
        try:
            raw = binance.publicGetKlines({
                'symbol': symbol,
                'interval': INTERVAL,
                'startTime': since,
                'limit': LIMIT
            })

            if not raw:
                print("⚠️ No more data returned. Ending early.")
                break

            collected.extend(raw)
            since = int(raw[-1][0]) + 60_000
            last_time = datetime.fromtimestamp(int(raw[-1][0]) / 1000, tz=timezone.utc)
            print(f"✅ {symbol}: {len(collected):,} candles so far | Last timestamp: {last_time}")
            time.sleep(SLEEP_BETWEEN_REQUESTS)

        except Exception as e:
            print(f"❌ Error: {e}. Retrying in 5 seconds...")
            time.sleep(5)

    # Format dataframe
    df = pd.DataFrame(collected, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'count',
        'taker_base_vol', 'taker_quote_vol', 'ignore']
    )

    for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['vwap'] = df['quote_volume'] / df['volume']
    df['vwap'] = df['vwap'].replace([float('inf'), -float('inf')], pd.NA)
    df['timestamp'] = pd.to_datetime(df['timestamp'].astype('int64'), unit='ms', utc=True)
    df['symbol'] = symbol
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count', 'symbol']]

    # Save individual file
    file_path = os.path.join(INDIVIDUAL_FOLDER, f"{symbol}.csv")
    df.to_csv(file_path, index=False)
    print(f"📄 Saved {len(df):,} rows to {file_path}")

    # Add to full dataset
    all_data.append(df)

    # Progress reporting
    percent = ((i + 1) / total_codes) * 100
    elapsed = time.time() - start_time
    estimated_total_time = (elapsed / (i + 1)) * total_codes
    eta = timedelta(seconds=int(estimated_total_time - elapsed))
    print(f"📊 Progress: {percent:.2f}% complete | ETA: {eta}")

# --- SAVE FINAL COMBINED DATASET ---
print("\n📦 Concatenating all individual CSVs from disk...")

csv_paths = [os.path.join(INDIVIDUAL_FOLDER, f) for f in os.listdir(INDIVIDUAL_FOLDER) if f.endswith(".csv")]
dataframes = []
for path in csv_paths:
    try:
        df = pd.read_csv(path, parse_dates=['timestamp'])
        dataframes.append(df)
    except Exception as e:
        print(f"⚠️ Skipped {path} due to read error: {e}")

final_df = pd.concat(dataframes, ignore_index=True)
final_df.to_csv(FULL_DATASET_CSV, index=False)

print(f"\n💾 Done! Saved {len(final_df):,} total rows to {FULL_DATASET_CSV}")