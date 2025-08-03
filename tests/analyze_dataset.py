import pandas as pd
from collections import defaultdict

# Config
CSV_PATH = "/home/jarred/git/ServoTrader/data/latest_100k_dataset.csv"
CHUNK_SIZE = 100_000  # Adjust based on your system (50k–500k is usually safe)

def analyze_large_csv(csv_path):
    print(f"📂 Reading large CSV in chunks from: {csv_path}")
    symbol_counts = defaultdict(int)
    total_rows = 0
    chunks_read = 0

    try:
        for chunk in pd.read_csv(csv_path, chunksize=CHUNK_SIZE):
            if "symbol" not in chunk.columns:
                print("❌ Missing 'symbol' column.")
                return

            chunk_counts = chunk["symbol"].value_counts()
            for symbol, count in chunk_counts.items():
                symbol_counts[symbol] += count

            total_rows += len(chunk)
            chunks_read += 1
            print(f"✅ Processed chunk {chunks_read}: {len(chunk)} rows (Total so far: {total_rows:,})")

    except Exception as e:
        print(f"❌ Error while reading: {e}")
        return

    print(f"\n📈 Finished processing {total_rows:,} rows across {chunks_read} chunks.")
    print(f"✅ Found {len(symbol_counts)} unique crypto codes:\n")

    for symbol, count in sorted(symbol_counts.items(), key=lambda x: -x[1]):
        print(f"  {symbol}: {count:,} datapoints")

if __name__ == "__main__":
    analyze_large_csv(CSV_PATH)
