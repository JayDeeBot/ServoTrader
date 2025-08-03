import pandas as pd
import os
from collections import defaultdict
from datetime import timedelta
from tqdm import tqdm
import time

# --- Config ---
INPUT_PATH = "/home/jarred/git/ServoTrader/data/modern_historical_crypto_data.csv"
OUTPUT_DIR = "/home/jarred/git/ServoTrader/data/split_10k_chunks_modern"
CHUNK_SIZE = 10_000
LINES_PER_SYMBOL = 1_000_000  # assumption

# Ensure output folder exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- First pass: Count how many lines per symbol ---
print("🔍 Counting lines per symbol...")

symbol_counts = defaultdict(int)
start_time = time.perf_counter()

for chunk in pd.read_csv(INPUT_PATH, chunksize=100_000):
    counts = chunk['symbol'].value_counts()
    for symbol, count in counts.items():
        symbol_counts[symbol] += count

total_symbols = len(symbol_counts)
total_lines = sum(symbol_counts.values())

print(f"✅ Found {total_symbols} symbols across {total_lines:,} lines\n")

# --- Second pass: Buffer and split ---
print("✂️ Splitting data into 10k chunks per symbol...\n")

# Prepare write buffers
symbol_buffers = defaultdict(list)
symbol_chunk_index = defaultdict(int)

# Track progress
symbol_progress = defaultdict(int)
symbol_chunk_targets = {sym: symbol_counts[sym] // CHUNK_SIZE for sym in symbol_counts}
completed_chunks = 0
total_target_chunks = sum(symbol_chunk_targets.values())

pbar = tqdm(total=total_target_chunks, desc="Progress", unit="chunk")

# Second pass: read and dispatch
for chunk in pd.read_csv(INPUT_PATH, chunksize=100_000, parse_dates=['timestamp']):
    for _, row in chunk.iterrows():
        symbol = row['symbol']
        symbol_buffers[symbol].append(row)
        if len(symbol_buffers[symbol]) == CHUNK_SIZE:
            index = symbol_chunk_index[symbol]
            filename = f"{index:03d}.csv"  # e.g. 000.csv, 001.csv ...
            path = os.path.join(OUTPUT_DIR, filename)

            df = pd.DataFrame(symbol_buffers[symbol])
            df.to_csv(path, mode='a', index=False, header=not os.path.exists(path))  # Append to file

            symbol_buffers[symbol].clear()
            symbol_chunk_index[symbol] += 1
            pbar.update(1)

# Write remaining buffers if any
for symbol, buffer in symbol_buffers.items():
    if buffer:
        index = symbol_chunk_index[symbol]
        filename = f"{index:03d}.csv"
        path = os.path.join(OUTPUT_DIR, filename)

        df = pd.DataFrame(buffer)
        df.to_csv(path, mode='a', index=False, header=not os.path.exists(path))
        pbar.update(1)

pbar.close()

end_time = time.perf_counter()
elapsed = timedelta(seconds=int(end_time - start_time))

print(f"\n✅ Done! Total chunks: {total_target_chunks}")
print(f"🕒 Time taken: {elapsed}")
print(f"📂 Output folder: {OUTPUT_DIR}")
