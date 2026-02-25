#!/usr/bin/env python3
"""
fetch_hourly_historical_data.py

Fetch extended historical hourly OHLCV data from Binance for prediction model training.

This script downloads 1-hour candles going back ~3 years (Binance limit for hourly data)
to provide sufficient training data for hourly prediction models.

Key Features:
- Fetches 1-hour (1h) candles
- Goes back ~3 years for hourly data
- Supports single symbol or multiple symbols from JSON
- Saves individual CSV files per symbol
- Optionally saves combined dataset

Usage:
    # Fetch single symbol (e.g., BTCUSDT)
    python fetch_hourly_historical_data.py --symbol BTCUSDT
    
    # Fetch multiple symbols from JSON file
    python fetch_hourly_historical_data.py --codes-json /path/to/crypto_codes.json
    
    # Specify output directory
    python fetch_hourly_historical_data.py --symbol BTCUSDT --output-dir /path/to/output

Author: Jarred Deluca
Project: ServoTrader
License: MIT
"""

import ccxt
import pandas as pd
import time
import json
import os
import argparse
from datetime import datetime, timezone, timedelta


# =============================================================================
# Configuration
# =============================================================================

# Binance API credentials (public endpoints don't require auth)
BINANCE_API_KEY = ""  # Optional
BINANCE_SECRET = ""   # Optional

# Default settings
DEFAULT_OUTPUT_DIR = "/home/jarred/git/ServoTrader/data/hourly_historical"
DEFAULT_INTERVAL = '1h'  # Hourly candles
# Binance typically keeps ~3 years of hourly data
DEFAULT_START_DATE = (datetime.now(timezone.utc) - timedelta(days=1095)).strftime('%Y-%m-%dT00:00:00Z')
MAX_CANDLES = 50000  # 3 years = ~26,280 hours
LIMIT = 1000  # Max candles per request
SLEEP_BETWEEN_REQUESTS = 0.3  # Seconds between API calls


# =============================================================================
# Main Functions
# =============================================================================

def init_binance():
    """Initialize Binance exchange connection."""
    config = {'enableRateLimit': True}
    
    if BINANCE_API_KEY and BINANCE_SECRET:
        config['apiKey'] = BINANCE_API_KEY
        config['secret'] = BINANCE_SECRET
    
    return ccxt.binance(config)


def fetch_symbol_data(binance, symbol: str, interval: str, start_date: str) -> pd.DataFrame:
    """
    Fetch historical OHLCV data for a single symbol.
    
    Args:
        binance: ccxt Binance exchange instance
        symbol: Trading pair symbol (e.g., 'BTCUSDT')
        interval: Candle interval (e.g., '1h')
        start_date: Start date in ISO format
        
    Returns:
        DataFrame with OHLCV data
    """
    print(f"\n🚀 Fetching {interval} candles for {symbol} from {start_date}...")
    
    since = binance.parse8601(start_date)
    collected = []
    
    while len(collected) < MAX_CANDLES:
        try:
            raw = binance.publicGetKlines({
                'symbol': symbol,
                'interval': interval,
                'startTime': since,
                'limit': LIMIT
            })
            
            if not raw:
                print(f"   ✅ No more data available. Total: {len(collected):,} candles")
                break
            
            collected.extend(raw)
            
            # Update since to fetch next batch
            last_timestamp = int(raw[-1][0])
            since = last_timestamp + get_interval_ms(interval)
            
            # Check if we've reached current time
            last_time = datetime.fromtimestamp(last_timestamp / 1000, tz=timezone.utc)
            now = datetime.now(timezone.utc)
            
            if last_time >= now - timedelta(hours=2):
                print(f"   ✅ Reached current time. Total: {len(collected):,} candles")
                break
            
            # Progress update every 5000 candles
            if len(collected) % 5000 == 0:
                print(f"   📊 {symbol}: {len(collected):,} candles | Last: {last_time.strftime('%Y-%m-%d %H:00')}")
            
            time.sleep(SLEEP_BETWEEN_REQUESTS)
            
        except Exception as e:
            print(f"   ❌ Error: {e}. Retrying in 5 seconds...")
            time.sleep(5)
    
    # Convert to DataFrame
    df = pd.DataFrame(collected, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'count',
        'taker_base_vol', 'taker_quote_vol', 'ignore'
    ])
    
    # Convert types
    for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Calculate VWAP
    df['vwap'] = df['quote_volume'] / df['volume']
    df['vwap'] = df['vwap'].replace([float('inf'), -float('inf')], pd.NA)
    
    # Convert timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'].astype('int64'), unit='ms', utc=True)
    
    # Add symbol column
    df['symbol'] = symbol
    
    # Select final columns
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count', 'symbol']]
    
    # Remove duplicates (if any)
    df = df.drop_duplicates(subset=['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    return df


def get_interval_ms(interval: str) -> int:
    """Convert interval string to milliseconds."""
    multipliers = {
        'm': 60 * 1000,
        'h': 60 * 60 * 1000,
        'd': 24 * 60 * 60 * 1000,
        'w': 7 * 24 * 60 * 60 * 1000,
    }
    
    unit = interval[-1]
    value = int(interval[:-1])
    
    return value * multipliers.get(unit, 60 * 1000)


def load_symbols_from_json(json_path: str) -> list:
    """Load symbol list from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Handle different JSON formats
    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        for key in ['crypto_codes', 'codes', 'symbols', 'tickers']:
            if key in data:
                return data[key]
    
    raise ValueError(f"Could not find symbol list in {json_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Fetch historical hourly OHLCV data from Binance'
    )
    parser.add_argument(
        '--symbol', type=str, default=None,
        help='Single symbol to fetch (e.g., BTCUSDT)'
    )
    parser.add_argument(
        '--codes-json', type=str, default=None,
        help='Path to JSON file with symbol list'
    )
    parser.add_argument(
        '--output-dir', type=str, default=DEFAULT_OUTPUT_DIR,
        help=f'Output directory for CSV files (default: {DEFAULT_OUTPUT_DIR})'
    )
    parser.add_argument(
        '--interval', type=str, default=DEFAULT_INTERVAL,
        help=f'Candle interval (default: {DEFAULT_INTERVAL})'
    )
    parser.add_argument(
        '--start-date', type=str, default=DEFAULT_START_DATE,
        help=f'Start date in ISO format (default: ~3 years ago)'
    )
    parser.add_argument(
        '--combined-csv', type=str, default=None,
        help='Path to save combined dataset (optional)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.symbol and not args.codes_json:
        print("❌ Error: Must specify either --symbol or --codes-json")
        print("\nExamples:")
        print("  python fetch_hourly_historical_data.py --symbol BTCUSDT")
        print("  python fetch_hourly_historical_data.py --codes-json /path/to/codes.json")
        return
    
    # Get symbol list
    if args.symbol:
        symbols = [args.symbol]
    else:
        symbols = load_symbols_from_json(args.codes_json)
    
    print("=" * 60)
    print("🚀 ServoTrader Hourly Historical Data Fetcher")
    print("=" * 60)
    print(f"   Symbols: {len(symbols)}")
    print(f"   Interval: {args.interval}")
    print(f"   Start date: {args.start_date}")
    print(f"   Output dir: {args.output_dir}")
    print(f"\n   ⚠️  Note: Binance typically keeps ~3 years of hourly data")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize Binance
    binance = init_binance()
    
    # Fetch data for each symbol
    all_data = []
    total_symbols = len(symbols)
    start_time = time.time()
    
    for i, symbol in enumerate(symbols):
        try:
            df = fetch_symbol_data(binance, symbol, args.interval, args.start_date)
            
            # Save individual CSV
            file_path = os.path.join(args.output_dir, f"{symbol}.csv")
            df.to_csv(file_path, index=False)
            print(f"   💾 Saved {len(df):,} rows to {file_path}")
            
            # Add to combined data
            all_data.append(df)
            
            # Progress reporting
            elapsed = time.time() - start_time
            percent = ((i + 1) / total_symbols) * 100
            
            if i > 0:
                avg_time = elapsed / (i + 1)
                remaining = avg_time * (total_symbols - i - 1)
                eta = timedelta(seconds=int(remaining))
                print(f"   📊 Progress: {percent:.1f}% | ETA: {eta}")
            
        except Exception as e:
            print(f"   ❌ Failed to fetch {symbol}: {e}")
            continue
    
    # Save combined dataset if requested
    if args.combined_csv and all_data:
        print(f"\n💾 Saving combined dataset to {args.combined_csv}...")
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df.to_csv(args.combined_csv, index=False)
        print(f"   ✅ Saved {len(combined_df):,} total rows")
    
    # Summary
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("✅ Fetch Complete!")
    print("=" * 60)
    print(f"   Symbols processed: {len(all_data)}/{total_symbols}")
    print(f"   Total time: {timedelta(seconds=int(total_time))}")
    print(f"   Output directory: {args.output_dir}")
    
    if all_data:
        total_rows = sum(len(df) for df in all_data)
        print(f"   Total candles: {total_rows:,}")
        
        # Show date range for first symbol
        sample_df = all_data[0]
        print(f"\n   Sample ({symbols[0]}):")
        print(f"     Date range: {sample_df['timestamp'].min()} to {sample_df['timestamp'].max()}")
        print(f"     Hourly candles: {len(sample_df):,}")
        print(f"     Days of data: {len(sample_df) / 24:.1f}")


if __name__ == '__main__':
    main()