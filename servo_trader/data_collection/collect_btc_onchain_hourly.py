#!/usr/bin/env python3
"""
collect_btc_onchain_hourly.py

Collects BTC on-chain metrics and resamples to hourly frequency for hourly prediction models.

**IMPORTANT LIMITATION**: Most on-chain data is only available at DAILY granularity.
This script collects daily on-chain metrics and forward-fills them to hourly frequency.

This means:
- Each hour within a day will have the SAME on-chain values
- On-chain features will only update once per day (at midnight UTC)
- For true hourly on-chain data, premium APIs like Glassnode are required

For hourly predictions, the primary signal will come from OHLCV technical indicators.
On-chain data serves as supplementary daily context.

Author: Jarred Deluca
Created: 2025
License: MIT
"""

import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_CSV = "/home/jarred/git/ServoTrader/data/btc_onchain_hourly_metrics.csv"
START_DATE = "2020-01-01"  # ~3 years to match hourly OHLCV availability
END_DATE   = datetime.now().strftime("%Y-%m-%d")

# Rate limits
RATE_BLOCKCHAIN = 1.0
RATE_GLASSNODE = 1.0

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def section(title):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")


def safe_get(url, params=None, headers=None, timeout=30, retries=3, delay=2.0):
    """HTTP GET with retry logic."""
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            if r.status_code == 200:
                return r
            elif r.status_code == 429:
                wait = delay * (attempt + 1) * 3
                print(f"    ⏳ Rate limited. Waiting {wait:.0f}s...")
                time.sleep(wait)
            else:
                print(f"    ⚠️  HTTP {r.status_code} on attempt {attempt+1}")
                time.sleep(delay)
        except Exception as e:
            print(f"    ❌ Request error (attempt {attempt+1}): {e}")
            time.sleep(delay * (attempt + 1))
    return None


# ============================================================================
# COLLECT DAILY ON-CHAIN DATA (Same as daily script)
# ============================================================================

def collect_blockchain_data():
    """Collect daily on-chain data from Blockchain.com API."""
    
    section("COLLECTING DAILY ON-CHAIN DATA FROM BLOCKCHAIN.COM")
    
    print("  ⚠️  Note: On-chain data is DAILY only - will be resampled to hourly")
    
    charts = {
        'market-price': 'price_usd',
        'total-bitcoins': 'circulating_supply',
        'n-transactions': 'transaction_count',
        'n-unique-addresses': 'unique_addresses',
        'estimated-transaction-volume': 'estimated_volume_btc',
        'output-volume': 'output_volume_btc',
        'hash-rate': 'hash_rate_ths',
        'difficulty': 'difficulty',
        'miners-revenue': 'miners_revenue_usd',
        'blocks-size': 'total_blocks_size_mb',
        'avg-block-size': 'avg_block_size_bytes',
        'n-transactions-per-block': 'tx_per_block',
        'total-fees': 'total_fees_btc',
        'transaction-fees-usd': 'total_fees_usd',
        'cost-per-transaction': 'cost_per_tx_usd',
        'cost-per-transaction-percent': 'cost_per_tx_percent',
        'mempool-size': 'mempool_size_bytes',
        'mempool-count': 'mempool_tx_count',
        'median-confirmation-time': 'median_confirm_time_min',
        'utxo-count': 'utxo_count',
    }
    
    all_data = {}
    
    for chart_name, col_name in charts.items():
        print(f"  📊 Fetching {chart_name}...")
        
        url = f"https://api.blockchain.info/charts/{chart_name}"
        params = {
            'timespan': '3years',
            'format': 'json',
            'sampled': 'false'
        }
        
        r = safe_get(url, params=params)
        
        if r:
            try:
                data = r.json()
                values = data.get('values', [])
                
                if values:
                    df = pd.DataFrame(values)
                    df['date'] = pd.to_datetime(df['x'], unit='s').dt.date
                    df = df.rename(columns={'y': col_name})
                    df = df[['date', col_name]]
                    
                    # Handle high-frequency data
                    if len(df) > 10000:
                        print(f"    ⚠️  High-frequency data detected ({len(df)} records)")
                        df = df.groupby('date')[col_name].mean().reset_index()
                        print(f"    ✅ Resampled to daily: {len(df)} records")
                    
                    all_data[col_name] = df
                    print(f"    ✅ {len(df)} daily records")
                else:
                    print(f"    ⚠️  No data returned")
            
            except Exception as e:
                print(f"    ❌ Error parsing: {e}")
        
        time.sleep(RATE_BLOCKCHAIN)
    
    if not all_data:
        print("  ❌ No data collected")
        return pd.DataFrame()
    
    # Merge all charts
    df = list(all_data.values())[0]
    for dataset in list(all_data.values())[1:]:
        df = df.merge(dataset, on='date', how='outer')
    
    df = df.sort_values('date').reset_index(drop=True)
    
    print(f"\n  ✅ Collected {len(df)} days × {len(df.columns)-1} metrics")
    
    return df


def engineer_onchain_features(df):
    """Engineer features from raw on-chain data."""
    
    section("ENGINEERING ON-CHAIN FEATURES")
    
    # Hash ribbon
    if 'hash_rate_ths' in df.columns:
        df['hash_rate_ma30'] = df['hash_rate_ths'].rolling(30).mean()
        df['hash_rate_ma60'] = df['hash_rate_ths'].rolling(60).mean()
        df['hash_ribbon'] = df['hash_rate_ma30'] - df['hash_rate_ma60']
        df['hash_rate_pct_change'] = df['hash_rate_ths'].pct_change()
    
    # Network growth
    if 'unique_addresses' in df.columns:
        df['unique_addresses_ma7'] = df['unique_addresses'].rolling(7).mean()
        df['unique_addresses_pct_change'] = df['unique_addresses'].pct_change()
    
    if 'transaction_count' in df.columns:
        df['transaction_count_ma7'] = df['transaction_count'].rolling(7).mean()
        df['transaction_count_pct_change'] = df['transaction_count'].pct_change()
    
    # Volume trends
    if 'estimated_volume_btc' in df.columns:
        df['volume_btc_ma7'] = df['estimated_volume_btc'].rolling(7).mean()
        df['volume_btc_pct_change'] = df['estimated_volume_btc'].pct_change()
    
    # Fee metrics
    if 'total_fees_btc' in df.columns and 'transaction_count' in df.columns:
        df['avg_fee_per_tx_usd'] = df['total_fees_usd'] / (df['transaction_count'] + 1e-10)
    
    # Velocity
    if 'estimated_volume_btc' in df.columns and 'circulating_supply' in df.columns:
        df['velocity'] = df['estimated_volume_btc'] / (df['circulating_supply'] + 1e-10)
    
    # Mining profitability
    if 'miners_revenue_usd' in df.columns and 'hash_rate_ths' in df.columns:
        df['revenue_per_hash'] = df['miners_revenue_usd'] / (df['hash_rate_ths'] + 1e-10)
    
    # Mempool congestion
    if 'mempool_tx_count' in df.columns and 'tx_per_block' in df.columns:
        df['mempool_to_block_ratio'] = df['mempool_tx_count'] / (df['tx_per_block'] + 1e-10)
    
    # UTXO growth
    if 'utxo_count' in df.columns:
        df['utxo_count_pct_change'] = df['utxo_count'].pct_change()
    
    # Difficulty adjustment
    if 'difficulty' in df.columns:
        df['difficulty_pct_change'] = df['difficulty'].pct_change()
    
    # Transaction density
    if 'tx_per_block' in df.columns:
        df['tx_per_block_ma7'] = df['tx_per_block'].rolling(7).mean()
    
    print(f"  ✅ Engineered features: {len(df.columns)} total columns")
    
    return df


# ============================================================================
# RESAMPLE TO HOURLY
# ============================================================================

def resample_to_hourly(daily_df):
    """Resample daily on-chain data to hourly by forward-filling."""
    
    section("RESAMPLING TO HOURLY FREQUENCY")
    
    print("  ⚠️  IMPORTANT: On-chain metrics are DAILY aggregates")
    print("     Each hour within a day will have the SAME values")
    print("     Values update once per day at midnight UTC")
    
    # Convert date to datetime
    daily_df['date'] = pd.to_datetime(daily_df['date'])
    daily_df = daily_df.set_index('date')
    
    # Create hourly index
    start = daily_df.index.min()
    end = daily_df.index.max() + timedelta(days=1)
    hourly_index = pd.date_range(start=start, end=end, freq='1H')
    
    # Reindex to hourly and forward-fill
    hourly_df = daily_df.reindex(hourly_index, method='ffill')
    hourly_df.index.name = 'timestamp'
    hourly_df = hourly_df.reset_index()
    
    print(f"\n  ✅ Resampled from {len(daily_df)} days to {len(hourly_df)} hours")
    print(f"     Ratio: {len(hourly_df) / len(daily_df):.1f}x (should be ~24)")
    
    return hourly_df


# ============================================================================
# MAIN
# ============================================================================

def main():
    section("BTC HOURLY ON-CHAIN DATA COLLECTOR")
    
    print(f"  ⚠️  CRITICAL LIMITATION:")
    print(f"     On-chain data is DAILY only (no hourly granularity available)")
    print(f"     This script forward-fills daily values to hourly frequency")
    print(f"     For hourly predictions, rely primarily on OHLCV technical indicators")
    
    print(f"\n  Date range: {START_DATE} → {END_DATE}")
    print(f"  Output: {OUTPUT_CSV}")
    
    # Collect daily data
    daily_df = collect_blockchain_data()
    
    if daily_df.empty:
        print("\n❌ No data collected. Exiting.")
        return
    
    # Engineer features on daily data
    daily_df = engineer_onchain_features(daily_df)
    
    # Resample to hourly
    hourly_df = resample_to_hourly(daily_df)
    
    # Save
    section("SAVING OUTPUT")
    hourly_df.to_csv(OUTPUT_CSV, index=False)
    print(f"  ✅ Saved to: {OUTPUT_CSV}")
    print(f"     Rows: {len(hourly_df):,} hours")
    print(f"     Cols: {len(hourly_df.columns)}")
    print(f"     Date range: {hourly_df['timestamp'].min()} → {hourly_df['timestamp'].max()}")
    
    # Feature completeness
    print(f"\n  📊 Feature Completeness:")
    missing = hourly_df.isnull().sum()
    missing_pct = (missing / len(hourly_df)) * 100
    high_missing = missing_pct[missing_pct > 10].sort_values(ascending=False)
    
    if len(high_missing) > 0:
        print(f"     Features with >10% missing:")
        for col in high_missing.index[:10]:
            print(f"       {col:45s} {missing_pct[col]:5.1f}%")
    else:
        print(f"     ✅ All features have <10% missing data")
    
    section("✅ COLLECTION COMPLETE")
    
    print(f"\n  ℹ️  Usage Notes:")
    print(f"     1. On-chain features update once per day (midnight UTC)")
    print(f"     2. For intraday predictions, these serve as daily context")
    print(f"     3. Primary hourly signal comes from OHLCV technical indicators")
    print(f"     4. For true hourly on-chain data, premium APIs required")
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()