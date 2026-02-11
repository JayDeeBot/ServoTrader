#!/usr/bin/env python3
"""
collect_btc_onchain_daily_v2.py

Collects daily on-chain metrics for Bitcoin from multiple free data sources.
Uses Blockchain.com as primary source since CoinMetrics Community API is now restricted.

Author: Jarred Deluca
Created: 2025
License: MIT
"""

import requests
import pandas as pd
import time
from datetime import datetime, timedelta
import os
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_CSV = "/home/jarred/git/ServoTrader/data/btc_onchain_daily_metrics.csv"
START_DATE = "2017-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")

# API Rate Limits
BLOCKCHAIN_RATE_LIMIT = 1.0  # 1 second between requests
GLASSNODE_RATE_LIMIT = 1.0   # 1 second between requests

# ============================================================================
# BLOCKCHAIN.COM API FUNCTIONS (PRIMARY SOURCE)
# ============================================================================

def get_blockchain_chart_data(chart_name, timespan="all", rolling_average=None):
    """
    Fetches historical chart data from Blockchain.com.
    
    Available charts:
    - market-price: Bitcoin price USD
    - total-bitcoins: Total BTC in circulation
    - n-transactions: Daily transaction count
    - n-transactions-total: Cumulative transactions
    - hash-rate: Network hash rate (TH/s)
    - difficulty: Mining difficulty
    - miners-revenue: Miner revenue USD
    - avg-block-size: Average block size (bytes)
    - n-transactions-per-block: Transactions per block
    - median-confirmation-time: Median confirmation time (min)
    - mempool-size: Mempool size (bytes)
    - mempool-count: Mempool transaction count
    - n-unique-addresses: Unique addresses used
    - estimated-transaction-volume: Estimated transaction volume BTC
    - output-volume: Total output volume BTC
    - utxo-count: Total UTXO count
    - transaction-fees: Total transaction fees BTC
    - transaction-fees-usd: Total transaction fees USD
    - cost-per-transaction: Cost per transaction USD
    - cost-per-transaction-percent: Cost as % of transaction value
    - blocks-size: Total blocks size
    """
    
    url = f"https://api.blockchain.info/charts/{chart_name}"
    params = {
        "timespan": timespan,
        "format": "json",
        "sampled": "false"
    }
    
    if rolling_average:
        params["rollingAverage"] = rolling_average
    
    print(f"  🔍 Fetching: {chart_name}...")
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if "values" in data:
            df = pd.DataFrame(data["values"])
            df["date"] = pd.to_datetime(df["x"], unit="s").dt.date
            df = df.rename(columns={"y": chart_name})
            
            # For high-frequency data (like mempool), resample to daily
            if len(df) > 10000:  # Likely sub-daily data
                print(f"    ⚠️ High-frequency data detected ({len(df)} records), resampling to daily...")
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date")
                df = df.resample("D").mean()  # Daily mean
                df = df.reset_index()
                df["date"] = df["date"].dt.date
            
            df = df[["date", chart_name]]
            print(f"    ✅ Retrieved {len(df)} daily records")
            return df
        else:
            print(f"    ⚠️ No data in response")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"    ❌ Error: {e}")
        return None


def get_all_blockchain_data():
    """
    Fetches comprehensive historical data from Blockchain.com charts.
    """
    
    print("\n📊 STEP 1: Fetching Blockchain.com data...")
    print("=" * 80)
    
    charts_to_fetch = {
        # Price & Market Data
        "market-price": "price_usd",
        "total-bitcoins": "circulating_supply",
        
        # Network Activity
        "n-transactions": "transaction_count",
        "n-unique-addresses": "unique_addresses",
        "estimated-transaction-volume": "estimated_volume_btc",
        "output-volume": "output_volume_btc",
        
        # Mining & Security
        "hash-rate": "hash_rate_ths",
        "difficulty": "difficulty",
        "miners-revenue": "miners_revenue_usd",
        "blocks-size": "total_blocks_size_mb",
        
        # Block Metrics
        "avg-block-size": "avg_block_size_bytes",
        "n-transactions-per-block": "tx_per_block",
        
        # Fee Metrics
        "transaction-fees": "total_fees_btc",
        "transaction-fees-usd": "total_fees_usd",
        "cost-per-transaction": "cost_per_tx_usd",
        "cost-per-transaction-percent": "cost_per_tx_percent",
        
        # Mempool
        "mempool-size": "mempool_size_bytes",
        "mempool-count": "mempool_tx_count",
        "median-confirmation-time": "median_confirm_time_min",
        
        # UTXO
        "utxo-count": "utxo_count",
    }
    
    dfs = []
    
    for chart_name, column_name in charts_to_fetch.items():
        df = get_blockchain_chart_data(chart_name, timespan="all")
        if df is not None:
            df = df.rename(columns={chart_name: column_name})
            dfs.append(df)
        time.sleep(BLOCKCHAIN_RATE_LIMIT)
    
    if not dfs:
        print("  ❌ No data retrieved from Blockchain.com")
        return pd.DataFrame()
    
    # Merge all dataframes on date
    print(f"\n  🔗 Merging {len(dfs)} datasets...")
    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = pd.merge(merged_df, df, on="date", how="outer")
    
    print(f"  ✅ Blockchain.com: {len(merged_df)} days, {len(merged_df.columns)-1} metrics")
    return merged_df


# ============================================================================
# GLASSNODE FREE API
# ============================================================================

def get_glassnode_metric(metric_name, asset="BTC"):
    """
    Fetches free metrics from Glassnode.
    
    Free metrics (no API key needed):
    - addresses/active_count
    - blockchain/utxo_count
    - market/price_usd_close
    - blockchain/block_count
    - mining/hash_rate_mean
    - mining/difficulty_latest
    """
    
    url = f"https://api.glassnode.com/v1/metrics/{metric_name}"
    params = {
        "a": asset,
        "i": "24h",  # Daily
        "s": "2017-01-01",
        "u": int(time.time())
    }
    
    print(f"  🔍 Fetching Glassnode: {metric_name}...")
    
    try:
        response = requests.get(url, params=params, timeout=30)
        
        # Check if we got a valid response
        if response.status_code == 401:
            print(f"    ⚠️ API key required for {metric_name}")
            return None
        
        response.raise_for_status()
        data = response.json()
        
        if data:
            df = pd.DataFrame(data)
            df["date"] = pd.to_datetime(df["t"], unit="s").dt.date
            metric_col = metric_name.replace("/", "_")
            df = df.rename(columns={"v": metric_col})
            df = df[["date", metric_col]]
            print(f"    ✅ Retrieved {len(df)} records")
            return df
        else:
            print(f"    ⚠️ No data returned")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"    ⚠️ Skipping (requires API key or unavailable)")
        return None


def get_all_glassnode_data():
    """
    Attempts to fetch free Glassnode metrics (most require API key).
    """
    
    print("\n📊 STEP 2: Attempting Glassnode free metrics...")
    print("=" * 80)
    print("  ℹ️ Most Glassnode metrics require paid API key, attempting free ones...")
    
    free_metrics = [
        "addresses/active_count",
        "blockchain/utxo_count", 
        "market/price_usd_close",
    ]
    
    dfs = []
    
    for metric in free_metrics:
        df = get_glassnode_metric(metric)
        if df is not None:
            dfs.append(df)
        time.sleep(GLASSNODE_RATE_LIMIT)
    
    if not dfs:
        print("  ⚠️ No free Glassnode metrics available")
        return pd.DataFrame()
    
    # Merge
    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = pd.merge(merged_df, df, on="date", how="outer")
    
    print(f"  ✅ Glassnode: {len(merged_df)} days, {len(merged_df.columns)-1} metrics")
    return merged_df


# ============================================================================
# ALTERNATIVE.ME FEAR & GREED INDEX
# ============================================================================

def get_fear_greed_index():
    """
    Fetches historical Fear & Greed Index data.
    """
    
    url = "https://api.alternative.me/fng/?limit=0"
    
    print("\n📊 STEP 3: Fetching Fear & Greed Index...")
    print("=" * 80)
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if "data" in data:
            df = pd.DataFrame(data["data"])
            df["date"] = pd.to_datetime(df["timestamp"], unit="s").dt.date
            df = df.rename(columns={"value": "fear_greed_index"})
            df["fear_greed_index"] = pd.to_numeric(df["fear_greed_index"], errors="coerce")
            df = df[["date", "fear_greed_index"]]
            print(f"  ✅ Fear & Greed: {len(df)} days")
            return df
        else:
            print("  ⚠️ No data in response")
            return pd.DataFrame()
            
    except requests.exceptions.RequestException as e:
        print(f"  ❌ Error: {e}")
        return pd.DataFrame()


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def engineer_features(df):
    """
    Engineers additional features from the collected data.
    """
    
    print("\n📊 STEP 5: Engineering features...")
    print("=" * 80)
    
    # Ensure data is sorted by date
    df = df.sort_values("date").reset_index(drop=True)
    
    # --- Hash Rate Features ---
    if "hash_rate_ths" in df.columns:
        df["hash_rate_ma30"] = df["hash_rate_ths"].rolling(window=30).mean()
        df["hash_rate_ma60"] = df["hash_rate_ths"].rolling(window=60).mean()
        df["hash_ribbon"] = df["hash_rate_ma30"] - df["hash_rate_ma60"]
        df["hash_rate_pct_change"] = df["hash_rate_ths"].pct_change() * 100
        print("  ✅ Hash rate features (MA30, MA60, ribbon, % change)")
    
    # --- Address Activity Features ---
    if "unique_addresses" in df.columns:
        df["unique_addresses_ma7"] = df["unique_addresses"].rolling(window=7).mean()
        df["unique_addresses_pct_change"] = df["unique_addresses"].pct_change() * 100
        print("  ✅ Address activity features")
    
    # --- Transaction Features ---
    if "transaction_count" in df.columns:
        df["transaction_count_ma7"] = df["transaction_count"].rolling(window=7).mean()
        df["transaction_count_pct_change"] = df["transaction_count"].pct_change() * 100
        print("  ✅ Transaction count features")
    
    # --- Volume Features ---
    if "estimated_volume_btc" in df.columns:
        df["volume_btc_ma7"] = df["estimated_volume_btc"].rolling(window=7).mean()
        df["volume_btc_pct_change"] = df["estimated_volume_btc"].pct_change() * 100
        print("  ✅ Volume features")
    
    # --- Fee Features ---
    if "total_fees_usd" in df.columns and "transaction_count" in df.columns:
        df["avg_fee_per_tx_usd"] = df["total_fees_usd"] / df["transaction_count"].replace(0, np.nan)
        print("  ✅ Fee per transaction")
    
    # --- Network Velocity ---
    if "estimated_volume_btc" in df.columns and "circulating_supply" in df.columns:
        df["velocity"] = df["estimated_volume_btc"] / df["circulating_supply"].replace(0, np.nan)
        print("  ✅ Network velocity")
    
    # --- Mining Profitability ---
    if "miners_revenue_usd" in df.columns and "hash_rate_ths" in df.columns:
        df["revenue_per_hash"] = df["miners_revenue_usd"] / df["hash_rate_ths"].replace(0, np.nan)
        print("  ✅ Revenue per hash")
    
    # --- Mempool Congestion ---
    if "mempool_size_bytes" in df.columns and "avg_block_size_bytes" in df.columns:
        df["mempool_to_block_ratio"] = df["mempool_size_bytes"] / df["avg_block_size_bytes"].replace(0, np.nan)
        print("  ✅ Mempool congestion ratio")
    
    # --- UTXO Features ---
    if "utxo_count" in df.columns:
        df["utxo_count_pct_change"] = df["utxo_count"].pct_change() * 100
        print("  ✅ UTXO growth rate")
    
    # --- Price Features ---
    if "price_usd" in df.columns:
        df["price_ma7"] = df["price_usd"].rolling(window=7).mean()
        df["price_ma30"] = df["price_usd"].rolling(window=30).mean()
        df["price_pct_change"] = df["price_usd"].pct_change() * 100
        df["log_returns"] = np.log(df["price_usd"] / df["price_usd"].shift(1))
        print("  ✅ Price features (MA7, MA30, returns)")
    
    # --- Difficulty Adjustment Rate ---
    if "difficulty" in df.columns:
        df["difficulty_pct_change"] = df["difficulty"].pct_change() * 100
        print("  ✅ Difficulty change rate")
    
    # --- Transaction Density ---
    if "tx_per_block" in df.columns:
        df["tx_per_block_ma7"] = df["tx_per_block"].rolling(window=7).mean()
        print("  ✅ Transaction density")
    
    print(f"\n  ✅ Feature engineering complete: {len(df.columns)} total features")
    
    return df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function.
    """
    
    print("\n" + "=" * 80)
    print("BITCOIN DAILY ON-CHAIN METRICS COLLECTOR v2.0")
    print("=" * 80)
    print(f"Date range: {START_DATE} to {END_DATE}")
    print(f"Output file: {OUTPUT_CSV}")
    print(f"Primary source: Blockchain.com (CoinMetrics no longer free)")
    print("=" * 80)
    
    # Create output directory
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    
    # Step 1: Fetch Blockchain.com data (PRIMARY SOURCE)
    blockchain_df = get_all_blockchain_data()
    
    if blockchain_df.empty:
        print("\n❌ CRITICAL: No data from Blockchain.com. Cannot proceed.")
        return
    
    # Step 2: Fetch Glassnode data (if available)
    glassnode_df = get_all_glassnode_data()
    
    # Step 3: Fetch Fear & Greed Index
    fear_greed_df = get_fear_greed_index()
    
    # Step 4: Merge all datasets
    print("\n📊 STEP 4: Merging datasets...")
    print("=" * 80)
    
    merged_df = blockchain_df.copy()
    
    if not glassnode_df.empty:
        merged_df = pd.merge(merged_df, glassnode_df, on="date", how="left")
        print(f"  ✅ Merged Glassnode data")
    
    if not fear_greed_df.empty:
        merged_df = pd.merge(merged_df, fear_greed_df, on="date", how="left")
        print(f"  ✅ Merged Fear & Greed Index")
    
    # Step 5: Engineer features
    final_df = engineer_features(merged_df)
    
    # Step 6: Filter date range
    print("\n📊 STEP 6: Filtering date range...")
    print("=" * 80)
    final_df["date"] = pd.to_datetime(final_df["date"])
    final_df = final_df[
        (final_df["date"] >= START_DATE) & 
        (final_df["date"] <= END_DATE)
    ]
    final_df["date"] = final_df["date"].dt.date
    print(f"  ✅ Filtered to {len(final_df)} days")
    
    # Step 7: Sort and clean
    final_df = final_df.sort_values("date").reset_index(drop=True)
    
    # Step 8: Save to CSV
    print("\n📊 STEP 7: Saving to CSV...")
    print("=" * 80)
    final_df.to_csv(OUTPUT_CSV, index=False)
    print(f"  ✅ Saved {len(final_df)} rows × {len(final_df.columns)} columns")
    
    # Step 9: Display summary
    print("\n" + "=" * 80)
    print("COLLECTION SUMMARY")
    print("=" * 80)
    print(f"Total records: {len(final_df):,}")
    print(f"Date range: {final_df['date'].min()} to {final_df['date'].max()}")
    print(f"Total features: {len(final_df.columns)}")
    
    print(f"\n📊 Feature Completeness Report:")
    print("-" * 80)
    
    for col in final_df.columns:
        if col != "date":
            non_null = final_df[col].notna().sum()
            completeness = (non_null / len(final_df)) * 100
            missing = len(final_df) - non_null
            print(f"  {col:40s} {completeness:6.1f}% complete ({missing:4d} missing)")
    
    print("\n" + "=" * 80)
    print("✅ DATA COLLECTION COMPLETE!")
    print("=" * 80)
    
    # Display sample
    print("\n📋 First 5 rows preview:")
    print(final_df.head().to_string())
    
    print(f"\n📁 Full dataset saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()