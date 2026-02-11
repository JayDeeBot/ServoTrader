#!/usr/bin/env python3
"""
On-Chain Data Fetcher for ServoTrader Prediction Model

This module fetches free on-chain Bitcoin metrics from Blockchain.com's Charts API.
These metrics are critical for improving daily prediction accuracy, as research shows
on-chain data can push accuracy from ~55% to ~82% for daily predictions.

Available Metrics (all free, no API key required):
- hash-rate: Network hash rate (TH/s) - security metric
- n-unique-addresses: Active addresses per day - adoption/activity
- n-transactions: Confirmed transactions per day - network usage
- estimated-transaction-volume-usd: Transaction volume in USD - economic activity
- difficulty: Mining difficulty - network security
- transaction-fees-usd: Total fees in USD - demand indicator
- miners-revenue: Total miner revenue - economic health
- market-cap: Market capitalization - market size
- avg-block-size: Average block size - network capacity usage

Derived Features (computed from raw metrics):
- Momentum indicators (1d, 7d changes)
- Moving average ratios (7d, 30d)
- Volatility measures
- Cross-metric ratios (tx_per_address, avg_tx_value, etc.)

Author: ServoTrader
Version: 2.0 (for v3.1 CNN-LSTM integration)
"""

import os
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Tuple

# Cache directory for on-chain data
CACHE_DIR = Path.home() / ".servo_trader" / "onchain_cache"


class OnChainDataFetcher:
    """
    Fetches and caches on-chain Bitcoin metrics from Blockchain.com API.
    
    The Blockchain.com Charts API is free and requires no authentication.
    Rate limiting is handled internally with delays between requests.
    """
    
    # Available chart endpoints and their descriptions
    AVAILABLE_METRICS = {
        # Network Activity
        'hash-rate': 'Network hash rate in TH/s',
        'n-unique-addresses': 'Number of unique addresses used per day',
        'n-transactions': 'Number of confirmed transactions per day',
        'n-payments': 'Number of payments per day',
        
        # Economic Activity  
        'estimated-transaction-volume-usd': 'Total transaction volume in USD',
        'transaction-fees-usd': 'Total transaction fees in USD',
        'miners-revenue': 'Total miner revenue in USD',
        
        # Network Health
        'difficulty': 'Mining difficulty',
        'avg-block-size': 'Average block size in bytes',
        'mempool-size': 'Mempool size in bytes',
        
        # Market Metrics
        'market-cap': 'Market capitalization in USD',
        'trade-volume': 'Exchange trading volume in USD',
    }
    
    # Default metrics to fetch for prediction model
    DEFAULT_METRICS = [
        'hash-rate',
        'n-unique-addresses', 
        'n-transactions',
        'estimated-transaction-volume-usd',
        'difficulty',
        'transaction-fees-usd',
        'miners-revenue',
        'market-cap',
        'avg-block-size',
    ]
    
    BASE_URL = "https://api.blockchain.info/charts"
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize the on-chain data fetcher.
        
        Args:
            cache_dir: Directory to store cached data. Defaults to ~/.servo_trader/onchain_cache
        """
        self.cache_dir = Path(cache_dir) if cache_dir else CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_cache_path(self, metric: str) -> Path:
        """Get the cache file path for a metric."""
        return self.cache_dir / f"blockchain_{metric.replace('-', '_')}.csv"
    
    def _fetch_metric(self, metric: str, timespan: str = "all", 
                      start: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Fetch a single metric from Blockchain.com API.
        
        Args:
            metric: The chart name (e.g., 'hash-rate', 'n-unique-addresses')
            timespan: Duration of data ('all', '1year', '6months', etc.)
            start: Start date in format 'YYYY-MM-DD' (optional)
            
        Returns:
            DataFrame with columns ['date', metric] or None if fetch fails
        """
        params = {
            'format': 'json',
            'sampled': 'false',  # Get all data points
            'timespan': timespan,
        }
        
        if start:
            params['start'] = start
            
        url = f"{self.BASE_URL}/{metric}"
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if 'values' not in data:
                print(f"   ⚠️  No 'values' key in response for {metric}")
                return None
                
            values = data['values']
            if not values:
                print(f"   ⚠️  Empty values for {metric}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(values)
            df.columns = ['timestamp', metric]
            
            # Convert Unix timestamp to datetime (timezone-naive)
            df['date'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
            # Convert to timezone-naive for consistency
            df['date'] = df['date'].dt.tz_localize(None)
            
            # IMPORTANT: Resample to daily frequency
            # The API may return data at sub-daily intervals
            df = df.set_index('date')
            df = df.drop(columns=['timestamp'])
            
            # Resample to daily, taking the last value of each day
            df = df.resample('D').last()
            df = df.dropna()
            df = df.reset_index()
            
            # Handle scientific notation and ensure numeric
            df[metric] = pd.to_numeric(df[metric], errors='coerce')
            
            return df
            
        except requests.exceptions.RequestException as e:
            print(f"   ❌ Error fetching {metric}: {e}")
            return None
        except Exception as e:
            print(f"   ❌ Unexpected error for {metric}: {e}")
            return None
    
    def fetch_all_metrics(self, metrics: Optional[List[str]] = None,
                          use_cache: bool = True,
                          cache_max_age_hours: int = 24) -> pd.DataFrame:
        """
        Fetch multiple on-chain metrics and merge them into a single DataFrame.
        
        Args:
            metrics: List of metric names to fetch. Defaults to DEFAULT_METRICS.
            use_cache: Whether to use cached data if available and fresh.
            cache_max_age_hours: Maximum age of cache before refreshing.
            
        Returns:
            DataFrame with 'date' index and columns for each metric.
        """
        if metrics is None:
            metrics = self.DEFAULT_METRICS
            
        print(f"\n📊 Fetching {len(metrics)} on-chain metrics from Blockchain.com...")
        
        all_dfs = []
        fetched_metrics = []
        
        for i, metric in enumerate(metrics):
            cache_path = self._get_cache_path(metric)
            
            # Check cache
            if use_cache and cache_path.exists():
                cache_age = time.time() - cache_path.stat().st_mtime
                if cache_age < cache_max_age_hours * 3600:
                    print(f"   [{i+1}/{len(metrics)}] Loading {metric} from cache...")
                    try:
                        df = pd.read_csv(cache_path, parse_dates=['date'])
                        # Ensure timezone-naive
                        if df['date'].dt.tz is not None:
                            df['date'] = df['date'].dt.tz_localize(None)
                        all_dfs.append(df)
                        fetched_metrics.append(metric)
                        continue
                    except Exception as e:
                        print(f"   ⚠️  Cache read failed for {metric}: {e}")
            
            # Fetch from API
            print(f"   [{i+1}/{len(metrics)}] Fetching {metric}...")
            df = self._fetch_metric(metric)
            
            if df is not None:
                # Cache the data
                try:
                    df.to_csv(cache_path, index=False)
                except Exception as e:
                    print(f"   ⚠️  Failed to cache {metric}: {e}")
                all_dfs.append(df)
                fetched_metrics.append(metric)
                
            # Rate limiting - be nice to the free API
            time.sleep(0.5)
        
        if not all_dfs:
            print("   ❌ No on-chain data could be fetched!")
            return pd.DataFrame()
        
        # Merge all DataFrames on date
        merged = all_dfs[0]
        for df in all_dfs[1:]:
            merged = pd.merge(merged, df, on='date', how='outer')
        
        # Sort by date
        merged = merged.sort_values('date').reset_index(drop=True)
        
        # Fill NaN values with forward/backward fill
        for col in merged.columns:
            if col != 'date':
                merged[col] = merged[col].ffill().bfill()
        
        print(f"   ✅ On-chain data loaded: {len(merged)} days, {len(fetched_metrics)} metrics")
        print(f"   📅 Date range: {merged['date'].min().date()} to {merged['date'].max().date()}")
        
        # Validate this looks like daily data
        expected_days = (merged['date'].max() - merged['date'].min()).days + 1
        actual_days = len(merged)
        if actual_days > expected_days * 1.1:  # Allow 10% tolerance
            print(f"   ⚠️  WARNING: Got {actual_days} rows but expected ~{expected_days} days")
            print(f"   ⚠️  Data may not be properly resampled to daily. Fixing...")
            merged = merged.drop_duplicates(subset=['date'], keep='last')
            print(f"   ✅ After deduplication: {len(merged)} days")
        
        return merged
    
    def compute_onchain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute derived features from raw on-chain metrics.
        
        Creates momentum, volatility, and ratio features that are more
        predictive than raw values.
        
        Args:
            df: DataFrame with raw on-chain metrics (must have 'date' column)
            
        Returns:
            DataFrame with original and derived features
        """
        print("\n🔧 Computing derived on-chain features...")
        
        df = df.copy()
        base_metrics = [c for c in df.columns if c != 'date']
        initial_cols = len(df.columns)
        
        for metric in base_metrics:
            if metric not in df.columns:
                continue
                
            # 1-day percentage change (momentum)
            df[f'{metric}_pct_1d'] = df[metric].pct_change(1).fillna(0)
            
            # 7-day percentage change (weekly momentum)
            df[f'{metric}_pct_7d'] = df[metric].pct_change(7).fillna(0)
            
            # Ratio to 7-day MA (short-term deviation)
            ma7 = df[metric].rolling(7, min_periods=1).mean()
            df[f'{metric}_ma7_ratio'] = (df[metric] / ma7.replace(0, np.nan)).fillna(1)
            
            # Ratio to 30-day MA (medium-term deviation)
            ma30 = df[metric].rolling(30, min_periods=1).mean()
            df[f'{metric}_ma30_ratio'] = (df[metric] / ma30.replace(0, np.nan)).fillna(1)
            
            # 7-day momentum (current vs 7 days ago direction)
            df[f'{metric}_momentum_7d'] = np.sign(df[metric] - df[metric].shift(7)).fillna(0)
            
            # 7-day volatility (normalized std dev)
            std7 = df[metric].rolling(7, min_periods=1).std()
            df[f'{metric}_vol_7d'] = (std7 / ma7.replace(0, np.nan)).fillna(0)
        
        # Clip extreme values
        for col in df.columns:
            if col != 'date' and col not in base_metrics:
                df[col] = df[col].clip(-10, 10)
        
        # Cross-metric ratios (these are particularly predictive)
        if 'n-transactions' in df.columns and 'n-unique-addresses' in df.columns:
            df['tx_per_address'] = (df['n-transactions'] / 
                                    df['n-unique-addresses'].replace(0, np.nan)).fillna(0)
        
        if 'estimated-transaction-volume-usd' in df.columns and 'n-transactions' in df.columns:
            df['avg_tx_value'] = (df['estimated-transaction-volume-usd'] / 
                                  df['n-transactions'].replace(0, np.nan)).fillna(0)
        
        if 'transaction-fees-usd' in df.columns and 'n-transactions' in df.columns:
            df['avg_fee_per_tx'] = (df['transaction-fees-usd'] / 
                                    df['n-transactions'].replace(0, np.nan)).fillna(0)
        
        if 'miners-revenue' in df.columns and 'hash-rate' in df.columns:
            df['revenue_per_hash'] = (df['miners-revenue'] / 
                                      df['hash-rate'].replace(0, np.nan)).fillna(0)
        
        # Hash rate growth indicator (important for network security)
        if 'hash-rate' in df.columns:
            hr_7d_ago = df['hash-rate'].shift(7)
            df['hashrate_growth_7d'] = ((df['hash-rate'] - hr_7d_ago) / 
                                        hr_7d_ago.replace(0, np.nan)).fillna(0)
        
        # Network Value to Transactions (NVT-like ratio)
        if 'market-cap' in df.columns and 'estimated-transaction-volume-usd' in df.columns:
            df['nvt_ratio'] = (df['market-cap'] / 
                               df['estimated-transaction-volume-usd'].replace(0, np.nan)).fillna(0)
            # Clip extreme NVT values
            df['nvt_ratio'] = df['nvt_ratio'].clip(0, 1000)
        
        # Fee ratio (fees as % of miner revenue)
        if 'transaction-fees-usd' in df.columns and 'miners-revenue' in df.columns:
            df['fee_ratio'] = (df['transaction-fees-usd'] / 
                               df['miners-revenue'].replace(0, np.nan)).fillna(0)
        
        # Replace infinities
        df = df.replace([np.inf, -np.inf], 0)
        
        derived_count = len(df.columns) - initial_cols
        print(f"   ✅ Created {derived_count} derived features")
        print(f"   📊 Total on-chain features: {len(df.columns) - 1}")
        
        return df


def load_onchain_data(start_date: Optional[str] = None,
                      metrics: Optional[List[str]] = None,
                      compute_features: bool = True,
                      use_cache: bool = True) -> pd.DataFrame:
    """
    Convenience function to load and process on-chain data.
    
    Args:
        start_date: Start date in 'YYYY-MM-DD' format. If None, loads all available.
        metrics: List of metrics to fetch. If None, uses default set.
        compute_features: Whether to compute derived features.
        use_cache: Whether to use cached data.
        
    Returns:
        DataFrame with on-chain data and features.
    """
    fetcher = OnChainDataFetcher()
    
    # Fetch raw metrics
    df = fetcher.fetch_all_metrics(metrics=metrics, use_cache=use_cache)
    
    if df.empty:
        return df
    
    # Filter by start date if provided
    if start_date:
        start_dt = pd.to_datetime(start_date)
        df = df[df['date'] >= start_dt].reset_index(drop=True)
    
    # Compute derived features
    if compute_features:
        df = fetcher.compute_onchain_features(df)
    
    return df


def get_onchain_feature_names(include_raw: bool = True,
                              include_derived: bool = True) -> List[str]:
    """
    Get list of on-chain feature column names.
    
    Args:
        include_raw: Include raw metric names.
        include_derived: Include derived feature names.
        
    Returns:
        List of feature column names.
    """
    raw_metrics = OnChainDataFetcher.DEFAULT_METRICS
    
    features = []
    
    if include_raw:
        features.extend(raw_metrics)
    
    if include_derived:
        for metric in raw_metrics:
            features.extend([
                f'{metric}_pct_1d',
                f'{metric}_pct_7d',
                f'{metric}_ma7_ratio',
                f'{metric}_ma30_ratio',
                f'{metric}_momentum_7d',
                f'{metric}_vol_7d',
            ])
        # Cross-metric ratios
        features.extend([
            'tx_per_address',
            'avg_tx_value',
            'avg_fee_per_tx',
            'revenue_per_hash',
            'hashrate_growth_7d',
            'nvt_ratio',
            'fee_ratio',
        ])
    
    return features


# ============================================================
# Test / Demo
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🔗 On-Chain Data Fetcher Test")
    print("=" * 60)
    
    # Test loading data
    df = load_onchain_data(compute_features=True)
    
    if not df.empty:
        print(f"\n📊 Data Summary:")
        print(f"   Rows: {len(df)}")
        print(f"   Columns: {len(df.columns)}")
        print(f"   Date range: {df['date'].min().date()} to {df['date'].max().date()}")
        
        print(f"\n📋 Feature columns ({len(df.columns) - 1}):")
        for i, col in enumerate(df.columns):
            if col != 'date':
                print(f"   {i}. {col}")
        
        print(f"\n📈 Sample data (last 5 rows):")
        print(df.tail())
    else:
        print("\n❌ Failed to load on-chain data")