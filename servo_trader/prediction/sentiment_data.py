#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sentiment_data.py

Sentiment data retrieval and processing for cryptocurrency prediction.

This module handles:
- Fetching Fear & Greed Index from Alternative.me API (FREE)
- Caching data locally to avoid repeated API calls
- Merging sentiment data with price data
- Creating sentiment-derived features

API Documentation: https://alternative.me/crypto/fear-and-greed-index/#api

Author: Jarred Deluca
Project: ServoTrader - Prediction Subsystem (v2.1)
"""

import os
import json
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple
from pathlib import Path


# =============================================================================
# Constants
# =============================================================================

FEAR_GREED_API_URL = "https://api.alternative.me/fng/"
DEFAULT_CACHE_DIR = os.path.expanduser("~/.servo_trader/sentiment_cache")


# =============================================================================
# Fear & Greed Index Functions
# =============================================================================

def fetch_fear_greed_index(limit: int = 0, cache_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch Fear & Greed Index data from Alternative.me API.
    
    This is a FREE API with no authentication required.
    
    Args:
        limit: Number of records to fetch (0 = all available, ~2000+ days)
        cache_dir: Directory to cache the data (None = use default)
        
    Returns:
        DataFrame with columns: date, fear_greed, sentiment
    """
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
    
    # Create cache directory if it doesn't exist
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    cache_file = os.path.join(cache_dir, "fear_greed_index.csv")
    
    # Check if we have recent cached data (less than 6 hours old)
    if os.path.exists(cache_file):
        cache_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))
        if cache_age < timedelta(hours=6):
            print(f"📂 Loading cached Fear & Greed data (age: {cache_age})")
            return pd.read_csv(cache_file, parse_dates=['date'])
    
    # Fetch from API
    print("🌐 Fetching Fear & Greed Index from Alternative.me API...")
    
    try:
        url = f"{FEAR_GREED_API_URL}?limit={limit}"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if 'data' not in data:
            raise ValueError("Invalid API response: 'data' field missing")
        
        records = data['data']
        print(f"   Retrieved {len(records)} days of sentiment data")
        
        # Convert to DataFrame
        df = pd.DataFrame(records)
        df['value'] = df['value'].astype(int)
        df['date'] = pd.to_datetime(df['timestamp'].astype(int), unit='s')
        
        # Rename columns
        df = df.rename(columns={
            'value': 'fear_greed',
            'value_classification': 'sentiment'
        })
        
        # Select and sort columns
        df = df[['date', 'fear_greed', 'sentiment']].sort_values('date').reset_index(drop=True)
        
        # Cache the data
        df.to_csv(cache_file, index=False)
        print(f"   Cached to: {cache_file}")
        
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"⚠️ API request failed: {e}")
        
        # Try to load from cache even if stale
        if os.path.exists(cache_file):
            print("   Loading stale cache as fallback...")
            return pd.read_csv(cache_file, parse_dates=['date'])
        
        raise RuntimeError(f"Failed to fetch Fear & Greed data and no cache available: {e}")


def get_fear_greed_for_date_range(start_date: datetime, 
                                   end_date: datetime,
                                   cache_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Get Fear & Greed data for a specific date range.
    
    Args:
        start_date: Start of date range
        end_date: End of date range
        cache_dir: Cache directory
        
    Returns:
        DataFrame filtered to the date range
    """
    df = fetch_fear_greed_index(limit=0, cache_dir=cache_dir)
    
    # Filter to date range
    mask = (df['date'] >= pd.Timestamp(start_date)) & (df['date'] <= pd.Timestamp(end_date))
    filtered = df[mask].copy()
    
    print(f"   Fear & Greed data range: {filtered['date'].min()} to {filtered['date'].max()}")
    print(f"   Days with sentiment data: {len(filtered)}")
    
    return filtered


def create_sentiment_features(fg_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features from Fear & Greed data.
    
    Args:
        fg_df: DataFrame with fear_greed column
        
    Returns:
        DataFrame with additional sentiment features
    """
    df = fg_df.copy()
    
    # Normalize to 0-1 scale
    df['fg_normalized'] = df['fear_greed'] / 100.0
    
    # Binary indicators for extreme sentiment
    df['fg_extreme_fear'] = (df['fear_greed'] < 25).astype(float)
    df['fg_extreme_greed'] = (df['fear_greed'] > 75).astype(float)
    df['fg_fear'] = (df['fear_greed'] < 40).astype(float)
    df['fg_greed'] = (df['fear_greed'] > 60).astype(float)
    
    # Change from previous day
    df['fg_change'] = df['fear_greed'].diff().fillna(0)
    df['fg_change_pct'] = df['fear_greed'].pct_change().fillna(0)
    
    # Moving averages (trend of sentiment)
    df['fg_ma_7'] = df['fear_greed'].rolling(window=7, min_periods=1).mean()
    df['fg_ma_14'] = df['fear_greed'].rolling(window=14, min_periods=1).mean()
    
    # Sentiment momentum (current vs moving average)
    df['fg_momentum_7'] = df['fear_greed'] - df['fg_ma_7']
    df['fg_momentum_14'] = df['fear_greed'] - df['fg_ma_14']
    
    # Volatility of sentiment
    df['fg_volatility_7'] = df['fear_greed'].rolling(window=7, min_periods=1).std().fillna(0)
    
    return df


def merge_sentiment_with_price(price_df: pd.DataFrame, 
                                fg_df: pd.DataFrame,
                                timestamp_col: str = 'timestamp') -> pd.DataFrame:
    """
    Merge Fear & Greed sentiment data with price data.
    
    For intraday data, the sentiment value is propagated to all bars
    within the same day.
    
    Args:
        price_df: Price DataFrame with timestamp column
        fg_df: Fear & Greed DataFrame with date and features
        timestamp_col: Name of timestamp column in price_df
        
    Returns:
        Price DataFrame with sentiment features added
    """
    result = price_df.copy()
    
    # Extract date from timestamp for merging
    result['_merge_date'] = pd.to_datetime(result[timestamp_col]).dt.date
    
    # Prepare sentiment data for merge
    fg_merge = fg_df.copy()
    fg_merge['_merge_date'] = pd.to_datetime(fg_merge['date']).dt.date
    
    # Drop the original date column to avoid confusion
    fg_cols = [col for col in fg_merge.columns if col != 'date']
    fg_merge = fg_merge[fg_cols]
    
    # Merge on date
    result = result.merge(fg_merge, on='_merge_date', how='left')
    
    # Drop the merge key
    result = result.drop(columns=['_merge_date'])
    
    # Forward fill any missing sentiment values (for dates before FG data starts)
    sentiment_cols = [col for col in result.columns if col.startswith('fg_') or col == 'fear_greed' or col == 'sentiment']
    for col in sentiment_cols:
        if col in result.columns:
            result[col] = result[col].ffill()
            # Back fill for any remaining NaNs at the start
            result[col] = result[col].bfill()
    
    # Fill any remaining NaNs with neutral values
    if 'fear_greed' in result.columns:
        result['fear_greed'] = result['fear_greed'].fillna(50)
    if 'fg_normalized' in result.columns:
        result['fg_normalized'] = result['fg_normalized'].fillna(0.5)
    
    # Fill binary indicators with 0
    for col in sentiment_cols:
        if col in result.columns and result[col].isna().any():
            result[col] = result[col].fillna(0)
    
    return result


def get_sentiment_feature_names() -> list:
    """
    Get the list of sentiment feature names for model input.
    
    Returns:
        List of sentiment feature column names
    """
    return [
        'fg_normalized',        # Fear & Greed 0-1 scale
        'fg_extreme_fear',      # Binary: extreme fear indicator
        'fg_extreme_greed',     # Binary: extreme greed indicator
        'fg_change',            # Day-over-day change
        'fg_momentum_7',        # Current vs 7-day MA
        'fg_volatility_7',      # 7-day volatility of sentiment
    ]


# =============================================================================
# Main / Testing
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Sentiment Data Module")
    print("=" * 60)
    
    # Test fetching Fear & Greed data
    print("\n1. Fetching Fear & Greed Index...")
    fg_df = fetch_fear_greed_index(limit=0)
    
    print(f"\n   Data shape: {fg_df.shape}")
    print(f"   Date range: {fg_df['date'].min()} to {fg_df['date'].max()}")
    print(f"   Sample data:")
    print(fg_df.tail(10).to_string())
    
    # Test creating features
    print("\n2. Creating sentiment features...")
    fg_features = create_sentiment_features(fg_df)
    print(f"   Features created: {list(fg_features.columns)}")
    
    # Test sentiment distribution
    print("\n3. Sentiment distribution:")
    print(fg_df['sentiment'].value_counts())
    
    # Test with mock price data
    print("\n4. Testing merge with price data...")
    mock_price = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='5min'),
        'close': np.random.randn(100).cumsum() + 45000,
        'volume': np.random.rand(100) * 1000
    })
    
    merged = merge_sentiment_with_price(mock_price, fg_features)
    print(f"   Merged shape: {merged.shape}")
    print(f"   Sentiment features in merged: {[c for c in merged.columns if 'fg_' in c]}")
    
    print("\n✅ Sentiment data module test passed!")