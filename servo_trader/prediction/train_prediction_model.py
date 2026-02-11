#!/usr/bin/env python3
"""
ServoTrader Prediction Model Training Script (v2.5 - On-Chain Data Integration)

This script trains a LightGBM model to predict Bitcoin price direction.
Version 2.5 adds integration with free on-chain blockchain metrics from Blockchain.com.

Key Features:
- Daily prediction support with proper indicator timescales
- Fear & Greed Index sentiment features
- On-chain blockchain metrics (hash rate, active addresses, tx volume, etc.)
- Boruta feature selection (optional)
- Optuna hyperparameter optimization (optional)

Usage:
    # Full training with all features and optimizations
    python train_prediction_model.py --horizon 1440
    
    # Quick test without optimization
    python train_prediction_model.py --horizon 1440 --no-boruta --no-optuna
    
    # Without on-chain data
    python train_prediction_model.py --horizon 1440 --no-onchain

Author: ServoTrader
Version: 2.5 (On-Chain Data Integration)
"""

import os
import sys
import argparse
import pickle
import json
import yaml
import warnings
from datetime import datetime
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, mean_squared_error, mean_absolute_error, r2_score
)
import lightgbm as lgb

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Try to import boruta
try:
    from boruta import BorutaPy
    BORUTA_AVAILABLE = True
except ImportError:
    BORUTA_AVAILABLE = False
    print("⚠️ Boruta not installed. Install with: pip install boruta")

# Try to import optuna
try:
    import optuna
    from optuna.samplers import TPESampler
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("⚠️ Optuna not installed. Install with: pip install optuna")


# ============================================================
# Configuration
# ============================================================

# Default paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models" / "prediction"
DAILY_DATA_DIR = DATA_DIR / "daily_historical"

# Sentiment data cache
SENTIMENT_CACHE_DIR = Path.home() / ".servo_trader" / "sentiment_cache"

# Default parameters
DEFAULT_HORIZON = 60  # 60 minutes
DEFAULT_DIRECTION_THRESHOLD = 0.001  # 0.1% threshold for flat classification
DEFAULT_TRAIN_RATIO = 0.7
DEFAULT_VAL_RATIO = 0.15


# ============================================================
# Sentiment Data Functions (Fear & Greed Index)
# ============================================================

def fetch_fear_greed_index(use_cache: bool = True, cache_max_age_hours: int = 24) -> pd.DataFrame:
    """
    Fetch Fear & Greed Index from Alternative.me API.
    
    Returns:
        DataFrame with columns: date, fg_value, fg_classification
    """
    import requests
    
    cache_path = SENTIMENT_CACHE_DIR / "fear_greed_index.csv"
    SENTIMENT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check cache
    if use_cache and cache_path.exists():
        import time
        cache_age = time.time() - cache_path.stat().st_mtime
        if cache_age < cache_max_age_hours * 3600:
            try:
                df = pd.read_csv(cache_path)
                # Convert date column if it exists
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    return df
                else:
                    print("   ⚠️ Cache file missing 'date' column, refetching...")
            except Exception as e:
                print(f"   ⚠️ Error reading cache: {e}, refetching...")
    
    print("🌐 Fetching Fear & Greed Index from Alternative.me API...")
    
    try:
        # Fetch all historical data (limit=0 means all)
        url = "https://api.alternative.me/fng/?limit=0&format=json"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if 'data' not in data:
            print("   ⚠️ No data in response")
            return pd.DataFrame()
        
        records = []
        for item in data['data']:
            records.append({
                'date': pd.to_datetime(int(item['timestamp']), unit='s'),
                'fg_value': int(item['value']),
                'fg_classification': item['value_classification']
            })
        
        df = pd.DataFrame(records)
        df['date'] = df['date'].dt.normalize()
        df = df.sort_values('date').reset_index(drop=True)
        
        # Cache the data
        df.to_csv(cache_path, index=False)
        print(f"   Retrieved {len(df)} days of sentiment data")
        print(f"   Cached to: {cache_path}")
        
        return df
        
    except Exception as e:
        print(f"   ❌ Error fetching Fear & Greed Index: {e}")
        # Try to return cached data even if stale
        if cache_path.exists():
            print("   📂 Using stale cache as fallback...")
            try:
                df = pd.read_csv(cache_path)
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    return df
            except Exception:
                pass
        return pd.DataFrame()


def compute_sentiment_features(fg_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute derived features from Fear & Greed Index.
    
    Args:
        fg_df: DataFrame with columns: date, fg_value, fg_classification
        
    Returns:
        DataFrame with additional sentiment features
    """
    df = fg_df.copy()
    
    # Moving averages
    df['fg_ma_7'] = df['fg_value'].rolling(7).mean()
    df['fg_ma_14'] = df['fg_value'].rolling(14).mean()
    df['fg_ma_30'] = df['fg_value'].rolling(30).mean()
    
    # Momentum
    df['fg_momentum_7'] = df['fg_value'] - df['fg_value'].shift(7)
    df['fg_momentum_14'] = df['fg_value'] - df['fg_value'].shift(14)
    
    # Volatility
    df['fg_volatility_7'] = df['fg_value'].rolling(7).std()
    df['fg_volatility_14'] = df['fg_value'].rolling(14).std()
    
    # Change
    df['fg_change'] = df['fg_value'].diff()
    df['fg_pct_change'] = df['fg_value'].pct_change()
    
    # Extreme indicators (binary)
    df['fg_extreme_fear'] = (df['fg_value'] < 25).astype(int)
    df['fg_extreme_greed'] = (df['fg_value'] > 75).astype(int)
    
    # Distance from neutral (50)
    df['fg_distance_neutral'] = df['fg_value'] - 50
    
    # Fill NaN values
    for col in df.columns:
        if col not in ['date', 'fg_classification']:
            df[col] = df[col].fillna(method='ffill').fillna(method='bfill').fillna(50)
    
    return df


# ============================================================
# On-Chain Data Functions
# ============================================================

def load_onchain_data_for_training(use_cache: bool = True) -> pd.DataFrame:
    """
    Load on-chain data for training.
    
    Returns:
        DataFrame with on-chain features indexed by date.
    """
    try:
        # Import the on-chain data module
        from servo_trader.prediction.onchain_data import load_onchain_data
        
        df = load_onchain_data(use_cache=use_cache, compute_features=True)
        return df
        
    except ImportError:
        # Fallback: try importing from local directory
        try:
            from onchain_data import load_onchain_data
            df = load_onchain_data(use_cache=use_cache, compute_features=True)
            return df
        except ImportError:
            print("   ⚠️ On-chain data module not found. Skipping on-chain features.")
            return pd.DataFrame()
    except Exception as e:
        print(f"   ❌ Error loading on-chain data: {e}")
        return pd.DataFrame()


# ============================================================
# Technical Indicator Functions
# ============================================================

def compute_daily_technical_indicators(df: pd.DataFrame, horizon_days: int = 1) -> pd.DataFrame:
    """
    Compute technical indicators appropriate for daily timeframe.
    
    Args:
        df: DataFrame with OHLCV data
        horizon_days: Prediction horizon in days
        
    Returns:
        DataFrame with technical indicators added
    """
    print(f"🔧 Computing daily technical indicators...")
    print(f"   Using {horizon_days}-day prediction horizon")
    
    df = df.copy()
    
    # Ensure we have required columns
    required = ['open', 'high', 'low', 'close', 'volume']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # -------------------------
    # Price-based indicators
    # -------------------------
    
    # Simple Moving Averages (daily appropriate periods)
    for period in [7, 14, 21, 30, 50]:
        df[f'sma_{period}'] = df['close'].rolling(period).mean()
        df[f'sma_{period}_pct'] = (df['close'] / df[f'sma_{period}'] - 1) * 100
    
    # Exponential Moving Averages
    for period in [7, 14, 21, 30, 50]:
        df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        df[f'ema_{period}_pct'] = (df['close'] / df[f'ema_{period}'] - 1) * 100
    
    # Volatility (daily)
    for period in [7, 14, 30]:
        df[f'volatility_{period}'] = df['close'].pct_change().rolling(period).std() * 100
    
    # -------------------------
    # Momentum indicators
    # -------------------------
    
    # RSI (Relative Strength Index)
    for period in [7, 14]:
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        df[f'rsi_{period}'] = df[f'rsi_{period}'].fillna(50)
    
    # MACD (12, 26, 9 are standard daily parameters)
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Stochastic Oscillator (14, 3, 3)
    low_14 = df['low'].rolling(14).min()
    high_14 = df['high'].rolling(14).max()
    df['stoch_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14 + 1e-10)
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()
    df['stoch_k_normalized'] = df['stoch_k'] / 100
    
    # Momentum
    for period in [7, 14, 30]:
        df[f'momentum_{period}'] = df['close'].pct_change(period) * 100
    
    # Rate of Change
    for period in [7, 14]:
        df[f'roc_{period}'] = (df['close'] / df['close'].shift(period) - 1) * 100
    
    # -------------------------
    # Volatility indicators
    # -------------------------
    
    # Bollinger Bands (20 day, 2 std)
    sma_20 = df['close'].rolling(20).mean()
    std_20 = df['close'].rolling(20).std()
    df['bb_upper'] = sma_20 + 2 * std_20
    df['bb_lower'] = sma_20 - 2 * std_20
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / sma_20 * 100
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
    
    # Average True Range
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(14).mean()
    df['atr_pct'] = df['atr_14'] / df['close'] * 100
    
    # -------------------------
    # Volume indicators
    # -------------------------
    
    # Volume moving average ratio
    for period in [7, 20]:
        vol_ma = df['volume'].rolling(period).mean()
        df[f'volume_ratio_{period}'] = df['volume'] / vol_ma.replace(0, np.nan)
    
    # On-Balance Volume trend
    obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    df['obv_trend'] = obv.diff(7)  # 7-day OBV change
    
    # -------------------------
    # Time features (for daily data)
    # -------------------------
    
    if 'timestamp' in df.columns:
        ts = pd.to_datetime(df['timestamp'])
        # Use .dt accessor for Series
        day_of_week = ts.dt.dayofweek
        day_of_month = ts.dt.day
        month = ts.dt.month
    elif df.index.name == 'timestamp' or isinstance(df.index, pd.DatetimeIndex):
        ts = pd.to_datetime(df.index)
        # DatetimeIndex has direct access
        day_of_week = ts.dayofweek
        day_of_month = ts.day
        month = ts.month
    else:
        # Try to create from index
        ts = pd.to_datetime(df.index)
        day_of_week = ts.dayofweek
        day_of_month = ts.day
        month = ts.month
    
    # Day of week (cyclical encoding)
    df['day_sin'] = np.sin(2 * np.pi * day_of_week / 7)
    df['day_cos'] = np.cos(2 * np.pi * day_of_week / 7)
    
    # Day of month
    df['dom_sin'] = np.sin(2 * np.pi * day_of_month / 31)
    df['dom_cos'] = np.cos(2 * np.pi * day_of_month / 31)
    
    # Month (cyclical)
    df['month_sin'] = np.sin(2 * np.pi * month / 12)
    df['month_cos'] = np.cos(2 * np.pi * month / 12)
    
    # Fill NaN values
    df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    # Replace infinities
    df = df.replace([np.inf, -np.inf], 0)
    
    return df


# ============================================================
# Data Preparation
# ============================================================

def detect_data_frequency(df: pd.DataFrame) -> str:
    """
    Detect if data is minute-level or daily.
    
    Returns:
        'minute' or 'daily'
    """
    if 'timestamp' in df.columns:
        ts = pd.to_datetime(df['timestamp'])
    else:
        ts = df.index
    
    # Calculate median time difference
    time_diff = ts.diff().median()
    
    if time_diff <= pd.Timedelta(hours=1):
        return 'minute'
    else:
        return 'daily'


def resample_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample minute-level data to daily OHLCV bars.
    """
    print("📅 Resampling data to daily bars...")
    
    if 'timestamp' in df.columns:
        df = df.set_index('timestamp')
    
    df.index = pd.to_datetime(df.index)
    
    # Resample to daily
    daily = df.resample('D').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    # Add additional columns if they exist
    if 'vwap' in df.columns:
        # Approximate VWAP for daily
        daily['vwap'] = daily['close']  # Simplified
    
    if 'count' in df.columns:
        daily['count'] = df['count'].resample('D').sum()
    
    daily = daily.reset_index()
    daily = daily.rename(columns={'index': 'timestamp'})
    
    print(f"   Original rows: {len(df)}")
    print(f"   Daily bars: {len(daily)}")
    print(f"   Date range: {daily['timestamp'].min()} to {daily['timestamp'].max()}")
    
    return daily


def prepare_data(
    data_path: str,
    horizon_minutes: int = 60,
    direction_threshold: float = 0.001,
    include_sentiment: bool = True,
    include_onchain: bool = True,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, StandardScaler, List[str], pd.DataFrame]:
    """
    Load and prepare data for training.
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test,
        y_train_binary, y_val_binary, y_test_binary,
        scaler, feature_columns, full_df
    """
    print("\n" + "=" * 60)
    print("🚀 Starting Data Preparation Pipeline")
    print("=" * 60 + "\n")
    
    # Load data
    print(f"📂 Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"   Loaded {len(df):,} rows")
    
    # Detect data frequency
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        freq = detect_data_frequency(df)
        print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    else:
        freq = 'daily'
    
    print(f"   Detected data frequency: {freq.capitalize()} bars")
    
    # Determine if we need daily resampling
    horizon_days = horizon_minutes // 1440
    use_daily = horizon_minutes >= 1440
    
    if use_daily:
        if freq == 'minute':
            df = resample_to_daily(df)
        else:
            print("✅ Data is already daily. No resampling needed.")
        
        # Compute daily technical indicators
        df = compute_daily_technical_indicators(df, horizon_days=horizon_days)
    else:
        # Use minute-level indicators (not shown here for brevity)
        print("⚠️ Using minute-level data without resampling")
    
    # -------------------------
    # Add sentiment features
    # -------------------------
    if include_sentiment:
        print("📊 Loading Fear & Greed sentiment data...")
        fg_df = fetch_fear_greed_index(use_cache=True)
        
        if not fg_df.empty:
            # Compute sentiment features
            fg_df = compute_sentiment_features(fg_df)
            print(f"   Sentiment data range: {fg_df['date'].min()} to {fg_df['date'].max()}")
            print(f"   Total sentiment days: {len(fg_df)}")
            
            # Merge with main data
            print("🧠 Merging sentiment features...")
            if 'timestamp' in df.columns:
                df['date'] = df['timestamp'].dt.normalize()
            else:
                df['date'] = pd.to_datetime(df.index).normalize()
            
            # Ensure both date columns are timezone-naive for merging
            if df['date'].dt.tz is not None:
                df['date'] = df['date'].dt.tz_localize(None)
            if fg_df['date'].dt.tz is not None:
                fg_df['date'] = fg_df['date'].dt.tz_localize(None)
            
            # Drop classification column (non-numeric)
            fg_cols = [c for c in fg_df.columns if c != 'fg_classification']
            df = df.merge(fg_df[fg_cols], on='date', how='left')
            
            # Fill missing sentiment values
            sentiment_cols = [c for c in fg_df.columns if c.startswith('fg_') and c != 'fg_classification']
            for col in sentiment_cols:
                if col in df.columns:
                    df[col] = df[col].fillna(method='ffill').fillna(method='bfill').fillna(50)
            
            coverage = df['fg_value'].notna().sum()
            print(f"   Sentiment coverage: {len(df):,} / {len(df):,} rows ({coverage/len(df)*100:.1f}%)")
        else:
            print("   ⚠️ No sentiment data available")
    
    # -------------------------
    # Add on-chain features
    # -------------------------
    if include_onchain:
        print("⛓️ Loading on-chain blockchain metrics...")
        onchain_df = load_onchain_data_for_training(use_cache=True)
        
        if not onchain_df.empty:
            print(f"   On-chain data range: {onchain_df['date'].min()} to {onchain_df['date'].max()}")
            print(f"   On-chain features: {len(onchain_df.columns) - 1}")
            
            # Merge with main data
            print("🔗 Merging on-chain features...")
            if 'date' not in df.columns:
                if 'timestamp' in df.columns:
                    df['date'] = df['timestamp'].dt.normalize()
                else:
                    df['date'] = pd.to_datetime(df.index).normalize()
            
            # Ensure both date columns are timezone-naive for merging
            if df['date'].dt.tz is not None:
                df['date'] = df['date'].dt.tz_localize(None)
            if onchain_df['date'].dt.tz is not None:
                onchain_df['date'] = onchain_df['date'].dt.tz_localize(None)
            
            df = df.merge(onchain_df, on='date', how='left')
            
            # Fill missing on-chain values
            onchain_cols = [c for c in onchain_df.columns if c != 'date']
            for col in onchain_cols:
                if col in df.columns:
                    df[col] = df[col].fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            print(f"   On-chain coverage: {df[onchain_cols[0]].notna().sum():,} / {len(df):,} rows")
        else:
            print("   ⚠️ No on-chain data available")
    
    # -------------------------
    # Identify feature columns
    # -------------------------
    exclude_cols = {
        'timestamp', 'date', 'symbol', 'pair', 'ticker', 'asset', 'base', 'quote',
        'open', 'high', 'low', 'close', 'volume', 'vwap', 'count',
        'fg_classification', 'target', 'target_pct', 'direction'
    }
    
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    # Only include numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in feature_cols if c in numeric_cols]
    
    print(f"   Total columns: {len(df.columns)}")
    print(f"   Feature columns: {len(feature_cols)}")
    
    # -------------------------
    # Prepare features
    # -------------------------
    print("📊 Preparing features...")
    
    # -------------------------
    # Create labels
    # -------------------------
    if use_daily:
        horizon_label = f"{horizon_days} day(s)"
    else:
        horizon_label = f"{horizon_minutes} minutes"
    
    print(f"🎯 Creating labels (horizon: {horizon_label})...")
    
    # Calculate future return
    if use_daily:
        df['target_pct'] = df['close'].shift(-horizon_days) / df['close'] - 1
    else:
        df['target_pct'] = df['close'].shift(-horizon_minutes) / df['close'] - 1
    
    # Create direction labels
    print(f"   Direction threshold: ±{direction_threshold*100:.2f}% (samples within excluded)")
    
    # Binary direction (0 = down, 1 = up)
    df['direction'] = np.where(df['target_pct'] > direction_threshold, 1, 
                               np.where(df['target_pct'] < -direction_threshold, 0, np.nan))
    
    # Count distribution
    down_count = (df['direction'] == 0).sum()
    up_count = (df['direction'] == 1).sum()
    flat_count = df['direction'].isna().sum()
    
    print(f"   Direction distribution (binary):")
    print(f"     Down: {down_count:,} ({down_count/len(df)*100:.1f}%)")
    print(f"     Up: {up_count:,} ({up_count/len(df)*100:.1f}%)")
    print(f"     Excluded (flat): {flat_count:,} ({flat_count/len(df)*100:.1f}%)")
    
    # Remove samples with NaN labels or targets
    valid_mask = df['direction'].notna() & df['target_pct'].notna()
    df_valid = df[valid_mask].copy()
    
    print(f"   Created {len(df_valid):,} valid samples (excluding flat)")
    
    # -------------------------
    # Extract features and labels
    # -------------------------
    X = df_valid[feature_cols].values
    y_pct = df_valid['target_pct'].values
    y_binary = df_valid['direction'].values.astype(int)
    
    print(f"   Feature shape: {X.shape}")
    
    # -------------------------
    # Temporal train/val/test split
    # -------------------------
    print("✂️ Splitting data (temporal split, no shuffling)...")
    
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    X_train = X[:train_end]
    X_val = X[train_end:val_end]
    X_test = X[val_end:]
    
    y_train = y_pct[:train_end]
    y_val = y_pct[train_end:val_end]
    y_test = y_pct[val_end:]
    
    y_train_binary = y_binary[:train_end]
    y_val_binary = y_binary[train_end:val_end]
    y_test_binary = y_binary[val_end:]
    
    # Get timestamps for reference
    if 'timestamp' in df_valid.columns:
        timestamps = df_valid['timestamp'].values
    else:
        timestamps = df_valid.index.values
    
    print(f"   Train: {len(X_train):,} samples ({timestamps[:train_end][-1]} to {timestamps[train_end-1]})")
    print(f"   Val:   {len(X_val):,} samples ({timestamps[train_end]} to {timestamps[val_end-1]})")
    print(f"   Test:  {len(X_test):,} samples ({timestamps[val_end]} to {timestamps[-1]})")
    
    # -------------------------
    # Scale features
    # -------------------------
    print("📏 Fitting scaler on training data...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    print("   Scaler fitted successfully")
    
    print("\n" + "=" * 60)
    print("✅ Data Preparation Complete")
    print("=" * 60 + "\n")
    
    return (X_train, X_val, X_test, 
            y_train, y_val, y_test,
            y_train_binary, y_val_binary, y_test_binary,
            scaler, feature_cols, df_valid)


# ============================================================
# Boruta Feature Selection
# ============================================================

def run_boruta_feature_selection(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: List[str],
    max_iter: int = 100,
    percentile: int = 100
) -> Tuple[np.ndarray, List[str], Dict]:
    """
    Run Boruta feature selection.
    
    Returns:
        selected_mask, selected_feature_names, boruta_results_dict
    """
    print("\n" + "=" * 60)
    print("🔍 Running Boruta Feature Selection")
    print("=" * 60)
    print(f"   Max iterations: {max_iter}")
    print(f"   Percentile threshold: {percentile}")
    print(f"   Input features: {len(feature_names)}")
    
    # Create random forest estimator
    rf = RandomForestClassifier(
        n_jobs=-1,
        class_weight='balanced',
        max_depth=5,
        random_state=42
    )
    
    # Run Boruta
    print("\n   Running Boruta (this may take a few minutes)...")
    start_time = datetime.now()
    
    boruta = BorutaPy(
        rf,
        n_estimators='auto',
        verbose=0,
        random_state=42,
        max_iter=max_iter,
        perc=percentile
    )
    
    boruta.fit(X_train, y_train)
    
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"   Completed in {elapsed:.1f} seconds")
    
    # Get results
    confirmed_mask = boruta.support_
    tentative_mask = boruta.support_weak_
    
    confirmed_features = [f for f, m in zip(feature_names, confirmed_mask) if m]
    tentative_features = [f for f, m in zip(feature_names, tentative_mask) if m]
    rejected_features = [f for f, c, t in zip(feature_names, confirmed_mask, tentative_mask) 
                         if not c and not t]
    
    print(f"\n📊 Boruta Results:")
    print(f"   ✅ Confirmed features: {len(confirmed_features)}")
    print(f"   ⚠️ Tentative features: {len(tentative_features)}")
    print(f"   ❌ Rejected features: {len(rejected_features)}")
    
    # Use confirmed + tentative features
    selected_mask = confirmed_mask | tentative_mask
    n_selected = selected_mask.sum()
    
    # HANDLE EDGE CASE: If Boruta rejects ALL features, fall back to top features by importance
    if n_selected == 0:
        print(f"\n   ⚠️ WARNING: Boruta rejected ALL features!")
        print(f"   📊 Falling back to top 20 features by Random Forest importance...")
        
        # Fit RF to get feature importances
        rf.fit(X_train, y_train)
        importances = rf.feature_importances_
        
        # Get indices of top 20 features (or all if less than 20)
        n_fallback = min(20, len(feature_names))
        top_indices = np.argsort(importances)[-n_fallback:]
        
        # Create new mask with top features
        selected_mask = np.zeros(len(feature_names), dtype=bool)
        selected_mask[top_indices] = True
        
        selected_features = [feature_names[i] for i in top_indices]
        selected_features_with_imp = [(feature_names[i], importances[i]) for i in top_indices]
        selected_features_with_imp.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n   Selected top {n_fallback} features by importance:")
        for i, (f, imp) in enumerate(selected_features_with_imp[:10], 1):
            print(f"      {i}. {f} ({imp:.4f})")
        
        # Update results to reflect fallback
        results = {
            'confirmed': [],
            'tentative': [],
            'rejected': feature_names,
            'selected': selected_features,
            'n_confirmed': 0,
            'n_tentative': 0,
            'n_rejected': len(feature_names),
            'fallback_used': True,
            'fallback_reason': 'All features rejected by Boruta'
        }
        
        print(f"\n   Using {len(selected_features)} features (fallback to RF importance)")
        
        return selected_mask, selected_features, results
    
    # Normal case: Boruta selected some features
    if confirmed_features:
        # Get feature importances for ranking
        rf.fit(X_train, y_train)
        importances = rf.feature_importances_
        
        print(f"\n   Confirmed features (ranked by importance):")
        confirmed_importances = [(f, importances[i]) for i, f in enumerate(feature_names) 
                                 if confirmed_mask[i]]
        confirmed_importances.sort(key=lambda x: x[1], reverse=True)
        for i, (f, imp) in enumerate(confirmed_importances[:10], 1):
            print(f"      {i}. {f}")
    
    if tentative_features:
        print(f"\n   Tentative features:")
        for f in tentative_features[:5]:
            print(f"      - {f}")
    
    selected_features = [f for f, m in zip(feature_names, selected_mask) if m]
    
    print(f"\n   Using {len(selected_features)} features (confirmed + tentative)")
    
    results = {
        'confirmed': confirmed_features,
        'tentative': tentative_features,
        'rejected': rejected_features,
        'selected': selected_features,
        'n_confirmed': len(confirmed_features),
        'n_tentative': len(tentative_features),
        'n_rejected': len(rejected_features),
        'fallback_used': False
    }
    
    return selected_mask, selected_features, results


# ============================================================
# Optuna Hyperparameter Optimization
# ============================================================

def run_optuna_optimization(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 50,
    timeout: Optional[int] = None
) -> Tuple[Dict, Dict]:
    """
    Run Optuna hyperparameter optimization for LightGBM.
    
    Returns:
        best_params, optuna_results_dict
    """
    print("\n" + "=" * 60)
    print("🔬 Running Optuna Hyperparameter Optimization")
    print("=" * 60)
    print(f"   Number of trials: {n_trials}")
    print(f"   Timeout: {timeout}")
    print(f"   Training samples: {len(X_train):,}")
    print(f"   Validation samples: {len(X_val):,}")
    print(f"   Features: {X_train.shape[1]}")
    
    best_accuracy = [0.0]  # Use list to allow modification in nested function
    
    def objective(trial):
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'verbosity': -1,
            'feature_pre_filter': False,
            
            'num_leaves': trial.suggest_int('num_leaves', 15, 127),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
        }
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        
        # Evaluate
        y_pred = (model.predict(X_val) > 0.5).astype(int)
        accuracy = accuracy_score(y_val, y_pred)
        
        if accuracy > best_accuracy[0]:
            best_accuracy[0] = accuracy
        
        return accuracy
    
    # Create study
    print("\n   Starting optimization...")
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    
    # Progress callback
    def callback(study, trial):
        if (trial.number + 1) % 10 == 0 or trial.number == 0:
            print(f"   Trial {trial.number + 1}/{n_trials}: "
                  f"Accuracy = {trial.value:.4f} ({trial.value*100:.2f}%) | "
                  f"Best so far = {study.best_value:.4f} ({study.best_value*100:.2f}%)")
    
    start_time = datetime.now()
    study.optimize(objective, n_trials=n_trials, timeout=timeout, 
                   callbacks=[callback], show_progress_bar=False)
    elapsed = (datetime.now() - start_time).total_seconds()
    
    print(f"\n   Optimization completed in {elapsed:.1f} seconds")
    
    # Get best parameters
    best_params = study.best_params
    best_trial = study.best_trial
    
    print(f"\n📊 Optuna Results:")
    print(f"   Best trial: #{best_trial.number}")
    print(f"   Best validation accuracy: {best_trial.value:.4f} ({best_trial.value*100:.2f}%)")
    print(f"\n   Best hyperparameters:")
    for key, value in best_params.items():
        print(f"      {key}: {value:.6f}" if isinstance(value, float) else f"      {key}: {value}")
    
    # Calculate improvement over default
    default_acc = 0.5  # Approximate default accuracy
    improvement = best_trial.value - default_acc
    print(f"\n   Improvement over default: +{improvement*100:.2f}% points")
    
    results = {
        'best_trial': best_trial.number,
        'best_accuracy': best_trial.value,
        'best_params': best_params,
        'n_trials': n_trials,
        'elapsed_seconds': elapsed
    }
    
    return best_params, results


# ============================================================
# Model Training
# ============================================================

def train_lightgbm_models(
    X_train: np.ndarray,
    y_train_pct: np.ndarray,
    y_train_binary: np.ndarray,
    X_val: np.ndarray,
    y_val_pct: np.ndarray,
    y_val_binary: np.ndarray,
    optuna_params: Optional[Dict] = None
) -> Tuple[lgb.Booster, lgb.Booster]:
    """
    Train LightGBM regressor and classifier models.
    
    Returns:
        regressor, classifier
    """
    print("\n" + "=" * 60)
    print("🌲 Training LightGBM Models")
    print("=" * 60)
    print(f"\n   Train samples: {len(X_train):,}")
    print(f"   Val samples: {len(X_val):,}")
    print(f"   Features: {X_train.shape[1]}")
    
    # Base parameters
    base_params = {
        'verbosity': -1,
        'feature_pre_filter': False,
        'force_col_wise': True,
    }
    
    # Regressor parameters
    reg_params = {
        **base_params,
        'objective': 'regression',
        'metric': 'l2',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
    }
    
    # Classifier parameters
    clf_params = {
        **base_params,
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
    }
    
    # Apply Optuna parameters if provided
    if optuna_params:
        print("   Using Optuna-optimized regressor parameters")
        print("   Using Optuna-optimized classifier parameters")
        reg_params.update(optuna_params)
        clf_params.update(optuna_params)
        # Override objective/metric
        reg_params['objective'] = 'regression'
        reg_params['metric'] = 'l2'
        clf_params['objective'] = 'binary'
        clf_params['metric'] = 'binary_logloss'
    
    # -------------------------
    # Train Regressor
    # -------------------------
    print("\n📈 Training LightGBM Regressor...")
    
    train_reg = lgb.Dataset(X_train, label=y_train_pct)
    val_reg = lgb.Dataset(X_val, label=y_val_pct, reference=train_reg)
    
    regressor = lgb.train(
        reg_params,
        train_reg,
        num_boost_round=1000,
        valid_sets=[train_reg, val_reg],
        valid_names=['train', 'val'],
        callbacks=[
            lgb.early_stopping(100, verbose=True),
            lgb.log_evaluation(period=0)
        ]
    )
    
    # Evaluate regressor
    y_val_pred = regressor.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val_pct, y_val_pred))
    mae = mean_absolute_error(y_val_pct, y_val_pred)
    r2 = r2_score(y_val_pct, y_val_pred)
    
    print(f"\n   Regressor Results:")
    print(f"   - Best iteration: {regressor.best_iteration}")
    print(f"   - RMSE: {rmse:.6f}")
    print(f"   - MAE: {mae:.6f}")
    print(f"   - R²: {r2:.4f}")
    
    # -------------------------
    # Train Classifier
    # -------------------------
    print("\n📊 Training LightGBM Classifier...")
    
    train_clf = lgb.Dataset(X_train, label=y_train_binary)
    val_clf = lgb.Dataset(X_val, label=y_val_binary, reference=train_clf)
    
    classifier = lgb.train(
        clf_params,
        train_clf,
        num_boost_round=1000,
        valid_sets=[train_clf, val_clf],
        valid_names=['train', 'val'],
        callbacks=[
            lgb.early_stopping(100, verbose=True),
            lgb.log_evaluation(period=0)
        ]
    )
    
    # Evaluate classifier
    y_val_pred_proba = classifier.predict(X_val)
    y_val_pred = (y_val_pred_proba > 0.5).astype(int)
    accuracy = accuracy_score(y_val_binary, y_val_pred)
    f1 = f1_score(y_val_binary, y_val_pred)
    
    print(f"\n   Classifier Results:")
    print(f"   - Best iteration: {classifier.best_iteration}")
    print(f"   - Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   - F1 Score: {f1:.4f}")
    
    print("\n✅ LightGBM training complete")
    
    return regressor, classifier


# ============================================================
# Evaluation
# ============================================================

def evaluate_models(
    regressor: lgb.Booster,
    classifier: lgb.Booster,
    X_test: np.ndarray,
    y_test_pct: np.ndarray,
    y_test_binary: np.ndarray,
    feature_names: List[str]
) -> Dict:
    """
    Evaluate trained models on test set.
    
    Returns:
        Dictionary of evaluation metrics
    """
    print("\n" + "=" * 60)
    print("📋 EVALUATION: Testing Model Performance")
    print("=" * 60)
    
    print("\nMaking predictions on test set...")
    
    # Regression predictions
    y_pred_pct = regressor.predict(X_test)
    
    # Classification predictions
    y_pred_proba = classifier.predict(X_test)
    y_pred_binary = (y_pred_proba > 0.5).astype(int)
    
    # -------------------------
    # Regression metrics
    # -------------------------
    rmse = np.sqrt(mean_squared_error(y_test_pct, y_pred_pct))
    mae = mean_absolute_error(y_test_pct, y_pred_pct)
    mape = np.mean(np.abs((y_test_pct - y_pred_pct) / (y_test_pct + 1e-10))) * 100
    r2 = r2_score(y_test_pct, y_pred_pct)
    
    print(f"📈 Regression Metrics:")
    print(f"   RMSE:  {rmse:.6f}")
    print(f"   MAE:   {mae:.6f}")
    print(f"   MAPE:  {mape:.2f}%")
    print(f"   R²:    {r2:.4f}")
    
    # -------------------------
    # Classification metrics
    # -------------------------
    accuracy = accuracy_score(y_test_binary, y_pred_binary)
    f1 = f1_score(y_test_binary, y_pred_binary)
    precision = precision_score(y_test_binary, y_pred_binary)
    recall = recall_score(y_test_binary, y_pred_binary)
    
    print(f"\n📊 Classification Metrics (Binary: Down/Up):")
    print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   F1 Score:  {f1:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    
    # Per-class accuracy
    cm = confusion_matrix(y_test_binary, y_pred_binary)
    class_accuracy = cm.diagonal() / cm.sum(axis=1)
    
    print(f"\n📊 Per-Class Accuracy:")
    print(f"   Down: {class_accuracy[0]:.4f} ({class_accuracy[0]*100:.2f}%)")
    print(f"   Up: {class_accuracy[1]:.4f} ({class_accuracy[1]*100:.2f}%)")
    
    # Confusion matrix
    print(f"\n📊 Confusion Matrix:")
    print(f"          Predicted")
    print(f"              Down      Up")
    print(f"Actual Down: [{cm[0,0]:4d} {cm[0,1]:4d}]")
    print(f"Actual Up  : [{cm[1,0]:4d} {cm[1,1]:4d}]")
    
    # -------------------------
    # Feature importances
    # -------------------------
    clf_importance = classifier.feature_importance(importance_type='gain')
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': clf_importance
    }).sort_values('importance', ascending=False)
    
    print(f"\n📊 Top 10 Feature Importances:")
    for _, row in importance_df.head(10).iterrows():
        print(f"   {row['feature']}: {row['importance']:.2f}")
    
    metrics = {
        'regression': {
            'rmse': float(rmse),
            'mae': float(mae),
            'mape': float(mape),
            'r2': float(r2)
        },
        'classification': {
            'accuracy': float(accuracy),
            'f1': float(f1),
            'precision': float(precision),
            'recall': float(recall),
            'down_accuracy': float(class_accuracy[0]),
            'up_accuracy': float(class_accuracy[1]),
            'confusion_matrix': cm.tolist()
        },
        'feature_importance': importance_df.to_dict('records')
    }
    
    return metrics


# ============================================================
# Save Models
# ============================================================

def save_models(
    regressor: lgb.Booster,
    classifier: lgb.Booster,
    scaler: StandardScaler,
    feature_names: List[str],
    config: Dict,
    metrics: Dict,
    boruta_results: Optional[Dict] = None,
    optuna_results: Optional[Dict] = None,
    output_dir: Path = MODEL_DIR
) -> None:
    """
    Save trained models and metadata.
    """
    print("\n" + "-" * 60)
    print("💾 Saving models...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save LightGBM models
    reg_path = output_dir / "lgb_regressor.txt"
    clf_path = output_dir / "lgb_classifier.txt"
    
    regressor.save_model(str(reg_path))
    classifier.save_model(str(clf_path))
    
    print(f"💾 LightGBM regressor saved to: {reg_path}")
    print(f"💾 LightGBM classifier saved to: {clf_path}")
    
    # Save metadata
    metadata = {
        'scaler': scaler,
        'feature_names': feature_names,
        'n_features': len(feature_names),
        'created_at': datetime.now().isoformat()
    }
    
    meta_path = output_dir / "model_metadata.pkl"
    with open(meta_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"💾 Metadata saved to: {meta_path}")
    
    print(f"\n✅ All models saved to: {output_dir}")
    
    # Save config
    config_path = output_dir / "training_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"💾 Config saved to: {config_path}")
    
    # Save metrics
    metrics_path = output_dir / "evaluation_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"💾 Metrics saved to: {metrics_path}")
    
    # Save Boruta results if available
    if boruta_results:
        boruta_path = output_dir / "boruta_results.json"
        with open(boruta_path, 'w') as f:
            json.dump(boruta_results, f, indent=2)
        print(f"💾 Boruta results saved to: {boruta_path}")
    
    # Save Optuna results if available
    if optuna_results:
        optuna_path = output_dir / "optuna_results.json"
        with open(optuna_path, 'w') as f:
            json.dump(optuna_results, f, indent=2)
        print(f"💾 Optuna results saved to: {optuna_path}")


# ============================================================
# Main
# ============================================================

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train ServoTrader Prediction Model')
    
    # Data arguments
    parser.add_argument('--data', type=str, default=None,
                        help='Path to training data CSV')
    parser.add_argument('--horizon', type=int, default=DEFAULT_HORIZON,
                        help='Prediction horizon in minutes (1440 = 1 day)')
    parser.add_argument('--threshold', type=float, default=DEFAULT_DIRECTION_THRESHOLD,
                        help='Direction classification threshold')
    
    # Feature arguments
    parser.add_argument('--no-sentiment', action='store_true',
                        help='Disable Fear & Greed sentiment features')
    parser.add_argument('--no-onchain', action='store_true',
                        help='Disable on-chain blockchain features')
    
    # Optimization arguments
    parser.add_argument('--no-boruta', action='store_true',
                        help='Disable Boruta feature selection')
    parser.add_argument('--boruta-trials', type=int, default=100,
                        help='Max Boruta iterations')
    parser.add_argument('--no-optuna', action='store_true',
                        help='Disable Optuna hyperparameter optimization')
    parser.add_argument('--optuna-trials', type=int, default=50,
                        help='Number of Optuna trials')
    
    # Output arguments
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for models')
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "=" * 60)
    print("🚀 ServoTrader Prediction Model Training (v2.5 - On-Chain Data)")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Determine data path
    if args.data:
        data_path = args.data
    else:
        # Default: use daily data for daily predictions
        if args.horizon >= 1440:
            data_path = DAILY_DATA_DIR / "BTCUSDT.csv"
        else:
            data_path = DATA_DIR / "BTCUSDT_5m.csv"
    
    # Check if data exists
    if not Path(data_path).exists():
        print(f"❌ Data file not found: {data_path}")
        sys.exit(1)
    
    # Print settings
    horizon_hours = args.horizon / 60
    print(f"\n📋 Settings:")
    print(f"   Prediction horizon: {args.horizon} minutes ({horizon_hours:.1f} hours)")
    
    if args.horizon >= 1440:
        print(f"   📅 Daily resampling: ENABLED (horizon >= 1 day)")
        print(f"   Prediction horizon: {args.horizon // 1440} day(s)")
    
    print(f"   Direction threshold: ±{args.threshold*100:.1f}%")
    print(f"   Include sentiment: {not args.no_sentiment}")
    print(f"   Include on-chain: {not args.no_onchain}")
    print(f"   Use Boruta: {not args.no_boruta and BORUTA_AVAILABLE}")
    if not args.no_boruta and BORUTA_AVAILABLE:
        print(f"   Boruta trials: {args.boruta_trials}")
    print(f"   Use Optuna: {not args.no_optuna and OPTUNA_AVAILABLE}")
    if not args.no_optuna and OPTUNA_AVAILABLE:
        print(f"   Optuna trials: {args.optuna_trials}")
    
    print("\n" + "-" * 60 + "\n")
    
    # -------------------------
    # Prepare data
    # -------------------------
    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     y_train_binary, y_val_binary, y_test_binary,
     scaler, feature_names, df) = prepare_data(
        data_path=str(data_path),
        horizon_minutes=args.horizon,
        direction_threshold=args.threshold,
        include_sentiment=not args.no_sentiment,
        include_onchain=not args.no_onchain
    )
    
    # Track which features are used
    selected_features = feature_names.copy()
    boruta_results = None
    optuna_results = None
    optuna_params = None
    
    # -------------------------
    # Boruta feature selection
    # -------------------------
    if not args.no_boruta and BORUTA_AVAILABLE:
        selected_mask, selected_features, boruta_results = run_boruta_feature_selection(
            X_train, y_train_binary, feature_names,
            max_iter=args.boruta_trials
        )
        
        # Apply feature selection
        print("\n📏 Applying feature selection...")
        print(f"   Original features: {X_train.shape[1]}")
        X_train = X_train[:, selected_mask]
        X_val = X_val[:, selected_mask]
        X_test = X_test[:, selected_mask]
        print(f"   Selected features: {X_train.shape[1]}")
        print(f"   New feature shape: {X_train.shape}")
    
    # -------------------------
    # Optuna optimization
    # -------------------------
    if not args.no_optuna and OPTUNA_AVAILABLE:
        optuna_params, optuna_results = run_optuna_optimization(
            X_train, y_train_binary, X_val, y_val_binary,
            n_trials=args.optuna_trials
        )
    
    # -------------------------
    # Train models
    # -------------------------
    regressor, classifier = train_lightgbm_models(
        X_train, y_train, y_train_binary,
        X_val, y_val, y_val_binary,
        optuna_params=optuna_params
    )
    
    # -------------------------
    # Evaluate
    # -------------------------
    metrics = evaluate_models(
        regressor, classifier,
        X_test, y_test, y_test_binary,
        selected_features
    )
    
    # -------------------------
    # Save models
    # -------------------------
    output_dir = Path(args.output) if args.output else MODEL_DIR
    
    config = {
        'version': '2.5',
        'horizon_minutes': args.horizon,
        'direction_threshold': args.threshold,
        'include_sentiment': not args.no_sentiment,
        'include_onchain': not args.no_onchain,
        'use_boruta': not args.no_boruta and BORUTA_AVAILABLE,
        'use_optuna': not args.no_optuna and OPTUNA_AVAILABLE,
        'n_features': len(selected_features),
        'feature_names': selected_features,
        'data_path': str(data_path)
    }
    
    save_models(
        regressor, classifier, scaler, selected_features,
        config, metrics,
        boruta_results=boruta_results,
        optuna_results=optuna_results,
        output_dir=output_dir
    )
    
    # -------------------------
    # Print summary
    # -------------------------
    print("\n" + "=" * 60)
    print("✅ Training Complete!")
    print("=" * 60)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nModel saved to: {output_dir}")
    
    print(f"\n📊 Summary:")
    print(f"   Test Accuracy: {metrics['classification']['accuracy']*100:.2f}%")
    print(f"   Test F1: {metrics['classification']['f1']:.4f}")
    print(f"   Test R²: {metrics['regression']['r2']:.4f}")
    print(f"   Features used: {len(selected_features)} "
          f"(sentiment: {not args.no_sentiment}, "
          f"onchain: {not args.no_onchain}, "
          f"boruta: {not args.no_boruta and BORUTA_AVAILABLE}, "
          f"optuna: {not args.no_optuna and OPTUNA_AVAILABLE})")
    
    if boruta_results:
        print(f"\n📊 Boruta Summary:")
        print(f"   Confirmed features: {boruta_results['n_confirmed']}")
        print(f"   Tentative features: {boruta_results['n_tentative']}")
        print(f"   Rejected features: {boruta_results['n_rejected']}")
    
    if optuna_results:
        print(f"\n📊 Optuna Summary:")
        print(f"   Best validation accuracy: {optuna_results['best_accuracy']*100:.2f}%")
        print(f"   Trials completed: {optuna_results['n_trials']}")
        print(f"   Best trial: #{optuna_results['best_trial']}")


if __name__ == "__main__":
    main()