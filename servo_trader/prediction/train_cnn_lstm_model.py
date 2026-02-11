#!/usr/bin/env python3
"""
ServoTrader Prediction Model Training Script (v3.1 - Simplified CNN-LSTM)

Key improvements over v3.0:
- Simplified architecture (~40K parameters vs 176K)
- Aggressive Boruta feature selection (reduces to 30-40 features)
- Better regularization to prevent overfitting
- Shorter sequence length for more training samples
- Optuna hyperparameter optimization with TimeSeriesSplit cross-validation
- Support for on-chain only mode

Architecture Changes from v3.0 to v3.1:
- CNN layers: [32, 64] filters (was [64, 128])
- LSTM layers: [50, 25] units (was [100, 50])
- Dense layer: [32] units (was [64, 32])
- Dropout: 0.5 (was 0.3)
- L2 regularization: 0.01 (was 0.001)
- Sequence length: 15 (was 30)
- Target features: 30-40 (was 117)

Usage:
    # Default training (all features)
    python train_cnn_lstm_v31.py --horizon 1440
    
    # ON-CHAIN ONLY MODE - Use only blockchain features
    python train_cnn_lstm_v31.py --horizon 1440 --onchain-only --optuna
    
    # With Optuna hyperparameter optimization
    python train_cnn_lstm_v31.py --horizon 1440 --optuna --optuna-trials 20
    
    # Without Boruta (use all features)
    python train_cnn_lstm_v31.py --horizon 1440 --no-boruta
    
    # Custom feature selection
    python train_cnn_lstm_v31.py --horizon 1440 --no-technical --no-sentiment

Author: ServoTrader
Version: 3.1 (Simplified CNN-LSTM with Boruta + Optuna CV)
"""

import os
import sys
import argparse
import pickle
import json
import yaml
import shutil
import warnings
from datetime import datetime
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any

# Parse --cpu flag BEFORE importing TensorFlow
if '--cpu' in sys.argv:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    print("🖥️  CPU-only mode enabled (--cpu flag)")

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Early Boruta check (before TensorFlow to catch issues early)
def check_boruta_installation():
    """Check if Boruta is properly installed and compatible."""
    try:
        from boruta import BorutaPy
        # Quick compatibility test
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        boruta = BorutaPy(rf, n_estimators='auto', max_iter=10, random_state=42)
        print("✅ Boruta installation verified")
        return True
    except ImportError as e:
        print(f"⚠️  Boruta ImportError: {e}")
        print("   Try: pip install Boruta")
        return False
    except Exception as e:
        print(f"⚠️  Boruta compatibility issue: {type(e).__name__}: {e}")
        print("   This may be a numpy/scikit-learn version conflict")
        return False

BORUTA_PRECHECK = check_boruta_installation()

# Check for Optuna
def check_optuna_installation():
    """Check if Optuna is properly installed."""
    try:
        import optuna
        print(f"✅ Optuna {optuna.__version__} available")
        return True
    except ImportError:
        print("⚠️  Optuna not installed. Hyperparameter tuning disabled.")
        print("   Install with: pip install optuna")
        return False

OPTUNA_AVAILABLE = check_optuna_installation()

# Import optuna if available (for use in objective function)
if OPTUNA_AVAILABLE:
    import optuna

# Check for on-chain data module
def check_onchain_module():
    """Check if on-chain data module is available."""
    try:
        from onchain_data import load_onchain_data, OnChainDataFetcher
        print("✅ On-chain data module found")
        return True
    except ImportError:
        # Try relative import
        try:
            import importlib.util
            import sys
            # Check same directory as this script
            script_dir = Path(__file__).parent
            onchain_path = script_dir / "onchain_data.py"
            if onchain_path.exists():
                spec = importlib.util.spec_from_file_location("onchain_data", onchain_path)
                module = importlib.util.module_from_spec(spec)
                sys.modules['onchain_data'] = module
                spec.loader.exec_module(module)
                print("✅ On-chain data module found (same directory)")
                return True
        except Exception:
            pass
        print("⚠️  On-chain data module not found (onchain_data.py)")
        print("   On-chain features will be disabled")
        return False

ONCHAIN_AVAILABLE = check_onchain_module()

# Import TensorFlow/Keras
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model, regularizers
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    TF_AVAILABLE = True
    KERAS_VERSION = tf.__version__
    
    # Configure GPU memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ TensorFlow {KERAS_VERSION} with GPU support")
    else:
        print(f"✅ TensorFlow {KERAS_VERSION} (CPU mode)")
        
except ImportError:
    TF_AVAILABLE = False
    print("❌ TensorFlow not installed. Please run: pip install tensorflow")
    sys.exit(1)


# ============================================================
# Configuration - v3.1 Simplified Defaults
# ============================================================

# Default paths (update these for your project structure)
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models" / "prediction"
DAILY_DATA_DIR = DATA_DIR / "daily_historical"
SENTIMENT_CACHE_DIR = Path.home() / ".servo_trader" / "sentiment_cache"

# v3.1 Simplified Defaults (vs v3.0)
DEFAULT_HORIZON = 1440             # 1 day in minutes
DEFAULT_SEQUENCE_LENGTH = 15       # was 30 - shorter for more samples
DEFAULT_DIRECTION_THRESHOLD = 0.001
DEFAULT_TRAIN_RATIO = 0.7
DEFAULT_VAL_RATIO = 0.15
DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 32

# v3.1 Simplified Architecture
DEFAULT_CNN_FILTERS = "32,64"      # was "64,128"
DEFAULT_LSTM_UNITS = "50,25"       # was "100,50"
DEFAULT_DENSE_UNITS = "32"         # was "64,32"
DEFAULT_DROPOUT = 0.5              # was 0.3
DEFAULT_L2_REG = 0.01              # was 0.001
DEFAULT_LEARNING_RATE = 0.001

# Boruta defaults
DEFAULT_MAX_FEATURES = 40          # Target 30-40 features
DEFAULT_BORUTA_MAX_ITER = 100
DEFAULT_BORUTA_PERC = 90


# ============================================================
# Boruta Feature Selection
# ============================================================

def boruta_feature_selection(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    max_features: int = 40,
    max_iter: int = 100,
    perc: int = 90,
    verbose: bool = True
) -> Tuple[np.ndarray, List[str], List[int]]:
    """
    Perform aggressive Boruta feature selection.
    
    Args:
        X: Feature matrix (2D: samples x features)
        y: Target labels
        feature_names: List of feature names
        max_features: Maximum features to keep
        max_iter: Boruta iterations
        perc: Percentile for shadow feature comparison
        verbose: Print progress
        
    Returns:
        X_selected: Selected features
        selected_names: Names of selected features
        selected_indices: Indices of selected features
    """
    print("\n" + "=" * 60)
    print("🎯 Boruta Feature Selection (Aggressive)")
    print("=" * 60)
    print(f"   Input features: {X.shape[1]}")
    print(f"   Target max features: {max_features}")
    print(f"   Max iterations: {max_iter}")
    print(f"   Percentile: {perc}")
    
    # Try to import Boruta (use precheck result)
    BORUTA_AVAILABLE = BORUTA_PRECHECK
    
    if BORUTA_AVAILABLE:
        try:
            from boruta import BorutaPy
            print("   ✅ Using Boruta feature selection")
        except Exception as e:
            BORUTA_AVAILABLE = False
            print(f"   ⚠️  Boruta import failed at runtime: {e}")
    else:
        print("   ⚠️  Boruta not available. Using Random Forest importance fallback.")
    
    if BORUTA_AVAILABLE:
        try:
            # Use Random Forest as the estimator
            rf = RandomForestClassifier(
                n_estimators=100,
                max_depth=7,
                n_jobs=-1,
                random_state=42,
                class_weight='balanced'
            )
            
            # Initialize Boruta with aggressive settings
            boruta = BorutaPy(
                rf,
                n_estimators='auto',
                perc=perc,
                max_iter=max_iter,
                random_state=42,
                verbose=0
            )
            
            # Fit Boruta
            print("\n   Running Boruta feature selection...")
            boruta.fit(X, y)
            
            # Get confirmed and tentative features
            confirmed = np.where(boruta.support_)[0]
            tentative = np.where(boruta.support_weak_)[0]
            
            print(f"\n   ✅ Confirmed features: {len(confirmed)}")
            print(f"   ⚠️  Tentative features: {len(tentative)}")
            print(f"   ❌ Rejected features: {X.shape[1] - len(confirmed) - len(tentative)}")
            
            # Include both confirmed and tentative
            selected_indices = np.sort(np.concatenate([confirmed, tentative]))
            
            # If we have more than max_features, rank by importance and take top N
            if len(selected_indices) > max_features:
                print(f"\n   📉 Reducing from {len(selected_indices)} to {max_features} features...")
                # Fit RF on selected features to get importance ranking
                X_sel = X[:, selected_indices]
                rf.fit(X_sel, y)
                importances = rf.feature_importances_
                # Get top N by importance
                top_idx = np.argsort(importances)[-max_features:]
                selected_indices = selected_indices[top_idx]
            
            # If we have fewer than desired, that's fine (Boruta was strict)
            if len(selected_indices) < 10:
                print(f"\n   ⚠️  Boruta was very strict ({len(selected_indices)} features).")
                print("   📈 Falling back to top features by RF importance...")
                # Fall through to importance-based selection
                raise ValueError("Too few features selected")
                
        except Exception as e:
            print(f"\n   ⚠️  Boruta failed: {e}")
            print("   📈 Using Random Forest importance fallback...")
            BORUTA_AVAILABLE = False
    
    if not BORUTA_AVAILABLE:
        # Fallback: Use Random Forest importance
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=7,
            n_jobs=-1,
            random_state=42,
            class_weight='balanced'
        )
        rf.fit(X, y)
        importances = rf.feature_importances_
        
        # Select top N features
        n_select = min(max_features, X.shape[1])
        selected_indices = np.argsort(importances)[-n_select:]
        selected_indices = np.sort(selected_indices)
        print(f"   Selected top {len(selected_indices)} features by importance")
    
    # Extract selected features
    selected_names = [feature_names[i] for i in selected_indices]
    X_selected = X[:, selected_indices]
    
    print(f"\n   📊 Final feature count: {len(selected_names)}")
    print("\n   Top 10 selected features:")
    
    # Show feature importances for selected features
    rf_final = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    rf_final.fit(X_selected, y)
    final_importances = rf_final.feature_importances_
    sorted_idx = np.argsort(final_importances)[::-1]
    
    for i, idx in enumerate(sorted_idx[:10]):
        print(f"      {i+1}. {selected_names[idx]}: {final_importances[idx]:.4f}")
    
    print("=" * 60 + "\n")
    
    return X_selected, selected_names, list(selected_indices)


# ============================================================
# Sentiment Data Functions
# ============================================================

def fetch_fear_greed_index(use_cache: bool = True, cache_max_age_hours: int = 24) -> pd.DataFrame:
    """Fetch Fear & Greed Index from Alternative.me API."""
    cache_file = SENTIMENT_CACHE_DIR / "fear_greed_index.csv"
    
    # Check cache
    if use_cache and cache_file.exists():
        cache_age = (datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime))
        if cache_age.total_seconds() < cache_max_age_hours * 3600:
            print(f"   📁 Loading cached sentiment data ({cache_age.seconds // 3600}h old)")
            return pd.read_csv(cache_file, parse_dates=['timestamp'])
    
    print("   🌐 Fetching Fear & Greed Index from API...")
    
    try:
        import requests
        url = "https://api.alternative.me/fng/?limit=0&format=json"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        records = []
        for item in data.get('data', []):
            records.append({
                'timestamp': pd.to_datetime(int(item['timestamp']), unit='s'),
                'fear_greed_value': int(item['value']),
                'fear_greed_class': item['value_classification']
            })
        
        df = pd.DataFrame(records)
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Cache the data
        SENTIMENT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_file, index=False)
        print(f"   ✅ Retrieved {len(df)} days of sentiment data")
        
        return df
        
    except Exception as e:
        print(f"   ⚠️  Failed to fetch sentiment data: {e}")
        if cache_file.exists():
            print("   📁 Using existing cache as fallback")
            return pd.read_csv(cache_file, parse_dates=['timestamp'])
        return pd.DataFrame()


def add_sentiment_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Fear & Greed sentiment features to dataframe.
    
    Note: Fear & Greed Index only available from ~2018. For older dates,
    values are forward/backward filled from available data.
    """
    sentiment_df = fetch_fear_greed_index()
    
    if sentiment_df.empty:
        print("   ⚠️  No sentiment data available, adding neutral defaults...")
        # Add neutral sentiment for all rows
        df['fear_greed_value'] = 50.0
        df['fear_greed_encoded'] = 2  # Neutral
        df['fear_greed_normalized'] = 0.5
        df['fear_greed_ma7'] = 0.5
        df['fear_greed_change'] = 0.0
        return df
    
    # Ensure timestamp is datetime and timezone-naive
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    if df['timestamp'].dt.tz is not None:
        df['timestamp'] = df['timestamp'].dt.tz_localize(None)
    
    sentiment_df['timestamp'] = pd.to_datetime(sentiment_df['timestamp'])
    if sentiment_df['timestamp'].dt.tz is not None:
        sentiment_df['timestamp'] = sentiment_df['timestamp'].dt.tz_localize(None)
    
    # Get date ranges
    data_start = df['timestamp'].min()
    data_end = df['timestamp'].max()
    sentiment_start = sentiment_df['timestamp'].min()
    sentiment_end = sentiment_df['timestamp'].max()
    
    # Create date column for merging
    df['date'] = df['timestamp'].dt.date
    sentiment_df['date'] = sentiment_df['timestamp'].dt.date
    
    # Merge on date
    sentiment_cols = ['date', 'fear_greed_value', 'fear_greed_class']
    df = df.merge(sentiment_df[sentiment_cols], on='date', how='left')
    
    # Count direct coverage before filling
    direct_coverage = df['fear_greed_value'].notna().sum()
    total_rows = len(df)
    
    # Fill missing values (for dates before/after sentiment data available)
    df['fear_greed_value'] = df['fear_greed_value'].ffill().bfill().fillna(50)
    
    # Create numeric encoding for class
    class_map = {
        'Extreme Fear': 0, 'Fear': 1, 'Neutral': 2, 'Greed': 3, 'Extreme Greed': 4
    }
    df['fear_greed_encoded'] = df['fear_greed_class'].map(class_map).fillna(2)
    
    # Normalize fear/greed to 0-1
    df['fear_greed_normalized'] = df['fear_greed_value'] / 100.0
    
    # Calculate sentiment momentum
    df['fear_greed_ma7'] = df['fear_greed_value'].rolling(7, min_periods=1).mean() / 100.0
    df['fear_greed_change'] = df['fear_greed_value'].pct_change().fillna(0).clip(-1, 1)
    
    # Drop intermediate columns
    df = df.drop(columns=['date', 'fear_greed_class'], errors='ignore')
    
    coverage = direct_coverage / total_rows * 100
    print(f"   ✅ Sentiment coverage: {coverage:.1f}% direct, 100% after filling")
    
    if data_start < sentiment_start:
        print(f"   ℹ️  Note: Data starts {data_start.strftime('%Y-%m-%d')}, "
              f"but sentiment only from {sentiment_start.strftime('%Y-%m-%d')}")
        print(f"   ℹ️  Older dates filled with earliest available sentiment values")
    
    return df


# ============================================================
# On-Chain Data Integration
# ============================================================

def add_onchain_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add on-chain blockchain metrics to the dataframe.
    
    Fetches data from Blockchain.com API (free, no API key required).
    Includes: hash rate, active addresses, transactions, fees, difficulty, etc.
    """
    if not ONCHAIN_AVAILABLE:
        print("   ⚠️  On-chain module not available, skipping...")
        return df
    
    try:
        from onchain_data import load_onchain_data
    except ImportError:
        print("   ⚠️  Could not import on-chain module, skipping...")
        return df
    
    print("\n🔗 Adding on-chain blockchain features...")
    
    # Load on-chain data with derived features
    onchain_df = load_onchain_data(compute_features=True, use_cache=True)
    
    if onchain_df.empty:
        print("   ⚠️  No on-chain data available, skipping...")
        return df
    
    # Store original row count
    original_rows = len(df)
    
    # Ensure timestamp columns are datetime and timezone-naive
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    if df['timestamp'].dt.tz is not None:
        df['timestamp'] = df['timestamp'].dt.tz_localize(None)
    
    onchain_df['date'] = pd.to_datetime(onchain_df['date'])
    if onchain_df['date'].dt.tz is not None:
        onchain_df['date'] = onchain_df['date'].dt.tz_localize(None)
    
    # Create date column for merging (date only, no time) - ensure timezone-naive
    df['_merge_date'] = df['timestamp'].dt.normalize()
    onchain_df['_merge_date'] = onchain_df['date'].dt.normalize()
    
    # Remove duplicate dates in on-chain data (keep last value per day)
    onchain_df = onchain_df.drop_duplicates(subset=['_merge_date'], keep='last')
    
    # Get on-chain feature columns (exclude 'date' and '_merge_date')
    onchain_cols = [c for c in onchain_df.columns if c not in ['date', '_merge_date']]
    
    # Merge on normalized date - use LEFT join to preserve original data
    merge_cols = ['_merge_date'] + onchain_cols
    df = df.merge(onchain_df[merge_cols], on='_merge_date', how='left')
    
    # Verify no row multiplication
    if len(df) != original_rows:
        print(f"   ⚠️  Row count changed during merge: {original_rows} -> {len(df)}")
        print(f"   ⚠️  This indicates duplicate dates in on-chain data. Fixing...")
        # If rows increased, there were duplicate merge keys - take first occurrence
        df = df.drop_duplicates(subset=['timestamp'], keep='first')
        print(f"   ✅ Fixed row count: {len(df)}")
    
    # Fill missing values with forward/backward fill
    for col in onchain_cols:
        if col in df.columns:
            df[col] = df[col].ffill().bfill().fillna(0)
    
    # Drop the temporary merge column
    df = df.drop(columns=['_merge_date'], errors='ignore')
    
    # Count coverage
    sample_col = onchain_cols[0] if onchain_cols else None
    if sample_col and sample_col in df.columns:
        coverage = df[sample_col].notna().sum() / len(df) * 100
        print(f"   ✅ On-chain coverage: {coverage:.1f}%")
        print(f"   📊 Added {len(onchain_cols)} on-chain features")
        print(f"   📋 Final row count: {len(df)} (unchanged from original)")
    
    return df


# ============================================================
# Technical Indicators
# ============================================================

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators to dataframe."""
    df = df.copy()
    
    # Ensure we have the required columns
    required = ['open', 'high', 'low', 'close', 'volume']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Price-based features
    df['returns'] = df['close'].pct_change().fillna(0)
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1)).fillna(0)
    
    # Moving averages
    for period in [5, 10, 20]:
        df[f'sma_{period}'] = df['close'].rolling(period, min_periods=1).mean()
        df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        df[f'close_to_sma_{period}'] = df['close'] / df[f'sma_{period}']
    
    # Volatility
    df['volatility_5'] = df['returns'].rolling(5, min_periods=1).std()
    df['volatility_10'] = df['returns'].rolling(10, min_periods=1).std()
    df['volatility_20'] = df['returns'].rolling(20, min_periods=1).std()
    
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
    rs = gain / (loss + 1e-10)
    df['rsi_14'] = 100 - (100 / (1 + rs))
    df['rsi_14'] = df['rsi_14'].fillna(50)
    
    # MACD
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(20, min_periods=1).mean()
    bb_std = df['close'].rolling(20, min_periods=1).std()
    df['bb_upper'] = df['bb_middle'] + 2 * bb_std
    df['bb_lower'] = df['bb_middle'] - 2 * bb_std
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
    
    # ATR
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift(1))
    low_close = abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(14, min_periods=1).mean()
    df['atr_normalized'] = df['atr_14'] / df['close']
    
    # Volume features
    df['volume_sma_10'] = df['volume'].rolling(10, min_periods=1).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_sma_10'] + 1)
    df['volume_change'] = df['volume'].pct_change().fillna(0).clip(-5, 5)
    
    # Price patterns
    df['high_low_range'] = (df['high'] - df['low']) / df['close']
    df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)
    
    # Momentum
    for period in [5, 10, 20]:
        df[f'momentum_{period}'] = df['close'].pct_change(period).fillna(0)
    
    # Fill any remaining NaN
    df = df.ffill().bfill()
    
    return df


# ============================================================
# Data Preparation
# ============================================================

def create_sequences(X: np.ndarray, y: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create sequences for LSTM input."""
    sequences = []
    labels = []
    
    for i in range(len(X) - seq_length):
        sequences.append(X[i:i + seq_length])
        labels.append(y[i + seq_length])
    
    return np.array(sequences), np.array(labels)


def prepare_data(
    data_path: str,
    horizon_minutes: int = 1440,
    sequence_length: int = 15,
    direction_threshold: float = 0.001,
    include_technical: bool = True,
    include_sentiment: bool = True,
    include_onchain: bool = True,
    use_boruta: bool = True,
    max_features: int = 40,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           RobustScaler, List[str], pd.DataFrame, List[int]]:
    """
    Prepare data for training with optional Boruta feature selection.
    
    Returns:
        X_train, X_val, X_test: Sequence data (3D: samples, seq_length, features)
        y_train, y_val, y_test: Labels
        scaler: Fitted scaler
        feature_names: List of feature names after selection
        df: Processed dataframe
        selected_indices: Indices of selected features (if Boruta used)
    """
    print("\n" + "=" * 60)
    print("📊 Data Preparation (v3.1)")
    print("=" * 60)
    
    # Load data
    print(f"\n📂 Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"   Rows: {len(df):,}")
    print(f"   Columns: {list(df.columns)}")
    
    # Ensure timestamp
    if 'timestamp' not in df.columns:
        if 'date' in df.columns:
            df['timestamp'] = pd.to_datetime(df['date'])
        else:
            df['timestamp'] = pd.date_range(start='2015-01-01', periods=len(df), freq='D')
    else:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Standardize column names
    col_map = {
        'Open': 'open', 'High': 'high', 'Low': 'low',
        'Close': 'close', 'Volume': 'volume', 'VWAP': 'vwap'
    }
    df = df.rename(columns=col_map)
    
    # Ensure required columns exist
    required = ['open', 'high', 'low', 'close', 'volume']
    for col in required:
        if col not in df.columns:
            if col == 'volume':
                df['volume'] = 1000000  # Default volume
            else:
                raise ValueError(f"Missing required column: {col}")
    
    # Add technical indicators (unless disabled)
    if include_technical:
        print("\n🔧 Adding technical indicators...")
        df = add_technical_indicators(df)
    else:
        print("\n⏭️  Skipping technical indicators (disabled)")
        # Keep only basic OHLCV columns for target calculation
        # We need 'close' for target creation
    
    # Add sentiment features
    if include_sentiment:
        print("\n📰 Adding sentiment features...")
        df = add_sentiment_features(df)
    else:
        print("\n⏭️  Skipping sentiment features (disabled)")
    
    # Add on-chain features
    if include_onchain:
        print("\n🔗 Adding on-chain features...")
        df = add_onchain_features(df)
    
    # Create target variable (direction)
    print(f"\n🎯 Creating target (horizon={horizon_minutes} min, threshold={direction_threshold})")
    
    if horizon_minutes >= 1440:
        # Daily: use next day's close
        df['future_return'] = df['close'].pct_change().shift(-1)
    else:
        # Intraday: use N-minute return
        periods = horizon_minutes
        df['future_return'] = df['close'].pct_change(periods).shift(-periods)
    
    # Binary direction: 1 = up, 0 = down
    df['direction'] = (df['future_return'] > direction_threshold).astype(int)
    
    # Remove rows with NaN target
    df = df.dropna(subset=['direction', 'future_return'])
    
    up_pct = df['direction'].mean() * 100
    print(f"   Class balance: Up={up_pct:.1f}%, Down={100-up_pct:.1f}%")
    
    # Sanity check: Bitcoin typically has 45-55% up days
    if up_pct < 10 or up_pct > 90:
        print(f"\n   ⚠️  WARNING: Class balance looks wrong!")
        print(f"   ⚠️  Expected ~45-55% up days, got {up_pct:.1f}%")
        print(f"   ⚠️  This may indicate a data processing error.")
        print(f"   ⚠️  Checking data integrity...")
        
        # Debug: Show sample of returns
        print(f"\n   📊 Debug - Future return stats:")
        print(f"      Mean: {df['future_return'].mean()*100:.4f}%")
        print(f"      Std: {df['future_return'].std()*100:.4f}%")
        print(f"      Min: {df['future_return'].min()*100:.4f}%")
        print(f"      Max: {df['future_return'].max()*100:.4f}%")
        print(f"      Sample values: {df['future_return'].head(10).tolist()}")
    
    # Select feature columns (exclude metadata and target)
    exclude_cols = ['timestamp', 'date', 'direction', 'future_return', 'symbol']
    feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ['float64', 'int64', 'float32', 'int32']]
    
    print(f"\n📋 Initial features: {len(feature_cols)}")
    
    # Prepare feature matrix
    X_raw = df[feature_cols].values
    y_raw = df['direction'].values
    
    # Temporal train/val/test split
    n_samples = len(X_raw)
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))
    
    X_train_raw = X_raw[:train_end]
    y_train_raw = y_raw[:train_end]
    X_val_raw = X_raw[train_end:val_end]
    y_val_raw = y_raw[train_end:val_end]
    X_test_raw = X_raw[val_end:]
    y_test_raw = y_raw[val_end:]
    
    print(f"\n✂️  Temporal split (before sequences):")
    print(f"   Train: {len(X_train_raw):,} | Val: {len(X_val_raw):,} | Test: {len(X_test_raw):,}")
    
    # Apply Boruta feature selection on training data
    selected_indices = list(range(len(feature_cols)))  # Default: all features
    
    if use_boruta:
        X_train_selected, selected_names, selected_indices = boruta_feature_selection(
            X_train_raw, y_train_raw, feature_cols,
            max_features=max_features,
            max_iter=DEFAULT_BORUTA_MAX_ITER,
            perc=DEFAULT_BORUTA_PERC
        )
        feature_cols = selected_names
        
        # Apply same selection to val and test
        X_val_selected = X_val_raw[:, selected_indices]
        X_test_selected = X_test_raw[:, selected_indices]
    else:
        X_train_selected = X_train_raw
        X_val_selected = X_val_raw
        X_test_selected = X_test_raw
        print(f"\n⏭️  Skipping Boruta (using all {len(feature_cols)} features)")
    
    # Scale features
    print("\n📏 Scaling features with RobustScaler...")
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_val_scaled = scaler.transform(X_val_selected)
    X_test_scaled = scaler.transform(X_test_selected)
    
    # Create sequences
    print(f"\n🔄 Creating sequences (length={sequence_length})...")
    X_train, y_train = create_sequences(X_train_scaled, y_train_raw, sequence_length)
    X_val, y_val = create_sequences(X_val_scaled, y_val_raw, sequence_length)
    X_test, y_test = create_sequences(X_test_scaled, y_test_raw, sequence_length)
    
    print(f"   Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")
    
    train_up = y_train.sum()
    train_down = len(y_train) - train_up
    print(f"   Training class balance: Down={train_down} ({train_down/len(y_train)*100:.1f}%), "
          f"Up={train_up} ({train_up/len(y_train)*100:.1f}%)")
    
    print("\n" + "=" * 60)
    print("✅ Data Preparation Complete")
    print("=" * 60)
    
    return (X_train, X_val, X_test, y_train, y_val, y_test,
            scaler, feature_cols, df, selected_indices)


# ============================================================
# CNN-LSTM Model Architecture (v3.1 Simplified)
# ============================================================

def build_cnn_lstm_model_v31(
    sequence_length: int,
    n_features: int,
    cnn_filters: List[int] = [32, 64],
    lstm_units: List[int] = [50, 25],
    dense_units: List[int] = [32],
    dropout_rate: float = 0.5,
    l2_reg: float = 0.01,
    learning_rate: float = 0.001
) -> Model:
    """
    Build simplified CNN-LSTM model for v3.1.
    
    Target: ~40K parameters (vs 176K in v3.0)
    """
    print("\n" + "=" * 60)
    print("🏗️  Building CNN-LSTM Model (v3.1 Simplified)")
    print("=" * 60)
    print(f"   Sequence length: {sequence_length}")
    print(f"   Features: {n_features}")
    print(f"   CNN filters: {cnn_filters}")
    print(f"   LSTM units: {lstm_units}")
    print(f"   Dense units: {dense_units}")
    print(f"   Dropout: {dropout_rate}")
    print(f"   L2 reg: {l2_reg}")
    
    # Input layer
    inputs = layers.Input(shape=(sequence_length, n_features), name='input')
    x = inputs
    
    # CNN layers
    for i, filters in enumerate(cnn_filters):
        x = layers.Conv1D(
            filters=filters,
            kernel_size=3,
            padding='same',
            kernel_regularizer=regularizers.l2(l2_reg),
            name=f'conv1d_{i+1}'
        )(x)
        x = layers.BatchNormalization(name=f'bn_cnn_{i+1}')(x)
        x = layers.ReLU(name=f'relu_cnn_{i+1}')(x)
        x = layers.Dropout(dropout_rate, name=f'dropout_cnn_{i+1}')(x)
        
        # Max pooling (only if sequence is long enough)
        if x.shape[1] > 2:
            x = layers.MaxPooling1D(pool_size=2, name=f'maxpool_{i+1}')(x)
    
    # LSTM layers
    for i, units in enumerate(lstm_units):
        return_sequences = i < len(lstm_units) - 1
        x = layers.LSTM(
            units=units,
            return_sequences=return_sequences,
            kernel_regularizer=regularizers.l2(l2_reg),
            recurrent_regularizer=regularizers.l2(l2_reg),
            name=f'lstm_{i+1}'
        )(x)
        x = layers.Dropout(dropout_rate, name=f'dropout_lstm_{i+1}')(x)
    
    # Dense layers
    for i, units in enumerate(dense_units):
        x = layers.Dense(
            units,
            kernel_regularizer=regularizers.l2(l2_reg),
            name=f'dense_{i+1}'
        )(x)
        x = layers.BatchNormalization(name=f'bn_dense_{i+1}')(x)
        x = layers.ReLU(name=f'relu_dense_{i+1}')(x)
        x = layers.Dropout(dropout_rate, name=f'dropout_dense_{i+1}')(x)
    
    # Output layer
    outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
    
    # Build model
    model = Model(inputs=inputs, outputs=outputs, name='cnn_lstm_v31')
    
    # Compile
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Print summary
    total_params = model.count_params()
    print(f"\n   Total parameters: {total_params:,}")
    print(f"   Target was: ~40,000")
    
    if total_params > 50000:
        print(f"   ⚠️  Model has more parameters than target!")
    else:
        print(f"   ✅ Model within target parameter budget")
    
    print("=" * 60 + "\n")
    
    return model


# ============================================================
# Training
# ============================================================

def train_model(
    model: Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 100,
    batch_size: int = 32,
    patience: int = 15,
    output_dir: Path = None
) -> Tuple[Model, Dict]:
    """Train the model with early stopping."""
    print("\n" + "=" * 60)
    print("🚀 Training Model")
    print("=" * 60)
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Early stopping patience: {patience}")
    
    # Class weights for imbalanced data
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}
    print(f"   Class weights: {class_weight_dict}")
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=patience,
            restore_best_weights=True,
            verbose=1,
            mode='max'
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Add checkpoint if output_dir specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            ModelCheckpoint(
                filepath=str(output_dir / 'cnn_lstm_v31_best.keras'),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            )
        )
    
    print("\n" + "-" * 60)
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )
    
    # Convert history to dict
    history_dict = {
        'loss': [float(x) for x in history.history['loss']],
        'accuracy': [float(x) for x in history.history['accuracy']],
        'val_loss': [float(x) for x in history.history['val_loss']],
        'val_accuracy': [float(x) for x in history.history['val_accuracy']]
    }
    
    print("\n" + "=" * 60)
    print("✅ Training Complete")
    print("=" * 60)
    
    return model, history_dict


# ============================================================
# Evaluation
# ============================================================

def evaluate_model(
    model: Model,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, float]:
    """Evaluate model on test set."""
    print("\n" + "=" * 60)
    print("📊 Model Evaluation")
    print("=" * 60)
    
    # Predictions
    y_proba = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_proba > 0.5).astype(int)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    
    try:
        auc_roc = roc_auc_score(y_test, y_proba)
    except:
        auc_roc = 0.5
    
    # Per-class accuracy
    cm = confusion_matrix(y_test, y_pred)
    if cm.shape == (2, 2):
        down_correct = cm[0, 0]
        down_total = cm[0, :].sum()
        up_correct = cm[1, 1]
        up_total = cm[1, :].sum()
        down_acc = down_correct / max(down_total, 1)
        up_acc = up_correct / max(up_total, 1)
    else:
        down_acc = up_acc = 0.5
    
    metrics = {
        'accuracy': float(accuracy),
        'f1': float(f1),
        'precision': float(precision),
        'recall': float(recall),
        'auc_roc': float(auc_roc),
        'down_accuracy': float(down_acc),
        'up_accuracy': float(up_acc)
    }
    
    print(f"\n   Test Accuracy: {accuracy*100:.2f}%")
    print(f"   F1 Score: {f1:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")
    print(f"   AUC-ROC: {auc_roc:.4f}")
    print(f"\n   Down Accuracy: {down_acc*100:.2f}% ({down_correct}/{down_total})")
    print(f"   Up Accuracy: {up_acc*100:.2f}% ({up_correct}/{up_total})")
    
    print("\n   Confusion Matrix:")
    print(f"              Pred Down  Pred Up")
    print(f"   Actual Down    {cm[0,0]:5d}    {cm[0,1]:5d}")
    print(f"   Actual Up      {cm[1,0]:5d}    {cm[1,1]:5d}")
    
    print("=" * 60 + "\n")
    
    return metrics


# ============================================================
# Optuna Hyperparameter Optimization
# ============================================================

def create_optuna_objective(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_features: int,
    sequence_length: int,
    n_cv_folds: int = 3
):
    """
    Create an Optuna objective function with TimeSeriesSplit cross-validation.
    
    Uses k-fold cross-validation to get more robust hyperparameter evaluation
    and reduce overfitting to a single validation set.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data (combined with train for CV)
        n_features: Number of features
        sequence_length: Sequence length
        n_cv_folds: Number of cross-validation folds
        
    Returns:
        Objective function for Optuna
    """
    from sklearn.model_selection import TimeSeriesSplit
    
    # Combine train and val for cross-validation
    X_combined = np.concatenate([X_train, X_val], axis=0)
    y_combined = np.concatenate([y_train, y_val], axis=0)
    
    # Create TimeSeriesSplit (respects temporal ordering)
    tscv = TimeSeriesSplit(n_splits=n_cv_folds)
    
    def objective(trial):
        # Clear TensorFlow session to prevent memory issues
        tf.keras.backend.clear_session()
        
        # Suggest hyperparameters
        # CNN architecture
        n_cnn_layers = trial.suggest_int('n_cnn_layers', 1, 3)
        cnn_filters = []
        for i in range(n_cnn_layers):
            filters = trial.suggest_categorical(f'cnn_filters_{i}', [16, 32, 64, 128])
            cnn_filters.append(filters)
        
        # LSTM architecture
        n_lstm_layers = trial.suggest_int('n_lstm_layers', 1, 2)
        lstm_units = []
        for i in range(n_lstm_layers):
            units = trial.suggest_categorical(f'lstm_units_{i}', [25, 50, 75, 100])
            lstm_units.append(units)
        
        # Dense layers
        n_dense_layers = trial.suggest_int('n_dense_layers', 1, 2)
        dense_units = []
        for i in range(n_dense_layers):
            units = trial.suggest_categorical(f'dense_units_{i}', [16, 32, 64])
            dense_units.append(units)
        
        # Regularization
        dropout_rate = trial.suggest_float('dropout', 0.2, 0.6, step=0.1)
        l2_reg = trial.suggest_float('l2_reg', 1e-4, 1e-1, log=True)
        
        # Training parameters
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        
        # Cross-validation scores
        cv_scores = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_combined)):
            X_cv_train, X_cv_val = X_combined[train_idx], X_combined[val_idx]
            y_cv_train, y_cv_val = y_combined[train_idx], y_combined[val_idx]
            
            # Build model with suggested parameters
            model = build_cnn_lstm_model_v31(
                sequence_length=sequence_length,
                n_features=n_features,
                cnn_filters=cnn_filters,
                lstm_units=lstm_units,
                dense_units=dense_units,
                dropout_rate=dropout_rate,
                l2_reg=l2_reg,
                learning_rate=learning_rate
            )
            
            # Class weights
            class_weights = compute_class_weight(
                'balanced',
                classes=np.unique(y_cv_train),
                y=y_cv_train
            )
            class_weight_dict = {i: w for i, w in enumerate(class_weights)}
            
            # Callbacks for early stopping
            callbacks = [
                EarlyStopping(
                    monitor='val_accuracy',
                    patience=7,
                    restore_best_weights=True,
                    mode='max'
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=4,
                    min_lr=1e-6
                )
            ]
            
            # Train with reduced verbosity
            history = model.fit(
                X_cv_train, y_cv_train,
                validation_data=(X_cv_val, y_cv_val),
                epochs=30,  # Reduced for CV
                batch_size=batch_size,
                class_weight=class_weight_dict,
                callbacks=callbacks,
                verbose=0
            )
            
            # Best validation accuracy for this fold
            best_val_acc = max(history.history['val_accuracy'])
            cv_scores.append(best_val_acc)
            
            # Clear session between folds
            tf.keras.backend.clear_session()
        
        # Return mean CV score
        mean_cv_score = np.mean(cv_scores)
        std_cv_score = np.std(cv_scores)
        
        # Report intermediate values for pruning
        trial.report(mean_cv_score, step=n_cv_folds)
        
        # Store CV details in trial
        trial.set_user_attr('cv_scores', cv_scores)
        trial.set_user_attr('cv_std', std_cv_score)
        
        # Prune if needed
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        return mean_cv_score
    
    return objective


def run_optuna_optimization(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_features: int,
    sequence_length: int,
    n_trials: int = 20,
    n_cv_folds: int = 3,
    timeout: int = None
) -> Dict[str, Any]:
    """
    Run Optuna hyperparameter optimization with cross-validation.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        n_features: Number of features
        sequence_length: Sequence length
        n_trials: Number of optimization trials (default: 20)
        n_cv_folds: Number of CV folds (default: 3)
        timeout: Maximum time in seconds (optional)
        
    Returns:
        Dictionary with best parameters and study results
    """
    if not OPTUNA_AVAILABLE:
        print("   ⚠️  Optuna not available, skipping optimization")
        return None
    
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    
    print("\n" + "=" * 60)
    print("🔍 Optuna Hyperparameter Optimization (with Cross-Validation)")
    print("=" * 60)
    print(f"   Trials: {n_trials}")
    print(f"   CV Folds: {n_cv_folds} (TimeSeriesSplit)")
    print(f"   Timeout: {timeout}s" if timeout else "   Timeout: None")
    print(f"   Note: Each trial trains {n_cv_folds} models for robust evaluation")
    
    # Create objective function
    objective = create_optuna_objective(
        X_train, y_train, X_val, y_val,
        n_features, sequence_length, n_cv_folds
    )
    
    # Create study with TPE sampler and median pruner
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=3, n_warmup_steps=5)
    )
    
    # Suppress Optuna's default logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    # Custom callback to show progress
    def progress_callback(study, trial):
        cv_std = trial.user_attrs.get('cv_std', 0)
        val_acc_str = f"{trial.value:.4f}" if trial.value is not None else "N/A"
        std_str = f"±{cv_std:.4f}" if cv_std else ""
        print(f"   Trial {trial.number + 1}/{n_trials}: "
              f"CV_acc={val_acc_str}{std_str} | "
              f"best={study.best_value:.4f}")
    
    print("\n   Running optimization...")
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        callbacks=[progress_callback],
        show_progress_bar=False
    )
    
    # Extract best parameters
    best_params = study.best_params
    best_value = study.best_value
    best_trial = study.best_trial
    
    # Reconstruct architecture from best params
    n_cnn_layers = best_params['n_cnn_layers']
    cnn_filters = [best_params[f'cnn_filters_{i}'] for i in range(n_cnn_layers)]
    
    n_lstm_layers = best_params['n_lstm_layers']
    lstm_units = [best_params[f'lstm_units_{i}'] for i in range(n_lstm_layers)]
    
    n_dense_layers = best_params['n_dense_layers']
    dense_units = [best_params[f'dense_units_{i}'] for i in range(n_dense_layers)]
    
    # Get CV details from best trial
    cv_scores = best_trial.user_attrs.get('cv_scores', [])
    cv_std = best_trial.user_attrs.get('cv_std', 0)
    
    print(f"\n   ✅ Optimization Complete!")
    print(f"\n   📊 Best Cross-Validation Accuracy: {best_value*100:.2f}% (±{cv_std*100:.2f}%)")
    if cv_scores:
        print(f"      Fold scores: {[f'{s*100:.1f}%' for s in cv_scores]}")
    print(f"\n   🏆 Best Hyperparameters:")
    print(f"      CNN filters: {cnn_filters}")
    print(f"      LSTM units: {lstm_units}")
    print(f"      Dense units: {dense_units}")
    print(f"      Dropout: {best_params['dropout']:.2f}")
    print(f"      L2 reg: {best_params['l2_reg']:.6f}")
    print(f"      Learning rate: {best_params['learning_rate']:.6f}")
    print(f"      Batch size: {best_params['batch_size']}")
    
    print("=" * 60 + "\n")
    
    return {
        'best_params': best_params,
        'best_value': best_value,
        'cv_std': cv_std,
        'cv_scores': cv_scores,
        'cnn_filters': cnn_filters,
        'lstm_units': lstm_units,
        'dense_units': dense_units,
        'dropout': best_params['dropout'],
        'l2_reg': best_params['l2_reg'],
        'learning_rate': best_params['learning_rate'],
        'batch_size': best_params['batch_size'],
        'n_trials': len(study.trials),
        'n_cv_folds': n_cv_folds,
        'study': study
    }


# ============================================================
# Save Model
# ============================================================

def save_model(
    model: Model,
    scaler: RobustScaler,
    feature_names: List[str],
    config: Dict,
    metrics: Dict,
    history: Dict,
    selected_indices: List[int],
    output_dir: Path
) -> None:
    """Save model, metadata, and training artifacts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving model to: {output_dir}")
    
    # Save Keras model
    model_path = output_dir / 'cnn_lstm_v31_model.keras'
    model.save(model_path)
    print(f"   ✅ Model: {model_path.name}")
    
    # Save metadata
    metadata = {
        'scaler': scaler,
        'feature_names': feature_names,
        'selected_indices': selected_indices
    }
    metadata_path = output_dir / 'cnn_lstm_v31_metadata.pkl'
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"   ✅ Metadata: {metadata_path.name}")
    
    # Save config
    config_path = output_dir / 'cnn_lstm_v31_config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"   ✅ Config: {config_path.name}")
    
    # Save metrics
    metrics_path = output_dir / 'cnn_lstm_v31_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"   ✅ Metrics: {metrics_path.name}")
    
    # Save history
    history_path = output_dir / 'cnn_lstm_v31_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"   ✅ History: {history_path.name}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Train CNN-LSTM v3.1 (Simplified) for crypto price prediction'
    )
    
    # Data arguments
    parser.add_argument('--data', type=str, default=None,
                        help='Path to CSV data file')
    parser.add_argument('--horizon', type=int, default=DEFAULT_HORIZON,
                        help=f'Prediction horizon in minutes (default: {DEFAULT_HORIZON})')
    parser.add_argument('--threshold', type=float, default=DEFAULT_DIRECTION_THRESHOLD,
                        help=f'Direction threshold (default: {DEFAULT_DIRECTION_THRESHOLD})')
    
    # Feature arguments
    parser.add_argument('--no-sentiment', action='store_true',
                        help='Disable sentiment features')
    parser.add_argument('--no-onchain', action='store_true',
                        help='Disable on-chain blockchain features')
    parser.add_argument('--no-technical', action='store_true',
                        help='Disable technical indicator features')
    parser.add_argument('--onchain-only', action='store_true',
                        help='Use ONLY on-chain features (disables technical and sentiment)')
    parser.add_argument('--clear-onchain-cache', action='store_true',
                        help='Clear on-chain data cache before fetching')
    parser.add_argument('--no-boruta', action='store_true',
                        help='Disable Boruta feature selection')
    parser.add_argument('--max-features', type=int, default=DEFAULT_MAX_FEATURES,
                        help=f'Max features after Boruta (default: {DEFAULT_MAX_FEATURES})')
    
    # Optuna arguments
    parser.add_argument('--optuna', action='store_true',
                        help='Enable Optuna hyperparameter optimization')
    parser.add_argument('--optuna-trials', type=int, default=20,
                        help='Number of Optuna trials (default: 20)')
    parser.add_argument('--optuna-cv-folds', type=int, default=3,
                        help='Number of cross-validation folds for Optuna (default: 3)')
    parser.add_argument('--optuna-timeout', type=int, default=None,
                        help='Optuna timeout in seconds (default: None)')
    
    # Architecture arguments (v3.1 simplified defaults)
    parser.add_argument('--sequence-length', type=int, default=DEFAULT_SEQUENCE_LENGTH,
                        help=f'Sequence length (default: {DEFAULT_SEQUENCE_LENGTH})')
    parser.add_argument('--cnn-filters', type=str, default=DEFAULT_CNN_FILTERS,
                        help=f'CNN filters (default: {DEFAULT_CNN_FILTERS})')
    parser.add_argument('--lstm-units', type=str, default=DEFAULT_LSTM_UNITS,
                        help=f'LSTM units (default: {DEFAULT_LSTM_UNITS})')
    parser.add_argument('--dense-units', type=str, default=DEFAULT_DENSE_UNITS,
                        help=f'Dense units (default: {DEFAULT_DENSE_UNITS})')
    parser.add_argument('--dropout', type=float, default=DEFAULT_DROPOUT,
                        help=f'Dropout rate (default: {DEFAULT_DROPOUT})')
    parser.add_argument('--l2-reg', type=float, default=DEFAULT_L2_REG,
                        help=f'L2 regularization (default: {DEFAULT_L2_REG})')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS,
                        help=f'Training epochs (default: {DEFAULT_EPOCHS})')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE,
                        help=f'Batch size (default: {DEFAULT_BATCH_SIZE})')
    parser.add_argument('--learning-rate', type=float, default=DEFAULT_LEARNING_RATE,
                        help=f'Learning rate (default: {DEFAULT_LEARNING_RATE})')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience (default: 15)')
    
    # Output arguments
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for model')
    
    # Other
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU-only mode')
    
    args = parser.parse_args()
    
    # Print banner
    print("\n" + "=" * 60)
    print("🚀 ServoTrader CNN-LSTM Training (v3.1 Simplified)")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Find data file
    if args.data:
        data_path = Path(args.data)
    else:
        # Try to find data in common locations
        possible_paths = [
            DAILY_DATA_DIR / "BTCUSDT.csv",
            DATA_DIR / "BTCUSDT.csv",
            Path("data/daily_historical/BTCUSDT.csv"),
            Path("data/BTCUSDT.csv"),
            Path("BTCUSDT.csv")
        ]
        data_path = None
        for p in possible_paths:
            if p.exists():
                data_path = p
                break
        
        if data_path is None:
            print("\n❌ No data file found!")
            print("Please specify data path with --data argument")
            print("Or place BTCUSDT.csv in one of these locations:")
            for p in possible_paths:
                print(f"   - {p}")
            sys.exit(1)
    
    if not data_path.exists():
        print(f"\n❌ Data file not found: {data_path}")
        sys.exit(1)
    
    print(f"\n📂 Data: {data_path}")
    print(f"🎯 Horizon: {args.horizon} minutes")
    print(f"📊 Max features: {args.max_features}")
    print(f"🔢 Sequence length: {args.sequence_length}")
    
    # Show feature configuration
    if args.onchain_only:
        print(f"🔗 Feature mode: ON-CHAIN ONLY")
    else:
        features_enabled = []
        if not args.no_technical:
            features_enabled.append("Technical")
        if not args.no_sentiment:
            features_enabled.append("Sentiment")
        if not args.no_onchain:
            features_enabled.append("On-Chain")
        print(f"📊 Features: {', '.join(features_enabled) if features_enabled else 'None'}")
    
    # Parse architecture strings
    cnn_filters = [int(x) for x in args.cnn_filters.split(',')]
    lstm_units = [int(x) for x in args.lstm_units.split(',')]
    dense_units = [int(x) for x in args.dense_units.split(',')]
    
    # Clear on-chain cache if requested
    if args.clear_onchain_cache:
        print("\n🗑️  Clearing on-chain cache...")
        cache_dir = Path.home() / ".servo_trader" / "onchain_cache"
        if cache_dir.exists():
            import shutil
            shutil.rmtree(cache_dir)
            print(f"   ✅ Cleared: {cache_dir}")
        else:
            print(f"   ℹ️  Cache directory doesn't exist: {cache_dir}")
    
    # Handle --onchain-only flag
    if args.onchain_only:
        print("\n🔗 ON-CHAIN ONLY MODE: Using only blockchain features")
        args.no_technical = True
        args.no_sentiment = True
        args.no_onchain = False  # Ensure on-chain is enabled
    
    # Prepare data
    (X_train, X_val, X_test, y_train, y_val, y_test,
     scaler, feature_names, df, selected_indices) = prepare_data(
        data_path=str(data_path),
        horizon_minutes=args.horizon,
        sequence_length=args.sequence_length,
        direction_threshold=args.threshold,
        include_technical=not args.no_technical,
        include_sentiment=not args.no_sentiment,
        include_onchain=not args.no_onchain,
        use_boruta=not args.no_boruta,
        max_features=args.max_features,
        train_ratio=DEFAULT_TRAIN_RATIO,
        val_ratio=DEFAULT_VAL_RATIO
    )
    
    # Get number of features
    n_features = X_train.shape[2]
    
    # Output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = MODEL_DIR
    
    # Run Optuna optimization if requested
    optuna_results = None
    if args.optuna:
        if OPTUNA_AVAILABLE:
            optuna_results = run_optuna_optimization(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                n_features=n_features,
                sequence_length=args.sequence_length,
                n_trials=args.optuna_trials,
                n_cv_folds=args.optuna_cv_folds,
                timeout=args.optuna_timeout
            )
            
            if optuna_results:
                # Use optimized parameters
                cnn_filters = optuna_results['cnn_filters']
                lstm_units = optuna_results['lstm_units']
                dense_units = optuna_results['dense_units']
                args.dropout = optuna_results['dropout']
                args.l2_reg = optuna_results['l2_reg']
                args.learning_rate = optuna_results['learning_rate']
                args.batch_size = optuna_results['batch_size']
                
                print("\n📊 Using Optuna-optimized parameters for final training...")
        else:
            print("\n⚠️  Optuna not available, using default parameters...")
    
    # Build model
    model = build_cnn_lstm_model_v31(
        sequence_length=args.sequence_length,
        n_features=n_features,
        cnn_filters=cnn_filters,
        lstm_units=lstm_units,
        dense_units=dense_units,
        dropout_rate=args.dropout,
        l2_reg=args.l2_reg,
        learning_rate=args.learning_rate
    )
    
    # Print model summary
    model.summary()
    
    # Train model
    model, history = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        output_dir=output_dir
    )
    
    # Evaluate
    metrics = evaluate_model(model, X_test, y_test)
    
    # Save config
    config = {
        'version': '3.1',
        'architecture': 'CNN-LSTM-Simplified',
        'horizon_minutes': args.horizon,
        'sequence_length': args.sequence_length,
        'direction_threshold': args.threshold,
        'include_technical': not args.no_technical,
        'include_sentiment': not args.no_sentiment,
        'include_onchain': not args.no_onchain,
        'onchain_only': args.onchain_only,
        'use_boruta': not args.no_boruta,
        'use_optuna': args.optuna,
        'optuna_trials': args.optuna_trials if args.optuna else 0,
        'optuna_cv_folds': args.optuna_cv_folds if args.optuna else 0,
        'optuna_best_cv_acc': optuna_results['best_value'] if optuna_results else None,
        'optuna_cv_std': optuna_results.get('cv_std', 0) if optuna_results else None,
        'max_features': args.max_features,
        'cnn_filters': cnn_filters,
        'lstm_units': lstm_units,
        'dense_units': dense_units,
        'dropout': args.dropout,
        'l2_reg': args.l2_reg,
        'learning_rate': args.learning_rate,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'n_features': n_features,
        'feature_names': feature_names,
        'total_parameters': model.count_params(),
        'data_path': str(data_path)
    }
    
    # Save everything
    save_model(
        model=model,
        scaler=scaler,
        feature_names=feature_names,
        config=config,
        metrics=metrics,
        history=history,
        selected_indices=selected_indices,
        output_dir=output_dir
    )
    
    # Final summary
    print("\n" + "=" * 60)
    print("✅ Training Complete (v3.1 Simplified)")
    print("=" * 60)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if optuna_results:
        print(f"\n🔍 Optuna Optimization:")
        print(f"   Trials: {optuna_results['n_trials']}")
        print(f"   CV Folds: {optuna_results.get('n_cv_folds', 'N/A')}")
        cv_std = optuna_results.get('cv_std', 0)
        print(f"   Best CV Accuracy: {optuna_results['best_value']*100:.2f}% (±{cv_std*100:.2f}%)")
        cv_scores = optuna_results.get('cv_scores', [])
        if cv_scores:
            print(f"   Fold scores: {[f'{s*100:.1f}%' for s in cv_scores]}")
    
    print(f"\n📊 Final Results:")
    print(f"   Test Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"   Test F1: {metrics['f1']:.4f}")
    print(f"   Test AUC-ROC: {metrics['auc_roc']:.4f}")
    print(f"   Down Accuracy: {metrics['down_accuracy']*100:.2f}%")
    print(f"   Up Accuracy: {metrics['up_accuracy']*100:.2f}%")
    
    best_epoch = np.argmax(history['val_accuracy']) + 1
    best_val_acc = max(history['val_accuracy'])
    print(f"\n📈 Best: Epoch {best_epoch} with {best_val_acc*100:.2f}% val accuracy")
    
    print(f"\n🏗️  Final Architecture:")
    print(f"   CNN filters: {cnn_filters}")
    print(f"   LSTM units: {lstm_units}")
    print(f"   Dense units: {dense_units}")
    print(f"   Dropout: {args.dropout}")
    print(f"   L2 reg: {args.l2_reg}")
    print(f"   Learning rate: {args.learning_rate}")
    print(f"   Batch size: {args.batch_size}")
    
    print(f"\n📊 v3.1 vs v3.0 Comparison:")
    print(f"   Parameters: {model.count_params():,} vs 176,219 (v3.0)")
    print(f"   Features: {n_features} vs 117 (v3.0)")
    print(f"   Sequence length: {args.sequence_length} vs 30 (v3.0)")
    print(f"   Param/Sample ratio: {model.count_params() / len(X_train):.1f}:1 vs 87:1 (v3.0)")
    
    print(f"\n💾 Model saved to: {output_dir}")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()