#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data_pipeline.py

Simplified data pipeline for LightGBM-only cryptocurrency prediction.

This module handles:
- Loading CSV data
- Computing technical indicators
- Fetching and integrating sentiment data (Fear & Greed Index)
- Creating labels (returns and direction)
- Train/validation/test splitting (temporal, no shuffling)
- Feature scaling

Updated for v2.2: Added Fear & Greed Index sentiment features

Author: Jarred Deluca
Project: ServoTrader - Prediction Subsystem (v2.2 - With Sentiment)
"""

import os
import numpy as np
import pandas as pd
import pickle
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.preprocessing import RobustScaler

from feature_engineering import (
    compute_all_features,
    get_static_feature_names
)

from sentiment_data import (
    fetch_fear_greed_index,
    create_sentiment_features,
    merge_sentiment_with_price,
    get_sentiment_feature_names
)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class PredictionDataset:
    """Container for prediction model datasets."""
    X: np.ndarray               # Features [N, num_features]
    y_return: np.ndarray        # Target: return percentage [N]
    y_direction: np.ndarray     # Target: direction class [N] (0=down, 1=flat, 2=up)
    timestamps: np.ndarray      # Timestamps for each sample
    close_prices: np.ndarray    # Close prices for backtesting


# =============================================================================
# Data Pipeline Class
# =============================================================================

class CryptoPredictionDataPipeline:
    """
    Simplified data pipeline for LightGBM-only prediction.
    
    Now includes Fear & Greed Index sentiment features.
    """
    
    def __init__(self,
                 csv_path: str,
                 prediction_horizon: int = 60,
                 direction_threshold: float = 0.003,
                 train_ratio: float = 0.70,
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.15,
                 feature_config: Optional[Dict] = None,
                 include_sentiment: bool = True,
                 sentiment_cache_dir: Optional[str] = None):
        """
        Initialize the data pipeline.
        
        Args:
            csv_path: Path to the CSV file with OHLCV data
            prediction_horizon: Minutes ahead to predict (must be multiple of 5)
            direction_threshold: Threshold for flat classification (e.g., 0.003 = 0.3%)
            train_ratio: Fraction of data for training
            val_ratio: Fraction of data for validation
            test_ratio: Fraction of data for testing
            feature_config: Optional dict with indicator parameters
            include_sentiment: Whether to include Fear & Greed Index features
            sentiment_cache_dir: Directory to cache sentiment data
        """
        self.csv_path = csv_path
        self.prediction_horizon = prediction_horizon
        self.horizon_steps = prediction_horizon // 5  # Convert minutes to 5-min bars
        self.direction_threshold = direction_threshold
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.feature_config = feature_config or self._default_feature_config()
        self.include_sentiment = include_sentiment
        self.sentiment_cache_dir = sentiment_cache_dir
        
        # Will be populated after data loading
        self.raw_df: Optional[pd.DataFrame] = None
        self.feature_df: Optional[pd.DataFrame] = None
        self.sentiment_df: Optional[pd.DataFrame] = None
        self.feature_names = get_static_feature_names(include_sentiment=include_sentiment)
        
        # Scaler for feature normalization
        self.scaler: Optional[RobustScaler] = None
        
        # Datasets
        self.train_data: Optional[PredictionDataset] = None
        self.val_data: Optional[PredictionDataset] = None
        self.test_data: Optional[PredictionDataset] = None
        
    def _default_feature_config(self) -> Dict:
        """Get default feature engineering configuration."""
        return {
            'rsi_period': 14,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'bb_period': 20,
            'bb_std': 2,
            'atr_period': 14,
            'ema_periods': [5, 20, 60],
            'volatility_period': 20,
            'volume_sma_period': 20,
            'trend_slope_period': 20
        }
    
    def load_data(self) -> pd.DataFrame:
        """Load raw data from CSV file."""
        print(f"📂 Loading data from: {self.csv_path}")
        
        self.raw_df = pd.read_csv(self.csv_path)
        self.raw_df['timestamp'] = pd.to_datetime(self.raw_df['timestamp'])
        self.raw_df = self.raw_df.sort_values('timestamp').reset_index(drop=True)
        
        print(f"   Loaded {len(self.raw_df):,} rows")
        print(f"   Date range: {self.raw_df['timestamp'].min()} to {self.raw_df['timestamp'].max()}")
        
        return self.raw_df
    
    def load_sentiment_data(self) -> pd.DataFrame:
        """Load and prepare Fear & Greed Index sentiment data."""
        if not self.include_sentiment:
            return None
            
        print("📊 Loading Fear & Greed sentiment data...")
        
        # Fetch Fear & Greed data (will use cache if available)
        fg_df = fetch_fear_greed_index(limit=0, cache_dir=self.sentiment_cache_dir)
        
        # Create derived features
        self.sentiment_df = create_sentiment_features(fg_df)
        
        print(f"   Sentiment data range: {self.sentiment_df['date'].min()} to {self.sentiment_df['date'].max()}")
        print(f"   Total sentiment days: {len(self.sentiment_df)}")
        
        return self.sentiment_df
    
    def compute_features(self) -> pd.DataFrame:
        """Compute all technical indicators and features."""
        if self.raw_df is None:
            self.load_data()
            
        print("🔧 Computing technical indicators...")
        
        self.feature_df = compute_all_features(self.raw_df, self.feature_config)
        
        # Add sentiment features if enabled
        if self.include_sentiment:
            if self.sentiment_df is None:
                self.load_sentiment_data()
            
            print("🧠 Merging sentiment features...")
            self.feature_df = merge_sentiment_with_price(
                self.feature_df, 
                self.sentiment_df,
                timestamp_col='timestamp'
            )
            
            # Count how many rows have sentiment data
            if 'fear_greed' in self.feature_df.columns:
                sentiment_coverage = (self.feature_df['fear_greed'].notna()).sum()
                print(f"   Sentiment coverage: {sentiment_coverage:,} / {len(self.feature_df):,} rows ({100*sentiment_coverage/len(self.feature_df):.1f}%)")
        
        print(f"   Total columns: {len(self.feature_df.columns)}")
        print(f"   Feature columns: {len(self.feature_names)}")
        
        return self.feature_df
    
    def create_labels(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create prediction labels (return and direction).
        
        Binary classification: Down (0) or Up (1)
        Samples within threshold are excluded from training.
        
        Returns:
            Tuple of (returns, direction, valid_mask)
        """
        if self.feature_df is None:
            self.compute_features()
            
        print(f"🎯 Creating labels (horizon: {self.prediction_horizon} min = {self.horizon_steps} bars)...")
        print(f"   Direction threshold: ±{self.direction_threshold*100:.2f}% (samples within excluded)")
        
        close = self.feature_df['close'].values
        
        # Future return: (close[t+h] - close[t]) / close[t]
        future_close = np.roll(close, -self.horizon_steps)
        returns = (future_close - close) / close
        
        # Set last horizon_steps to NaN (no future data)
        returns[-self.horizon_steps:] = np.nan
        
        # Binary direction classification
        # 0 = Down (return < -threshold)
        # 1 = Up (return > +threshold)
        # Samples within [-threshold, +threshold] are excluded
        direction = np.full_like(returns, -1, dtype=np.int64)  # -1 = undefined/excluded
        direction[returns < -self.direction_threshold] = 0     # Down
        direction[returns > self.direction_threshold] = 1      # Up
        
        # Valid mask: not NaN and has a direction (not flat)
        valid_mask = ~np.isnan(returns) & (direction >= 0)
        
        # Print distribution
        print(f"   Direction distribution (binary):")
        total_valid = np.sum(valid_mask)
        for label, name in [(0, 'Down'), (1, 'Up')]:
            count = np.sum(direction[valid_mask] == label)
            pct = 100 * count / total_valid if total_valid > 0 else 0
            print(f"     {name}: {count:,} ({pct:.1f}%)")
        
        # Count excluded (flat) samples
        flat_mask = ~np.isnan(returns) & (direction < 0)
        flat_count = np.sum(flat_mask)
        flat_pct = 100 * flat_count / np.sum(~np.isnan(returns))
        print(f"     Excluded (flat): {flat_count:,} ({flat_pct:.1f}%)")
        
        return returns, direction, valid_mask
    
    def prepare_features_and_labels(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                                    np.ndarray, np.ndarray]:
        """
        Prepare features and labels for training.
        
        For binary classification, samples within the threshold (flat) are excluded.
        
        Returns:
            Tuple of (X, y_return, y_direction, timestamps, close_prices)
        """
        if self.feature_df is None:
            self.compute_features()
            
        print(f"📊 Preparing features...")
        
        # Get labels (now returns valid_mask for binary classification)
        y_return, y_direction, direction_valid_mask = self.create_labels()
        
        # Extract features (only the static feature columns)
        X = self.feature_df[self.feature_names].values
        timestamps = self.feature_df['timestamp'].values
        close_prices = self.feature_df['close'].values
        
        # Check for any NaN in features
        feature_valid = ~np.any(np.isnan(X), axis=1)
        
        # Combined valid mask: valid direction (not flat) AND valid features
        valid_mask = direction_valid_mask & feature_valid
        
        X = X[valid_mask].astype(np.float32)
        y_return = y_return[valid_mask].astype(np.float32)
        y_direction = y_direction[valid_mask].astype(np.int64)
        timestamps = timestamps[valid_mask]
        close_prices = close_prices[valid_mask].astype(np.float32)
        
        print(f"   Created {len(X):,} valid samples (excluding flat)")
        print(f"   Feature shape: {X.shape}")
        
        return X, y_return, y_direction, timestamps, close_prices
    
    def split_data(self, X: np.ndarray, y_return: np.ndarray, y_direction: np.ndarray,
                   timestamps: np.ndarray, close_prices: np.ndarray
                   ) -> Tuple[PredictionDataset, PredictionDataset, PredictionDataset]:
        """Split data into train/validation/test sets (temporal split)."""
        print("✂️ Splitting data (temporal split, no shuffling)...")
        
        n_total = len(X)
        n_train = int(n_total * self.train_ratio)
        n_val = int(n_total * self.val_ratio)
        
        # Train set
        train_data = PredictionDataset(
            X=X[:n_train],
            y_return=y_return[:n_train],
            y_direction=y_direction[:n_train],
            timestamps=timestamps[:n_train],
            close_prices=close_prices[:n_train]
        )
        
        # Validation set
        val_data = PredictionDataset(
            X=X[n_train:n_train + n_val],
            y_return=y_return[n_train:n_train + n_val],
            y_direction=y_direction[n_train:n_train + n_val],
            timestamps=timestamps[n_train:n_train + n_val],
            close_prices=close_prices[n_train:n_train + n_val]
        )
        
        # Test set
        test_data = PredictionDataset(
            X=X[n_train + n_val:],
            y_return=y_return[n_train + n_val:],
            y_direction=y_direction[n_train + n_val:],
            timestamps=timestamps[n_train + n_val:],
            close_prices=close_prices[n_train + n_val:]
        )
        
        print(f"   Train: {len(train_data.X):,} samples "
              f"({train_data.timestamps[0]} to {train_data.timestamps[-1]})")
        print(f"   Val:   {len(val_data.X):,} samples "
              f"({val_data.timestamps[0]} to {val_data.timestamps[-1]})")
        print(f"   Test:  {len(test_data.X):,} samples "
              f"({test_data.timestamps[0]} to {test_data.timestamps[-1]})")
        
        return train_data, val_data, test_data
    
    def fit_scaler(self, train_data: PredictionDataset) -> None:
        """Fit scaler on training data."""
        print("📏 Fitting scaler on training data...")
        
        self.scaler = RobustScaler()
        self.scaler.fit(train_data.X)
        
        print("   Scaler fitted successfully")
    
    def transform_data(self, data: PredictionDataset) -> PredictionDataset:
        """Apply scaling to a dataset."""
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Call fit_scaler() first.")
        
        X_scaled = self.scaler.transform(data.X).astype(np.float32)
        
        return PredictionDataset(
            X=X_scaled,
            y_return=data.y_return,
            y_direction=data.y_direction,
            timestamps=data.timestamps,
            close_prices=data.close_prices
        )
    
    def prepare_data(self) -> Tuple[PredictionDataset, PredictionDataset, PredictionDataset]:
        """Full data preparation pipeline."""
        print("\n" + "=" * 60)
        print("🚀 Starting Data Preparation Pipeline")
        print("=" * 60 + "\n")
        
        # Load and process
        self.load_data()
        self.compute_features()
        
        # Prepare features and labels
        X, y_return, y_direction, timestamps, close_prices = self.prepare_features_and_labels()
        
        # Split data
        train_data, val_data, test_data = self.split_data(
            X, y_return, y_direction, timestamps, close_prices
        )
        
        # Fit scaler on training data
        self.fit_scaler(train_data)
        
        # Transform all datasets
        self.train_data = self.transform_data(train_data)
        self.val_data = self.transform_data(val_data)
        self.test_data = self.transform_data(test_data)
        
        print("\n" + "=" * 60)
        print("✅ Data Preparation Complete")
        print("=" * 60 + "\n")
        
        return self.train_data, self.val_data, self.test_data
    
    def save_scaler(self, path: str) -> None:
        """Save fitted scaler to disk."""
        scaler_data = {
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'prediction_horizon': self.prediction_horizon,
            'direction_threshold': self.direction_threshold
        }
        with open(path, 'wb') as f:
            pickle.dump(scaler_data, f)
        print(f"💾 Scaler saved to: {path}")
    
    def load_scaler(self, path: str) -> None:
        """Load scaler from disk."""
        with open(path, 'rb') as f:
            scaler_data = pickle.load(f)
        
        self.scaler = scaler_data['scaler']
        self.feature_names = scaler_data['feature_names']
        self.prediction_horizon = scaler_data['prediction_horizon']
        self.direction_threshold = scaler_data['direction_threshold']
        
        print(f"📂 Scaler loaded from: {path}")
    
    def get_feature_names(self) -> List[str]:
        """Get feature names."""
        return self.feature_names


# =============================================================================
# Utility Functions
# =============================================================================

def prepare_single_sample(df: pd.DataFrame, 
                          pipeline: CryptoPredictionDataPipeline) -> np.ndarray:
    """
    Prepare a single sample for inference (e.g., live prediction).
    
    Args:
        df: DataFrame with OHLCV data (needs enough rows for feature calculation)
        pipeline: Fitted data pipeline with scaler
        
    Returns:
        Scaled features ready for model input [1, num_features]
    """
    # Compute features
    feature_df = compute_all_features(df, pipeline.feature_config)
    
    # Extract last row's features
    X = feature_df[pipeline.feature_names].values[-1:].astype(np.float32)
    
    # Apply scaling
    X_scaled = pipeline.scaler.transform(X)
    
    return X_scaled


if __name__ == "__main__":
    # Test the data pipeline
    print("Testing simplified data pipeline module...")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 10000
    
    dates = pd.date_range(start='2021-01-01', periods=n_samples, freq='5min')
    close = 30000 + np.cumsum(np.random.randn(n_samples) * 10)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': close + np.random.randn(n_samples) * 5,
        'high': close + np.abs(np.random.randn(n_samples) * 20),
        'low': close - np.abs(np.random.randn(n_samples) * 20),
        'close': close,
        'vwap': close + np.random.randn(n_samples) * 2,
        'volume': np.abs(np.random.randn(n_samples) * 100) + 100,
        'count': np.random.randint(1000, 10000, n_samples),
        'symbol': 'BTCUSDT'
    })
    
    # Save temp CSV
    temp_path = '/tmp/test_btc.csv'
    df.to_csv(temp_path, index=False)
    
    # Initialize pipeline with new settings
    pipeline = CryptoPredictionDataPipeline(
        csv_path=temp_path,
        prediction_horizon=60,      # 60 minutes
        direction_threshold=0.003   # 0.3%
    )
    
    # Prepare data
    train_data, val_data, test_data = pipeline.prepare_data()
    
    print(f"\nDataset shapes:")
    print(f"  Train X: {train_data.X.shape}")
    print(f"  Val X: {val_data.X.shape}")
    print(f"  Test X: {test_data.X.shape}")
    
    # Clean up
    os.remove(temp_path)
    
    print("\n✅ Simplified data pipeline test passed!")