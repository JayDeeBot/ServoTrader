#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
feature_engineering.py

Technical indicator calculations for cryptocurrency price prediction.

This module provides functions to compute various technical indicators
from OHLCV data, including momentum, volatility, trend, and volume indicators.

Updated for v2.2: Added sentiment features (Fear & Greed Index)

Author: Jarred Deluca
Project: ServoTrader - Prediction Subsystem
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from scipy.stats import linregress


# =============================================================================
# Helper Functions
# =============================================================================

def _safe_divide(numerator: pd.Series, denominator: pd.Series, fill_value: float = 0.0) -> pd.Series:
    """Safely divide two series, replacing inf/nan with fill_value."""
    result = numerator / denominator
    result = result.replace([np.inf, -np.inf], fill_value)
    result = result.fillna(fill_value)
    return result


def _slope_func(x: np.ndarray) -> float:
    """Compute the slope of a linear regression line."""
    x = np.asarray(x, dtype=np.float64)
    if np.any(np.isnan(x)) or np.all(x == x[0]) or len(x) < 2:
        return 0.0
    return linregress(np.arange(len(x)), x).slope


# =============================================================================
# Momentum Indicators
# =============================================================================

def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute Relative Strength Index (RSI)."""
    delta = close.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = _safe_divide(avg_gain, avg_loss, fill_value=0.0)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.fillna(50.0)


def compute_macd(close: pd.Series, 
                 fast_period: int = 12, 
                 slow_period: int = 26, 
                 signal_period: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Compute MACD (Moving Average Convergence Divergence)."""
    ema_fast = close.ewm(span=fast_period, min_periods=fast_period, adjust=False).mean()
    ema_slow = close.ewm(span=slow_period, min_periods=slow_period, adjust=False).mean()
    
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, min_periods=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line
    
    return macd_line.fillna(0.0), signal_line.fillna(0.0), histogram.fillna(0.0)


# =============================================================================
# Volatility Indicators
# =============================================================================

def compute_bollinger_bands(close: pd.Series, 
                            period: int = 20, 
                            num_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Compute Bollinger Bands."""
    middle_band = close.rolling(window=period, min_periods=period).mean()
    std = close.rolling(window=period, min_periods=period).std()
    
    upper_band = middle_band + (num_std * std)
    lower_band = middle_band - (num_std * std)
    
    band_width = upper_band - lower_band
    bb_pct = _safe_divide(close - lower_band, band_width, fill_value=0.5)
    bb_pct = bb_pct.clip(0.0, 1.0)
    
    middle_band = middle_band.fillna(close)
    upper_band = upper_band.fillna(close)
    lower_band = lower_band.fillna(close)
    bb_pct = bb_pct.fillna(0.5)
    
    return upper_band, middle_band, lower_band, bb_pct


def compute_atr(high: pd.Series, 
                low: pd.Series, 
                close: pd.Series, 
                period: int = 14) -> pd.Series:
    """Compute Average True Range (ATR)."""
    prev_close = close.shift(1)
    
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return atr.fillna(true_range)


def compute_volatility(returns: pd.Series, period: int = 20) -> pd.Series:
    """Compute rolling volatility (standard deviation of returns)."""
    volatility = returns.rolling(window=period, min_periods=1).std()
    return volatility.fillna(0.0)


# =============================================================================
# Trend Indicators
# =============================================================================

def compute_ema(close: pd.Series, period: int) -> pd.Series:
    """Compute Exponential Moving Average (EMA)."""
    ema = close.ewm(span=period, min_periods=period, adjust=False).mean()
    return ema.fillna(close)


def compute_trend_slope(close: pd.Series, period: int = 20) -> pd.Series:
    """Compute the trend slope using linear regression."""
    slope = close.rolling(window=period, min_periods=period).apply(_slope_func, raw=True)
    return slope.fillna(0.0)


def compute_price_position(high: pd.Series, 
                           low: pd.Series, 
                           close: pd.Series, 
                           period: int = 20) -> pd.Series:
    """Compute price position within recent high-low range."""
    high_roll = high.rolling(window=period, min_periods=1).max()
    low_roll = low.rolling(window=period, min_periods=1).min()
    
    price_range = high_roll - low_roll
    position = _safe_divide(close - low_roll, price_range, fill_value=0.5)
    
    return position.clip(0.0, 1.0)


# =============================================================================
# Volume Indicators
# =============================================================================

def compute_volume_sma_ratio(volume: pd.Series, period: int = 20) -> pd.Series:
    """Compute volume relative to its moving average."""
    volume_sma = volume.rolling(window=period, min_periods=1).mean()
    ratio = _safe_divide(volume, volume_sma, fill_value=1.0)
    return ratio


# =============================================================================
# Time Features
# =============================================================================

def compute_time_features(timestamps: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Compute cyclical time features from timestamps."""
    if not pd.api.types.is_datetime64_any_dtype(timestamps):
        timestamps = pd.to_datetime(timestamps)
    
    hours = timestamps.dt.hour
    day_of_week = timestamps.dt.dayofweek
    
    hour_sin = np.sin(2 * np.pi * hours / 24)
    hour_cos = np.cos(2 * np.pi * hours / 24)
    day_of_week_norm = day_of_week / 6.0
    
    return (pd.Series(hour_sin, index=timestamps.index),
            pd.Series(hour_cos, index=timestamps.index),
            pd.Series(day_of_week_norm, index=timestamps.index))


# =============================================================================
# Main Feature Engineering Function
# =============================================================================

def compute_all_features(df: pd.DataFrame, 
                         config: Optional[Dict] = None) -> pd.DataFrame:
    """
    Compute all technical indicators and features from OHLCV data.
    
    Args:
        df: DataFrame with columns: timestamp, open, high, low, close, vwap, volume, count
        config: Optional configuration dict with indicator parameters
        
    Returns:
        DataFrame with all original columns plus computed features
    """
    if config is None:
        config = {
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
    
    result = df.copy()
    
    if 'timestamp' in result.columns:
        result = result.sort_values('timestamp').reset_index(drop=True)
    
    close = result['close']
    high = result['high']
    low = result['low']
    volume = result['volume']
    
    # ----- Returns (multiple periods) -----
    result['returns'] = close.pct_change().fillna(0.0)
    result['returns_5'] = close.pct_change(periods=5).fillna(0.0)   # 25 min return
    result['returns_20'] = close.pct_change(periods=20).fillna(0.0) # 100 min return
    
    # ----- RSI -----
    result['rsi_14'] = compute_rsi(close, period=config['rsi_period'])
    result['rsi_14_norm'] = result['rsi_14'] / 100.0
    
    # ----- MACD -----
    macd, macd_signal, macd_hist = compute_macd(
        close, 
        fast_period=config['macd_fast'],
        slow_period=config['macd_slow'],
        signal_period=config['macd_signal']
    )
    result['macd'] = macd
    result['macd_signal'] = macd_signal
    result['macd_histogram'] = macd_hist
    
    # Normalize MACD by price for comparability
    result['macd'] = _safe_divide(result['macd'], close, 0.0)
    result['macd_signal'] = _safe_divide(result['macd_signal'], close, 0.0)
    result['macd_histogram'] = _safe_divide(result['macd_histogram'], close, 0.0)
    
    # ----- Bollinger Bands -----
    bb_upper, bb_middle, bb_lower, bb_pct = compute_bollinger_bands(
        close, 
        period=config['bb_period'],
        num_std=config['bb_std']
    )
    result['bb_upper'] = bb_upper
    result['bb_middle'] = bb_middle
    result['bb_lower'] = bb_lower
    result['bb_pct'] = bb_pct
    result['bb_upper_pct'] = _safe_divide(bb_upper - close, close, 0.0)
    result['bb_lower_pct'] = _safe_divide(bb_lower - close, close, 0.0)
    
    # ----- ATR -----
    atr = compute_atr(high, low, close, period=config['atr_period'])
    result['atr_14'] = atr
    result['atr_14_pct'] = _safe_divide(atr, close, 0.0)
    
    # ----- EMAs -----
    for period in config['ema_periods']:
        ema = compute_ema(close, period)
        result[f'ema_{period}'] = ema
        result[f'ema_{period}_pct'] = _safe_divide(ema - close, close, 0.0)
    
    # ----- Volatility -----
    result['volatility_20'] = compute_volatility(result['returns'], period=config['volatility_period'])
    
    # ----- Volume Indicators -----
    result['volume_sma_ratio'] = compute_volume_sma_ratio(volume, period=config['volume_sma_period'])
    
    # ----- Price Position -----
    result['price_position'] = compute_price_position(high, low, close, period=config['bb_period'])
    
    # ----- Trend Slope -----
    result['trend_slope'] = compute_trend_slope(close, period=config['trend_slope_period'])
    # Normalize trend slope
    result['trend_slope'] = _safe_divide(result['trend_slope'], close, 0.0)
    
    # ----- Time Features -----
    if 'timestamp' in result.columns:
        hour_sin, hour_cos, day_of_week = compute_time_features(result['timestamp'])
        result['hour_sin'] = hour_sin.values
        result['hour_cos'] = hour_cos.values
        result['day_of_week'] = day_of_week.values
    
    # ----- Final cleanup -----
    result = result.replace([np.inf, -np.inf], 0.0)
    result = result.fillna(0.0)
    
    return result


def get_static_feature_names(include_sentiment: bool = True) -> list:
    """
    Get the list of feature names for LightGBM input.
    
    Args:
        include_sentiment: Whether to include sentiment features
    
    Returns:
        List of feature column names
    """
    features = [
        # Returns
        'returns',
        'returns_5',
        'returns_20',
        # RSI
        'rsi_14_norm',
        # MACD
        'macd',
        'macd_signal',
        'macd_histogram',
        # Bollinger Bands
        'bb_upper_pct',
        'bb_lower_pct',
        'bb_pct',
        # Volatility
        'atr_14_pct',
        'volatility_20',
        # EMAs
        'ema_5_pct',
        'ema_20_pct',
        'ema_60_pct',
        # Volume
        'volume_sma_ratio',
        # Price action
        'price_position',
        'trend_slope',
        # Time
        'hour_sin',
        'hour_cos',
        'day_of_week'
    ]
    
    if include_sentiment:
        # Sentiment features from Fear & Greed Index
        sentiment_features = [
            'fg_normalized',        # Fear & Greed 0-1 scale
            'fg_extreme_fear',      # Binary: extreme fear indicator
            'fg_extreme_greed',     # Binary: extreme greed indicator
            'fg_change',            # Day-over-day change
            'fg_momentum_7',        # Current vs 7-day MA
            'fg_volatility_7',      # 7-day volatility of sentiment
        ]
        features.extend(sentiment_features)
    
    return features


if __name__ == "__main__":
    print("Testing feature engineering module...")
    
    np.random.seed(42)
    n_samples = 1000
    
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
    
    result = compute_all_features(df)
    
    print(f"\nFeature columns: {len(get_static_feature_names())}")
    print(f"Features: {get_static_feature_names()}")
    
    print("\n✅ Feature engineering module test passed!")