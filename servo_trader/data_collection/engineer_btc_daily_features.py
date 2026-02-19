#!/usr/bin/env python3
"""
engineer_btc_daily_features.py

Comprehensive feature engineering for Bitcoin daily price prediction.
Combines OHLCV, on-chain, and sentiment data with extensive technical indicators.

Author: Jarred Deluca
Created: 2025
License: MIT
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================

# Input files
OHLCV_FILE     = "/home/jarred/git/ServoTrader/data/daily_historical/BTCUSDT.csv"
ONCHAIN_FILE   = "/home/jarred/git/ServoTrader/data/btc_onchain_daily_metrics.csv"
SENTIMENT_FILE = "/home/jarred/git/ServoTrader/data/btc_sentiment_daily.csv"

# Output file
OUTPUT_FILE = "/home/jarred/git/ServoTrader/data/btc_daily_features_engineered.csv"

# Feature engineering parameters
LOOKBACK_PERIODS = [3, 5, 7, 10, 14, 21, 30, 60, 90]  # Rolling window sizes

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def section(title):
    """Print section header."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")


# ============================================================================
# TECHNICAL INDICATORS
# ============================================================================

def calculate_rsi(series, period=14):
    """Calculate Relative Strength Index."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(series, fast=12, slow=26, signal=9):
    """Calculate MACD, Signal, and Histogram."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram


def calculate_bollinger_bands(series, period=20, std_dev=2):
    """Calculate Bollinger Bands."""
    ma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = ma + (std * std_dev)
    lower = ma - (std * std_dev)
    bandwidth = (upper - lower) / ma
    percent_b = (series - lower) / (upper - lower + 1e-10)
    return upper, lower, ma, bandwidth, percent_b


def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range."""
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr


def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index (trend strength)."""
    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    plus_dm = pd.Series(plus_dm, index=close.index)
    minus_dm = pd.Series(minus_dm, index=close.index)
    
    # Smoothed values
    atr = tr.rolling(window=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / (atr + 1e-10))
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / (atr + 1e-10))
    
    # ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.rolling(window=period).mean()
    
    return adx, plus_di, minus_di


def calculate_stochastic(high, low, close, k_period=14, d_period=3):
    """Calculate Stochastic Oscillator."""
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    
    k = 100 * ((close - lowest_low) / (highest_high - lowest_low + 1e-10))
    d = k.rolling(window=d_period).mean()
    
    return k, d


def calculate_obv(close, volume):
    """Calculate On-Balance Volume."""
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    return obv


def calculate_cci(high, low, close, period=20):
    """Calculate Commodity Channel Index."""
    tp = (high + low + close) / 3
    ma = tp.rolling(window=period).mean()
    md = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
    cci = (tp - ma) / (0.015 * md + 1e-10)
    return cci


def calculate_williams_r(high, low, close, period=14):
    """Calculate Williams %R."""
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    wr = -100 * ((highest_high - close) / (highest_high - lowest_low + 1e-10))
    return wr


def calculate_roc(series, period=12):
    """Calculate Rate of Change."""
    roc = ((series - series.shift(period)) / (series.shift(period) + 1e-10)) * 100
    return roc


def calculate_mfi(high, low, close, volume, period=14):
    """Calculate Money Flow Index."""
    tp = (high + low + close) / 3
    mf = tp * volume
    
    mf_pos = mf.where(tp > tp.shift(1), 0)
    mf_neg = mf.where(tp < tp.shift(1), 0)
    
    mf_ratio = (mf_pos.rolling(window=period).sum() / 
                (mf_neg.rolling(window=period).sum() + 1e-10))
    mfi = 100 - (100 / (1 + mf_ratio))
    return mfi


# ============================================================================
# LOAD AND MERGE DATA
# ============================================================================

def load_and_merge_data():
    """Load all three datasets and merge on date."""
    
    section("LOADING DATA")
    
    # Load OHLCV
    print("  📊 Loading OHLCV data...")
    ohlcv = pd.read_csv(OHLCV_FILE)
    ohlcv['date'] = pd.to_datetime(ohlcv['timestamp']).dt.date
    ohlcv = ohlcv.sort_values('date').reset_index(drop=True)
    ohlcv = ohlcv.drop(columns=['timestamp', 'symbol'], errors='ignore')
    print(f"     ✅ {len(ohlcv)} rows  |  {ohlcv['date'].min()} → {ohlcv['date'].max()}")
    
    # Load on-chain
    print("  ⛓️  Loading on-chain data...")
    onchain = pd.read_csv(ONCHAIN_FILE)
    onchain['date'] = pd.to_datetime(onchain['date']).dt.date
    onchain = onchain.sort_values('date').reset_index(drop=True)
    # Remove duplicate price column (we'll use OHLCV price)
    onchain = onchain.drop(columns=['price_usd', 'price_ma7', 'price_ma30', 
                                     'price_pct_change', 'log_returns'], errors='ignore')
    print(f"     ✅ {len(onchain)} rows  |  {onchain['date'].min()} → {onchain['date'].max()}")
    
    # Load sentiment
    print("  💭 Loading sentiment data...")
    sentiment = pd.read_csv(SENTIMENT_FILE)
    sentiment['date'] = pd.to_datetime(sentiment['date']).dt.date
    sentiment = sentiment.sort_values('date').reset_index(drop=True)
    # Remove duplicate fear_greed_index (already in onchain)
    sentiment = sentiment.drop(columns=['fear_greed_value'], errors='ignore')
    print(f"     ✅ {len(sentiment)} rows  |  {sentiment['date'].min()} → {sentiment['date'].max()}")
    
    # Merge
    section("MERGING DATASETS")
    df = ohlcv.copy()
    df = df.merge(onchain, on='date', how='left')
    print(f"  ✅ Merged on-chain data  |  {len(df)} rows, {len(df.columns)} columns")
    
    df = df.merge(sentiment, on='date', how='left')
    print(f"  ✅ Merged sentiment data  |  {len(df)} rows, {len(df.columns)} columns")
    
    df = df.sort_values('date').reset_index(drop=True)
    
    # Report missing data
    print(f"\n  📋 Missing Data Summary:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    for col in missing[missing > 0].index:
        print(f"     {col:45s} {missing[col]:5d} missing ({missing_pct[col]:5.1f}%)")
    
    return df


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def engineer_technical_indicators(df):
    """Calculate extensive technical indicators from OHLCV data."""
    
    section("ENGINEERING TECHNICAL INDICATORS")
    
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # Price-based indicators
    print("  📈 Price indicators...")
    
    # RSI (multiple periods)
    for period in [7, 14, 21, 30]:
        df[f'rsi_{period}'] = calculate_rsi(close, period)
    
    # MACD
    df['macd'], df['macd_signal'], df['macd_histogram'] = calculate_macd(close)
    
    # Bollinger Bands
    for period in [14, 20, 30]:
        upper, lower, ma, bandwidth, percent_b = calculate_bollinger_bands(close, period)
        df[f'bb_upper_{period}'] = upper
        df[f'bb_lower_{period}'] = lower
        df[f'bb_middle_{period}'] = ma
        df[f'bb_bandwidth_{period}'] = bandwidth
        df[f'bb_percent_b_{period}'] = percent_b
    
    # Moving Averages
    for period in [5, 10, 20, 50, 100, 200]:
        df[f'sma_{period}'] = close.rolling(window=period).mean()
        df[f'ema_{period}'] = close.ewm(span=period, adjust=False).mean()
    
    # Price position relative to MAs
    df['price_above_sma_20'] = (close > df['sma_20']).astype(int)
    df['price_above_sma_50'] = (close > df['sma_50']).astype(int)
    df['price_above_sma_200'] = (close > df['sma_200']).astype(int)
    
    # Golden Cross / Death Cross indicators
    df['sma_50_above_200'] = (df['sma_50'] > df['sma_200']).astype(int)
    
    # ATR (volatility)
    for period in [7, 14, 21]:
        df[f'atr_{period}'] = calculate_atr(high, low, close, period)
    
    # ADX (trend strength)
    df['adx_14'], df['plus_di_14'], df['minus_di_14'] = calculate_adx(high, low, close, 14)
    
    # Stochastic
    df['stoch_k'], df['stoch_d'] = calculate_stochastic(high, low, close)
    
    # CCI
    df['cci_20'] = calculate_cci(high, low, close, 20)
    
    # Williams %R
    df['williams_r_14'] = calculate_williams_r(high, low, close, 14)
    
    # Rate of Change
    for period in [5, 10, 20]:
        df[f'roc_{period}'] = calculate_roc(close, period)
    
    # MFI (Money Flow Index)
    df['mfi_14'] = calculate_mfi(high, low, close, volume, 14)
    
    print("     ✅ Technical indicators calculated")
    
    # Volume indicators
    print("  📊 Volume indicators...")
    
    # OBV
    df['obv'] = calculate_obv(close, volume)
    df['obv_ma_20'] = df['obv'].rolling(window=20).mean()
    
    # Volume moving averages
    for period in [5, 10, 20, 30]:
        df[f'volume_ma_{period}'] = volume.rolling(window=period).mean()
    
    # Volume ratios
    df['volume_ratio_5'] = volume / (df['volume_ma_5'] + 1e-10)
    df['volume_ratio_20'] = volume / (df['volume_ma_20'] + 1e-10)
    
    # VWAP deviation
    df['vwap_deviation'] = (close - df['vwap']) / (df['vwap'] + 1e-10)
    
    print("     ✅ Volume indicators calculated")
    
    # Momentum and trend
    print("  🚀 Momentum indicators...")
    
    # Price momentum (N-day returns)
    for period in [1, 3, 5, 7, 10, 14, 21, 30]:
        df[f'returns_{period}d'] = close.pct_change(period)
    
    # Volatility (rolling std)
    for period in [5, 10, 20, 30]:
        df[f'volatility_{period}d'] = close.pct_change().rolling(window=period).std()
    
    # Higher highs, lower lows detection
    df['higher_high'] = ((high > high.shift(1)) & (high.shift(1) > high.shift(2))).astype(int)
    df['lower_low'] = ((low < low.shift(1)) & (low.shift(1) < low.shift(2))).astype(int)
    
    # Price range metrics
    df['daily_range'] = high - low
    df['daily_range_pct'] = (high - low) / (close + 1e-10)
    
    # Gap detection
    df['gap_up'] = (low > high.shift(1)).astype(int)
    df['gap_down'] = (high < low.shift(1)).astype(int)
    
    print("     ✅ Momentum indicators calculated")
    
    return df


def engineer_cross_sectional_features(df):
    """Create ratio and interaction features between different metrics."""
    
    section("ENGINEERING CROSS-SECTIONAL FEATURES")
    
    # Price vs on-chain ratios
    if 'hash_rate_ths' in df.columns:
        df['price_to_hashrate'] = df['close'] / (df['hash_rate_ths'] + 1e-10)
    
    if 'transaction_count' in df.columns:
        df['price_to_tx_count'] = df['close'] / (df['transaction_count'] + 1e-10)
    
    if 'unique_addresses' in df.columns:
        df['price_to_addresses'] = df['close'] / (df['unique_addresses'] + 1e-10)
    
    # Volume vs transaction ratios
    if 'estimated_volume_btc' in df.columns:
        df['exchange_to_onchain_volume'] = df['volume'] / (df['estimated_volume_btc'] + 1e-10)
    
    # Volatility vs volume
    df['vol_volatility_product'] = df['volatility_20d'] * df['volume_ratio_20']
    
    # Sentiment vs price momentum
    if 'fear_greed_index' in df.columns:
        df['sentiment_momentum_interaction'] = df['fear_greed_index'] * df['returns_7d']
    
    # RSI divergence from sentiment
    if 'fear_greed_index' in df.columns:
        df['rsi_sentiment_divergence'] = df['rsi_14'] - df['fear_greed_index']
    
    print("  ✅ Cross-sectional features calculated")
    
    return df


def engineer_lag_features(df):
    """Create lagged features for temporal patterns."""
    
    section("ENGINEERING LAG FEATURES")
    
    # Lag important features
    lag_cols = ['close', 'volume', 'rsi_14', 'macd', 'atr_14']
    if 'fear_greed_index' in df.columns:
        lag_cols.append('fear_greed_index')
    if 'hash_rate_ths' in df.columns:
        lag_cols.append('hash_rate_ths')
    
    for col in lag_cols:
        if col in df.columns:
            for lag in [1, 3, 7]:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    
    print(f"  ✅ Lag features calculated for {len(lag_cols)} variables")
    
    return df


def engineer_rolling_statistics(df):
    """Calculate rolling statistics for key features."""
    
    section("ENGINEERING ROLLING STATISTICS")
    
    # Rolling stats on returns
    for period in [7, 14, 30]:
        df[f'returns_mean_{period}d'] = df['returns_1d'].rolling(window=period).mean()
        df[f'returns_std_{period}d'] = df['returns_1d'].rolling(window=period).std()
        df[f'returns_max_{period}d'] = df['returns_1d'].rolling(window=period).max()
        df[f'returns_min_{period}d'] = df['returns_1d'].rolling(window=period).min()
        df[f'returns_skew_{period}d'] = df['returns_1d'].rolling(window=period).skew()
    
    # Rolling stats on volume
    for period in [7, 14, 30]:
        df[f'volume_std_{period}d'] = df['volume'].rolling(window=period).std()
        df[f'volume_max_{period}d'] = df['volume'].rolling(window=period).max()
    
    print("  ✅ Rolling statistics calculated")
    
    return df


def engineer_market_regime_features(df):
    """Detect market regime (trending, ranging, volatile)."""
    
    section("ENGINEERING MARKET REGIME FEATURES")
    
    # Trend regime (based on ADX)
    if 'adx_14' in df.columns:
        df['trending_regime'] = (df['adx_14'] > 25).astype(int)
        df['strong_trend'] = (df['adx_14'] > 40).astype(int)
    
    # Volatility regime (based on ATR percentile)
    if 'atr_14' in df.columns:
        df['atr_percentile'] = df['atr_14'].rolling(window=90).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1]
        )
        df['high_volatility_regime'] = (df['atr_percentile'] > 0.75).astype(int)
        df['low_volatility_regime'] = (df['atr_percentile'] < 0.25).astype(int)
    
    # Bull/Bear regime (based on MA positions)
    if 'sma_50' in df.columns and 'sma_200' in df.columns:
        df['bull_regime'] = ((df['close'] > df['sma_50']) & 
                              (df['sma_50'] > df['sma_200'])).astype(int)
        df['bear_regime'] = ((df['close'] < df['sma_50']) & 
                              (df['sma_50'] < df['sma_200'])).astype(int)
    
    # Consolidation regime (low volatility + ranging price)
    if 'volatility_20d' in df.columns:
        vol_threshold = df['volatility_20d'].rolling(window=90).quantile(0.25)
        df['consolidation_regime'] = (df['volatility_20d'] < vol_threshold).astype(int)
    
    print("  ✅ Market regime features calculated")
    
    return df


def engineer_cyclical_features(df):
    """Add cyclical time-based features."""
    
    section("ENGINEERING CYCLICAL FEATURES")
    
    # Convert date to datetime for extraction
    df['datetime'] = pd.to_datetime(df['date'])
    
    # Day of week (Monday = 0, Sunday = 6)
    df['day_of_week'] = df['datetime'].dt.dayofweek
    
    # Day of month
    df['day_of_month'] = df['datetime'].dt.day
    
    # Month
    df['month'] = df['datetime'].dt.month
    
    # Quarter
    df['quarter'] = df['datetime'].dt.quarter
    
    # Halving cycle approximation (Bitcoin halving roughly every 4 years)
    # First halving: Nov 28, 2012
    days_since_first_halving = (df['datetime'] - pd.Timestamp('2012-11-28')).dt.days
    df['days_since_halving'] = days_since_first_halving % (4 * 365)  # Approximate 4-year cycle
    df['halving_cycle_position'] = df['days_since_halving'] / (4 * 365)  # 0 to 1
    
    # Is it weekend? (Saturday or Sunday)
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Is it month end? (last 3 days of month)
    df['is_month_end'] = (df['day_of_month'] >= 28).astype(int)
    
    # Drop temporary datetime column
    df = df.drop(columns=['datetime'])
    
    print("  ✅ Cyclical features calculated")
    
    return df


def create_target_variables(df):
    """Create target variables for prediction."""
    
    section("CREATING TARGET VARIABLES")
    
    # Next day price change (for regression)
    df['target_next_day_return'] = df['close'].pct_change().shift(-1)
    df['target_next_day_close'] = df['close'].shift(-1)
    
    # Next day direction (for classification)
    df['target_next_day_direction'] = (df['target_next_day_return'] > 0).astype(int)
    
    # Next day high/low (for range prediction)
    df['target_next_day_high'] = df['high'].shift(-1)
    df['target_next_day_low'] = df['low'].shift(-1)
    
    # Multi-day targets (3-day, 7-day)
    df['target_3day_return'] = (df['close'].shift(-3) - df['close']) / df['close']
    df['target_7day_return'] = (df['close'].shift(-7) - df['close']) / df['close']
    
    df['target_3day_direction'] = (df['target_3day_return'] > 0).astype(int)
    df['target_7day_direction'] = (df['target_7day_return'] > 0).astype(int)
    
    print("  ✅ Target variables created")
    print(f"     - Next day return (regression)")
    print(f"     - Next day direction (binary classification)")
    print(f"     - Next day high/low (range)")
    print(f"     - 3-day and 7-day returns/directions")
    
    return df


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution."""
    
    section("BTC DAILY FEATURE ENGINEERING")
    print(f"  Input files:")
    print(f"    - OHLCV:     {OHLCV_FILE}")
    print(f"    - On-chain:  {ONCHAIN_FILE}")
    print(f"    - Sentiment: {SENTIMENT_FILE}")
    print(f"  Output file:   {OUTPUT_FILE}")
    
    # Load and merge
    df = load_and_merge_data()
    
    # Engineer features
    df = engineer_technical_indicators(df)
    df = engineer_cross_sectional_features(df)
    df = engineer_lag_features(df)
    df = engineer_rolling_statistics(df)
    df = engineer_market_regime_features(df)
    df = engineer_cyclical_features(df)
    
    # Create targets
    df = create_target_variables(df)
    
    # Final cleanup
    section("FINAL PROCESSING")
    
    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)
    
    # Report final stats
    print(f"\n  📊 Final Dataset Statistics:")
    print(f"     Total rows:    {len(df):,}")
    print(f"     Total columns: {len(df.columns):,}")
    print(f"     Date range:    {df['date'].min()} → {df['date'].max()}")
    
    # Report missing data
    total_missing = df.isnull().sum().sum()
    total_cells = len(df) * len(df.columns)
    missing_pct = (total_missing / total_cells) * 100
    print(f"     Missing data:  {total_missing:,} cells ({missing_pct:.2f}%)")
    
    # Columns with most missing data
    print(f"\n  ⚠️  Features with >20% missing data:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    high_missing = missing_pct[missing_pct > 20].sort_values(ascending=False)
    
    if len(high_missing) > 0:
        for col in high_missing.index[:10]:  # Show top 10
            print(f"     {col:45s} {missing_pct[col]:5.1f}%")
    else:
        print(f"     None")
    
    # Save
    section("SAVING OUTPUT")
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"  ✅ Saved to: {OUTPUT_FILE}")
    
    # Feature categories summary
    section("FEATURE CATEGORIES SUMMARY")
    
    ohlcv_base = ['open', 'high', 'low', 'close', 'vwap', 'volume', 'count']
    technical = [c for c in df.columns if any(x in c for x in ['rsi', 'macd', 'bb_', 'sma', 'ema', 
                                                                  'atr', 'adx', 'stoch', 'cci', 
                                                                  'williams', 'roc', 'mfi', 'obv'])]
    onchain = [c for c in df.columns if any(x in c for x in ['hash', 'difficulty', 'transaction', 
                                                               'addresses', 'mempool', 'utxo', 'miners'])]
    sentiment = [c for c in df.columns if any(x in c for x in ['fear_greed', 'trends', 'wikipedia', 
                                                                 'github', 'reddit', 'stocktwits', 'gdelt'])]
    lag = [c for c in df.columns if 'lag' in c]
    rolling = [c for c in df.columns if any(x in c for x in ['_mean_', '_std_', '_max_', '_min_', '_skew_'])]
    regime = [c for c in df.columns if 'regime' in c]
    cyclical = [c for c in df.columns if c in ['day_of_week', 'day_of_month', 'month', 'quarter', 
                                                 'days_since_halving', 'halving_cycle_position', 
                                                 'is_weekend', 'is_month_end']]
    targets = [c for c in df.columns if 'target' in c]
    
    print(f"  📈 OHLCV Base:          {len(ohlcv_base):3d} features")
    print(f"  📊 Technical:           {len(technical):3d} features")
    print(f"  ⛓️  On-chain:            {len(onchain):3d} features")
    print(f"  💭 Sentiment:           {len(sentiment):3d} features")
    print(f"  ⏮️  Lag:                 {len(lag):3d} features")
    print(f"  📉 Rolling Stats:       {len(rolling):3d} features")
    print(f"  🔄 Market Regime:       {len(regime):3d} features")
    print(f"  📅 Cyclical:            {len(cyclical):3d} features")
    print(f"  🎯 Targets:             {len(targets):3d} variables")
    print(f"  {'─'*60}")
    print(f"  📦 Total Features:      {len(df.columns):3d} columns")
    
    # Show sample
    print(f"\n  📋 First 3 rows (select columns):")
    sample_cols = ['date', 'close', 'volume', 'rsi_14', 'macd', 'fear_greed_index', 
                   'hash_rate_ths', 'target_next_day_return', 'target_next_day_direction']
    sample_cols = [c for c in sample_cols if c in df.columns]
    print(df[sample_cols].head(3).to_string(index=False))
    
    section("✅ FEATURE ENGINEERING COMPLETE")
    print(f"\n  🎯 Next steps:")
    print(f"     1. Handle missing data (forward-fill, drop, or impute)")
    print(f"     2. Run Boruta feature selection")
    print(f"     3. Train prediction models (LightGBM, XGBoost)")
    print(f"\n")


if __name__ == "__main__":
    main()