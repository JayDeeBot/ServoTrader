#!/usr/bin/env python3
"""
engineer_btc_hourly_features.py

Comprehensive feature engineering for Bitcoin HOURLY price prediction.
Combines OHLCV, on-chain, and sentiment data with extensive technical indicators.

Key differences from daily version:
- Lookback periods in HOURS (24h, 48h, 168h) instead of days
- Hour-of-day cyclical features instead of day-of-week
- Targets for next hour prediction
- Faster-moving indicators appropriate for hourly timeframe

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
OHLCV_FILE     = "/home/jarred/git/ServoTrader/data/hourly_historical/BTCUSDT.csv"
ONCHAIN_FILE   = "/home/jarred/git/ServoTrader/data/btc_onchain_hourly_metrics.csv"
SENTIMENT_FILE = "/home/jarred/git/ServoTrader/data/btc_sentiment_hourly.csv"

# Output file
OUTPUT_FILE = "/home/jarred/git/ServoTrader/data/btc_hourly_features_engineered.csv"

# Feature engineering parameters (in HOURS)
LOOKBACK_PERIODS = [6, 12, 24, 48, 72, 168, 336]  # 6h, 12h, 1d, 2d, 3d, 1w, 2w

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def section(title):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")


# ============================================================================
# TECHNICAL INDICATORS (Same formulas, different parameters)
# ============================================================================

def calculate_rsi(series, period=14):
    """Calculate RSI."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(series, fast=12, slow=26, signal=9):
    """Calculate MACD."""
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
    """Calculate ATR."""
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr


def calculate_adx(high, low, close, period=14):
    """Calculate ADX."""
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    plus_dm = pd.Series(plus_dm, index=close.index)
    minus_dm = pd.Series(minus_dm, index=close.index)
    
    atr = tr.rolling(window=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / (atr + 1e-10))
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / (atr + 1e-10))
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.rolling(window=period).mean()
    
    return adx, plus_di, minus_di


def calculate_stochastic(high, low, close, k_period=14, d_period=3):
    """Calculate Stochastic."""
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    k = 100 * ((close - lowest_low) / (highest_high - lowest_low + 1e-10))
    d = k.rolling(window=d_period).mean()
    return k, d


def calculate_obv(close, volume):
    """Calculate OBV."""
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    return obv


# ============================================================================
# LOAD AND MERGE DATA
# ============================================================================

def load_and_merge_data():
    """Load all three datasets and merge on timestamp."""
    
    section("LOADING DATA")
    
    # Load OHLCV
    print("  📊 Loading OHLCV data...")
    ohlcv = pd.read_csv(OHLCV_FILE)
    ohlcv['timestamp'] = pd.to_datetime(ohlcv['timestamp'])
    # Remove timezone to avoid merge issues
    if ohlcv['timestamp'].dt.tz is not None:
        ohlcv['timestamp'] = ohlcv['timestamp'].dt.tz_localize(None)
    ohlcv = ohlcv.sort_values('timestamp').reset_index(drop=True)
    ohlcv = ohlcv.drop(columns=['symbol'], errors='ignore')
    print(f"     ✅ {len(ohlcv)} rows  |  {ohlcv['timestamp'].min()} → {ohlcv['timestamp'].max()}")
    
    # Load on-chain
    print("  ⛓️  Loading on-chain data...")
    onchain = pd.read_csv(ONCHAIN_FILE)
    onchain['timestamp'] = pd.to_datetime(onchain['timestamp'])
    # Remove timezone if present
    if onchain['timestamp'].dt.tz is not None:
        onchain['timestamp'] = onchain['timestamp'].dt.tz_localize(None)
    onchain = onchain.sort_values('timestamp').reset_index(drop=True)
    # Remove duplicate price columns
    onchain = onchain.drop(columns=['price_usd'], errors='ignore')
    print(f"     ✅ {len(onchain)} rows  |  {onchain['timestamp'].min()} → {onchain['timestamp'].max()}")
    print(f"     ⚠️  Note: On-chain data is forward-filled from daily values")
    
    # Load sentiment
    print("  💭 Loading sentiment data...")
    sentiment = pd.read_csv(SENTIMENT_FILE)
    sentiment['timestamp'] = pd.to_datetime(sentiment['timestamp'])
    # Remove timezone if present
    if sentiment['timestamp'].dt.tz is not None:
        sentiment['timestamp'] = sentiment['timestamp'].dt.tz_localize(None)
    sentiment = sentiment.sort_values('timestamp').reset_index(drop=True)
    print(f"     ✅ {len(sentiment)} rows  |  {sentiment['timestamp'].min()} → {sentiment['timestamp'].max()}")
    print(f"     ⚠️  Note: Sentiment data is forward-filled from daily values")
    
    # Merge
    section("MERGING DATASETS")
    df = ohlcv.copy()
    df = df.merge(onchain, on='timestamp', how='left')
    print(f"  ✅ Merged on-chain  |  {len(df)} rows, {len(df.columns)} columns")
    
    df = df.merge(sentiment, on='timestamp', how='left')
    print(f"  ✅ Merged sentiment  |  {len(df)} rows, {len(df.columns)} columns")
    
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Report missing data
    print(f"\n  📋 Missing Data Summary:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    for col in missing[missing > 0].index[:10]:
        print(f"     {col:45s} {missing[col]:5d} missing ({missing_pct[col]:5.1f}%)")
    
    return df


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def engineer_technical_indicators(df):
    """Calculate technical indicators optimized for hourly data."""
    
    section("ENGINEERING TECHNICAL INDICATORS (HOURLY)")
    
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # RSI (hourly-appropriate periods)
    print("  📈 Technical indicators...")
    for period in [6, 14, 24, 48]:  # 6h, 14h, 1d, 2d
        df[f'rsi_{period}'] = calculate_rsi(close, period)
    
    # MACD (hourly settings)
    df['macd'], df['macd_signal'], df['macd_histogram'] = calculate_macd(close, fast=12, slow=26, signal=9)
    
    # Bollinger Bands
    for period in [12, 24, 48]:  # 12h, 1d, 2d
        upper, lower, ma, bandwidth, percent_b = calculate_bollinger_bands(close, period)
        df[f'bb_upper_{period}'] = upper
        df[f'bb_lower_{period}'] = lower
        df[f'bb_middle_{period}'] = ma
        df[f'bb_bandwidth_{period}'] = bandwidth
        df[f'bb_percent_b_{period}'] = percent_b
    
    # Moving Averages
    for period in [6, 12, 24, 48, 72, 168]:  # 6h to 1 week
        df[f'sma_{period}'] = close.rolling(window=period).mean()
        df[f'ema_{period}'] = close.ewm(span=period, adjust=False).mean()
    
    # Price position relative to MAs
    df['price_above_sma_24'] = (close > df['sma_24']).astype(int)
    df['price_above_sma_168'] = (close > df['sma_168']).astype(int)
    
    # ATR
    for period in [12, 24, 48]:
        df[f'atr_{period}'] = calculate_atr(high, low, close, period)
    
    # ADX
    df['adx_14'], df['plus_di_14'], df['minus_di_14'] = calculate_adx(high, low, close, 14)
    
    # Stochastic
    df['stoch_k'], df['stoch_d'] = calculate_stochastic(high, low, close)
    
    # OBV
    df['obv'] = calculate_obv(close, volume)
    df['obv_ma_24'] = df['obv'].rolling(window=24).mean()
    
    # Volume
    for period in [6, 12, 24, 48]:
        df[f'volume_ma_{period}'] = volume.rolling(window=period).mean()
    
    df['volume_ratio_24'] = volume / (df['volume_ma_24'] + 1e-10)
    
    # VWAP deviation
    df['vwap_deviation'] = (close - df['vwap']) / (df['vwap'] + 1e-10)
    
    print("     ✅ Technical indicators calculated")
    
    # Momentum
    print("  🚀 Momentum indicators...")
    for period in [1, 6, 12, 24, 48, 72, 168]:  # 1h to 1 week
        df[f'returns_{period}h'] = close.pct_change(period)
    
    # Volatility
    for period in [6, 12, 24, 48]:
        df[f'volatility_{period}h'] = close.pct_change().rolling(window=period).std()
    
    # Price patterns
    df['higher_high'] = ((high > high.shift(1)) & (high.shift(1) > high.shift(2))).astype(int)
    df['lower_low'] = ((low < low.shift(1)) & (low.shift(1) < low.shift(2))).astype(int)
    df['daily_range'] = high - low
    df['daily_range_pct'] = (high - low) / (close + 1e-10)
    
    print("     ✅ Momentum indicators calculated")
    
    return df


def engineer_rolling_statistics(df):
    """Calculate rolling statistics for hourly data."""
    
    section("ENGINEERING ROLLING STATISTICS (HOURLY)")
    
    # Rolling stats on returns
    for period in [24, 48, 168]:  # 1d, 2d, 1w
        df[f'returns_mean_{period}h'] = df['returns_1h'].rolling(window=period).mean()
        df[f'returns_std_{period}h'] = df['returns_1h'].rolling(window=period).std()
        df[f'returns_max_{period}h'] = df['returns_1h'].rolling(window=period).max()
        df[f'returns_min_{period}h'] = df['returns_1h'].rolling(window=period).min()
    
    # Rolling stats on volume
    for period in [24, 48]:
        df[f'volume_std_{period}h'] = df['volume'].rolling(window=period).std()
    
    print("  ✅ Rolling statistics calculated")
    
    return df


def engineer_market_regime_features(df):
    """Detect market regime."""
    
    section("ENGINEERING MARKET REGIME FEATURES")
    
    # Trend regime
    if 'adx_14' in df.columns:
        df['trending_regime'] = (df['adx_14'] > 25).astype(int)
        df['strong_trend'] = (df['adx_14'] > 40).astype(int)
    
    # Volatility regime
    if 'atr_24' in df.columns:
        df['atr_percentile'] = df['atr_24'].rolling(window=168).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1]
        )
        df['high_volatility_regime'] = (df['atr_percentile'] > 0.75).astype(int)
    
    # Bull/Bear regime
    if 'sma_48' in df.columns and 'sma_168' in df.columns:
        df['bull_regime'] = ((df['close'] > df['sma_48']) & 
                             (df['sma_48'] > df['sma_168'])).astype(int)
        df['bear_regime'] = ((df['close'] < df['sma_48']) & 
                             (df['sma_48'] < df['sma_168'])).astype(int)
    
    print("  ✅ Market regime features calculated")
    
    return df


def engineer_cyclical_features(df):
    """Add hourly cyclical features."""
    
    section("ENGINEERING CYCLICAL FEATURES (HOURLY)")
    
    # Hour of day (0-23)
    df['hour_of_day'] = df['timestamp'].dt.hour
    
    # Day of week (0-6)
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    # Is trading hours (assuming 9-16 NYC time, UTC offset)
    df['is_trading_hours'] = df['hour_of_day'].between(14, 21).astype(int)  # 9am-4pm EST in UTC
    
    # Is weekend
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Morning/afternoon/evening/night
    df['is_morning'] = df['hour_of_day'].between(6, 11).astype(int)
    df['is_afternoon'] = df['hour_of_day'].between(12, 17).astype(int)
    df['is_evening'] = df['hour_of_day'].between(18, 23).astype(int)
    df['is_night'] = df['hour_of_day'].between(0, 5).astype(int)
    
    print("  ✅ Cyclical features calculated")
    
    return df


def create_target_variables(df):
    """Create target variables for next-hour prediction."""
    
    section("CREATING TARGET VARIABLES (NEXT HOUR)")
    
    # Next hour prediction
    df['target_next_hour_return'] = df['close'].pct_change().shift(-1)
    df['target_next_hour_close'] = df['close'].shift(-1)
    df['target_next_hour_direction'] = (df['target_next_hour_return'] > 0).astype(int)
    
    # Next hour high/low
    df['target_next_hour_high'] = df['high'].shift(-1)
    df['target_next_hour_low'] = df['low'].shift(-1)
    
    # Multi-hour targets
    df['target_6hour_return'] = (df['close'].shift(-6) - df['close']) / df['close']
    df['target_24hour_return'] = (df['close'].shift(-24) - df['close']) / df['close']
    
    df['target_6hour_direction'] = (df['target_6hour_return'] > 0).astype(int)
    df['target_24hour_direction'] = (df['target_24hour_return'] > 0).astype(int)
    
    print("  ✅ Target variables created")
    print(f"     - Next hour return/direction")
    print(f"     - Next hour high/low")
    print(f"     - 6-hour and 24-hour returns/directions")
    
    return df


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution."""
    
    section("BTC HOURLY FEATURE ENGINEERING")
    print(f"  Input files:")
    print(f"    - OHLCV:     {OHLCV_FILE}")
    print(f"    - On-chain:  {ONCHAIN_FILE}")
    print(f"    - Sentiment: {SENTIMENT_FILE}")
    print(f"  Output: {OUTPUT_FILE}")
    
    print(f"\n  ⚠️  Note: On-chain & sentiment are forward-filled from daily data")
    print(f"     Primary hourly signal comes from OHLCV technical indicators")
    
    # Load and merge
    df = load_and_merge_data()
    
    # Engineer features
    df = engineer_technical_indicators(df)
    df = engineer_rolling_statistics(df)
    df = engineer_market_regime_features(df)
    df = engineer_cyclical_features(df)
    
    # Create targets
    df = create_target_variables(df)
    
    # Final processing
    section("FINAL PROCESSING")
    
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"\n  📊 Final Dataset:")
    print(f"     Rows:    {len(df):,}")
    print(f"     Columns: {len(df.columns):,}")
    print(f"     Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"     Hours of data: {len(df)}")
    print(f"     Days of data: {len(df) / 24:.1f}")
    
    # Missing data
    total_missing = df.isnull().sum().sum()
    total_cells = len(df) * len(df.columns)
    missing_pct = (total_missing / total_cells) * 100
    print(f"     Missing: {total_missing:,} cells ({missing_pct:.2f}%)")
    
    # Save
    section("SAVING OUTPUT")
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"  ✅ Saved to: {OUTPUT_FILE}")
    
    # Feature summary
    section("FEATURE CATEGORIES SUMMARY")
    
    ohlcv_base = ['open', 'high', 'low', 'close', 'vwap', 'volume', 'count']
    technical = [c for c in df.columns if any(x in c for x in ['rsi', 'macd', 'bb_', 'sma', 'ema', 'atr', 'adx', 'stoch', 'obv'])]
    onchain = [c for c in df.columns if any(x in c for x in ['hash', 'difficulty', 'transaction', 'addresses', 'mempool', 'utxo', 'miners'])]
    sentiment = [c for c in df.columns if any(x in c for x in ['fear_greed', 'trends', 'wikipedia', 'github'])]
    momentum = [c for c in df.columns if 'returns' in c and not c.startswith('target')]
    rolling = [c for c in df.columns if any(x in c for x in ['_mean_', '_std_', '_max_', '_min_'])]
    regime = [c for c in df.columns if 'regime' in c]
    cyclical = [c for c in df.columns if c in ['hour_of_day', 'day_of_week', 'is_trading_hours', 'is_weekend', 'is_morning', 'is_afternoon', 'is_evening', 'is_night']]
    targets = [c for c in df.columns if 'target' in c]
    
    print(f"  📈 OHLCV Base:       {len(ohlcv_base):3d} features")
    print(f"  📊 Technical:        {len(technical):3d} features")
    print(f"  ⛓️  On-chain:         {len(onchain):3d} features")
    print(f"  💭 Sentiment:        {len(sentiment):3d} features")
    print(f"  🚀 Momentum:         {len(momentum):3d} features")
    print(f"  📉 Rolling Stats:    {len(rolling):3d} features")
    print(f"  🔄 Market Regime:    {len(regime):3d} features")
    print(f"  📅 Cyclical:         {len(cyclical):3d} features")
    print(f"  🎯 Targets:          {len(targets):3d} variables")
    print(f"  {'─'*60}")
    print(f"  📦 Total:            {len(df.columns):3d} columns")
    
    section("✅ FEATURE ENGINEERING COMPLETE")
    
    print(f"\n  🎯 Next steps:")
    print(f"     1. Handle missing data")
    print(f"     2. Run Boruta feature selection")
    print(f"     3. Train hourly prediction models")
    print(f"\n  💡 Expected improvements over daily:")
    print(f"     - More training samples ({len(df):,} hours vs ~1,000 days)")
    print(f"     - Cleaner intraday signals")
    print(f"     - Better for short-term trading strategies")
    print(f"\n")


if __name__ == "__main__":
    main()