#!/usr/bin/env python3
"""
engineer_btc_5min_features.py

Feature engineering for 5-MINUTE BTC prediction - OHLCV ONLY.

Focus: Pure technical analysis from price/volume data
- No on-chain (daily granularity, useless for 5-min)
- No sentiment (daily granularity, useless for 5-min)
- ONLY OHLCV technical indicators

Optimized for 5-minute timeframe with appropriate lookback periods.

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

# Input file (your existing 5-min OHLCV data)
OHLCV_FILE = "/home/jarred/git/ServoTrader/data/individual_crypto_csvs_ancient_5_min/BTCUSDT.csv"

# Output
OUTPUT_FILE = "/home/jarred/git/ServoTrader/data/btc_5min_features_engineered.csv"

# Feature engineering parameters (in 5-MINUTE INTERVALS)
# 1 hour = 12 candles, 1 day = 288 candles, 1 week = 2016 candles
LOOKBACK_PERIODS = [6, 12, 24, 48, 72, 144, 288, 576, 2016]  # 30min to 1 week

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def section(title):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")


# ============================================================================
# TECHNICAL INDICATORS
# ============================================================================

def calculate_rsi(series, period=14):
    """RSI."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def calculate_macd(series, fast=12, slow=26, signal=9):
    """MACD."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram


def calculate_bollinger_bands(series, period=20, std_dev=2):
    """Bollinger Bands."""
    ma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = ma + (std * std_dev)
    lower = ma - (std * std_dev)
    bandwidth = (upper - lower) / (ma + 1e-10)
    percent_b = (series - lower) / (upper - lower + 1e-10)
    return upper, lower, ma, bandwidth, percent_b


def calculate_atr(high, low, close, period=14):
    """ATR."""
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def calculate_stochastic(high, low, close, k_period=14, d_period=3):
    """Stochastic."""
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    k = 100 * ((close - lowest_low) / (highest_high - lowest_low + 1e-10))
    d = k.rolling(window=d_period).mean()
    return k, d


def calculate_obv(close, volume):
    """OBV."""
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    return obv


# ============================================================================
# LOAD DATA
# ============================================================================

def load_data():
    """Load 5-min OHLCV data only."""
    
    section("LOADING 5-MINUTE OHLCV DATA")
    
    print(f"  📊 Loading: {OHLCV_FILE}")
    
    df = pd.read_csv(OHLCV_FILE)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Remove timezone if present
    if df['timestamp'].dt.tz is not None:
        df['timestamp'] = df['timestamp'].dt.tz_localize(None)
    
    df = df.sort_values('timestamp').reset_index(drop=True)
    df = df.drop(columns=['symbol'], errors='ignore')
    
    print(f"     ✅ {len(df):,} 5-min candles")
    print(f"     📅 {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"     📊 {len(df) / 288:.1f} days of data")
    
    # Verify required columns
    required = ['open', 'high', 'low', 'close', 'volume']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    print(f"\n  ✅ Columns: {list(df.columns)}")
    
    return df


# ============================================================================
# FEATURE ENGINEERING - TECHNICAL INDICATORS
# ============================================================================

def engineer_technical_indicators(df):
    """Technical indicators optimized for 5-minute data."""
    
    section("ENGINEERING TECHNICAL INDICATORS (5-MIN)")
    
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    print("  📈 RSI indicators...")
    # RSI - 5-min appropriate periods
    # 6 = 30min, 12 = 1h, 24 = 2h, 48 = 4h, 72 = 6h
    for period in [6, 12, 24, 48, 72]:
        df[f'rsi_{period}'] = calculate_rsi(close, period)
        print(f"     ✓ rsi_{period}")
    
    print("\n  📊 MACD...")
    # MACD - standard settings work well for 5-min
    df['macd'], df['macd_signal'], df['macd_histogram'] = calculate_macd(close, fast=12, slow=26, signal=9)
    print(f"     ✓ macd, macd_signal, macd_histogram")
    
    print("\n  📉 Bollinger Bands...")
    # Bollinger Bands - 12 = 1h, 24 = 2h, 48 = 4h, 72 = 6h
    for period in [12, 24, 48, 72]:
        upper, lower, ma, bandwidth, percent_b = calculate_bollinger_bands(close, period)
        df[f'bb_upper_{period}'] = upper
        df[f'bb_lower_{period}'] = lower
        df[f'bb_bandwidth_{period}'] = bandwidth
        df[f'bb_percent_b_{period}'] = percent_b
        print(f"     ✓ bb_{period} (upper, lower, bandwidth, percent_b)")
    
    print("\n  📈 Moving Averages...")
    # Moving Averages - 6=30min, 12=1h, 24=2h, 48=4h, 72=6h, 144=12h, 288=1d
    for period in [6, 12, 24, 48, 72, 144, 288]:
        df[f'sma_{period}'] = close.rolling(window=period).mean()
        df[f'ema_{period}'] = close.ewm(span=period, adjust=False).mean()
        print(f"     ✓ sma_{period}, ema_{period}")
    
    print("\n  📍 Price position indicators...")
    # Price position relative to MAs
    df['price_above_sma_72'] = (close > df['sma_72']).astype(int)  # 6 hours
    df['price_above_sma_288'] = (close > df['sma_288']).astype(int)  # 1 day
    print(f"     ✓ price_above_sma_72, price_above_sma_288")
    
    print("\n  📊 ATR (volatility)...")
    # ATR - 12=1h, 24=2h, 48=4h
    for period in [12, 24, 48]:
        df[f'atr_{period}'] = calculate_atr(high, low, close, period)
        print(f"     ✓ atr_{period}")
    
    print("\n  📈 Stochastic...")
    # Stochastic - 24 = 2 hours
    df['stoch_k'], df['stoch_d'] = calculate_stochastic(high, low, close, k_period=24, d_period=3)
    print(f"     ✓ stoch_k, stoch_d")
    
    print("\n  📊 Volume indicators...")
    # Volume - 6=30min, 12=1h, 24=2h, 48=4h
    for period in [6, 12, 24, 48]:
        df[f'volume_ma_{period}'] = volume.rolling(window=period).mean()
        print(f"     ✓ volume_ma_{period}")
    
    df['volume_ratio_24'] = volume / (df['volume_ma_24'] + 1e-10)  # vs 2h average
    print(f"     ✓ volume_ratio_24")
    
    # OBV
    df['obv'] = calculate_obv(close, volume)
    df['obv_ma_48'] = df['obv'].rolling(window=48).mean()  # 4h average
    print(f"     ✓ obv, obv_ma_48")
    
    # VWAP deviation (if vwap exists)
    if 'vwap' in df.columns:
        df['vwap_deviation'] = (close - df['vwap']) / (df['vwap'] + 1e-10)
        print(f"     ✓ vwap_deviation")
    
    print("\n  ✅ Technical indicators complete")
    
    return df


def engineer_momentum_features(df):
    """Momentum and returns features."""
    
    section("ENGINEERING MOMENTUM FEATURES")
    
    close = df['close']
    
    print("  🚀 Returns (pct change)...")
    # Returns - 1=5min, 3=15min, 6=30min, 12=1h, 24=2h, 48=4h, 72=6h
    for period in [1, 3, 6, 12, 24, 48, 72]:
        df[f'returns_{period}p'] = close.pct_change(period)  # 'p' = periods
        print(f"     ✓ returns_{period}p")
    
    print("\n  📉 Volatility...")
    # Volatility - 6=30min, 12=1h, 24=2h, 48=4h
    for period in [6, 12, 24, 48]:
        df[f'volatility_{period}p'] = close.pct_change().rolling(window=period).std()
        print(f"     ✓ volatility_{period}p")
    
    print("\n  ✅ Momentum features complete")
    
    return df


def engineer_price_patterns(df):
    """Price pattern features."""
    
    section("ENGINEERING PRICE PATTERNS")
    
    high = df['high']
    low = df['low']
    close = df['close']
    
    # Higher highs / lower lows
    df['higher_high'] = ((high > high.shift(1)) & (high.shift(1) > high.shift(2))).astype(int)
    df['lower_low'] = ((low < low.shift(1)) & (low.shift(1) < low.shift(2))).astype(int)
    
    # Range
    df['candle_range'] = high - low
    df['candle_range_pct'] = (high - low) / (close + 1e-10)
    
    print("  ✅ higher_high, lower_low, candle_range, candle_range_pct")
    
    return df


def engineer_rolling_statistics(df):
    """Rolling statistics."""
    
    section("ENGINEERING ROLLING STATISTICS")
    
    print("  📊 Rolling stats on returns...")
    
    # 24=2h, 48=4h, 144=12h
    for period in [24, 48, 144]:
        if f'returns_1p' in df.columns:
            df[f'returns_mean_{period}p'] = df['returns_1p'].rolling(window=period).mean()
            df[f'returns_std_{period}p'] = df['returns_1p'].rolling(window=period).std()
            print(f"     ✓ returns_mean_{period}p, returns_std_{period}p")
    
    print("\n  ✅ Rolling stats complete")
    
    return df


def engineer_cyclical_features(df):
    """Cyclical time features."""
    
    section("ENGINEERING CYCLICAL FEATURES")
    
    # 5-min specific: which 5-min slot within hour (0-11)
    df['slot_in_hour'] = (df['timestamp'].dt.minute // 5)
    
    # Hour of day (0-23)
    df['hour_of_day'] = df['timestamp'].dt.hour
    
    # Day of week (0-6)
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    # Trading hours (9am-4pm EST = 14:00-21:00 UTC)
    df['is_trading_hours'] = df['hour_of_day'].between(14, 21).astype(int)
    
    # Weekend
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    print("  ✅ slot_in_hour, hour_of_day, day_of_week, is_trading_hours, is_weekend")
    
    return df


# ============================================================================
# RL TARGETS
# ============================================================================

def create_rl_targets(df):
    """Create RL-specific targets."""
    
    section("CREATING RL TARGETS")
    
    print("  🎯 Future returns (for rewards)...")
    # 1=5min, 3=15min, 6=30min, 12=1h, 24=2h, 48=4h ahead
    for period in [1, 3, 6, 12, 24, 48]:
        df[f'target_return_{period}p'] = (df['close'].shift(-period) - df['close']) / df['close']
        print(f"     ✓ target_return_{period}p")
    
    print("\n  📉 Future volatility (for risk)...")
    # Future volatility over next 6=30min, 12=1h, 24=2h
    for period in [6, 12, 24]:
        future_vol = []
        for i in range(len(df)):
            if i + period < len(df):
                future_returns = df['close'].iloc[i+1:i+period+1].pct_change().std()
                future_vol.append(future_returns)
            else:
                future_vol.append(np.nan)
        df[f'target_volatility_{period}p'] = future_vol
        print(f"     ✓ target_volatility_{period}p")
    
    print("\n  🔼🔽 Binary direction (for classification)...")
    # Direction for 1=5min, 6=30min, 12=1h ahead
    for period in [1, 6, 12]:
        df[f'target_direction_{period}p'] = (df[f'target_return_{period}p'] > 0).astype(int)
        print(f"     ✓ target_direction_{period}p")
    
    print("\n  🎬 Optimal actions (for supervised pre-training)...")
    # Optimal actions for 6 periods (30min) and 12 periods (1h)
    # BUY=0, HOLD=1, SELL=2
    
    # 30min horizon
    threshold_buy = 0.001  # 0.1%
    threshold_sell = -0.001
    conditions = [
        df['target_return_6p'] > threshold_buy,
        df['target_return_6p'] < threshold_sell
    ]
    df['target_optimal_action_6p'] = np.select(conditions, [0, 2], default=1)
    print(f"     ✓ target_optimal_action_6p (BUY=0, HOLD=1, SELL=2)")
    
    # 1h horizon
    threshold_buy_12 = 0.002  # 0.2%
    threshold_sell_12 = -0.002
    conditions_12 = [
        df['target_return_12p'] > threshold_buy_12,
        df['target_return_12p'] < threshold_sell_12
    ]
    df['target_optimal_action_12p'] = np.select(conditions_12, [0, 2], default=1)
    print(f"     ✓ target_optimal_action_12p")
    
    print("\n  📈 Sharpe-like targets (risk-adjusted)...")
    # Sharpe for 6p=30min and 12p=1h
    for period in [6, 12]:
        df[f'target_sharpe_{period}p'] = (
            df[f'target_return_{period}p'] / (df[f'target_volatility_{period}p'] + 1e-10)
        )
        print(f"     ✓ target_sharpe_{period}p")
    
    print("\n  ✅ All RL targets created!")
    
    return df


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution."""
    
    section("BTC 5-MINUTE FEATURE ENGINEERING - OHLCV ONLY")
    
    print(f"  Input:  {OHLCV_FILE}")
    print(f"  Output: {OUTPUT_FILE}")
    print(f"\n  🎯 Focus: Pure technical analysis from OHLCV")
    print(f"     - NO on-chain (daily granularity)")
    print(f"     - NO sentiment (daily granularity)")
    print(f"     - ONLY price/volume technical indicators")
    print(f"\n  ⏱️  Timeframe: 5-minute candles")
    print(f"     - 1 hour = 12 candles")
    print(f"     - 1 day = 288 candles")
    print(f"     - 1 week = 2,016 candles")
    
    # Load
    df = load_data()
    
    # Engineer features
    df = engineer_technical_indicators(df)
    df = engineer_momentum_features(df)
    df = engineer_price_patterns(df)
    df = engineer_rolling_statistics(df)
    df = engineer_cyclical_features(df)
    
    # RL targets
    df = create_rl_targets(df)
    
    # Final processing
    section("FINAL PROCESSING")
    
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"\n  📊 Final Dataset:")
    print(f"     Rows:    {len(df):,}")
    print(f"     Columns: {len(df.columns):,}")
    print(f"     Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"     5-min candles: {len(df):,}")
    print(f"     Days:    {len(df) / 288:.1f}")
    
    # Missing data summary
    total_missing = df.isnull().sum().sum()
    if total_missing > 0:
        total_cells = len(df) * len(df.columns)
        missing_pct = (total_missing / total_cells) * 100
        print(f"\n  ⚠️  Missing: {total_missing:,} cells ({missing_pct:.2f}%)")
        print(f"     (Normal for indicators with lookback periods)")
    else:
        print(f"\n  ✅ No missing data")
    
    # Save
    section("SAVING OUTPUT")
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"  ✅ Saved to: {OUTPUT_FILE}")
    
    # Feature summary
    section("FEATURE SUMMARY")
    
    ohlcv_base = [c for c in ['open', 'high', 'low', 'close', 'vwap', 'volume', 'count'] if c in df.columns]
    technical = [c for c in df.columns if any(x in c for x in ['rsi', 'macd', 'bb_', 'sma', 'ema', 'atr', 'stoch', 'obv'])]
    momentum = [c for c in df.columns if 'returns' in c and not c.startswith('target')]
    volatility = [c for c in df.columns if 'volatility' in c and not c.startswith('target')]
    volume_features = [c for c in df.columns if 'volume' in c and c not in ohlcv_base]
    price_patterns = [c for c in df.columns if any(x in c for x in ['higher_high', 'lower_low', 'range'])]
    rolling = [c for c in df.columns if any(x in c for x in ['_mean_', '_std_']) and not c.startswith('target')]
    cyclical = [c for c in df.columns if any(x in c for x in ['slot', 'hour', 'day_of', 'trading', 'weekend'])]
    targets = [c for c in df.columns if c.startswith('target')]
    
    print(f"  📈 OHLCV Base:       {len(ohlcv_base):3d}")
    print(f"  📊 Technical:        {len(technical):3d}")
    print(f"  🚀 Momentum:         {len(momentum):3d}")
    print(f"  📉 Volatility:       {len(volatility):3d}")
    print(f"  📊 Volume:           {len(volume_features):3d}")
    print(f"  📈 Price Patterns:   {len(price_patterns):3d}")
    print(f"  📊 Rolling Stats:    {len(rolling):3d}")
    print(f"  📅 Cyclical:         {len(cyclical):3d}")
    print(f"  🎯 RL Targets:       {len(targets):3d}")
    print(f"  {'─'*60}")
    print(f"  📦 Total Features:   {len(df.columns) - len(targets) - 1:3d}  (excluding targets & timestamp)")
    print(f"  📦 Total Columns:    {len(df.columns):3d}")
    
    print(f"\n  🎯 RL Target Breakdown:")
    return_targets = [c for c in targets if 'return' in c]
    vol_targets = [c for c in targets if 'volatility' in c]
    dir_targets = [c for c in targets if 'direction' in c]
    action_targets = [c for c in targets if 'action' in c]
    sharpe_targets = [c for c in targets if 'sharpe' in c]
    
    print(f"     Returns:      {len(return_targets):2d} (for rewards)")
    print(f"     Volatility:   {len(vol_targets):2d} (for risk)")
    print(f"     Direction:    {len(dir_targets):2d} (for classification)")
    print(f"     Actions:      {len(action_targets):2d} (for pre-training)")
    print(f"     Sharpe:       {len(sharpe_targets):2d} (for risk-adjusted)")
    
    section("✅ FEATURE ENGINEERING COMPLETE")
    
    print(f"\n  🎯 Next steps:")
    print(f"     1. Optional: Run Boruta (or skip for Transformer)")
    print(f"     2. Build RL gym environment")
    print(f"     3. Train Transformer + PPO/DQN agent")
    print(f"     4. Use target_return_6p (30min) or target_return_12p (1h) for rewards")
    print(f"\n  💡 For RL:")
    print(f"     - State: All technical features (or Boruta-selected)")
    print(f"     - Actions: BUY (0), HOLD (1), SELL (2)")
    print(f"     - Reward: target_return_6p - trading_cost")
    print(f"\n")


if __name__ == "__main__":
    main()