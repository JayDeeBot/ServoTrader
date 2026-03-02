#!/usr/bin/env python3
"""
boruta_5min.py

Boruta feature selection for 5-MINUTE BTC prediction - OHLCV features only.

Optimized for RL trading with 5-minute candles.
- Tests only OHLCV technical indicators
- Samples data to manageable size
- Provides feature importance rankings

Author: Jarred Deluca
Created: 2025
License: MIT
"""

import pandas as pd
import numpy as np
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_FILE = "/home/jarred/git/ServoTrader/data/btc_5min_features_engineered.csv"
OUTPUT_REPORT = "/home/jarred/git/ServoTrader/data/boruta_5min_report.csv"
OUTPUT_SELECTED = "/home/jarred/git/ServoTrader/data/btc_5min_features_selected.csv"

# ============================================================================
# TARGET SELECTION
# ============================================================================
# For RL: Use binary classification target for Boruta
# 6p = 6 periods = 30 minutes ahead

TARGET_VARIABLE = "target_direction_6p"  # Binary: up=1, down=0 (30min ahead)

# Alternative targets:
# TARGET_VARIABLE = "target_direction_12p"  # 1 hour ahead
# TARGET_VARIABLE = "target_optimal_action_6p"  # Multi-class: BUY/HOLD/SELL

# ============================================================================
# BORUTA PARAMETERS (MODERATE)
# ============================================================================

BORUTA_ALPHA = 0.05              # Strict
BORUTA_PERC = 90                 # Strict
BORUTA_TWO_STEP = False
BORUTA_MAX_ITER = 100
BORUTA_RANDOM_STATE = 42

# ============================================================================
# RF PARAMETERS
# ============================================================================

RF_N_ESTIMATORS = 250
RF_MAX_DEPTH = 8
RF_MIN_SAMPLES_SPLIT = 20
RF_MIN_SAMPLES_LEAF = 10
RF_MAX_FEATURES = 'sqrt'
RF_RANDOM_STATE = 42

# ============================================================================
# SAMPLING
# ============================================================================
# 5-min data has ~1/5th the rows of 1-min
# Can use larger sample size

SAMPLE_SIZE = 50000              # 50k samples (enough for 5-min)
SAMPLE_RANDOM = True

# ============================================================================
# PRE-FILTERING
# ============================================================================

PREFILTER_TOP_N = 100            # Test top 100 features

# ============================================================================
# DATA HANDLING
# ============================================================================

MIN_DATA_THRESHOLD = 0.7
FORWARD_FILL_LIMIT = 288         # 1 day in 5-min candles

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def section(title):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")


def categorize_feature(feature_name):
    """Categorize feature type."""
    name_lower = feature_name.lower()
    
    # Base OHLCV
    if feature_name in ['open', 'high', 'low', 'close', 'vwap', 'volume', 'count']:
        return 'ohlcv_base'
    
    # Technical indicators
    if any(kw in name_lower for kw in ['rsi', 'macd', 'bb_', 'bollinger', 'sma', 'ema', 'atr', 'stoch', 'obv']):
        return 'technical'
    
    # Volume
    if 'volume' in name_lower and feature_name not in ['volume', 'count']:
        return 'volume'
    
    # Momentum/Returns
    if any(kw in name_lower for kw in ['returns', 'momentum']):
        return 'momentum'
    
    # Volatility
    if 'volatility' in name_lower or 'vol_' in name_lower:
        return 'volatility'
    
    # Price patterns
    if any(kw in name_lower for kw in ['higher_high', 'lower_low', 'range']):
        return 'price_patterns'
    
    # Rolling stats
    if any(kw in name_lower for kw in ['_mean_', '_std_', '_max_', '_min_']):
        return 'rolling_stats'
    
    # Cyclical
    if any(kw in name_lower for kw in ['slot', 'hour', 'day_of', 'trading', 'weekend']):
        return 'cyclical'
    
    return 'other'


# ============================================================================
# LOAD AND SAMPLE DATA
# ============================================================================

def load_and_sample_data():
    """Load and sample 5-min data."""
    
    section("LOADING 5-MINUTE DATA")
    
    df = pd.read_csv(INPUT_FILE)
    print(f"  📊 Full dataset: {len(df):,} rows × {len(df.columns)} columns")
    
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        print(f"  📅 Time range: {df['timestamp'].min()} → {df['timestamp'].max()}")
        df = df.sort_values('timestamp').reset_index(drop=True)
    
    section("SAMPLING DATA")
    
    print(f"  💡 5-min data benefits from sampling for faster Boruta")
    
    if len(df) > SAMPLE_SIZE:
        print(f"     Full size: {len(df):,} rows")
        print(f"     Sampling: {SAMPLE_SIZE:,} rows")
        
        if SAMPLE_RANDOM:
            print(f"     Method: Random sampling")
            df_sampled = df.sample(n=SAMPLE_SIZE, random_state=42)
        else:
            print(f"     Method: Most recent data")
            df_sampled = df.tail(SAMPLE_SIZE)
        
        df_sampled = df_sampled.sort_values('timestamp').reset_index(drop=True)
        print(f"     ✅ Sampled: {len(df_sampled):,} rows")
        df = df_sampled
    else:
        print(f"  ℹ️  Dataset size ({len(df):,} rows) - using full dataset")
    
    section("PREPARING FEATURES")
    
    # Identify feature columns
    target_cols = [c for c in df.columns if c.startswith('target')]
    feature_cols = [c for c in df.columns if c not in target_cols and c not in ['timestamp', 'date']]
    
    print(f"  📋 Dataset structure:")
    print(f"     Features: {len(feature_cols)}")
    print(f"     Targets:  {len(target_cols)}")
    print(f"     Total:    {len(df.columns)}")
    
    # Check for target
    if TARGET_VARIABLE not in df.columns:
        print(f"\n  ❌ ERROR: Target '{TARGET_VARIABLE}' not found!")
        available = [c for c in target_cols if 'direction' in c or 'action' in c]
        print(f"     Available targets: {available}")
        raise ValueError(f"Target '{TARGET_VARIABLE}' not in dataset")
    
    # Handle missing data
    section("HANDLING MISSING DATA")
    
    missing_before = df.isnull().sum()
    high_missing = missing_before[missing_before > len(df) * 0.3]
    
    if len(high_missing) > 0:
        print(f"  ⚠️  {len(high_missing)} features with >30% missing:")
        for col in high_missing.index[:5]:
            pct = (missing_before[col] / len(df)) * 100
            print(f"     {col:40s} {pct:5.1f}%")
    
    # Drop features with too much missing
    min_valid_rows = int(len(df) * MIN_DATA_THRESHOLD)
    cols_to_drop = missing_before[missing_before > (len(df) - min_valid_rows)].index.tolist()
    
    if cols_to_drop:
        print(f"\n  🗑️  Dropping {len(cols_to_drop)} features with <{MIN_DATA_THRESHOLD*100}% data")
        df = df.drop(columns=cols_to_drop)
        feature_cols = [c for c in feature_cols if c in df.columns]
    
    # Forward fill
    df[feature_cols] = df[feature_cols].fillna(method='ffill', limit=FORWARD_FILL_LIMIT)
    df[feature_cols] = df[feature_cols].fillna(method='bfill', limit=FORWARD_FILL_LIMIT)
    df[feature_cols] = df[feature_cols].fillna(0)
    
    # Drop rows with missing target
    rows_before = len(df)
    df = df[df[TARGET_VARIABLE].notna()].copy()
    if rows_before - len(df) > 0:
        print(f"  🗑️  Dropped {rows_before - len(df)} rows with missing target")
    
    remaining_nulls = df[feature_cols].isnull().sum().sum()
    if remaining_nulls > 0:
        print(f"  ⚠️  Warning: {remaining_nulls} nulls remain")
    else:
        print(f"  ✅ All missing values handled")
    
    print(f"\n  ✅ Clean dataset: {len(df):,} rows × {len(feature_cols)} features")
    
    return df, feature_cols


# ============================================================================
# PRE-FILTER
# ============================================================================

def prefilter_features(df, feature_cols):
    """Pre-filter to top N features."""
    
    if PREFILTER_TOP_N is None or PREFILTER_TOP_N >= len(feature_cols):
        print(f"  ℹ️  Testing all {len(feature_cols)} features")
        return feature_cols
    
    section("PRE-FILTERING FEATURES")
    
    print(f"  🔍 Using Random Forest to select top {PREFILTER_TOP_N} features...")
    
    X = df[feature_cols].values
    y = df[TARGET_VARIABLE].values
    
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    print(f"  🌲 Training quick RF...")
    rf.fit(X, y)
    
    importances = pd.Series(rf.feature_importances_, index=feature_cols)
    top_features = importances.nlargest(PREFILTER_TOP_N).index.tolist()
    
    print(f"\n  ✅ Selected top {len(top_features)} features")
    print(f"\n  📊 Top 10 by RF importance:")
    for i, feat in enumerate(top_features[:10], 1):
        print(f"     {i:2d}. {feat:40s} ({importances[feat]:.6f})")
    
    return top_features


# ============================================================================
# RUN BORUTA
# ============================================================================

def run_boruta_with_diagnostics(df, feature_cols):
    """Run Boruta with diagnostics."""
    
    section("BORUTA CONFIGURATION (5-MIN)")
    
    print(f"  🔬 Boruta Parameters:")
    print(f"     alpha:     {BORUTA_ALPHA}")
    print(f"     perc:      {BORUTA_PERC}")
    print(f"     two_step:  {BORUTA_TWO_STEP}")
    print(f"     max_iter:  {BORUTA_MAX_ITER}")
    
    print(f"\n  🌲 Random Forest Parameters:")
    print(f"     n_estimators:      {RF_N_ESTIMATORS}")
    print(f"     max_depth:         {RF_MAX_DEPTH}")
    print(f"     min_samples_split: {RF_MIN_SAMPLES_SPLIT}")
    print(f"     min_samples_leaf:  {RF_MIN_SAMPLES_LEAF}")
    
    section("RUNNING BORUTA")
    
    X = df[feature_cols].copy()
    y = df[TARGET_VARIABLE].values
    
    print(f"  📊 Input: {X.shape[0]:,} samples × {X.shape[1]} features")
    print(f"  🎯 Target: {TARGET_VARIABLE} (30min ahead direction)")
    
    # Target distribution
    unique, counts = np.unique(y, return_counts=True)
    print(f"\n  📈 Target distribution:")
    for val, count in zip(unique, counts):
        pct = (count / len(y)) * 100
        label = "DOWN" if val == 0 else "UP"
        print(f"     {label} ({val}): {count:,} ({pct:.1f}%)")
    
    # Initialize RF
    rf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_split=RF_MIN_SAMPLES_SPLIT,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        max_features=RF_MAX_FEATURES,
        random_state=RF_RANDOM_STATE,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    # Baseline
    print(f"\n  📊 Baseline RF accuracy (5-fold CV):")
    scores = cross_val_score(rf, X.values, y, cv=5, scoring='accuracy', n_jobs=-1)
    print(f"     Mean: {scores.mean():.3f} ± {scores.std():.3f}")
    
    if scores.mean() > 0.56:
        print(f"     ✅✅ Excellent signal for 5-min!")
    elif scores.mean() > 0.54:
        print(f"     ✅ Good signal for 5-min")
    elif scores.mean() > 0.52:
        print(f"     ✅ Moderate signal (normal for 5-min)")
    else:
        print(f"     ⚠️  Weak signal (near random)")
    
    # Boruta
    boruta = BorutaPy(
        estimator=rf,
        n_estimators=RF_N_ESTIMATORS,
        max_iter=BORUTA_MAX_ITER,
        alpha=BORUTA_ALPHA,
        perc=BORUTA_PERC,
        two_step=BORUTA_TWO_STEP,
        random_state=BORUTA_RANDOM_STATE,
        verbose=2
    )
    
    print(f"\n  🚀 Running Boruta (may take 10-20 minutes)...")
    print(f"  {'-'*76}")
    
    boruta.fit(X.values, y)
    
    print(f"  {'-'*76}")
    
    # Results
    confirmed = X.columns[boruta.support_].tolist()
    tentative = X.columns[boruta.support_weak_].tolist()
    rejected = X.columns[~(boruta.support_ | boruta.support_weak_)].tolist()
    rankings = boruta.ranking_
    
    print(f"\n  ✅ Boruta completed!")
    print(f"\n  📊 Results:")
    print(f"     ✅ Confirmed:  {len(confirmed):3d} ({len(confirmed)/len(feature_cols)*100:5.1f}%)")
    print(f"     ⚠️  Tentative:  {len(tentative):3d} ({len(tentative)/len(feature_cols)*100:5.1f}%)")
    print(f"     ❌ Rejected:   {len(rejected):3d} ({len(rejected)/len(feature_cols)*100:5.1f}%)")
    
    # Assessment
    if len(confirmed) >= 35:
        print(f"\n  ✅✅ Excellent! {len(confirmed)} features confirmed")
    elif len(confirmed) >= 20:
        print(f"\n  ✅ Good! {len(confirmed)} features confirmed")
    elif len(confirmed) >= 10:
        print(f"\n  ⚠️  Moderate. {len(confirmed)} features confirmed")
    else:
        print(f"\n  ⚠️  WARNING: Only {len(confirmed)} features confirmed")
    
    return boruta, confirmed, tentative, rejected, rankings


# ============================================================================
# CREATE REPORT & SAVE (abbreviated for space)
# ============================================================================

def create_and_save_outputs(df, feature_cols, confirmed, tentative, rejected, rankings):
    """Create report and save outputs."""
    
    section("GENERATING REPORT")
    
    report_data = []
    for i, feature in enumerate(feature_cols):
        if feature in confirmed:
            status = 'CONFIRMED'
        elif feature in tentative:
            status = 'TENTATIVE'
        else:
            status = 'REJECTED'
        
        report_data.append({
            'feature': feature,
            'status': status,
            'ranking': rankings[i],
            'category': categorize_feature(feature)
        })
    
    report_df = pd.DataFrame(report_data).sort_values('ranking')
    report_df['importance_rank'] = range(1, len(report_df) + 1)
    
    # Save report
    report_df.to_csv(OUTPUT_REPORT, index=False)
    print(f"  ✅ Report saved: {OUTPUT_REPORT}")
    
    # Display top confirmed
    confirmed_df = report_df[report_df['status'] == 'CONFIRMED']
    if len(confirmed_df) > 0:
        print(f"\n  🏆 Top 20 Confirmed Features:")
        for idx, row in confirmed_df.head(20).iterrows():
            print(f"     {row['importance_rank']:2d}. {row['feature']:45s} [{row['category']}]")
        
        print(f"\n  📈 Confirmed by Category:")
        cat_counts = confirmed_df.groupby('category').size().sort_values(ascending=False)
        for cat, count in cat_counts.items():
            print(f"     {cat:20s} {count:3d}")
    
    # Save selected dataset
    selected = confirmed + tentative
    if len(selected) > 0:
        timestamp_col = 'timestamp' if 'timestamp' in df.columns else 'date'
        target_cols = [c for c in df.columns if c.startswith('target')]
        cols = [timestamp_col] + selected + target_cols
        
        df[cols].to_csv(OUTPUT_SELECTED, index=False)
        print(f"\n  ✅ Selected dataset: {OUTPUT_SELECTED}")
        print(f"     Features: {len(selected)}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution."""
    
    section("5-MINUTE BORUTA - OHLCV FEATURES ONLY")
    
    print(f"  📋 Configuration:")
    print(f"     Target:        {TARGET_VARIABLE} (30min ahead)")
    print(f"     Sample size:   {SAMPLE_SIZE:,}")
    print(f"     Pre-filter:    Top {PREFILTER_TOP_N}")
    print(f"     Boruta alpha:  {BORUTA_ALPHA}")
    
    print(f"\n  💡 For RL with Transformer:")
    print(f"     - Boruta is OPTIONAL (Transformer has attention)")
    print(f"     - Use confirmed features OR all features")
    print(f"     - Reward = target_return_6p (30min) or target_return_12p (1h)")
    
    # Load and sample
    df, feature_cols = load_and_sample_data()
    
    # Pre-filter
    feature_cols = prefilter_features(df, feature_cols)
    
    # Run Boruta
    boruta, confirmed, tentative, rejected, rankings = run_boruta_with_diagnostics(df, feature_cols)
    
    # Create report and save
    create_and_save_outputs(df, feature_cols, confirmed, tentative, rejected, rankings)
    
    section("✅ COMPLETE")
    
    print(f"\n  Results: {len(confirmed)} confirmed, {len(tentative)} tentative")
    
    if len(confirmed) >= 20:
        print(f"\n  🎯 Next: Train RL agent with {len(confirmed)} selected features")
    else:
        print(f"\n  🎯 Next: Consider using ALL technical features")
    
    print(f"\n")


if __name__ == "__main__":
    main()