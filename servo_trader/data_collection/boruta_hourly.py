#!/usr/bin/env python3
"""
boruta_hourly.py

Boruta feature selection for HOURLY BTC prediction with optimized parameters.
Tailored for hourly data with more samples and cleaner signal than daily.

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
# CONFIGURATION FOR HOURLY DATA
# ============================================================================

# Input/Output
INPUT_FILE = "/home/jarred/git/ServoTrader/data/btc_hourly_features_engineered.csv"
OUTPUT_REPORT = "/home/jarred/git/ServoTrader/data/boruta_hourly_report.csv"
OUTPUT_SELECTED = "/home/jarred/git/ServoTrader/data/btc_hourly_features_selected.csv"

# Target variable for hourly prediction
TARGET_VARIABLE = "target_next_hour_direction"  # Changed from daily

# ============================================================================
# BORUTA PARAMETERS - MODERATE SETTINGS FOR HOURLY
# ============================================================================
# Since hourly has 26k samples (vs ~1k daily), we can be less lenient

BORUTA_ALPHA = 0.10              # Moderate (hourly has more data)
BORUTA_PERC = 80                 # Moderate 
BORUTA_TWO_STEP = False          # Keep single-step
BORUTA_MAX_ITER = 150            # Sufficient for hourly
BORUTA_RANDOM_STATE = 42

# ============================================================================
# RANDOM FOREST PARAMETERS - OPTIMIZED FOR HOURLY
# ============================================================================

RF_N_ESTIMATORS = 250            # Stable importance scores
RF_MAX_DEPTH = 10                # Moderate depth (not too deep)
RF_MIN_SAMPLES_SPLIT = 10        # Prevent overfitting
RF_MIN_SAMPLES_LEAF = 5          # Prevent overfitting
RF_MAX_FEATURES = 'sqrt'         # Good default
RF_RANDOM_STATE = 42

# ============================================================================
# PRE-FILTERING
# ============================================================================

PREFILTER_TOP_N = None           # Disabled - test all features

# Missing data handling
FORWARD_FILL_LIMIT = 24          # 24 hours (changed from 7 days)
MIN_DATA_THRESHOLD = 0.7

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def section(title):
    """Print section header."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")


def categorize_feature(feature_name):
    """Categorize feature by type."""
    name_lower = feature_name.lower()
    
    if feature_name in ['open', 'high', 'low', 'close', 'vwap', 'volume', 'count']:
        return 'ohlcv_base'
    
    technical_keywords = ['rsi', 'macd', 'bb_', 'bollinger', 'sma', 'ema', 'atr', 'adx', 
                          'stoch', 'cci', 'williams', 'roc', 'mfi', 'obv']
    if any(kw in name_lower for kw in technical_keywords):
        return 'technical'
    
    onchain_keywords = ['hash', 'difficulty', 'transaction', 'addresses', 'mempool', 
                        'utxo', 'miners', 'revenue', 'supply', 'velocity']
    if any(kw in name_lower for kw in onchain_keywords):
        return 'onchain'
    
    sentiment_keywords = ['fear_greed', 'fear', 'greed', 'trends', 'wikipedia', 
                          'github', 'reddit', 'stocktwits', 'gdelt', 'sentiment', 'social']
    if any(kw in name_lower for kw in sentiment_keywords):
        return 'sentiment'
    
    if 'volume' in name_lower and feature_name not in ['volume', 'count']:
        return 'volume'
    if any(kw in name_lower for kw in ['returns', 'momentum']):
        return 'momentum'
    if 'volatility' in name_lower or 'vol_' in name_lower:
        return 'volatility'
    if '_lag_' in name_lower:
        return 'lag'
    if any(kw in name_lower for kw in ['_mean_', '_std_', '_max_', '_min_', '_skew_']):
        return 'rolling_stats'
    if 'regime' in name_lower:
        return 'regime'
    if any(kw in name_lower for kw in ['hour_of', 'day_of', 'trading_hours', 'weekend', 'morning', 'afternoon', 'evening', 'night']):
        return 'cyclical'
    if '_to_' in name_lower or 'ratio' in name_lower or 'interaction' in name_lower or 'divergence' in name_lower:
        return 'cross_sectional'
    if any(kw in name_lower for kw in ['gap', 'higher_high', 'lower_low', 'range']):
        return 'price_patterns'
    
    return 'other'


# ============================================================================
# LOAD AND CLEAN DATA
# ============================================================================

def load_and_clean_data():
    """Load and clean hourly data."""
    
    section("LOADING HOURLY DATA")
    
    df = pd.read_csv(INPUT_FILE)
    print(f"  📊 Loaded: {len(df)} rows × {len(df.columns)} columns")
    
    # Hourly data uses 'timestamp' not 'date'
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        print(f"  📅 Time range: {df['timestamp'].min()} → {df['timestamp'].max()}")
        df = df.sort_values('timestamp').reset_index(drop=True)
    else:
        print(f"  ⚠️  Warning: No timestamp column found")
    
    section("HANDLING MISSING DATA")
    
    missing_before = df.isnull().sum()
    missing_pct_before = (missing_before / len(df)) * 100
    
    # Drop features with too much missing
    min_valid_rows = int(len(df) * MIN_DATA_THRESHOLD)
    cols_to_drop = missing_before[missing_before > (len(df) - min_valid_rows)].index.tolist()
    
    if cols_to_drop:
        print(f"  🗑️  Dropping {len(cols_to_drop)} features with <{MIN_DATA_THRESHOLD*100}% data")
        df = df.drop(columns=cols_to_drop)
    
    target_cols = [c for c in df.columns if c.startswith('target')]
    feature_cols = [c for c in df.columns if c not in target_cols and c not in ['timestamp', 'date']]
    
    print(f"\n  📋 Dataset structure:")
    print(f"     Features: {len(feature_cols)}")
    print(f"     Targets:  {len(target_cols)}")
    
    # Forward fill (hourly limit)
    df[feature_cols] = df[feature_cols].fillna(method='ffill', limit=FORWARD_FILL_LIMIT)
    df[feature_cols] = df[feature_cols].fillna(method='bfill', limit=FORWARD_FILL_LIMIT)
    df[feature_cols] = df[feature_cols].fillna(0)
    
    # Drop rows with missing target
    if TARGET_VARIABLE in df.columns:
        rows_before = len(df)
        df = df[df[TARGET_VARIABLE].notna()].copy()
        rows_dropped = rows_before - len(df)
        print(f"\n  🗑️  Dropped {rows_dropped} rows with missing target")
    else:
        print(f"\n  ❌ ERROR: Target variable '{TARGET_VARIABLE}' not found!")
        print(f"     Available targets: {target_cols}")
        raise ValueError(f"Target variable '{TARGET_VARIABLE}' not in dataset")
    
    # Check remaining nulls
    remaining_nulls = df[feature_cols].isnull().sum().sum()
    if remaining_nulls > 0:
        print(f"  ⚠️  Warning: {remaining_nulls} nulls remain")
    else:
        print(f"  ✅ All missing values handled")
    
    print(f"\n  ✅ Clean dataset: {len(df)} rows × {len(feature_cols)} features")
    
    return df, feature_cols, missing_pct_before


# ============================================================================
# PRE-FILTERING (OPTIONAL)
# ============================================================================

def prefilter_features(df, feature_cols):
    """Optional: Pre-filter to top N features."""
    
    if PREFILTER_TOP_N is None or PREFILTER_TOP_N >= len(feature_cols):
        print(f"  ℹ️  Pre-filtering disabled (testing all {len(feature_cols)} features)")
        return feature_cols
    
    section("PRE-FILTERING FEATURES")
    
    print(f"  🔍 Pre-filtering to top {PREFILTER_TOP_N} features...")
    
    X = df[feature_cols].values
    y = df[TARGET_VARIABLE].values
    
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X, y)
    
    importances = pd.Series(rf.feature_importances_, index=feature_cols)
    top_features = importances.nlargest(PREFILTER_TOP_N).index.tolist()
    
    print(f"  ✅ Selected top {len(top_features)} features")
    print(f"\n  📊 Top 10 by RF importance:")
    for i, feat in enumerate(top_features[:10], 1):
        print(f"     {i:2d}. {feat:50s} ({importances[feat]:.6f})")
    
    return top_features


# ============================================================================
# RUN BORUTA
# ============================================================================

def run_boruta_with_diagnostics(df, feature_cols):
    """Run Boruta with diagnostics."""
    
    section("BORUTA CONFIGURATION (HOURLY)")
    
    print(f"  🔬 Boruta Parameters:")
    print(f"     alpha:     {BORUTA_ALPHA} (p-value threshold)")
    print(f"     perc:      {BORUTA_PERC} (shadow percentile)")
    print(f"     two_step:  {BORUTA_TWO_STEP}")
    print(f"     max_iter:  {BORUTA_MAX_ITER}")
    
    print(f"\n  🌲 Random Forest Parameters:")
    print(f"     n_estimators:      {RF_N_ESTIMATORS}")
    print(f"     max_depth:         {RF_MAX_DEPTH}")
    print(f"     min_samples_split: {RF_MIN_SAMPLES_SPLIT}")
    print(f"     min_samples_leaf:  {RF_MIN_SAMPLES_LEAF}")
    print(f"     max_features:      {RF_MAX_FEATURES}")
    
    print(f"\n  💡 Note: Settings are moderate (not lenient) because hourly has:")
    print(f"     - 26k samples (vs ~1k for daily)")
    print(f"     - Cleaner intraday signal")
    
    section("RUNNING BORUTA")
    
    X = df[feature_cols].copy()
    y = df[TARGET_VARIABLE].values
    
    print(f"  📊 Input: {X.shape[0]:,} samples × {X.shape[1]} features")
    print(f"  🎯 Target: {TARGET_VARIABLE}")
    
    # Check target distribution
    unique, counts = np.unique(y, return_counts=True)
    print(f"\n  📈 Target distribution:")
    for val, count in zip(unique, counts):
        pct = (count / len(y)) * 100
        label = "DOWN" if val == 0 else "UP"
        print(f"     {label} ({val}): {count:,} ({pct:.1f}%)")
    
    # Check class imbalance
    imbalance_ratio = max(counts) / min(counts)
    if imbalance_ratio > 2:
        print(f"\n  ⚠️  Class imbalance: {imbalance_ratio:.2f}:1 ratio")
        print(f"     Using class_weight='balanced'")
    
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
    
    # Baseline check
    print(f"\n  📊 Baseline RF accuracy (5-fold CV):")
    scores = cross_val_score(rf, X.values, y, cv=5, scoring='accuracy')
    print(f"     Mean: {scores.mean():.3f} ± {scores.std():.3f}")
    
    if scores.mean() < 0.52:
        print(f"     ⚠️  WARNING: Weak signal (near random)")
    elif scores.mean() < 0.55:
        print(f"     ✅ Weak-moderate signal (typical for hourly)")
    elif scores.mean() < 0.60:
        print(f"     ✅✅ Good signal!")
    else:
        print(f"     ✅✅✅ Excellent signal!")
    
    # Initialize Boruta
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
    if len(confirmed) >= 30:
        print(f"\n  ✅✅ Excellent! {len(confirmed)} features confirmed")
        print(f"     This is great for hourly prediction")
    elif len(confirmed) >= 15:
        print(f"\n  ✅ Good! {len(confirmed)} features confirmed")
        print(f"     Sufficient for building a model")
    elif len(confirmed) >= 5:
        print(f"\n  ⚠️  Moderate. {len(confirmed)} features confirmed")
        print(f"     You can train a model but signal is weak")
    else:
        print(f"\n  ⚠️  WARNING: Only {len(confirmed)} features confirmed")
        print(f"     Try more lenient settings:")
        print(f"     - BORUTA_ALPHA = 0.15")
        print(f"     - BORUTA_PERC = 70")
    
    return boruta, confirmed, tentative, rejected, rankings


# ============================================================================
# CREATE REPORT
# ============================================================================

def create_report(df, feature_cols, boruta, confirmed, tentative, rejected, rankings, missing_pct_before):
    """Generate report."""
    
    section("GENERATING REPORT")
    
    report_data = []
    
    for i, feature in enumerate(feature_cols):
        if feature in confirmed:
            status = 'CONFIRMED'
            status_code = 1
        elif feature in tentative:
            status = 'TENTATIVE'
            status_code = 2
        else:
            status = 'REJECTED'
            status_code = 3
        
        category = categorize_feature(feature)
        ranking = rankings[i]
        missing_pct = missing_pct_before.get(feature, 0.0)
        
        if feature in df.columns:
            mean_val = df[feature].mean()
            std_val = df[feature].std()
        else:
            mean_val = std_val = np.nan
        
        report_data.append({
            'feature': feature,
            'status': status,
            'status_code': status_code,
            'ranking': ranking,
            'category': category,
            'missing_pct_before_clean': missing_pct,
            'mean': mean_val,
            'std': std_val,
        })
    
    report_df = pd.DataFrame(report_data)
    report_df = report_df.sort_values('ranking').reset_index(drop=True)
    report_df['importance_rank'] = range(1, len(report_df) + 1)
    
    print(f"  ✅ Report created: {len(report_df)} features")
    
    return report_df


# ============================================================================
# DISPLAY SUMMARY
# ============================================================================

def display_summary(report_df):
    """Display summary."""
    
    section("SUMMARY")
    
    confirmed = report_df[report_df['status'] == 'CONFIRMED']
    tentative = report_df[report_df['status'] == 'TENTATIVE']
    
    if len(confirmed) > 0:
        print(f"\n  🏆 Top Confirmed Features:")
        for idx, row in confirmed.head(20).iterrows():
            print(f"     {row['importance_rank']:2d}. {row['feature']:50s} [{row['category']}]")
    
    if len(confirmed) > 0:
        print(f"\n  📈 Confirmed by Category:")
        cat_counts = confirmed.groupby('category').size().sort_values(ascending=False)
        for cat, count in cat_counts.items():
            print(f"     {cat:20s} {count:3d}")


# ============================================================================
# SAVE OUTPUTS
# ============================================================================

def save_outputs(df, report_df, confirmed, tentative):
    """Save outputs."""
    
    section("SAVING OUTPUTS")
    
    report_df.to_csv(OUTPUT_REPORT, index=False)
    print(f"  ✅ Report: {OUTPUT_REPORT}")
    
    selected = confirmed + tentative
    if len(selected) > 0:
        timestamp_col = 'timestamp' if 'timestamp' in df.columns else 'date'
        cols = [timestamp_col] + selected + [c for c in df.columns if c.startswith('target')]
        df[cols].to_csv(OUTPUT_SELECTED, index=False)
        print(f"  ✅ Selected: {OUTPUT_SELECTED} ({len(selected)} features)")
    else:
        print(f"  ⚠️  No features selected - not saving dataset")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution."""
    
    section("HOURLY BTC BORUTA FEATURE SELECTION")
    
    print(f"  📋 Configuration (Moderate for Hourly):")
    print(f"     Boruta alpha:  {BORUTA_ALPHA}")
    print(f"     Boruta perc:   {BORUTA_PERC}")
    print(f"     Two-step:      {BORUTA_TWO_STEP}")
    print(f"     Max iter:      {BORUTA_MAX_ITER}")
    print(f"     RF depth:      {RF_MAX_DEPTH}")
    print(f"     RF trees:      {RF_N_ESTIMATORS}")
    print(f"     Pre-filter:    {PREFILTER_TOP_N if PREFILTER_TOP_N else 'Disabled'}")
    
    print(f"\n  💡 Expected results:")
    print(f"     - Baseline accuracy: 54-57%")
    print(f"     - Confirmed features: 30-50")
    print(f"     - Much better than daily!")
    
    df, feature_cols, missing_pct_before = load_and_clean_data()
    
    feature_cols = prefilter_features(df, feature_cols)
    
    boruta, confirmed, tentative, rejected, rankings = run_boruta_with_diagnostics(df, feature_cols)
    
    report_df = create_report(df, feature_cols, boruta, confirmed, tentative, rejected, rankings, missing_pct_before)
    
    display_summary(report_df)
    
    save_outputs(df, report_df, confirmed, tentative)
    
    section("✅ COMPLETE")
    print(f"\n  Results: {len(confirmed)} confirmed, {len(tentative)} tentative")
    print(f"\n  🎯 Next: Train hourly prediction model with selected features")
    print(f"\n")


if __name__ == "__main__":
    main()