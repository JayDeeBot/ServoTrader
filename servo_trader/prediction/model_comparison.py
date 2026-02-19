"""
ServoTrader — Model Comparison Framework
=========================================
Binary classification: Will BTC rise or fall the next day?

Models tested:
  1. Logistic Regression    — Simple baseline
  2. Random Forest          — Interpretable ensemble baseline
  3. LightGBM               — Fast gradient boosting (research frontrunner for BTC)
  4. XGBoost                — Gradient boosting (research shows consistent outperformance)
  5. GRU                    — Lightweight recurrent network
  6. CNN-LSTM               — Convolutional + recurrent hybrid

Context window: 3 days (configurable via CONTEXT_WINDOW)
Cross-validation: TimeSeriesSplit (no data leakage)

Usage:
    python model_comparison.py --data path/to/btc_daily_features_selected.csv
    python model_comparison.py --data path/to/features.csv --window 5 --target target_direction
"""

import argparse
import warnings
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix, classification_report
)
from sklearn.pipeline import Pipeline

# Gradient Boosting
import lightgbm as lgb
import xgboost as xgb

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    GRU, LSTM, Conv1D, MaxPooling1D, Dense, Dropout,
    BatchNormalization, Flatten, Input
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

CONTEXT_WINDOW   = 3       # Days of history to use as input features
N_CV_SPLITS      = 5       # TimeSeriesSplit folds
RANDOM_STATE     = 42
TEST_SIZE_RATIO  = 0.2     # Final holdout test size

# Deep learning settings
DL_EPOCHS        = 50
DL_BATCH_SIZE    = 32
DL_PATIENCE      = 10      # Early stopping patience
DL_DROPOUT       = 0.3

# Target column (auto-detected if not specified)
DEFAULT_TARGET   = 'target_next_day_direction'

# ============================================================================
# DATA LOADING & PREPARATION
# ============================================================================

def load_data(csv_path: str, target_col: str) -> tuple[pd.DataFrame, list, str]:
    """Load feature CSV, auto-detect target column if needed."""
    print(f"\n{'='*70}")
    print(f"  LOADING DATA")
    print(f"{'='*70}")

    df = pd.read_csv(csv_path)
    print(f"  Loaded: {csv_path}")
    print(f"  Shape:  {df.shape[0]} rows × {df.shape[1]} columns")

    # Parse dates
    date_cols = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]
    if date_cols:
        df[date_cols[0]] = pd.to_datetime(df[date_cols[0]])
        df = df.sort_values(date_cols[0]).reset_index(drop=True)
        print(f"  Date range: {df[date_cols[0]].min().date()} → {df[date_cols[0]].max().date()}")

    # Auto-detect target column
    target_candidates = [c for c in df.columns if 'direction' in c.lower() or 'target' in c.lower()]
    if target_col not in df.columns:
        if target_candidates:
            target_col = target_candidates[0]
            print(f"  ⚠  Target column not found, using: '{target_col}'")
        else:
            raise ValueError(f"Could not find target column. Available: {df.columns.tolist()}")
    else:
        print(f"  Target:  '{target_col}'")

    # Identify feature columns (exclude dates and all targets)
    non_feature = date_cols + [c for c in df.columns if 'target' in c.lower()]
    feature_cols = [c for c in df.columns if c not in non_feature]

    print(f"  Features: {len(feature_cols)}")
    print(f"  Class distribution:")
    counts = df[target_col].value_counts()
    for val, count in counts.items():
        label = 'UP  ' if val == 1 else 'DOWN'
        print(f"    {label} ({val}): {count} ({count/len(df)*100:.1f}%)")

    return df, feature_cols, target_col


def create_context_features(df: pd.DataFrame, feature_cols: list, target_col: str,
                             window: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (X, y) with a sliding context window.

    For tree models (tabular): features are flattened across `window` days.
    Shape: (samples, window * n_features)

    For sequence models (DL): reshaped to (samples, window, n_features)
    Both are returned; callers choose which shape to use.
    """
    X_raw = df[feature_cols].values.astype(np.float32)
    y_raw = df[target_col].values.astype(np.int32)

    # Replace inf / nan
    X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=1e6, neginf=-1e6)

    n_samples, n_features = X_raw.shape
    valid_samples = n_samples - window

    # Sequence shape: (samples, window, n_features)
    X_seq = np.zeros((valid_samples, window, n_features), dtype=np.float32)
    y_out = np.zeros(valid_samples, dtype=np.int32)

    for i in range(valid_samples):
        X_seq[i] = X_raw[i:i + window]
        y_out[i] = y_raw[i + window]

    # Flat shape: (samples, window * n_features)
    X_flat = X_seq.reshape(valid_samples, window * n_features)

    return X_flat, X_seq, y_out


# ============================================================================
# MODEL BUILDERS
# ============================================================================

def build_logistic_regression():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight='balanced',
            random_state=RANDOM_STATE
        ))
    ])


def build_random_forest():
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight='balanced',
        n_jobs=-1,
        random_state=RANDOM_STATE
    )


def build_lightgbm():
    return lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=6,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        class_weight='balanced',
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=-1
    )


def build_xgboost():
    return xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        use_label_encoder=False,
        eval_metric='logloss',
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbosity=0
    )


def build_gru(window: int, n_features: int) -> tf.keras.Model:
    """Gated Recurrent Unit — lighter alternative to LSTM."""
    model = Sequential([
        Input(shape=(window, n_features)),
        GRU(64, return_sequences=True),
        Dropout(DL_DROPOUT),
        BatchNormalization(),
        GRU(32, return_sequences=False),
        Dropout(DL_DROPOUT),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ], name='GRU')
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def build_cnn_lstm(window: int, n_features: int) -> tf.keras.Model:
    """Conv1D feature extraction followed by LSTM temporal modelling."""
    model = Sequential([
        Input(shape=(window, n_features)),
        Conv1D(filters=64, kernel_size=min(3, window), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=1),                  # pool_size=1 safe for small windows
        Dropout(DL_DROPOUT),
        LSTM(64, return_sequences=False),
        Dropout(DL_DROPOUT),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ], name='CNN_LSTM')
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


# ============================================================================
# EVALUATION HELPERS
# ============================================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_prob: np.ndarray = None) -> dict:
    """Compute classification metrics for one fold."""
    metrics = {
        'accuracy':  accuracy_score(y_true, y_pred),
        'f1':        f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall':    recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'up_acc':    0.0,
        'down_acc':  0.0,
    }

    # Per-class accuracy
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        metrics['down_acc'] = cm[0, 0] / cm[0].sum() if cm[0].sum() > 0 else 0
        metrics['up_acc']   = cm[1, 1] / cm[1].sum() if cm[1].sum() > 0 else 0

    # AUC
    if y_prob is not None:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_prob)
        except Exception:
            metrics['auc'] = 0.5
    else:
        metrics['auc'] = None

    return metrics


def get_dl_callbacks():
    return [
        EarlyStopping(monitor='val_loss', patience=DL_PATIENCE,
                      restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5,
                          min_lr=1e-5, verbose=0)
    ]


# ============================================================================
# CROSS-VALIDATION RUNNER
# ============================================================================

def run_cv(name: str, X_flat: np.ndarray, X_seq: np.ndarray,
           y: np.ndarray, window: int, n_splits: int) -> dict:
    """
    Run TimeSeriesSplit cross-validation for one model.
    Tree models use X_flat; DL models use X_seq.
    """
    is_dl = name in ('GRU', 'CNN-LSTM')
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics = []

    print(f"\n  Running {name} ({n_splits}-fold CV)...")
    t0 = time.time()

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_flat), 1):

        if is_dl:
            X_tr, X_val = X_seq[train_idx], X_seq[val_idx]
        else:
            X_tr, X_val = X_flat[train_idx], X_flat[val_idx]

        y_tr, y_val = y[train_idx], y[val_idx]

        # Scale for DL models
        if is_dl:
            scaler = StandardScaler()
            n_s, n_w, n_f = X_tr.shape
            X_tr  = scaler.fit_transform(X_tr.reshape(-1, n_f)).reshape(n_s, n_w, n_f)
            n_sv  = X_val.shape[0]
            X_val = scaler.transform(X_val.reshape(-1, n_f)).reshape(n_sv, n_w, n_f)

        # Build and train
        if name == 'GRU':
            model = build_gru(window, X_seq.shape[2])
            model.fit(
                X_tr, y_tr,
                validation_data=(X_val, y_val),
                epochs=DL_EPOCHS,
                batch_size=DL_BATCH_SIZE,
                callbacks=get_dl_callbacks(),
                class_weight={0: 1.0, 1: 1.0},
                verbose=0
            )
            y_prob = model.predict(X_val, verbose=0).flatten()
            y_pred = (y_prob >= 0.5).astype(int)
            tf.keras.backend.clear_session()

        elif name == 'CNN-LSTM':
            model = build_cnn_lstm(window, X_seq.shape[2])
            model.fit(
                X_tr, y_tr,
                validation_data=(X_val, y_val),
                epochs=DL_EPOCHS,
                batch_size=DL_BATCH_SIZE,
                callbacks=get_dl_callbacks(),
                verbose=0
            )
            y_prob = model.predict(X_val, verbose=0).flatten()
            y_pred = (y_prob >= 0.5).astype(int)
            tf.keras.backend.clear_session()

        elif name == 'LightGBM':
            model = build_lightgbm()
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False),
                           lgb.log_evaluation(period=-1)]
            )
            y_prob = model.predict_proba(X_val)[:, 1]
            y_pred = model.predict(X_val)

        elif name == 'XGBoost':
            model = build_xgboost()
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            y_prob = model.predict_proba(X_val)[:, 1]
            y_pred = model.predict(X_val)

        elif name == 'Random Forest':
            model = build_random_forest()
            model.fit(X_tr, y_tr)
            y_prob = model.predict_proba(X_val)[:, 1]
            y_pred = model.predict(X_val)

        elif name == 'Logistic Regression':
            model = build_logistic_regression()
            model.fit(X_tr, y_tr)
            y_prob = model.predict_proba(X_val)[:, 1]
            y_pred = model.predict(X_val)

        else:
            raise ValueError(f"Unknown model: {name}")

        fold_metrics.append(compute_metrics(y_val, y_pred, y_prob))
        acc = fold_metrics[-1]['accuracy']
        print(f"    Fold {fold}/{n_splits} — acc: {acc:.4f}", end='\r')

    elapsed = time.time() - t0

    # Aggregate folds
    agg = {}
    for key in fold_metrics[0]:
        vals = [m[key] for m in fold_metrics if m[key] is not None]
        agg[key]         = np.mean(vals)
        agg[f'{key}_std'] = np.std(vals)

    agg['time_seconds'] = elapsed
    agg['model']        = name

    print(f"    {name}: acc={agg['accuracy']:.4f} ± {agg['accuracy_std']:.4f}  "
          f"auc={agg.get('auc', 0):.4f}  [{elapsed:.0f}s]        ")
    return agg


# ============================================================================
# FINAL HOLDOUT EVALUATION
# ============================================================================

def final_holdout_eval(name: str, X_flat: np.ndarray, X_seq: np.ndarray,
                        y: np.ndarray, window: int,
                        test_ratio: float = TEST_SIZE_RATIO) -> dict:
    """Train on first (1-test_ratio) of data, evaluate on last test_ratio."""
    split = int(len(y) * (1 - test_ratio))
    is_dl = name in ('GRU', 'CNN-LSTM')

    if is_dl:
        X_tr, X_te = X_seq[:split], X_seq[split:]
        scaler = StandardScaler()
        n_s, n_w, n_f = X_tr.shape
        X_tr = scaler.fit_transform(X_tr.reshape(-1, n_f)).reshape(n_s, n_w, n_f)
        n_te = X_te.shape[0]
        X_te = scaler.transform(X_te.reshape(-1, n_f)).reshape(n_te, n_w, n_f)
    else:
        X_tr, X_te = X_flat[:split], X_flat[split:]

    y_tr, y_te = y[:split], y[split:]

    if name == 'GRU':
        model = build_gru(window, X_seq.shape[2])
        model.fit(X_tr, y_tr, epochs=DL_EPOCHS, batch_size=DL_BATCH_SIZE,
                  validation_split=0.1, callbacks=get_dl_callbacks(), verbose=0)
        y_prob = model.predict(X_te, verbose=0).flatten()
        y_pred = (y_prob >= 0.5).astype(int)
        tf.keras.backend.clear_session()

    elif name == 'CNN-LSTM':
        model = build_cnn_lstm(window, X_seq.shape[2])
        model.fit(X_tr, y_tr, epochs=DL_EPOCHS, batch_size=DL_BATCH_SIZE,
                  validation_split=0.1, callbacks=get_dl_callbacks(), verbose=0)
        y_prob = model.predict(X_te, verbose=0).flatten()
        y_pred = (y_prob >= 0.5).astype(int)
        tf.keras.backend.clear_session()

    elif name == 'LightGBM':
        model = build_lightgbm()
        model.fit(X_tr, y_tr, callbacks=[lgb.log_evaluation(period=-1)])
        y_prob = model.predict_proba(X_te)[:, 1]
        y_pred = model.predict(X_te)

    elif name == 'XGBoost':
        model = build_xgboost()
        model.fit(X_tr, y_tr, verbose=False)
        y_prob = model.predict_proba(X_te)[:, 1]
        y_pred = model.predict(X_te)

    elif name == 'Random Forest':
        model = build_random_forest()
        model.fit(X_tr, y_tr)
        y_prob = model.predict_proba(X_te)[:, 1]
        y_pred = model.predict(X_te)

    elif name == 'Logistic Regression':
        model = build_logistic_regression()
        model.fit(X_tr, y_tr)
        y_prob = model.predict_proba(X_te)[:, 1]
        y_pred = model.predict(X_te)

    metrics = compute_metrics(y_te, y_pred, y_prob)
    metrics['model'] = name
    return metrics, y_te, y_pred


# ============================================================================
# RESULTS TABLE
# ============================================================================

def print_results_table(cv_results: list, holdout_results: list):
    """Print a formatted comparison table."""
    print(f"\n{'='*70}")
    print(f"  CROSS-VALIDATION RESULTS  ({N_CV_SPLITS}-fold TimeSeriesSplit)")
    print(f"{'='*70}")
    print(f"  {'Model':<22} {'Accuracy':>10} {'F1':>8} {'AUC':>8} {'UP%':>8} {'DOWN%':>8} {'Time':>7}")
    print(f"  {'-'*68}")
    
    # Sort by accuracy descending
    cv_results_sorted = sorted(cv_results, key=lambda x: x['accuracy'], reverse=True)
    for r in cv_results_sorted:
        up   = r.get('up_acc', 0) * 100
        down = r.get('down_acc', 0) * 100
        auc  = r.get('auc', 0) or 0
        t    = r['time_seconds']
        print(f"  {r['model']:<22} {r['accuracy']:>8.4f}  {r['f1']:>6.4f}  {auc:>6.4f}  {up:>6.1f}%  {down:>6.1f}%  {t:>5.0f}s")

    print(f"\n{'='*70}")
    print(f"  HOLDOUT TEST RESULTS  (last {TEST_SIZE_RATIO*100:.0f}% of data)")
    print(f"{'='*70}")
    print(f"  {'Model':<22} {'Accuracy':>10} {'F1':>8} {'AUC':>8} {'UP%':>8} {'DOWN%':>8}")
    print(f"  {'-'*60}")
    
    holdout_sorted = sorted(holdout_results, key=lambda x: x['accuracy'], reverse=True)
    for r in holdout_sorted:
        up   = r.get('up_acc', 0) * 100
        down = r.get('down_acc', 0) * 100
        auc  = r.get('auc', 0) or 0
        print(f"  {r['model']:<22} {r['accuracy']:>8.4f}  {r['f1']:>6.4f}  {auc:>6.4f}  {up:>6.1f}%  {down:>6.1f}%")

    # Identify best model
    best = holdout_sorted[0]
    print(f"\n  🏆 Best model (holdout): {best['model']}  —  acc={best['accuracy']:.4f}  auc={best.get('auc',0):.4f}")
    print(f"{'='*70}\n")


def save_results_csv(cv_results: list, holdout_results: list, out_path: str):
    cv_df = pd.DataFrame(cv_results)
    cv_df['split'] = 'cv'
    ho_df = pd.DataFrame(holdout_results)
    ho_df['split'] = 'holdout'
    combined = pd.concat([cv_df, ho_df], ignore_index=True)
    combined.to_csv(out_path, index=False)
    print(f"  Results saved → {out_path}")


# ============================================================================
# MAIN
# ============================================================================

def main(args):
    print(f"\n{'='*70}")
    print(f"  SERVOTRADER — MODEL COMPARISON")
    print(f"  Context window: {args.window} days | CV splits: {N_CV_SPLITS}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")

    # Load data
    df, feature_cols, target_col = load_data(args.data, args.target)

    # Build context-windowed arrays
    print(f"\n  Building {args.window}-day context window...")
    X_flat, X_seq, y = create_context_features(df, feature_cols, target_col, args.window)
    print(f"  Tabular shape:  {X_flat.shape}  (samples × window×features)")
    print(f"  Sequence shape: {X_seq.shape}  (samples × window × features)")
    print(f"  Labels:         {y.shape}  | UP={y.sum()} ({y.mean()*100:.1f}%)  DOWN={(~y.astype(bool)).sum()}")

    # Model registry
    models_to_run = [
        'Logistic Regression',
        'Random Forest',
        'LightGBM',
        'XGBoost',
        'GRU',
        'CNN-LSTM',
    ]

    # Filter if user specified subset
    if args.models:
        models_to_run = [m for m in models_to_run
                         if any(a.lower() in m.lower() for a in args.models)]
        print(f"\n  Running subset: {models_to_run}")

    print(f"\n{'='*70}")
    print(f"  CROSS-VALIDATION")
    print(f"{'='*70}")

    cv_results = []
    holdout_results = []

    for model_name in models_to_run:
        try:
            # CV
            cv_agg = run_cv(model_name, X_flat, X_seq, y, args.window, N_CV_SPLITS)
            cv_results.append(cv_agg)

            # Holdout
            ho_metrics, y_te, y_pred = final_holdout_eval(
                model_name, X_flat, X_seq, y, args.window
            )
            holdout_results.append(ho_metrics)

        except Exception as e:
            print(f"  ❌ {model_name} failed: {e}")
            import traceback; traceback.print_exc()

    # Print results
    print_results_table(cv_results, holdout_results)

    # Save results CSV
    out_csv = Path(args.output) / 'model_comparison_results.csv'
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    save_results_csv(cv_results, holdout_results, str(out_csv))

    # Detailed classification report for each model
    print(f"\n{'='*70}")
    print(f"  DETAILED HOLDOUT REPORTS")
    print(f"{'='*70}")
    for ho, (model_name, y_te, y_pred_) in zip(
        holdout_results,
        [(m, *final_holdout_eval(m, X_flat, X_seq, y, args.window)[1:]) for m in models_to_run
         if m in [r['model'] for r in holdout_results]]
    ):
        print(f"\n  {model_name}:")
        print(classification_report(y_te, y_pred_, target_names=['DOWN', 'UP'], digits=3))

    print(f"\n  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ServoTrader Model Comparison')
    parser.add_argument(
        '--data', type=str, required=True,
        help='Path to Boruta-selected features CSV (e.g. btc_daily_features_selected.csv)'
    )
    parser.add_argument(
        '--target', type=str, default=DEFAULT_TARGET,
        help=f'Target column name (default: {DEFAULT_TARGET})'
    )
    parser.add_argument(
        '--window', type=int, default=CONTEXT_WINDOW,
        help=f'Context window in days (default: {CONTEXT_WINDOW})'
    )
    parser.add_argument(
        '--output', type=str, default='./results',
        help='Directory to save results CSV (default: ./results)'
    )
    parser.add_argument(
        '--models', nargs='+', default=None,
        help='Subset of models to run, e.g. --models lightgbm xgboost gru'
    )
    args = parser.parse_args()
    main(args)