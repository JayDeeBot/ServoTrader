#!/usr/bin/env python3
"""
ServoTrader CNN-LSTM Training Script (v3.2 — Pre-Selected Features)
=====================================================================

Simplified from v3.1. All feature engineering, on-chain fetching,
sentiment loading, and Boruta selection have been removed. This script
expects the pre-processed Boruta output CSV (btc_features_selected_tunable.csv)
where the confirmed features are already present as columns.

Key changes from v3.1:
  - No feature engineering (no technical indicators, sentiment, on-chain)
  - No Boruta (features already confirmed — loaded directly from CSV)
  - Auto-detects the 12 feature columns from the CSV
  - Default sequence length: 3 days (configurable)
  - Architecture adapted for small feature count (12 features, seq=3)
  - All training, Optuna, evaluation, and save logic preserved

Architecture defaults for 12 features / 3-day window:
  - CNN filters:  [32, 64]
  - LSTM units:   [50, 25]
  - Dense units:  [32]
  - Dropout:      0.5
  - L2 reg:       0.01

Usage:
    # Basic run
    python train_cnn_lstm_v32.py --data btc_features_selected_tunable.csv

    # With Optuna hyperparameter search (recommended after first baseline)
    python train_cnn_lstm_v32.py --data btc_features_selected_tunable.csv --optuna --optuna-trials 30

    # Try different context windows
    python train_cnn_lstm_v32.py --data btc_features_selected_tunable.csv --sequence-length 5
    python train_cnn_lstm_v32.py --data btc_features_selected_tunable.csv --sequence-length 7

    # CPU only
    python train_cnn_lstm_v32.py --data btc_features_selected_tunable.csv --cpu

Author: ServoTrader
Version: 3.2 (CNN-LSTM with Pre-Selected Features)
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
from typing import Tuple, List, Dict, Any, Optional

# ------------------------------------------------------------------
# --cpu flag must be parsed before TensorFlow is imported
# ------------------------------------------------------------------
if '--cpu' in sys.argv:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    print("CPU-only mode enabled")

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# ------------------------------------------------------------------
# Optional: Optuna
# ------------------------------------------------------------------
def _check_optuna() -> bool:
    try:
        import optuna
        print(f"Optuna {optuna.__version__} available")
        return True
    except ImportError:
        print("Optuna not installed - hyperparameter tuning disabled")
        print("   Install with: pip install optuna")
        return False

OPTUNA_AVAILABLE = _check_optuna()
if OPTUNA_AVAILABLE:
    import optuna


# ------------------------------------------------------------------
# TensorFlow
# ------------------------------------------------------------------
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model, regularizers
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"TensorFlow {tf.__version__} - GPU enabled ({len(gpus)} device(s))")
    else:
        print(f"TensorFlow {tf.__version__} - CPU mode")
except ImportError:
    print("TensorFlow not installed.  Run: pip install tensorflow")
    sys.exit(1)


# ==================================================================
# CONSTANTS & DEFAULTS
# ==================================================================

PROJECT_ROOT = Path(__file__).parent
MODEL_DIR    = PROJECT_ROOT / "models" / "prediction"

DEFAULT_SEQUENCE_LENGTH = 1       # 3-day look-back window
DEFAULT_EPOCHS          = 100
DEFAULT_BATCH_SIZE      = 32
DEFAULT_PATIENCE        = 15
DEFAULT_TRAIN_RATIO     = 0.70
DEFAULT_VAL_RATIO       = 0.15
DEFAULT_CNN_FILTERS     = "32,64"
DEFAULT_LSTM_UNITS      = "50,25"
DEFAULT_DENSE_UNITS     = "32"
DEFAULT_DROPOUT         = 0.5
DEFAULT_L2_REG          = 0.01
DEFAULT_LR              = 0.001

# Column that holds the binary direction label
TARGET_COL = "target_next_day_direction"

# Columns that are never input features
_NON_FEATURE_EXACT   = {"date", "timestamp", "open_time", "close_time", "symbol"}
_NON_FEATURE_PREFIXES = ("target_",)


# ==================================================================
# DATA LOADING
# ==================================================================

def load_preselected_data(
    csv_path:        str,
    sequence_length: int,
    train_ratio:     float,
    val_ratio:       float,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray,   # X_train, X_val, X_test
    np.ndarray, np.ndarray, np.ndarray,   # y_train, y_val, y_test
    RobustScaler,
    List[str],                             # feature_names
]:
    """
    Load the Boruta-selected features CSV and return scaled sequence arrays.

    Steps:
      1. Read CSV and sort chronologically
      2. Auto-detect feature columns (numeric, not date, not target_*)
      3. Temporal 70/15/15 train/val/test split
      4. RobustScaler fitted on train only
      5. Build sliding-window sequences of `sequence_length` days
    """
    print("\n" + "=" * 60)
    print("  Loading Pre-Selected Feature Data (v3.2)")
    print("=" * 60)

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Data file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"\n  File:    {csv_path.name}")
    print(f"  Rows:    {len(df):,}")
    print(f"  Columns: {df.shape[1]}")

    # Sort chronologically
    date_col = None
    for c in df.columns:
        if c.lower() in ("date", "timestamp", "open_time"):
            date_col = c
            break
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).reset_index(drop=True)
        print(f"  Range:   {df[date_col].min().date()} -> {df[date_col].max().date()}")
    else:
        print("  WARNING: No date column found - assuming rows are in chronological order")

    # Identify feature columns
    feature_cols = []
    for col in df.columns:
        if col.lower() in _NON_FEATURE_EXACT:
            continue
        if any(col.startswith(p) for p in _NON_FEATURE_PREFIXES):
            continue
        if df[col].dtype not in (np.float64, np.float32, np.int64, np.int32, float, int):
            continue
        feature_cols.append(col)

    print(f"\n  Features detected ({len(feature_cols)}):")
    for fc in feature_cols:
        print(f"    - {fc}")

    if TARGET_COL not in df.columns:
        available_targets = [c for c in df.columns if c.startswith("target_")]
        raise ValueError(
            f"Target column '{TARGET_COL}' not found.\n"
            f"Available target columns: {available_targets}"
        )

    # Drop rows with NaN in features or target
    cols_needed = feature_cols + [TARGET_COL]
    before = len(df)
    df = df.dropna(subset=cols_needed).reset_index(drop=True)
    if len(df) < before:
        print(f"\n  WARNING: Dropped {before - len(df)} rows with NaN values")

    X_raw = df[feature_cols].values.astype(np.float32)
    y_raw = df[TARGET_COL].values.astype(np.int32)

    # Class balance check
    up_pct = y_raw.mean() * 100
    print(f"\n  Class balance:  UP={up_pct:.1f}%  DOWN={100 - up_pct:.1f}%")
    if up_pct < 25 or up_pct > 75:
        print(f"  WARNING: Unusual class balance - check target column")

    # Temporal split
    n         = len(X_raw)
    train_end = int(n * train_ratio)
    val_end   = int(n * (train_ratio + val_ratio))

    X_tr_raw = X_raw[:train_end];        y_tr_raw = y_raw[:train_end]
    X_va_raw = X_raw[train_end:val_end]; y_va_raw = y_raw[train_end:val_end]
    X_te_raw = X_raw[val_end:];          y_te_raw = y_raw[val_end:]

    print(f"\n  Temporal split (before sequences):")
    print(f"    Train: {len(X_tr_raw):,}  |  Val: {len(X_va_raw):,}  |  Test: {len(X_te_raw):,}")

    # Scale (fit on train only)
    scaler   = RobustScaler()
    X_tr_sc  = scaler.fit_transform(X_tr_raw)
    X_va_sc  = scaler.transform(X_va_raw)
    X_te_sc  = scaler.transform(X_te_raw)

    # Build sliding-window sequences
    def make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int):
        xs, ys = [], []
        for i in range(len(X) - seq_len):
            xs.append(X[i:i + seq_len])
            ys.append(y[i + seq_len])
        return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.int32)

    print(f"\n  Building {sequence_length}-step sequences...")
    X_train, y_train = make_sequences(X_tr_sc, y_tr_raw, sequence_length)
    X_val,   y_val   = make_sequences(X_va_sc, y_va_raw, sequence_length)
    X_test,  y_test  = make_sequences(X_te_sc, y_te_raw, sequence_length)

    print(f"    Train: {X_train.shape}  |  Val: {X_val.shape}  |  Test: {X_test.shape}")

    # Param/sample ratio warning (rough estimate with default arch)
    rough_params = (32 * 3 * len(feature_cols) +
                    50 * 4 * (len(feature_cols) + 50) + 32 * 50 + 32)
    ratio = rough_params / max(len(X_train), 1)
    ratio_flag = "WARNING" if ratio > 10 else "OK"
    print(f"    Param/sample ratio: ~{ratio:.1f}:1  [{ratio_flag} - target < 10:1]")

    print("\n" + "=" * 60)
    print("  Data loading complete")
    print("=" * 60)

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_cols


# ==================================================================
# MODEL ARCHITECTURE
# ==================================================================

def build_cnn_lstm(
    sequence_length: int,
    n_features:      int,
    cnn_filters:     List[int] = [32, 64],
    lstm_units:      List[int] = [50, 25],
    dense_units:     List[int] = [32],
    dropout_rate:    float     = 0.5,
    l2_reg:          float     = 0.01,
    learning_rate:   float     = 0.001,
) -> Model:
    """
    CNN-LSTM classifier for binary direction prediction.

    CNN layers extract local patterns across time steps.
    LSTM layers model sequential dependencies across those patterns.
    MaxPooling is only applied when the remaining sequence is long
    enough to avoid collapsing it to zero steps.
    """
    print("\n" + "=" * 60)
    print("  Building CNN-LSTM (v3.2)")
    print("=" * 60)
    print(f"  Input shape:  ({sequence_length}, {n_features})")
    print(f"  CNN filters:  {cnn_filters}")
    print(f"  LSTM units:   {lstm_units}")
    print(f"  Dense units:  {dense_units}")
    print(f"  Dropout:      {dropout_rate}")
    print(f"  L2 reg:       {l2_reg}")
    print(f"  LR:           {learning_rate}")

    inp         = layers.Input(shape=(sequence_length, n_features), name="input")
    x           = inp
    current_len = sequence_length

    # CNN layers
    for i, filters in enumerate(cnn_filters):
        x = layers.Conv1D(
            filters     = filters,
            kernel_size = min(3, current_len),
            padding     = "same",
            kernel_regularizer = regularizers.l2(l2_reg),
            name        = f"conv_{i+1}",
        )(x)
        x = layers.BatchNormalization(name=f"bn_conv_{i+1}")(x)
        x = layers.ReLU(name=f"relu_conv_{i+1}")(x)
        x = layers.Dropout(dropout_rate, name=f"drop_conv_{i+1}")(x)

        # Only pool if sequence is long enough to survive it
        if current_len > 2:
            x           = layers.MaxPooling1D(pool_size=2, name=f"pool_{i+1}")(x)
            current_len = current_len // 2

    # LSTM layers
    for i, units in enumerate(lstm_units):
        return_seq = (i < len(lstm_units) - 1)
        x = layers.LSTM(
            units,
            return_sequences      = return_seq,
            kernel_regularizer    = regularizers.l2(l2_reg),
            recurrent_regularizer = regularizers.l2(l2_reg),
            name                  = f"lstm_{i+1}",
        )(x)
        x = layers.Dropout(dropout_rate, name=f"drop_lstm_{i+1}")(x)

    # Dense head
    for i, units in enumerate(dense_units):
        x = layers.Dense(
            units,
            kernel_regularizer = regularizers.l2(l2_reg),
            name               = f"dense_{i+1}",
        )(x)
        x = layers.BatchNormalization(name=f"bn_dense_{i+1}")(x)
        x = layers.ReLU(name=f"relu_dense_{i+1}")(x)
        x = layers.Dropout(dropout_rate, name=f"drop_dense_{i+1}")(x)

    out = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = Model(inputs=inp, outputs=out, name="cnn_lstm_v32")
    model.compile(
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate),
        loss      = "binary_crossentropy",
        metrics   = ["accuracy"],
    )

    total_params = model.count_params()
    print(f"\n  Total parameters: {total_params:,}")
    print("=" * 60 + "\n")

    return model


# ==================================================================
# TRAINING
# ==================================================================

def train_model(
    model:      Model,
    X_train:    np.ndarray,
    y_train:    np.ndarray,
    X_val:      np.ndarray,
    y_val:      np.ndarray,
    epochs:     int            = 100,
    batch_size: int            = 32,
    patience:   int            = 15,
    output_dir: Optional[Path] = None,
) -> Tuple[Model, Dict]:
    """Train with early stopping, LR reduction, and optional checkpoint."""
    print("\n" + "=" * 60)
    print("  Training")
    print("=" * 60)
    print(f"  Epochs:      {epochs}")
    print(f"  Batch size:  {batch_size}")
    print(f"  ES patience: {patience}")

    cw  = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    cwd = dict(enumerate(cw))
    print(f"  Class weights: {cwd}")

    callbacks = [
        EarlyStopping(
            monitor              = "val_accuracy",
            patience             = patience,
            restore_best_weights = True,
            mode                 = "max",
            verbose              = 1,
        ),
        ReduceLROnPlateau(
            monitor  = "val_loss",
            factor   = 0.5,
            patience = max(patience // 2, 5),
            min_lr   = 1e-6,
            verbose  = 1,
        ),
    ]

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            ModelCheckpoint(
                filepath       = str(output_dir / "cnn_lstm_v32_best.keras"),
                monitor        = "val_accuracy",
                save_best_only = True,
                mode           = "max",
                verbose        = 1,
            )
        )

    print()
    history = model.fit(
        X_train, y_train,
        validation_data = (X_val, y_val),
        epochs          = epochs,
        batch_size      = batch_size,
        class_weight    = cwd,
        callbacks       = callbacks,
        verbose         = 1,
    )

    history_dict = {k: [float(v) for v in vals]
                    for k, vals in history.history.items()}

    best_epoch = int(np.argmax(history_dict["val_accuracy"])) + 1
    best_val   = max(history_dict["val_accuracy"])
    print(f"\n  Best epoch: {best_epoch}  val_accuracy = {best_val*100:.2f}%")
    print("\n" + "=" * 60)
    print("  Training complete")
    print("=" * 60)

    return model, history_dict


# ==================================================================
# EVALUATION
# ==================================================================

def evaluate_model(
    model:  Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    """Full evaluation on held-out test set."""
    print("\n" + "=" * 60)
    print("  Evaluation")
    print("=" * 60)

    y_prob = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)

    acc       = accuracy_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_test, y_prob)
    except Exception:
        auc = 0.5

    cm = confusion_matrix(y_test, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        down_acc = tn / max(tn + fp, 1)
        up_acc   = tp / max(tp + fn, 1)
    else:
        tn = fp = fn = tp = 0
        down_acc = up_acc = 0.5

    print(f"\n  Accuracy:   {acc*100:.2f}%")
    print(f"  F1 score:   {f1:.4f}")
    print(f"  Precision:  {precision:.4f}")
    print(f"  Recall:     {recall:.4f}")
    print(f"  AUC-ROC:    {auc:.4f}")
    print(f"\n  Per-class accuracy:")
    print(f"    DOWN  {down_acc*100:.1f}%  ({tn} correct / {tn+fp} total)")
    print(f"    UP    {up_acc*100:.1f}%  ({tp} correct / {tp+fn} total)")
    print(f"\n  Confusion matrix:")
    print(f"               Pred DOWN   Pred UP")
    print(f"  Actual DOWN  {tn:9d}  {fp:8d}")
    print(f"  Actual UP    {fn:9d}  {tp:8d}")
    print("=" * 60 + "\n")

    return {
        "accuracy":      float(acc),
        "f1":            float(f1),
        "precision":     float(precision),
        "recall":        float(recall),
        "auc_roc":       float(auc),
        "down_accuracy": float(down_acc),
        "up_accuracy":   float(up_acc),
        "tn": int(tn), "fp": int(fp),
        "fn": int(fn), "tp": int(tp),
    }


# ==================================================================
# OPTUNA HYPERPARAMETER OPTIMISATION
# ==================================================================

def _build_optuna_objective(
    X_train:     np.ndarray,
    y_train:     np.ndarray,
    X_val:       np.ndarray,
    y_val:       np.ndarray,
    n_features:  int,
    seq_len:     int,
    n_cv_folds:  int,
):
    """
    Returns an Optuna objective evaluated via TimeSeriesSplit CV
    over the combined train+val set — avoids overfitting to a single
    validation split.
    """
    X_all = np.concatenate([X_train, X_val], axis=0)
    y_all = np.concatenate([y_train, y_val], axis=0)
    tscv  = TimeSeriesSplit(n_splits=n_cv_folds)

    def objective(trial):
        tf.keras.backend.clear_session()

        n_cnn  = trial.suggest_int("n_cnn_layers", 1, 2)
        cnn_f  = [trial.suggest_categorical(f"cnn_f_{i}", [16, 32, 64])
                  for i in range(n_cnn)]

        n_lstm = trial.suggest_int("n_lstm_layers", 1, 2)
        lstm_u = [trial.suggest_categorical(f"lstm_u_{i}", [25, 50, 75])
                  for i in range(n_lstm)]

        n_dense = trial.suggest_int("n_dense_layers", 1, 2)
        dense_u = [trial.suggest_categorical(f"dense_u_{i}", [16, 32, 64])
                   for i in range(n_dense)]

        dropout    = trial.suggest_float("dropout",    0.2, 0.6, step=0.1)
        l2_reg     = trial.suggest_float("l2_reg",     1e-4, 0.1, log=True)
        lr         = trial.suggest_float("lr",         1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

        cv_scores = []
        for tr_idx, va_idx in tscv.split(X_all):
            tf.keras.backend.clear_session()

            X_cv_tr, X_cv_va = X_all[tr_idx], X_all[va_idx]
            y_cv_tr, y_cv_va = y_all[tr_idx], y_all[va_idx]

            m = build_cnn_lstm(
                sequence_length = seq_len,
                n_features      = n_features,
                cnn_filters     = cnn_f,
                lstm_units      = lstm_u,
                dense_units     = dense_u,
                dropout_rate    = dropout,
                l2_reg          = l2_reg,
                learning_rate   = lr,
            )

            cw  = compute_class_weight("balanced",
                                       classes=np.unique(y_cv_tr), y=y_cv_tr)
            h = m.fit(
                X_cv_tr, y_cv_tr,
                validation_data = (X_cv_va, y_cv_va),
                epochs          = 30,
                batch_size      = batch_size,
                class_weight    = dict(enumerate(cw)),
                callbacks       = [
                    EarlyStopping(monitor="val_accuracy", patience=7,
                                  restore_best_weights=True, mode="max"),
                    ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                     patience=4, min_lr=1e-6),
                ],
                verbose = 0,
            )
            cv_scores.append(max(h.history["val_accuracy"]))

        mean_score = float(np.mean(cv_scores))
        std_score  = float(np.std(cv_scores))

        trial.set_user_attr("cv_scores", cv_scores)
        trial.set_user_attr("cv_std",    std_score)
        trial.report(mean_score, step=n_cv_folds)

        if trial.should_prune():
            raise optuna.TrialPruned()

        return mean_score

    return objective


def run_optuna(
    X_train:    np.ndarray,
    y_train:    np.ndarray,
    X_val:      np.ndarray,
    y_val:      np.ndarray,
    n_features: int,
    seq_len:    int,
    n_trials:   int            = 20,
    n_cv_folds: int            = 3,
    timeout:    Optional[int]  = None,
) -> Optional[Dict[str, Any]]:
    """Run Optuna search and return best hyperparameter dict."""
    if not OPTUNA_AVAILABLE:
        print("Optuna not available — skipping hyperparameter search")
        return None

    from optuna.pruners  import MedianPruner
    from optuna.samplers import TPESampler

    print("\n" + "=" * 60)
    print("  Optuna Hyperparameter Search (TimeSeriesSplit CV)")
    print("=" * 60)
    print(f"  Trials:   {n_trials}")
    print(f"  CV folds: {n_cv_folds}  (TimeSeriesSplit)")
    print(f"  Timeout:  {timeout}s" if timeout else "  Timeout:  none")
    print(f"  Each trial trains {n_cv_folds} models for robust evaluation\n")

    objective = _build_optuna_objective(
        X_train, y_train, X_val, y_val,
        n_features, seq_len, n_cv_folds,
    )

    study = optuna.create_study(
        direction = "maximize",
        sampler   = TPESampler(seed=42),
        pruner    = MedianPruner(n_startup_trials=3, n_warmup_steps=5),
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def _cb(study, trial):
        cv_std = trial.user_attrs.get("cv_std", 0.0)
        val    = trial.value if trial.value else 0.0
        print(f"  Trial {trial.number+1:>3}/{n_trials}  "
              f"cv_acc={val:.4f} +/-{cv_std:.4f}  "
              f"best={study.best_value:.4f}")

    study.optimize(
        objective,
        n_trials          = n_trials,
        timeout           = timeout,
        callbacks         = [_cb],
        show_progress_bar = False,
    )

    bp      = study.best_params
    best_t  = study.best_trial
    cv_std  = best_t.user_attrs.get("cv_std",    0.0)
    cv_sc   = best_t.user_attrs.get("cv_scores", [])

    n_cnn   = bp["n_cnn_layers"]
    n_lstm_ = bp["n_lstm_layers"]
    n_dn    = bp["n_dense_layers"]
    cnn_f   = [bp[f"cnn_f_{i}"]   for i in range(n_cnn)]
    lstm_u  = [bp[f"lstm_u_{i}"]  for i in range(n_lstm_)]
    dense_u = [bp[f"dense_u_{i}"] for i in range(n_dn)]

    print(f"\n  Best CV accuracy: {study.best_value*100:.2f}% +/-{cv_std*100:.2f}%")
    print(f"  Fold scores: {[f'{s*100:.1f}%' for s in cv_sc]}")
    print(f"\n  Best hyperparameters:")
    print(f"    CNN filters:  {cnn_f}")
    print(f"    LSTM units:   {lstm_u}")
    print(f"    Dense units:  {dense_u}")
    print(f"    Dropout:      {bp['dropout']:.2f}")
    print(f"    L2 reg:       {bp['l2_reg']:.6f}")
    print(f"    LR:           {bp['lr']:.6f}")
    print(f"    Batch size:   {bp['batch_size']}")
    print("=" * 60 + "\n")

    return {
        "cnn_filters": cnn_f,
        "lstm_units":  lstm_u,
        "dense_units": dense_u,
        "dropout":     bp["dropout"],
        "l2_reg":      bp["l2_reg"],
        "lr":          bp["lr"],
        "batch_size":  bp["batch_size"],
        "best_cv_acc": study.best_value,
        "cv_std":      cv_std,
        "cv_scores":   cv_sc,
        "n_trials":    len(study.trials),
        "n_cv_folds":  n_cv_folds,
    }


# ==================================================================
# SAVE ARTIFACTS
# ==================================================================

def save_artifacts(
    model:         Model,
    scaler:        RobustScaler,
    feature_names: List[str],
    config:        Dict,
    metrics:       Dict,
    history:       Dict,
    output_dir:    Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Saving to: {output_dir}")

    model.save(output_dir / "cnn_lstm_v32_model.keras")
    print("    Model saved  (cnn_lstm_v32_model.keras)")

    with open(output_dir / "cnn_lstm_v32_metadata.pkl", "wb") as f:
        pickle.dump({"scaler": scaler, "feature_names": feature_names}, f)
    print("    Metadata saved  (scaler + feature names)")

    with open(output_dir / "cnn_lstm_v32_config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    print("    Config saved")

    with open(output_dir / "cnn_lstm_v32_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("    Metrics saved")

    with open(output_dir / "cnn_lstm_v32_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print("    Training history saved")


# ==================================================================
# MAIN
# ==================================================================

def main():
    p = argparse.ArgumentParser(
        description="Train CNN-LSTM v3.2 on pre-selected Boruta features",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--data",            required=True,
                   help="Path to btc_features_selected_tunable.csv")
    p.add_argument("--sequence-length", type=int,   default=DEFAULT_SEQUENCE_LENGTH,
                   help="Context window in days (try 3, 5, 7)")
    p.add_argument("--cnn-filters",     default=DEFAULT_CNN_FILTERS,
                   help="Comma-separated CNN filter sizes")
    p.add_argument("--lstm-units",      default=DEFAULT_LSTM_UNITS,
                   help="Comma-separated LSTM unit counts")
    p.add_argument("--dense-units",     default=DEFAULT_DENSE_UNITS,
                   help="Comma-separated Dense unit counts")
    p.add_argument("--dropout",         type=float, default=DEFAULT_DROPOUT)
    p.add_argument("--l2-reg",          type=float, default=DEFAULT_L2_REG)
    p.add_argument("--learning-rate",   type=float, default=DEFAULT_LR)
    p.add_argument("--epochs",          type=int,   default=DEFAULT_EPOCHS)
    p.add_argument("--batch-size",      type=int,   default=DEFAULT_BATCH_SIZE)
    p.add_argument("--patience",        type=int,   default=DEFAULT_PATIENCE)
    p.add_argument("--optuna",          action="store_true",
                   help="Run Optuna search before final training")
    p.add_argument("--optuna-trials",   type=int,   default=20)
    p.add_argument("--optuna-cv-folds", type=int,   default=3,
                   help="TimeSeriesSplit folds per Optuna trial")
    p.add_argument("--optuna-timeout",  type=int,   default=None,
                   help="Wall-clock limit (seconds) for Optuna")
    p.add_argument("--output",          default=str(MODEL_DIR),
                   help="Directory to save model and artifacts")
    p.add_argument("--cpu",             action="store_true")

    args = p.parse_args()

    print("\n" + "=" * 60)
    print("  ServoTrader CNN-LSTM v3.2  (Pre-Selected Features)")
    print("=" * 60)
    print(f"  Start:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Data:   {args.data}")
    print(f"  Window: {args.sequence_length} days")

    cnn_filters = [int(x) for x in args.cnn_filters.split(",")]
    lstm_units  = [int(x) for x in args.lstm_units.split(",")]
    dense_units = [int(x) for x in args.dense_units.split(",")]

    # Load data
    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     scaler, feature_names) = load_preselected_data(
        csv_path        = args.data,
        sequence_length = args.sequence_length,
        train_ratio     = DEFAULT_TRAIN_RATIO,
        val_ratio       = DEFAULT_VAL_RATIO,
    )
    n_features = X_train.shape[2]

    # Optional Optuna search
    optuna_results = None
    if args.optuna:
        optuna_results = run_optuna(
            X_train, y_train, X_val, y_val,
            n_features  = n_features,
            seq_len     = args.sequence_length,
            n_trials    = args.optuna_trials,
            n_cv_folds  = args.optuna_cv_folds,
            timeout     = args.optuna_timeout,
        )
        if optuna_results:
            cnn_filters        = optuna_results["cnn_filters"]
            lstm_units         = optuna_results["lstm_units"]
            dense_units        = optuna_results["dense_units"]
            args.dropout       = optuna_results["dropout"]
            args.l2_reg        = optuna_results["l2_reg"]
            args.learning_rate = optuna_results["lr"]
            args.batch_size    = optuna_results["batch_size"]
            print("  Using Optuna-tuned parameters for final training")

    # Build model
    model = build_cnn_lstm(
        sequence_length = args.sequence_length,
        n_features      = n_features,
        cnn_filters     = cnn_filters,
        lstm_units      = lstm_units,
        dense_units     = dense_units,
        dropout_rate    = args.dropout,
        l2_reg          = args.l2_reg,
        learning_rate   = args.learning_rate,
    )
    model.summary()

    # Train
    model, history = train_model(
        model      = model,
        X_train    = X_train,
        y_train    = y_train,
        X_val      = X_val,
        y_val      = y_val,
        epochs     = args.epochs,
        batch_size = args.batch_size,
        patience   = args.patience,
        output_dir = Path(args.output),
    )

    # Evaluate
    metrics = evaluate_model(model, X_test, y_test)

    # Save
    config = {
        "version":           "3.2",
        "architecture":      "CNN-LSTM-PreSelected",
        "data_path":         str(args.data),
        "sequence_length":   args.sequence_length,
        "n_features":        n_features,
        "feature_names":     feature_names,
        "target_column":     TARGET_COL,
        "cnn_filters":       cnn_filters,
        "lstm_units":        lstm_units,
        "dense_units":       dense_units,
        "dropout":           args.dropout,
        "l2_reg":            args.l2_reg,
        "learning_rate":     args.learning_rate,
        "epochs":            args.epochs,
        "batch_size":        args.batch_size,
        "patience":          args.patience,
        "total_parameters":  model.count_params(),
        "use_optuna":        args.optuna,
        "optuna_trials":     args.optuna_trials   if args.optuna else 0,
        "optuna_cv_folds":   args.optuna_cv_folds if args.optuna else 0,
        "optuna_best_cv_acc":optuna_results["best_cv_acc"] if optuna_results else None,
        "optuna_cv_std":     optuna_results["cv_std"]      if optuna_results else None,
    }

    save_artifacts(
        model=model, scaler=scaler, feature_names=feature_names,
        config=config, metrics=metrics, history=history,
        output_dir=Path(args.output),
    )

    # Final summary
    best_epoch = int(np.argmax(history["val_accuracy"])) + 1
    best_val   = max(history["val_accuracy"])

    print("\n" + "=" * 60)
    print("  Training Complete (v3.2)")
    print("=" * 60)
    print(f"  Finished:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n  Test Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"  Test F1:       {metrics['f1']:.4f}")
    print(f"  AUC-ROC:       {metrics['auc_roc']:.4f}")
    print(f"  DOWN accuracy: {metrics['down_accuracy']*100:.1f}%")
    print(f"  UP accuracy:   {metrics['up_accuracy']*100:.1f}%")
    print(f"\n  Best epoch:    {best_epoch}  ({best_val*100:.2f}% val acc)")
    print(f"  Parameters:    {model.count_params():,}")
    print(f"  Param/sample:  {model.count_params()/len(X_train):.1f}:1")

    if optuna_results:
        print(f"\n  Optuna trials: {optuna_results['n_trials']}")
        print(f"  Best CV acc:   {optuna_results['best_cv_acc']*100:.2f}% "
              f"+/-{optuna_results['cv_std']*100:.2f}%")

    print(f"\n  Saved to: {args.output}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()