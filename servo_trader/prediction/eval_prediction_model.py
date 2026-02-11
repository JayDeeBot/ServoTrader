#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_prediction_model.py

Evaluation script for the LightGBM-only cryptocurrency prediction model.

Author: Jarred Deluca
Project: ServoTrader - Prediction Subsystem (v2 - LightGBM Only)
"""

import os
import sys
import argparse
import json
from datetime import datetime
from typing import Dict, Any

import numpy as np
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_pipeline import CryptoPredictionDataPipeline, PredictionDataset
from lgb_predictor import LightGBMPredictor


DEFAULT_MODEL_DIR = '/home/jarred/git/ServoTrader/models/prediction'
DEFAULT_CSV_PATH = '/home/jarred/git/ServoTrader/data/individual_crypto_csvs_ancient_5_min/BTCUSDT.csv'


def simple_backtest(y_true_returns: np.ndarray,
                    pred_directions: np.ndarray,
                    up_probs: np.ndarray,
                    trade_fee_pct: float = 0.001,
                    confidence_threshold: float = 0.5) -> Dict[str, Any]:
    """Run a simple backtest based on predictions (binary: 0=Down, 1=Up)."""
    n_samples = len(y_true_returns)
    
    positions = np.zeros(n_samples)
    strategy_returns = np.zeros(n_samples)
    
    # For binary classification, up_probs is a 1D array
    # Confidence is distance from 0.5
    confidences = np.abs(up_probs - 0.5) + 0.5
    
    for i in range(n_samples):
        if confidences[i] >= confidence_threshold:
            if pred_directions[i] == 1:  # UP
                positions[i] = 1
            elif pred_directions[i] == 0:  # DOWN
                positions[i] = -1
        
        position_change = abs(positions[i] - positions[i-1]) if i > 0 else abs(positions[i])
        fees = position_change * trade_fee_pct
        strategy_returns[i] = positions[i] * y_true_returns[i] - fees
    
    buy_hold_returns = y_true_returns
    
    cumulative_strategy = np.cumprod(1 + strategy_returns) - 1
    cumulative_buy_hold = np.cumprod(1 + buy_hold_returns) - 1
    
    total_return = cumulative_strategy[-1] if len(cumulative_strategy) > 0 else 0
    buy_hold_return = cumulative_buy_hold[-1] if len(cumulative_buy_hold) > 0 else 0
    
    bars_per_year = 365 * 24 * 12
    strategy_std = np.std(strategy_returns)
    sharpe = np.sqrt(bars_per_year) * np.mean(strategy_returns) / strategy_std if strategy_std > 0 else 0
    
    winning_trades = (strategy_returns > 0).sum()
    total_trades = (positions != 0).sum()
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    cumulative = np.cumprod(1 + strategy_returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (running_max - cumulative) / running_max
    max_drawdown = np.max(drawdown)
    
    return {
        'total_return': float(total_return),
        'buy_hold_return': float(buy_hold_return),
        'excess_return': float(total_return - buy_hold_return),
        'sharpe_ratio': float(sharpe),
        'win_rate': float(win_rate),
        'max_drawdown': float(max_drawdown),
        'total_trades': int(total_trades),
    }


def evaluate_predictions(predictor: LightGBMPredictor,
                         test_data: PredictionDataset,
                         run_backtest: bool = True) -> Dict[str, Any]:
    """Full evaluation of the prediction model."""
    print("\n" + "=" * 60)
    print("📋 Model Evaluation")
    print("=" * 60 + "\n")
    
    print("Making predictions...")
    pred_returns = predictor.predict_returns_only(test_data.X)
    pred_directions, pred_probs = predictor.predict_direction_only(test_data.X)
    
    print(f"Test samples: {len(pred_returns):,}")
    
    # Regression metrics
    print("\n📈 Regression Metrics:")
    rmse = np.sqrt(mean_squared_error(test_data.y_return, pred_returns))
    mae = mean_absolute_error(test_data.y_return, pred_returns)
    r2 = r2_score(test_data.y_return, pred_returns)
    
    non_zero_mask = test_data.y_return != 0
    mape = np.mean(np.abs((test_data.y_return[non_zero_mask] - pred_returns[non_zero_mask]) 
                          / test_data.y_return[non_zero_mask])) * 100 if non_zero_mask.sum() > 0 else 0
    
    print(f"   RMSE: {rmse:.6f}")
    print(f"   MAE: {mae:.6f}")
    print(f"   MAPE: {mape:.2f}%")
    print(f"   R²: {r2:.4f}")
    
    # Classification metrics
    print("\n📊 Classification Metrics:")
    accuracy = accuracy_score(test_data.y_direction, pred_directions)
    f1 = f1_score(test_data.y_direction, pred_directions, average='macro')
    
    print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   F1 (macro): {f1:.4f}")
    
    # Confusion matrix
    conf_matrix = confusion_matrix(test_data.y_direction, pred_directions)
    print("\n📊 Confusion Matrix:")
    print("          Predicted")
    print("          Down  Flat    Up")
    for i, name in enumerate(['Down', 'Flat', 'Up']):
        print(f"Actual {name:4s}: {conf_matrix[i]}")
    
    # Backtest
    backtest_metrics = {}
    if run_backtest:
        print("\n💰 Backtest Results:")
        backtest_metrics = simple_backtest(
            test_data.y_return,
            pred_directions,
            pred_probs
        )
        print(f"   Total Return: {backtest_metrics['total_return']*100:.2f}%")
        print(f"   Buy & Hold: {backtest_metrics['buy_hold_return']*100:.2f}%")
        print(f"   Excess Return: {backtest_metrics['excess_return']*100:.2f}%")
        print(f"   Sharpe Ratio: {backtest_metrics['sharpe_ratio']:.4f}")
        print(f"   Win Rate: {backtest_metrics['win_rate']:.2%}")
        print(f"   Max Drawdown: {backtest_metrics['max_drawdown']:.2%}")
    
    return {
        'regression': {'rmse': rmse, 'mae': mae, 'mape': mape, 'r2': r2},
        'classification': {'accuracy': accuracy, 'f1_macro': f1},
        'backtest': backtest_metrics,
    }


def main(args):
    """Main evaluation function."""
    print("\n" + "=" * 60)
    print("📊 ServoTrader Prediction Model Evaluation (v2)")
    print("=" * 60)
    
    model_dir = args.model_dir or DEFAULT_MODEL_DIR
    print(f"\nLoading model from: {model_dir}")
    
    predictor = LightGBMPredictor.load(model_dir)
    
    csv_path = args.csv or DEFAULT_CSV_PATH
    print(f"Loading data from: {csv_path}")
    
    pipeline = CryptoPredictionDataPipeline(
        csv_path=csv_path,
        prediction_horizon=predictor.horizon_minutes,
        direction_threshold=predictor.direction_threshold,
    )
    
    pipeline.scaler = predictor.scaler
    pipeline.feature_names = predictor.feature_names
    
    train_data, val_data, test_data = pipeline.prepare_data()
    
    eval_data = test_data if args.split == 'test' else val_data
    
    metrics = evaluate_predictions(predictor, eval_data, not args.no_backtest)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(metrics, f, indent=2, default=float)
        print(f"\n💾 Results saved to: {args.output}")
    
    print("\n✅ Evaluation Complete!")
    
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate LightGBM prediction model')
    parser.add_argument('--model-dir', type=str, default=None)
    parser.add_argument('--csv', type=str, default=None)
    parser.add_argument('--split', type=str, default='test', choices=['test', 'val'])
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--no-backtest', action='store_true')
    
    args = parser.parse_args()
    main(args)