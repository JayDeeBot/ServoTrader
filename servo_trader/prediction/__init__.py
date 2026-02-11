#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ServoTrader Prediction Subsystem (v2.5 - With On-Chain Data)

This package uses LightGBM on technical indicators, sentiment data,
and on-chain blockchain metrics for cryptocurrency price prediction.

Changes from v2.4:
- Added on-chain blockchain metrics from Blockchain.com API (FREE)
- New metrics: hash rate, active addresses, tx volume, difficulty, fees, miner revenue
- ~50 derived on-chain features (momentum, MA ratios, volatility, cross-metric ratios)
- Total features: 90+ (technical + sentiment + on-chain)

Changes from v2.3:
- Added Optuna hyperparameter optimization
- Improved model training with optimized parameters

Changes from v2.2:
- Added Boruta feature selection
- Automatic feature importance ranking

Changes from v2.1:
- Added Fear & Greed Index sentiment features
- 6 new features from free Alternative.me API

Modules:
- feature_engineering: Technical indicator calculations
- sentiment_data: Fear & Greed Index retrieval and processing
- onchain_data: On-chain blockchain metrics (NEW in v2.5)
- data_pipeline: Data loading, preprocessing, splitting
- lgb_predictor: LightGBM predictor class

Author: Jarred Deluca
Project: ServoTrader
"""

# Existing imports - Feature Engineering
from .feature_engineering import (
    compute_all_features,
    get_static_feature_names,
    compute_rsi,
    compute_macd,
    compute_bollinger_bands,
    compute_atr,
    compute_ema,
    compute_volatility,
    compute_trend_slope,
    compute_price_position,
    compute_volume_sma_ratio,
    compute_time_features
)

# Existing imports - Data Pipeline
from .data_pipeline import (
    CryptoPredictionDataPipeline,
    PredictionDataset,
    prepare_single_sample
)

# Existing imports - Predictor
from .lgb_predictor import (
    LightGBMPredictor,
    PredictionResult
)

# New imports - On-Chain Data (v2.5)
from .onchain_data import (
    OnChainDataFetcher,
    load_onchain_data,
    get_onchain_feature_names
)

__version__ = '2.5.0'
__author__ = 'Jarred Deluca'

__all__ = [
    # Feature Engineering
    'compute_all_features',
    'get_static_feature_names',
    'compute_rsi',
    'compute_macd',
    'compute_bollinger_bands',
    'compute_atr',
    'compute_ema',
    'compute_volatility',
    'compute_trend_slope',
    'compute_price_position',
    'compute_volume_sma_ratio',
    'compute_time_features',
    
    # Data Pipeline
    'CryptoPredictionDataPipeline',
    'PredictionDataset',
    'prepare_single_sample',
    
    # Predictor
    'LightGBMPredictor',
    'PredictionResult',
    
    # On-Chain Data (NEW in v2.5)
    'OnChainDataFetcher',
    'load_onchain_data',
    'get_onchain_feature_names',
]