#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lgb_predictor.py

LightGBM-only predictor for cryptocurrency price prediction.

This simplified version removes the GRU and uses LightGBM directly
on technical indicators for both regression and classification.

Author: Jarred Deluca
Project: ServoTrader - Prediction Subsystem (v2 - LightGBM Only)
"""

import os
import pickle
import numpy as np
import lightgbm as lgb
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime


# =============================================================================
# Prediction Result Data Class
# =============================================================================

@dataclass
class PredictionResult:
    """Container for prediction outputs (binary classification)."""
    predicted_return_pct: float         # Predicted return (e.g., 0.0023 for 0.23%)
    predicted_direction: int            # 0=down, 1=up (binary)
    confidence: float                   # Confidence score [0, 1]
    up_probability: float               # Probability of Up direction
    timestamp: Optional[datetime] = None
    horizon_minutes: int = 60
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'predicted_return_pct': self.predicted_return_pct,
            'predicted_direction': self.predicted_direction,
            'direction_label': ['DOWN', 'UP'][self.predicted_direction],
            'confidence': self.confidence,
            'up_probability': self.up_probability,
            'down_probability': 1.0 - self.up_probability,
            'timestamp': str(self.timestamp) if self.timestamp else None,
            'horizon_minutes': self.horizon_minutes
        }
    
    def __str__(self) -> str:
        direction_str = ['DOWN ↓', 'UP ↑'][self.predicted_direction]
        return (f"Prediction: {direction_str} | "
                f"Return: {self.predicted_return_pct*100:.4f}% | "
                f"Confidence: {self.confidence:.2%}")


# =============================================================================
# LightGBM Predictor Class
# =============================================================================

class LightGBMPredictor:
    """
    LightGBM-only predictor for cryptocurrency prices.
    
    Uses LightGBM for both return regression and direction classification.
    """
    
    def __init__(self,
                 lgb_regressor: lgb.Booster,
                 lgb_classifier: lgb.Booster,
                 scaler: Any,
                 feature_names: List[str],
                 horizon_minutes: int = 60,
                 direction_threshold: float = 0.003):
        """
        Initialize the predictor.
        
        Args:
            lgb_regressor: Trained LightGBM regressor for return prediction
            lgb_classifier: Trained LightGBM classifier for direction prediction
            scaler: Fitted feature scaler
            feature_names: List of feature column names
            horizon_minutes: Prediction horizon in minutes
            direction_threshold: Threshold used for direction classification
        """
        self.lgb_regressor = lgb_regressor
        self.lgb_classifier = lgb_classifier
        self.scaler = scaler
        self.feature_names = feature_names
        self.horizon_minutes = horizon_minutes
        self.direction_threshold = direction_threshold
    
    def predict(self, X: np.ndarray,
                timestamp: Optional[datetime] = None) -> PredictionResult:
        """
        Make a prediction for a single sample.
        
        Args:
            X: Features [1, num_features] (already scaled)
            timestamp: Optional timestamp for the prediction
            
        Returns:
            PredictionResult object
        """
        # Predict return
        predicted_return = self.lgb_regressor.predict(X)[0]
        
        # Predict direction probability (binary: probability of Up)
        up_prob = self.lgb_classifier.predict(X)[0]
        predicted_direction = 1 if up_prob >= 0.5 else 0
        confidence = up_prob if predicted_direction == 1 else (1 - up_prob)
        
        return PredictionResult(
            predicted_return_pct=float(predicted_return),
            predicted_direction=predicted_direction,
            confidence=float(confidence),
            up_probability=float(up_prob),
            timestamp=timestamp,
            horizon_minutes=self.horizon_minutes
        )
    
    def predict_batch(self, X: np.ndarray,
                      timestamps: Optional[np.ndarray] = None
                      ) -> List[PredictionResult]:
        """
        Make predictions for a batch of samples.
        
        Args:
            X: Features [batch, num_features] (already scaled)
            timestamps: Optional timestamps for each prediction
            
        Returns:
            List of PredictionResult objects
        """
        # Batch predictions
        predicted_returns = self.lgb_regressor.predict(X)
        up_probs = self.lgb_classifier.predict(X)
        
        # Create result objects
        results = []
        for i in range(len(X)):
            timestamp = timestamps[i] if timestamps is not None else None
            up_prob = up_probs[i]
            predicted_direction = 1 if up_prob >= 0.5 else 0
            confidence = up_prob if predicted_direction == 1 else (1 - up_prob)
            
            result = PredictionResult(
                predicted_return_pct=float(predicted_returns[i]),
                predicted_direction=predicted_direction,
                confidence=float(confidence),
                up_probability=float(up_prob),
                timestamp=timestamp,
                horizon_minutes=self.horizon_minutes
            )
            results.append(result)
        
        return results
    
    def predict_returns_only(self, X: np.ndarray) -> np.ndarray:
        """
        Predict returns only (for backtesting efficiency).
        
        Args:
            X: Features [batch, num_features]
            
        Returns:
            Predicted returns [batch]
        """
        return self.lgb_regressor.predict(X)
    
    def predict_direction_only(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict direction only (for backtesting efficiency).
        
        Binary classification: 0=Down, 1=Up
        
        Args:
            X: Features [batch, num_features]
            
        Returns:
            Tuple of (predicted_directions, up_probabilities)
        """
        up_probs = self.lgb_classifier.predict(X)
        directions = (up_probs >= 0.5).astype(np.int64)
        return directions, up_probs
    
    def get_feature_importance(self, importance_type: str = 'gain') -> Dict[str, float]:
        """
        Get feature importance from the classifier.
        
        Args:
            importance_type: 'gain' or 'split'
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        importance = self.lgb_classifier.feature_importance(importance_type=importance_type)
        return dict(zip(self.feature_names, importance))
    
    def save(self, save_dir: str) -> None:
        """
        Save all model components to a directory.
        
        Args:
            save_dir: Directory to save models
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save LightGBM regressor
        regressor_path = os.path.join(save_dir, 'lgb_regressor.txt')
        self.lgb_regressor.save_model(regressor_path)
        print(f"💾 LightGBM regressor saved to: {regressor_path}")
        
        # Save LightGBM classifier
        classifier_path = os.path.join(save_dir, 'lgb_classifier.txt')
        self.lgb_classifier.save_model(classifier_path)
        print(f"💾 LightGBM classifier saved to: {classifier_path}")
        
        # Save metadata
        metadata_path = os.path.join(save_dir, 'model_metadata.pkl')
        metadata = {
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'horizon_minutes': self.horizon_minutes,
            'direction_threshold': self.direction_threshold,
            'model_type': 'lightgbm_only_v2'
        }
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        print(f"💾 Metadata saved to: {metadata_path}")
        
        print(f"\n✅ All models saved to: {save_dir}")
    
    @classmethod
    def load(cls, load_dir: str) -> 'LightGBMPredictor':
        """
        Load a saved predictor from a directory.
        
        Args:
            load_dir: Directory containing saved models
            
        Returns:
            Loaded LightGBMPredictor instance
        """
        print(f"📂 Loading models from: {load_dir}")
        
        # Load LightGBM regressor
        regressor_path = os.path.join(load_dir, 'lgb_regressor.txt')
        lgb_regressor = lgb.Booster(model_file=regressor_path)
        print(f"📂 LightGBM regressor loaded")
        
        # Load LightGBM classifier
        classifier_path = os.path.join(load_dir, 'lgb_classifier.txt')
        lgb_classifier = lgb.Booster(model_file=classifier_path)
        print(f"📂 LightGBM classifier loaded")
        
        # Load metadata
        metadata_path = os.path.join(load_dir, 'model_metadata.pkl')
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        print(f"📂 Metadata loaded")
        
        print(f"\n✅ All models loaded successfully")
        
        return cls(
            lgb_regressor=lgb_regressor,
            lgb_classifier=lgb_classifier,
            scaler=metadata['scaler'],
            feature_names=metadata['feature_names'],
            horizon_minutes=metadata.get('horizon_minutes', 60),
            direction_threshold=metadata.get('direction_threshold', 0.003)
        )


if __name__ == "__main__":
    # Test the predictor
    print("Testing LightGBM predictor module...")
    
    import tempfile
    
    # Create dummy data
    np.random.seed(42)
    n_samples = 1000
    n_features = 21
    
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y_return = np.random.randn(n_samples).astype(np.float32) * 0.01
    y_direction = np.random.randint(0, 3, n_samples)
    
    feature_names = [f'feature_{i}' for i in range(n_features)]
    
    # Train dummy LightGBM models
    print("\nTraining dummy LightGBM models...")
    
    train_data_reg = lgb.Dataset(X[:800], label=y_return[:800])
    val_data_reg = lgb.Dataset(X[800:], label=y_return[800:], reference=train_data_reg)
    
    lgb_regressor = lgb.train(
        {'objective': 'regression', 'metric': 'mse', 'verbose': -1},
        train_data_reg,
        num_boost_round=10,
        valid_sets=[val_data_reg]
    )
    
    train_data_cls = lgb.Dataset(X[:800], label=y_direction[:800])
    val_data_cls = lgb.Dataset(X[800:], label=y_direction[800:], reference=train_data_cls)
    
    lgb_classifier = lgb.train(
        {'objective': 'multiclass', 'num_class': 3, 'metric': 'multi_logloss', 'verbose': -1},
        train_data_cls,
        num_boost_round=10,
        valid_sets=[val_data_cls]
    )
    
    # Create predictor
    predictor = LightGBMPredictor(
        lgb_regressor=lgb_regressor,
        lgb_classifier=lgb_classifier,
        scaler=None,
        feature_names=feature_names,
        horizon_minutes=60,
        direction_threshold=0.003
    )
    
    # Test single prediction
    print("\nTesting single prediction...")
    X_test = np.random.randn(1, n_features).astype(np.float32)
    result = predictor.predict(X_test)
    print(f"Result: {result}")
    
    # Test batch prediction
    print("\nTesting batch prediction...")
    X_batch = np.random.randn(32, n_features).astype(np.float32)
    results = predictor.predict_batch(X_batch)
    print(f"Batch size: {len(results)}")
    
    # Test save/load
    print("\nTesting save/load...")
    with tempfile.TemporaryDirectory() as tmpdir:
        predictor.save(tmpdir)
        loaded_predictor = LightGBMPredictor.load(tmpdir)
        
        # Verify predictions match
        result_original = predictor.predict(X_test)
        result_loaded = loaded_predictor.predict(X_test)
        
        assert abs(result_original.predicted_return_pct - result_loaded.predicted_return_pct) < 1e-5
        assert result_original.predicted_direction == result_loaded.predicted_direction
    
    print("\n✅ LightGBM predictor module test passed!")