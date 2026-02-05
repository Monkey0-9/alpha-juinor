"""
Predictive Model Wrapper
Handles loading of LightGBM models and generating probabilistic forecasts using SMC features.
"""
import logging
import os
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd

from features.ml_feature_engineer import calculate_smc_features

logger = logging.getLogger(__name__)

class PredictiveModel:
    def __init__(self, model_path="models/lgbm_predictor_v1.pkl"):
        self.model_path = model_path
        self.model = None
        self._load_model()

    def _load_model(self):
        """Attempts to load the trained model artifact."""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                logger.info(f"[ML-ALPHA] Loaded predictive model from {self.model_path}")
            except Exception as e:
                logger.warning(f"[ML-ALPHA] Failed to load model: {e}")
        else:
            # This is expected during initial setup phases
            logger.info(f"[ML-ALPHA] Model predictor not trained yet. Using neutral fallback.")

    def get_forecast(self, symbol_data: pd.DataFrame) -> dict:
        """
        Generate a probability forecast for the given symbol data.

        Args:
            symbol_data: DataFrame with OHLCV data.

        Returns:
            dict: {'probability': float (0.0-1.0), 'confidence': str, 'source': str}
        """
        # 1. Fallback if no model
        if self.model is None:
            return {
                'probability': 0.5,
                'confidence': 'neutral',
                'source': 'fallback_no_model'
            }

        try:
            # 2. Generate Features on the fly
            features_df = calculate_smc_features(symbol_data)

            # 3. Predict
            # Expecting a classifier with predict_proba
            if hasattr(self.model, 'predict_proba'):
                # Binary classification: [prob_0, prob_1]
                pred_prob = self.model.predict_proba(features_df)[0][1]
            else:
                # Regressor or direct prediction
                pred_prob = float(self.model.predict(features_df)[0])
                # Clip to 0-1 just in case
                pred_prob = max(0.0, min(1.0, pred_prob))

            # 4. Assess Confidence
            # High confidence if probability is far from 0.5
            conf_label = 'neutral'
            if pred_prob > 0.65:
                conf_label = 'high_bullish'
            elif pred_prob < 0.35:
                conf_label = 'high_bearish'

            return {
                'probability': float(pred_prob),
                'confidence': conf_label,
                'source': 'lightgbm_v1'
            }

        except Exception as e:
            logger.error(f"[ML-ALPHA] Prediction Logic Error: {e}")
            return {
                'probability': 0.5,
                'confidence': 'error',
                'source': 'error'
            }
