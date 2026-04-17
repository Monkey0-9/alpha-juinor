"""
Predictive Model Wrapper
Handles loading of LightGBM models and generating probabilistic forecasts using SMC features.
Now integrates with advanced intelligence engine for enhanced predictions.
"""

import logging
import os
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd

from mini_quant_fund.config.ml_config import MLConfig
from mini_quant_fund.features.ml_feature_engineer import calculate_smc_features

logger = logging.getLogger(__name__)

# Try to import enhanced model
try:
    from mini_quant_fund.ml_alpha.enhanced_predictive_model import get_enhanced_model

    ENHANCED_MODEL_AVAILABLE = True
except ImportError:
    ENHANCED_MODEL_AVAILABLE = False


class PredictiveModel:
    def __init__(self, model_path=MLConfig.MODEL_PATH, use_advanced: bool = True):
        self.model_path = model_path
        self.model = None
        self.use_advanced = use_advanced and ENHANCED_MODEL_AVAILABLE
        self.enhanced_model = None

        if self.use_advanced:
            try:
                self.enhanced_model = get_enhanced_model()
                logger.info("[ML-ALPHA] Using advanced intelligence engine")
            except Exception as e:
                logger.warning(f"[ML-ALPHA] Failed to initialize advanced model: {e}")
                self.use_advanced = False

        self._load_model()

    def _load_model(self):
        """Attempts to load the trained model artifact."""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, "rb") as f:
                    self.model = pickle.load(f)
                logger.info(
                    f"[ML-ALPHA] Loaded predictive model from {self.model_path}"
                )
            except Exception as e:
                logger.warning(f"[ML-ALPHA] Failed to load model: {e}")
        else:
            # This is expected during initial setup phases
            logger.info(
                f"[ML-ALPHA] Model predictor not trained yet. Using neutral fallback."
            )

    def get_forecast(self, symbol_data: pd.DataFrame) -> dict:
        """
        Generate a probability forecast for the given symbol data.

        Args:
            symbol_data: DataFrame with OHLCV data.

        Returns:
            dict: {'probability': float (0.0-1.0), 'confidence': str, 'source': str, 'regime': str, 'anomaly': bool}
        """
        # Try advanced model first
        if self.use_advanced and self.enhanced_model:
            try:
                enhanced_forecast = self.enhanced_model.get_forecast(symbol_data)
                if enhanced_forecast.get("probability") is not None:
                    return enhanced_forecast
            except Exception as e:
                logger.warning(
                    f"[ML-ALPHA] Advanced model failed: {e}, falling back to standard model"
                )

        # Fallback if no model
        if self.model is None:
            return {
                "probability": 0.5,
                "confidence": "neutral",
                "source": "fallback_no_model",
                "regime": "unknown",
                "anomaly": False,
            }

        try:
            # Generate Features on the fly
            features_df = calculate_smc_features(symbol_data)

            # Predict
            # Expecting a classifier with predict_proba
            if hasattr(self.model, "predict_proba"):
                # Binary classification: [prob_0, prob_1]
                pred_prob = self.model.predict_proba(features_df)[0][1]
            else:
                # Regressor or direct prediction
                pred_prob = float(self.model.predict(features_df)[0])
                # Clip to 0-1 just in case
                pred_prob = max(0.0, min(1.0, pred_prob))

            # Assess Confidence
            conf_label = "neutral"
            if pred_prob > MLConfig.BULLISH_THRESHOLD:
                conf_label = "high_bullish"
            elif pred_prob < MLConfig.BEARISH_THRESHOLD:
                conf_label = "high_bearish"

            return {
                "probability": float(pred_prob),
                "confidence": conf_label,
                "source": "lightgbm_v1",
                "regime": "unknown",
                "anomaly": False,
            }

        except Exception as e:
            logger.error(f"[ML-ALPHA] Prediction Logic Error: {e}")
            return {
                "probability": 0.5,
                "confidence": "error",
                "source": "error",
                "regime": "error",
                "anomaly": False,
            }

    def get_feature_importance(self) -> dict:
        """Get feature importance from the loaded model."""
        # Try advanced model first
        if self.use_advanced and self.enhanced_model:
            try:
                advanced_importance = self.enhanced_model.get_feature_importance()
                if advanced_importance:
                    return advanced_importance
            except Exception as e:
                logger.warning(f"[ML-ALPHA] Advanced feature importance failed: {e}")

        if self.model is None:
            return {}

        try:
            # LightGBM specific
            if hasattr(self.model, "feature_importance"):
                importance = self.model.feature_importance()
                names = self.model.feature_name()
                return dict(zip(names, importance))
            # Sklearn
            elif hasattr(self.model, "feature_importances_"):
                # Use indices if no names
                return {
                    f"feature_{i}": v
                    for i, v in enumerate(self.model.feature_importances_)
                }
            return {}
        except Exception as e:
            logger.error(f"Failed to get feature importance: {e}")
            return {}

    def get_model_status(self) -> dict:
        """Get current model status."""
        if self.use_advanced and self.enhanced_model:
            try:
                return self.enhanced_model.get_model_status()
            except Exception as e:
                logger.warning(f"[ML-ALPHA] Failed to get advanced model status: {e}")

        return {
            "is_trained": self.model is not None,
            "model_type": "lightgbm_v1" if self.model else "untrained",
            "use_advanced": self.use_advanced and self.enhanced_model is not None,
        }
