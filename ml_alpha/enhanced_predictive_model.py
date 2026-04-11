"""
Enhanced Predictive Model with Advanced Intelligence Integration
Uses advanced ensemble, feature selection, and adaptive learning
NOW WITH INTELLIGENCE CORE - Learns from every prediction
NOW WITH SMART BRAIN ENGINE - REAL REASONING AND THINKING
"""

import logging
import os
import pickle
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from config.ml_config import MLConfig
from ml_alpha.advanced_features import AdvancedFeatureEngineer
from ml_alpha.advanced_intelligence_engine import (
    AdaptiveLearningEngine,
    AnomalyDetector,
    FeatureSelector,
    IntelligenceSystem,
    get_intelligence_system,
)
from ml_alpha.intelligence_core import get_intelligence_core
from ml_alpha.smart_brain_engine import MarketState, get_smart_brain

logger = logging.getLogger(__name__)


class EnhancedPredictiveModel:
    """
    Enhanced predictive model with REAL INTELLIGENCE:
    - Multi-model ensemble (LightGBM, XGBoost, CatBoost, etc.)
    - Intelligent feature selection
    - Anomaly detection and regime detection
    - Adaptive learning and retraining
    - Cross-validation and performance tracking
    - INTELLIGENCE CORE: Meta-learning, pattern recognition, risk adaptation
    - SMART BRAIN ENGINE: Bayesian reasoning, causal inference, strategy learning
    """

    def __init__(self, model_path: str = MLConfig.MODEL_PATH):
        self.model_path = model_path
        self.intelligence_system = get_intelligence_system()
        self.feature_engineer = AdvancedFeatureEngineer()
        self.adaptive_learner = AdaptiveLearningEngine()
        self.anomaly_detector = AnomalyDetector()

        # INTELLIGENCE CORE - The Learning System
        self.intelligence_core = get_intelligence_core()

        # SMART BRAIN ENGINE - The Decision Maker
        self.smart_brain = get_smart_brain()

        self.is_trained = False
        self.selected_features: Optional[list] = None
        self.model_performance: Dict[str, Any] = {}
        self.training_history: list = []

        self._load_model()

    def _load_model(self):
        """Load trained ensemble from disk."""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, "rb") as f:
                    state = pickle.load(f)
                    self.intelligence_system = state.get(
                        "intelligence_system", self.intelligence_system
                    )
                    self.selected_features = state.get("selected_features")
                    self.model_performance = state.get("model_performance", {})
                    self.is_trained = state.get("is_trained", False)

                logger.info(f"[ENHANCED-MODEL] Loaded ensemble from {self.model_path}")
            except Exception as e:
                logger.warning(f"[ENHANCED-MODEL] Failed to load ensemble: {e}")
                self.is_trained = False

    def save_model(self):
        """Save trained ensemble to disk."""
        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            state = {
                "intelligence_system": self.intelligence_system,
                "selected_features": self.selected_features,
                "model_performance": self.model_performance,
                "is_trained": self.is_trained,
                "timestamp": datetime.now(),
            }
            with open(self.model_path, "wb") as f:
                pickle.dump(state, f)
            logger.info(f"[ENHANCED-MODEL] Saved ensemble to {self.model_path}")
        except Exception as e:
            logger.error(f"[ENHANCED-MODEL] Failed to save ensemble: {e}")

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_selection_method: str = "hybrid",
        validate: bool = True,
    ) -> Dict[str, Any]:
        """
        Train the enhanced ensemble model.

        Args:
            X: Feature matrix
            y: Target variable
            feature_selection_method: 'correlation', 'importance', 'pca', or 'hybrid'
            validate: Whether to perform cross-validation

        Returns:
            Training results including performance metrics
        """
        logger.info("[ENHANCED-MODEL] Starting ensemble training...")

        # Align X and y by index (in case they have different lengths)
        # Keep only indices that exist in both
        common_index = X.index.intersection(y.index)
        X = X.loc[common_index].copy()
        y = y.loc[common_index].copy()

        # Detect anomalies
        anomalies = self.anomaly_detector.detect_anomalies(X)
        X_clean = X[anomalies == 1].copy()
        y_clean = y[anomalies == 1].copy()

        if len(X_clean) < len(X) * 0.5:
            logger.warning(
                f"[ENHANCED-MODEL] {100*(1-len(X_clean)/len(X)):.1f}% data flagged as anomalous"
            )

        # Initialize intelligence system
        logger.info(
            f"[ENHANCED-MODEL] Initializing intelligence system with {len(X_clean)} samples..."
        )

        # Validate we have enough data
        if len(X_clean) < 10:
            logger.error(f"[ENHANCED-MODEL] Insufficient samples: {len(X_clean)} (minimum 10 required)")
            return {
                "success": False,
                "error": f"Insufficient samples: {len(X_clean)}",
                "selected_features": [],
                "model_count": 0,
                "model_weights": {},
                "performance": {},
            }

        init_results = self.intelligence_system.initialize(X_clean, y_clean)

        # Validate init_results
        if init_results is None or not isinstance(init_results, dict):
            logger.error("[ENHANCED-MODEL] Intelligence system initialization failed")
            return {
                "success": False,
                "error": "Initialization failed",
                "selected_features": [],
                "model_count": 0,
                "model_weights": {},
                "performance": {},
            }

        self.selected_features = init_results.get("selected_features", [])
        self.model_performance = init_results.get("performance", {})

        self.is_trained = True
        self.adaptive_learner.last_training_date = datetime.now()

        # Save model
        self.save_model()

        logger.info(
            f"[ENHANCED-MODEL] Training complete. Models: {list(self.model_performance.keys())}"
        )

        return {
            "success": True,
            "selected_features": self.selected_features,
            "model_count": len(self.model_performance),
            "model_weights": self.intelligence_system.ensemble.model_weights,
            "performance": {
                name: {
                    "r2": perf.r2_score,
                    "rmse": perf.rmse,
                    "mae": perf.mae,
                    "sharpe": perf.sharpe_ratio,
                }
                for name, perf in self.model_performance.items()
            },
        }

    def get_forecast(self, symbol_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate probability forecast with REAL REASONING using Smart Brain Engine.
        This integrates statistical models + intelligence core + smart brain reasoning.

        Args:
            symbol_data: DataFrame with OHLCV data

        Returns:
            dict with probability, confidence, reasoning chain, and smart decisions
        """
        if not self.is_trained:
            logger.warning(
                "[ENHANCED-MODEL] Model not trained, returning neutral forecast"
            )
            return {
                "probability": 0.5,
                "confidence": 0.0,
                "source": "untrained_model",
                "regime": "unknown",
                "anomaly": False,
                "reasoning": ["Model not trained yet"],
                "brain_action": "neutral",
            }

        # Initialize market_features for use across the method
        market_features = {}

        try:
            # Generate advanced features
            features_df = self.feature_engineer.compute_all_features(symbol_data)

            if features_df.empty:
                return {
                    "probability": 0.5,
                    "confidence": 0.0,
                    "source": "insufficient_data",
                    "regime": "unknown",
                    "anomaly": False,
                    "reasoning": ["Insufficient data"],
                    "brain_action": "neutral",
                }

            # Use only selected features
            if self.selected_features:
                available_features = [
                    f for f in self.selected_features if f in features_df.columns
                ]
                if not available_features:
                    available_features = self.selected_features[
                        : min(10, len(self.selected_features))
                    ]
                features_df = features_df[available_features].tail(1)
            else:
                features_df = features_df.tail(1)

            # Detect anomalies
            anomaly_score = self.anomaly_detector.detect_anomalies(features_df)[0]
            is_anomalous = anomaly_score == -1

            # Detect regime
            returns = symbol_data["close"].pct_change()
            regime = self.anomaly_detector.detect_regime(returns)

            # Get ensemble prediction
            pred, confidence, sys_regime = self.intelligence_system.predict(features_df)

            # Update adaptive learner
            self.adaptive_learner.update_performance(confidence)

            # Check if retraining needed
            if self.adaptive_learner.should_retrain():
                logger.info("[ENHANCED-MODEL] Adaptive learning triggered retraining")

            # INTELLIGENCE CORE ENHANCEMENT
            # Convert features to dict for core processing
            market_features = {
                "volatility": (
                    features_df.get("volatility_20", [0.01]).iloc[0]
                    if "volatility_20" in features_df
                    else 0.01
                ),
                "momentum": (
                    features_df.get("momentum_10", [0.0]).iloc[0]
                    if "momentum_10" in features_df
                    else 0.0
                ),
                "trend": (
                    features_df.get("price_trend_20", [0.5]).iloc[0]
                    if "price_trend_20" in features_df
                    else 0.5
                ),
                "sharpe_ratio": self.adaptive_learner.get_performance_trend()
                == "improving"
                and 0.5
                or -0.5,
                "drawdown": 0.0,  # Would be updated from portfolio metrics
            }

            # Process through Intelligence Core for enhancement
            core_enhancement = self.intelligence_core.process_prediction(
                prediction=pred,
                model_weights=self.intelligence_system.ensemble.model_weights,
                market_features=market_features,
                confidence=confidence,
            )

            # ==============================================
            # SMART BRAIN ENGINE - REAL REASONING
            # ==============================================
            # Create market state for brain reasoning
            market_state = MarketState(
                volatility=market_features.get("volatility", 0.01),
                momentum=market_features.get("momentum", 0.0),
                trend_strength=abs(market_features.get("trend", 0.5) - 0.5) * 2,
                regime=sys_regime,
                lstm_signal=float(pred) * 2 - 1.0,  # Convert 0-1 to -1 to 1
                uncertainty=1.0
                - float(confidence),  # Higher confidence = lower uncertainty
            )

            # Get brain reasoning
            brain_decision = self.smart_brain.think(market_state=market_state)

            # Blend statistical forecast with brain reasoning
            # Brain action: 'bullish' -> 1.0, 'bearish' -> 0.0, 'neutral' -> 0.5
            brain_action_value = {
                "bullish": 1.0,
                "neutral": 0.5,
                "bearish": 0.0,
            }.get(brain_decision.get("action", "neutral"), 0.5)

            # Weighted blend: 60% statistical ensemble + 40% brain reasoning
            blend_probability = (
                0.6 * float(core_enhancement["enhanced_prediction"])
                + 0.4 * brain_action_value
            )
            blend_confidence = 0.6 * float(
                core_enhancement["adjusted_confidence"]
            ) + 0.4 * brain_decision.get("confidence", 0.5)

            logger.info(
                f"[SMART-BRAIN] Decision: {brain_decision.get('action')} "
                f"(confidence: {brain_decision.get('confidence'):.2f})"
            )

            return {
                "probability": float(blend_probability),
                "confidence": float(blend_confidence),
                "source": "smart_brain_enhanced",
                "regime": core_enhancement["risk_regime"],
                "anomaly": bool(is_anomalous),
                "model_weights": self.intelligence_system.ensemble.model_weights,
                "performance_trend": self.adaptive_learner.get_performance_trend(),
                "pattern_detected": core_enhancement["pattern_detected"],
                "pattern_confidence": core_enhancement["pattern_confidence"],
                "core_accuracy": core_enhancement["core_accuracy"],
                "learning_progress": core_enhancement["learning_progress"],
                # SMART BRAIN ADDITIONS
                "brain_action": brain_decision.get("action", "neutral"),
                "brain_confidence": float(brain_decision.get("confidence", 0.5)),
                "brain_uncertainty": float(brain_decision.get("uncertainty", 0.2)),
                "reasoning_chain": brain_decision.get("reasoning", []),
                "alternate_views": brain_decision.get("alternate_views", []),
            }

        except Exception as e:
            logger.error(f"[ENHANCED-MODEL] Forecast error: {e}")
            return {
                "probability": 0.5,
                "confidence": 0.0,
                "source": "forecast_error",
                "regime": "error",
                "anomaly": False,
                "error": str(e),
                "reasoning": [f"Error: {str(e)}"],
                "brain_action": "neutral",
            }

    def learn_from_outcome(
        self,
        prediction: float,
        confidence: float,
        actual_outcome: float,
        profit_loss: float = 0.0,
        symbol_data: Optional[pd.DataFrame] = None,
        brain_decision_action: str = "neutral",
    ):
        """
        Feed prediction outcome back to system so it learns.
        This is what makes the system actually intelligent!
        Now also feeds back to Smart Brain for continuous reasoning improvement.
        """
        try:
            # Extract market features if available
            market_features = {}
            if symbol_data is not None:
                try:
                    features_df = self.feature_engineer.compute_all_features(
                        symbol_data
                    )
                    if not features_df.empty:
                        features_df = features_df.tail(1)
                        market_features = {
                            "volatility": (
                                features_df.get("volatility_20", [0.01]).iloc[0]
                                if "volatility_20" in features_df
                                else 0.01
                            ),
                            "momentum": (
                                features_df.get("momentum_10", [0.0]).iloc[0]
                                if "momentum_10" in features_df
                                else 0.0
                            ),
                            "trend": (
                                features_df.get("price_trend_20", [0.5]).iloc[0]
                                if "price_trend_20" in features_df
                                else 0.5
                            ),
                        }
                except:
                    pass

            # Feed back to Intelligence Core for learning
            self.intelligence_core.learn_from_outcome(
                prediction=prediction,
                confidence=confidence,
                actual_outcome=actual_outcome,
                profit_loss=profit_loss,
                market_features=market_features,
            )

            # Feed back to Smart Brain for continuous improvement
            actual_action = (
                "bullish"
                if actual_outcome > 0.5
                else "bearish" if actual_outcome < 0.5 else "neutral"
            )
            self.smart_brain.learn_from_decision_outcome(
                decision_action=brain_decision_action,
                actual_outcome=actual_action,
                profit_loss=profit_loss,
                market_features=market_features,
            )

            logger.info(
                f"[ENHANCED-MODEL] Learned from outcome: "
                f"pred={prediction:.2f}, actual={actual_outcome:.2f}, "
                f"brain_action={brain_decision_action}, actual_action={actual_action}, "
                f"correct={prediction > 0.5 == actual_outcome > 0.5}"
            )

        except Exception as e:
            logger.warning(f"[ENHANCED-MODEL] Learning failed: {e}")

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from ensemble."""
        try:
            if not self.is_trained:
                return {}

            # Aggregate importance across models
            importance_scores = {}
            total_weight = 0

            for model_name, perf in self.model_performance.items():
                weight = self.intelligence_system.ensemble.model_weights.get(
                    model_name, 0
                )
                total_weight += weight

                # Try to get feature importance from model
                if (
                    hasattr(perf, "model")
                    and perf.model
                    and hasattr(perf.model, "feature_importances_")
                ):
                    importances = perf.model.feature_importances_
                    for i, imp in enumerate(importances):
                        feat_name = f"feature_{i}"
                        importance_scores[feat_name] = importance_scores.get(
                            feat_name, 0
                        ) + (imp * weight)

            if total_weight > 0:
                importance_scores = {
                    k: v / total_weight for k, v in importance_scores.items()
                }

            return importance_scores
        except Exception as e:
            logger.error(f"[ENHANCED-MODEL] Feature importance error: {e}")
            return {}

    def get_model_status(self) -> Dict[str, Any]:
        """Get current model status and health WITH INTELLIGENCE CORE STATUS."""
        core_status = self.intelligence_core.get_system_intelligence()

        return {
            "is_trained": self.is_trained,
            "model_count": len(self.model_performance),
            "selected_features_count": (
                len(self.selected_features) if self.selected_features else 0
            ),
            "models": list(self.model_performance.keys()),
            "model_weights": self.intelligence_system.ensemble.model_weights,
            "last_training": self.adaptive_learner.last_training_date,
            "performance_trend": self.adaptive_learner.get_performance_trend(),
            "feature_importance": self.get_feature_importance(),
            # INTELLIGENCE CORE STATUS
            "intelligence_core": {
                "overall_accuracy": core_status["overall_accuracy"],
                "total_predictions_learned": core_status["total_predictions"],
                "patterns_discovered": core_status["patterns_learned"],
                "top_patterns": core_status["top_patterns"],
                "model_rankings": core_status["model_rankings"],
                "current_risk_regime": core_status["current_risk_regime"],
                "learning_duration_hours": core_status["learning_duration_hours"],
            },
        }


# Global instance
_enhanced_model: Optional[EnhancedPredictiveModel] = None


def get_enhanced_model() -> EnhancedPredictiveModel:
    """Get or create enhanced model instance."""
    global _enhanced_model
    if _enhanced_model is None:
        _enhanced_model = EnhancedPredictiveModel()
    return _enhanced_model
