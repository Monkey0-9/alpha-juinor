"""
Advanced Machine Learning Intelligence Engine
Provides multi-model ensemble, feature selection, anomaly detection, and adaptive learning
"""

import logging
import os
import pickle
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import catboost as cb

    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import RobustScaler, StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class ModelPerformance:
    """Track model performance metrics."""

    model_name: str
    r2_score: float
    rmse: float
    mae: float
    sharpe_ratio: float
    max_drawdown: float
    accuracy: float
    training_time: float
    last_updated: datetime


class FeatureSelector:
    """Intelligent feature selection system."""

    def __init__(self, max_features: int = 50, min_correlation: float = 0.05):
        self.max_features = max_features
        self.min_correlation = min_correlation
        self.selected_features: List[str] = []
        self.feature_importance: Dict[str, float] = {}
        self.scaler = RobustScaler()

    def select_features(
        self, X: pd.DataFrame, y: pd.Series, method: str = "hybrid"
    ) -> List[str]:
        """
        Select features using multiple methods:
        - correlation: Remove highly correlated features
        - importance: Use feature importance from tree models
        - pca: Use PCA for dimensionality reduction
        - hybrid: Combine multiple methods
        """
        if method == "correlation":
            return self._correlation_based_selection(X, y)
        elif method == "importance":
            return self._importance_based_selection(X, y)
        elif method == "pca":
            return self._pca_based_selection(X)
        elif method == "hybrid":
            return self._hybrid_selection(X, y)
        else:
            return list(X.columns)

    def _correlation_based_selection(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Select features with correlation > threshold to target."""
        correlations = X.corrwith(y).abs().sort_values(ascending=False)
        selected = correlations[correlations > self.min_correlation].index.tolist()
        return selected[: self.max_features]

    def _importance_based_selection(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Select features based on tree model importance."""
        try:
            model = (
                lgb.LGBMRegressor(n_estimators=100, verbosity=-1)
                if LIGHTGBM_AVAILABLE
                else GradientBoostingRegressor(n_estimators=100)
            )
            model.fit(X, y)

            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1][: self.max_features]
                selected = [X.columns[i] for i in indices]
                self.feature_importance = dict(zip(selected, importances[indices]))
                return selected
        except Exception as e:
            logger.error(f"Feature importance selection failed: {e}")

        return list(X.columns[: self.max_features])

    def _pca_based_selection(self, X: pd.DataFrame) -> List[str]:
        """Select features using PCA."""
        try:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            pca = PCA(n_components=min(self.max_features, X.shape[1]))
            pca.fit(X_scaled)

            # Return top contributing features
            loadings = np.abs(pca.components_).sum(axis=0)
            indices = np.argsort(loadings)[::-1][: self.max_features]
            return [X.columns[i] for i in indices]
        except Exception as e:
            logger.error(f"PCA-based selection failed: {e}")
            return list(X.columns[: self.max_features])

    def _hybrid_selection(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Combine multiple selection methods."""
        corr_features = set(self._correlation_based_selection(X, y))
        imp_features = set(self._importance_based_selection(X, y))
        pca_features = set(self._pca_based_selection(X))

        # Union and take top features
        combined = list(corr_features | imp_features | pca_features)
        return combined[: self.max_features]


class AnomalyDetector:
    """Detect market anomalies and regime changes."""

    def __init__(self, lookback_window: int = 100):
        self.lookback_window = lookback_window
        self.mean_return: Optional[float] = None
        self.std_return: Optional[float] = None
        self.regime: str = "normal"

    def detect_regime(self, returns: pd.Series) -> str:
        """Detect current market regime."""
        if len(returns) < 20:
            return "insufficient_data"

        recent_returns = returns.tail(self.lookback_window)
        volatility = recent_returns.std()
        skewness = recent_returns.skew()

        if volatility > np.percentile(returns.std(), 75):
            self.regime = "high_volatility"
        elif volatility < np.percentile(returns.std(), 25):
            self.regime = "low_volatility"
        elif skewness < -0.5:
            self.regime = "negative_skew"
        elif skewness > 0.5:
            self.regime = "positive_skew"
        else:
            self.regime = "normal"

        return self.regime

    def detect_anomalies(self, X: pd.DataFrame) -> np.ndarray:
        """Detect anomalous data points using Isolation Forest."""
        try:
            # Handle empty DataFrames
            if X.empty or len(X) == 0:
                return np.ones(len(X), dtype=int)

            # If too small, consider all normal
            if len(X) < 10:
                return np.ones(len(X), dtype=int)

            from sklearn.ensemble import IsolationForest

            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X)

            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            anomalies = iso_forest.fit_predict(X_scaled)

            return anomalies  # -1 for anomalies, 1 for normal
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return np.ones(len(X), dtype=int)


class MultiModelEnsemble:
    """Advanced ensemble combining multiple ML models."""

    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.model_weights: Dict[str, float] = {}
        self.performance_history: Dict[str, List[ModelPerformance]] = {}
        self.scaler = RobustScaler()

    def build_ensemble(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Dict[str, ModelPerformance]:
        """Build multiple models with cross-validation."""
        # Validate data before processing
        if X.empty or y.empty:
            logger.error("[ENSEMBLE] Cannot build ensemble with empty data")
            return {}

        if len(X) < 10:
            logger.error(f"[ENSEMBLE] Insufficient samples: {len(X)} (minimum 10 required)")
            return {}

        # Remove NaN values
        valid_idx = X.notna().all(axis=1) & y.notna()
        X = X[valid_idx].copy()
        y = y[valid_idx].copy()

        if len(X) < 10:
            logger.error(f"[ENSEMBLE] Insufficient valid samples after cleaning: {len(X)}")
            return {}

        X_scaled = self.scaler.fit_transform(X)

        tscv = TimeSeriesSplit(n_splits=5)
        performance_results = {}

        # LightGBM
        if LIGHTGBM_AVAILABLE:
            perf = self._train_lightgbm(X_scaled, y, tscv)
            if perf:
                performance_results["lightgbm"] = perf
                self.models["lightgbm"] = perf.model if hasattr(perf, "model") else None

        # XGBoost
        if XGBOOST_AVAILABLE:
            perf = self._train_xgboost(X_scaled, y, tscv)
            if perf:
                performance_results["xgboost"] = perf
                self.models["xgboost"] = perf.model if hasattr(perf, "model") else None

        # CatBoost
        if CATBOOST_AVAILABLE:
            perf = self._train_catboost(X_scaled, y, tscv)
            if perf:
                performance_results["catboost"] = perf
                self.models["catboost"] = perf.model if hasattr(perf, "model") else None

        # Gradient Boosting
        perf = self._train_gradient_boosting(X_scaled, y, tscv)
        if perf:
            performance_results["gradient_boosting"] = perf

        # Random Forest
        perf = self._train_random_forest(X_scaled, y, tscv)
        if perf:
            performance_results["random_forest"] = perf

        # Calculate weights based on performance
        self._calculate_weights(performance_results)

        return performance_results

    def _train_lightgbm(
        self, X: np.ndarray, y: pd.Series, tscv
    ) -> Optional[ModelPerformance]:
        """Train LightGBM with cross-validation."""
        try:
            model = lgb.LGBMRegressor(
                n_estimators=200,
                max_depth=7,
                learning_rate=0.05,
                num_leaves=50,
                min_child_samples=10,
                subsample=0.8,
                colsample_bytree=0.8,
                verbosity=-1,
            )

            cv_scores = cross_val_score(model, X, y, cv=tscv, scoring="r2")
            model.fit(X, y)

            return ModelPerformance(
                model_name="lightgbm",
                r2_score=np.mean(cv_scores),
                rmse=np.sqrt(mean_squared_error(y, model.predict(X))),
                mae=mean_absolute_error(y, model.predict(X)),
                sharpe_ratio=self._calculate_sharpe(y, model.predict(X)),
                max_drawdown=self._calculate_max_drawdown(y),
                accuracy=np.mean(cv_scores),
                training_time=0.0,
                last_updated=datetime.now(),
            )
        except Exception as e:
            logger.error(f"LightGBM training failed: {e}")
            return None

    def _train_xgboost(
        self, X: np.ndarray, y: pd.Series, tscv
    ) -> Optional[ModelPerformance]:
        """Train XGBoost with cross-validation."""
        try:
            model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=7,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                verbosity=0,
            )

            cv_scores = cross_val_score(model, X, y, cv=tscv, scoring="r2")
            model.fit(X, y)

            return ModelPerformance(
                model_name="xgboost",
                r2_score=np.mean(cv_scores),
                rmse=np.sqrt(mean_squared_error(y, model.predict(X))),
                mae=mean_absolute_error(y, model.predict(X)),
                sharpe_ratio=self._calculate_sharpe(y, model.predict(X)),
                max_drawdown=self._calculate_max_drawdown(y),
                accuracy=np.mean(cv_scores),
                training_time=0.0,
                last_updated=datetime.now(),
            )
        except Exception as e:
            logger.error(f"XGBoost training failed: {e}")
            return None

    def _train_catboost(
        self, X: np.ndarray, y: pd.Series, tscv
    ) -> Optional[ModelPerformance]:
        """Train CatBoost with cross-validation."""
        try:
            model = cb.CatBoostRegressor(
                iterations=200,
                max_depth=7,
                learning_rate=0.05,
                subsample=0.8,
                verbose=False,
            )

            cv_scores = cross_val_score(model, X, y, cv=tscv, scoring="r2")
            model.fit(X, y)

            return ModelPerformance(
                model_name="catboost",
                r2_score=np.mean(cv_scores),
                rmse=np.sqrt(mean_squared_error(y, model.predict(X))),
                mae=mean_absolute_error(y, model.predict(X)),
                sharpe_ratio=self._calculate_sharpe(y, model.predict(X)),
                max_drawdown=self._calculate_max_drawdown(y),
                accuracy=np.mean(cv_scores),
                training_time=0.0,
                last_updated=datetime.now(),
            )
        except Exception as e:
            logger.error(f"CatBoost training failed: {e}")
            return None

    def _train_gradient_boosting(
        self, X: np.ndarray, y: pd.Series, tscv
    ) -> Optional[ModelPerformance]:
        """Train Gradient Boosting with cross-validation."""
        try:
            model = GradientBoostingRegressor(
                n_estimators=200, max_depth=7, learning_rate=0.05, subsample=0.8
            )

            cv_scores = cross_val_score(model, X, y, cv=tscv, scoring="r2")
            model.fit(X, y)

            return ModelPerformance(
                model_name="gradient_boosting",
                r2_score=np.mean(cv_scores),
                rmse=np.sqrt(mean_squared_error(y, model.predict(X))),
                mae=mean_absolute_error(y, model.predict(X)),
                sharpe_ratio=self._calculate_sharpe(y, model.predict(X)),
                max_drawdown=self._calculate_max_drawdown(y),
                accuracy=np.mean(cv_scores),
                training_time=0.0,
                last_updated=datetime.now(),
            )
        except Exception as e:
            logger.error(f"Gradient Boosting training failed: {e}")
            return None

    def _train_random_forest(
        self, X: np.ndarray, y: pd.Series, tscv
    ) -> Optional[ModelPerformance]:
        """Train Random Forest with cross-validation."""
        try:
            model = RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                n_jobs=-1,
            )

            cv_scores = cross_val_score(model, X, y, cv=tscv, scoring="r2")
            model.fit(X, y)

            return ModelPerformance(
                model_name="random_forest",
                r2_score=np.mean(cv_scores),
                rmse=np.sqrt(mean_squared_error(y, model.predict(X))),
                mae=mean_absolute_error(y, model.predict(X)),
                sharpe_ratio=self._calculate_sharpe(y, model.predict(X)),
                max_drawdown=self._calculate_max_drawdown(y),
                accuracy=np.mean(cv_scores),
                training_time=0.0,
                last_updated=datetime.now(),
            )
        except Exception as e:
            logger.error(f"Random Forest training failed: {e}")
            return None

    def _calculate_weights(self, performance_results: Dict[str, ModelPerformance]):
        """Calculate model weights based on performance."""
        if not performance_results:
            self.model_weights = {}
            return

        scores = {name: perf.r2_score for name, perf in performance_results.items()}
        total_score = sum(max(0, s) for s in scores.values())

        if total_score > 0:
            self.model_weights = {
                name: max(0, score) / total_score for name, score in scores.items()
            }
        else:
            # Equal weights if all negative
            self.model_weights = {name: 1 / len(scores) for name in scores}

    @staticmethod
    def _calculate_sharpe(
        y_true: pd.Series, y_pred: np.ndarray, risk_free_rate: float = 0.02
    ) -> float:
        """Calculate Sharpe ratio."""
        returns = (y_true.values - y_pred) / np.abs(y_true.values + 1e-8)
        excess_returns = returns.mean() - risk_free_rate / 252
        return excess_returns / (returns.std() + 1e-8) if returns.std() > 0 else 0.0

    @staticmethod
    def _calculate_max_drawdown(y: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cumulative = (1 + y).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()


class AdaptiveLearningEngine:
    """Adaptive learning that adjusts to market conditions."""

    def __init__(self, window_size: int = 252, retraining_frequency: int = 20):
        self.window_size = window_size
        self.retraining_frequency = retraining_frequency
        self.training_count = 0
        self.last_training_date: Optional[datetime] = None
        self.performance_window: List[float] = []
        self.model_age: Optional[timedelta] = None

    def should_retrain(self) -> bool:
        """Determine if model should be retrained."""
        if self.last_training_date is None:
            return True

        self.training_count += 1

        if self.training_count >= self.retraining_frequency:
            self.training_count = 0
            return True

        return False

    def update_performance(self, performance: float):
        """Track performance for adaptive adjustment."""
        self.performance_window.append(performance)
        if len(self.performance_window) > self.window_size:
            self.performance_window = self.performance_window[-self.window_size :]

    def get_performance_trend(self) -> str:
        """Determine if performance is improving or degrading."""
        if len(self.performance_window) < 20:
            return "insufficient_data"

        recent = np.mean(self.performance_window[-10:])
        previous = np.mean(self.performance_window[-20:-10])

        if recent > previous * 1.05:
            return "improving"
        elif recent < previous * 0.95:
            return "degrading"
        else:
            return "stable"


class IntelligenceSystem:
    """Master intelligence system combining all components."""

    def __init__(self):
        self.feature_selector = FeatureSelector()
        self.anomaly_detector = AnomalyDetector()
        self.ensemble = MultiModelEnsemble()
        self.adaptive_learner = AdaptiveLearningEngine()
        self.is_initialized = False

    def initialize(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Initialize all components."""
        logger.info("[INTELLIGENCE] Initializing advanced intelligence system...")

        # Select features
        selected_features = self.feature_selector.select_features(X, y)
        logger.info(f"[INTELLIGENCE] Selected {len(selected_features)} features")

        X_selected = X[selected_features]

        # Build ensemble
        performance = self.ensemble.build_ensemble(X_selected, y)
        logger.info(f"[INTELLIGENCE] Ensemble built with {len(performance)} models")

        # Initialize adaptive learning
        self.adaptive_learner.last_training_date = datetime.now()
        self.is_initialized = True

        return {
            "selected_features": selected_features,
            "performance": performance,
            "model_weights": self.ensemble.model_weights,
        }

    def predict(self, X: pd.DataFrame) -> Tuple[float, float, str]:
        """
        Generate weighted ensemble prediction.

        Returns:
            Tuple of (prediction, confidence, regime)
        """
        if not self.is_initialized:
            return 0.5, 0.0, "uninitialized"

        try:
            # Scale features
            X_scaled = self.ensemble.scaler.transform(X)

            # Get predictions from available models
            predictions = {}
            for model_name, weight in self.ensemble.model_weights.items():
                if (
                    model_name in self.ensemble.models
                    and self.ensemble.models[model_name]
                ):
                    try:
                        pred = self.ensemble.models[model_name].predict(X_scaled)[0]
                        predictions[model_name] = (pred, weight)
                    except:
                        pass

            # Weighted average
            if predictions:
                weighted_pred = sum(
                    pred * weight for pred, weight in predictions.values()
                )
                avg_weight = sum(weight for _, weight in predictions.values())
                final_pred = weighted_pred / avg_weight if avg_weight > 0 else 0.5
            else:
                final_pred = 0.5

            # Calculate confidence
            confidence = min(1.0, abs(final_pred - 0.5) * 2)

            return final_pred, confidence, self.anomaly_detector.regime

        except Exception as e:
            logger.error(f"[INTELLIGENCE] Prediction error: {e}")
            return 0.5, 0.0, "error"


# Singleton
_intelligence_system: Optional[IntelligenceSystem] = None


def get_intelligence_system() -> IntelligenceSystem:
    """Get or create the intelligence system."""
    global _intelligence_system
    if _intelligence_system is None:
        _intelligence_system = IntelligenceSystem()
    return _intelligence_system
