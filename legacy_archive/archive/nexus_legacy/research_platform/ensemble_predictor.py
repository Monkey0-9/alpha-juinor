"""
Ensemble Predictor - Non-Linear ML for Alpha Discovery
=========================================================

Move beyond linear regression. Use gradient boosted trees.

Features:
1. XGBoost and LightGBM integration
2. Feature importance from alternative data
3. Weak signal ensembling
4. Walk-forward ML training
5. Regime-aware predictions

Non-linear alpha from alternative data.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import threading

logger = logging.getLogger(__name__)

# Try to import ML libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("[ML] XGBoost not installed. Install: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.warning("[ML] LightGBM not installed. Install: pip install lightgbm")


@dataclass
class FeatureImportance:
    """Feature importance from model."""
    feature_name: str
    importance_score: float
    importance_rank: int
    category: str  # options_flow, insider, order_flow, technical


@dataclass
class PredictionResult:
    """Result of ensemble prediction."""
    timestamp: datetime
    symbol: str

    # Predictions
    predicted_return: float  # Expected return
    confidence: float        # 0-1 confidence
    direction: int           # 1 = long, -1 = short, 0 = neutral

    # Model details
    model_name: str

    # Feature contributions
    top_features: List[FeatureImportance]

    # Regime context
    regime: Optional[str] = None


@dataclass
class EnsembleModel:
    """An ensemble model container."""
    name: str
    model_type: str  # xgboost, lightgbm, linear

    # Model object (stored as any)
    model: Any = None

    # Training info
    trained_at: Optional[datetime] = None
    training_samples: int = 0

    # Performance
    in_sample_r2: float = 0.0
    oos_r2: float = 0.0

    # Feature info
    feature_names: List[str] = field(default_factory=list)
    feature_importance: Dict[str, float] = field(default_factory=dict)


class EnsemblePredictor:
    """
    Non-linear ensemble predictor using gradient boosted trees.

    Combines weak signals from alternative data into
    strong predictive models.
    """

    def __init__(self):
        """Initialize the predictor."""
        self.models: Dict[str, EnsembleModel] = {}
        self.predictions_history: List[PredictionResult] = []

        self._lock = threading.Lock()

        available = []
        if XGBOOST_AVAILABLE:
            available.append("XGBoost")
        if LIGHTGBM_AVAILABLE:
            available.append("LightGBM")

        logger.info(
            f"[ENSEMBLE] Predictor initialized | "
            f"Available: {', '.join(available) if available else 'Linear only'}"
        )

    def create_feature_matrix(
        self,
        market_data: pd.DataFrame,
        options_flow_signals: Optional[pd.DataFrame] = None,
        insider_signals: Optional[pd.DataFrame] = None,
        order_flow_signals: Optional[pd.DataFrame] = None,
        technical_signals: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Create feature matrix from multiple signal sources.

        Combines weak signals into unified feature set.
        """
        features = market_data.copy()

        # Basic features from market data
        if 'Close' in features.columns:
            features['return_1d'] = features['Close'].pct_change()
            features['return_5d'] = features['Close'].pct_change(5)
            features['return_20d'] = features['Close'].pct_change(20)

            # Volatility
            features['vol_10d'] = features['return_1d'].rolling(10).std()
            features['vol_20d'] = features['return_1d'].rolling(20).std()

            # Momentum
            features['momentum_10d'] = features['Close'] / features['Close'].shift(10) - 1
            features['momentum_20d'] = features['Close'] / features['Close'].shift(20) - 1

        if 'Volume' in features.columns:
            features['volume_ratio'] = features['Volume'] / features['Volume'].rolling(20).mean()

        # Merge alternative data signals
        if options_flow_signals is not None:
            for col in options_flow_signals.columns:
                features[f'opt_{col}'] = options_flow_signals[col]

        if insider_signals is not None:
            for col in insider_signals.columns:
                features[f'ins_{col}'] = insider_signals[col]

        if order_flow_signals is not None:
            for col in order_flow_signals.columns:
                features[f'flow_{col}'] = order_flow_signals[col]

        if technical_signals is not None:
            for col in technical_signals.columns:
                features[f'tech_{col}'] = technical_signals[col]

        return features.dropna()

    def train_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_name: str = "default",
        model_type: str = "xgboost",
        n_estimators: int = 100,
        max_depth: int = 5,
        learning_rate: float = 0.1,
        test_size: float = 0.3
    ) -> EnsembleModel:
        """
        Train an ensemble model.

        Args:
            X: Feature matrix
            y: Target (e.g., forward returns)
            model_name: Name for this model
            model_type: "xgboost", "lightgbm", or "linear"
            n_estimators: Number of trees
            max_depth: Max tree depth
            learning_rate: Learning rate
            test_size: Fraction for testing
        """
        # Split data
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        feature_names = list(X.columns)

        if model_type == "xgboost" and XGBOOST_AVAILABLE:
            model = self._train_xgboost(
                X_train, y_train, n_estimators, max_depth, learning_rate
            )
            importance = dict(zip(
                feature_names,
                model.feature_importances_
            ))
        elif model_type == "lightgbm" and LIGHTGBM_AVAILABLE:
            model = self._train_lightgbm(
                X_train, y_train, n_estimators, max_depth, learning_rate
            )
            importance = dict(zip(
                feature_names,
                model.feature_importances_
            ))
        else:
            # Fallback to simple linear
            model, importance = self._train_linear(X_train, y_train, feature_names)

        # Calculate R2
        from sklearn.metrics import r2_score
        is_pred = model.predict(X_train)
        oos_pred = model.predict(X_test)

        is_r2 = r2_score(y_train, is_pred)
        oos_r2 = r2_score(y_test, oos_pred)

        ensemble_model = EnsembleModel(
            name=model_name,
            model_type=model_type,
            model=model,
            trained_at=datetime.utcnow(),
            training_samples=len(X_train),
            in_sample_r2=is_r2,
            oos_r2=oos_r2,
            feature_names=feature_names,
            feature_importance=importance
        )

        with self._lock:
            self.models[model_name] = ensemble_model

        logger.info(
            f"[ENSEMBLE] Model trained: {model_name} | "
            f"Type: {model_type} | "
            f"IS R2: {is_r2:.4f} | OOS R2: {oos_r2:.4f}"
        )

        return ensemble_model

    def _train_xgboost(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_estimators: int,
        max_depth: int,
        learning_rate: float
    ):
        """Train XGBoost model."""
        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            objective='reg:squarederror',
            verbosity=0
        )
        model.fit(X, y)
        return model

    def _train_lightgbm(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_estimators: int,
        max_depth: int,
        learning_rate: float
    ):
        """Train LightGBM model."""
        model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            verbose=-1
        )
        model.fit(X, y)
        return model

    def _train_linear(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: List[str]
    ):
        """Train simple linear model as fallback."""
        from sklearn.linear_model import Ridge

        model = Ridge(alpha=1.0)
        model.fit(X, y)

        importance = dict(zip(feature_names, np.abs(model.coef_)))

        return model, importance

    def predict(
        self,
        X: pd.DataFrame,
        model_name: str = "default",
        symbol: str = "UNKNOWN",
        regime: Optional[str] = None
    ) -> Optional[PredictionResult]:
        """Make a prediction using trained model."""
        with self._lock:
            if model_name not in self.models:
                logger.warning(f"[ENSEMBLE] Model not found: {model_name}")
                return None

            model_obj = self.models[model_name]

        try:
            # Ensure columns match
            missing = set(model_obj.feature_names) - set(X.columns)
            if missing:
                logger.warning(f"[ENSEMBLE] Missing features: {missing}")
                return None

            X_aligned = X[model_obj.feature_names]

            # Predict
            prediction = model_obj.model.predict(X_aligned)

            # Get latest prediction (if multiple rows)
            if hasattr(prediction, '__len__') and len(prediction) > 1:
                pred_value = float(prediction[-1])
            else:
                pred_value = float(prediction)

            # Direction and confidence
            direction = 1 if pred_value > 0.001 else (-1 if pred_value < -0.001 else 0)
            confidence = min(0.95, abs(pred_value) * 10)  # Scale to 0-1

            # Top features
            sorted_importance = sorted(
                model_obj.feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]

            top_features = [
                FeatureImportance(
                    feature_name=name,
                    importance_score=score,
                    importance_rank=i + 1,
                    category=self._categorize_feature(name)
                )
                for i, (name, score) in enumerate(sorted_importance)
            ]

            result = PredictionResult(
                timestamp=datetime.utcnow(),
                symbol=symbol,
                predicted_return=pred_value,
                confidence=confidence,
                direction=direction,
                model_name=model_name,
                top_features=top_features,
                regime=regime
            )

            with self._lock:
                self.predictions_history.append(result)

            return result

        except Exception as e:
            logger.error(f"[ENSEMBLE] Prediction failed: {e}")
            return None

    def _categorize_feature(self, feature_name: str) -> str:
        """Categorize feature by name prefix."""
        if feature_name.startswith('opt_'):
            return 'options_flow'
        elif feature_name.startswith('ins_'):
            return 'insider'
        elif feature_name.startswith('flow_'):
            return 'order_flow'
        elif feature_name.startswith('tech_'):
            return 'technical'
        else:
            return 'market'

    def walk_forward_train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_name: str,
        n_folds: int = 5,
        model_type: str = "xgboost"
    ) -> Dict[str, Any]:
        """
        Walk-forward training with cross-validation.

        Returns performance across folds.
        """
        fold_size = len(X) // n_folds
        results = []

        for i in range(1, n_folds):
            train_end = i * fold_size
            test_end = min((i + 1) * fold_size, len(X))

            X_train = X.iloc[:train_end]
            y_train = y.iloc[:train_end]
            X_test = X.iloc[train_end:test_end]
            y_test = y.iloc[train_end:test_end]

            # Train model
            model = self.train_model(
                X_train, y_train,
                model_name=f"{model_name}_fold{i}",
                model_type=model_type,
                test_size=0  # Already split
            )

            # Predict on test
            pred = model.model.predict(X_test)

            from sklearn.metrics import r2_score, mean_squared_error
            r2 = r2_score(y_test, pred)
            rmse = np.sqrt(mean_squared_error(y_test, pred))

            results.append({
                "fold": i,
                "train_size": len(X_train),
                "test_size": len(X_test),
                "r2": r2,
                "rmse": rmse
            })

        return {
            "model_name": model_name,
            "n_folds": n_folds,
            "folds": results,
            "avg_r2": np.mean([r["r2"] for r in results]),
            "avg_rmse": np.mean([r["rmse"] for r in results])
        }

    def get_model_summary(self) -> Dict[str, Any]:
        """Get summary of all models."""
        with self._lock:
            return {
                "models_count": len(self.models),
                "predictions_made": len(self.predictions_history),
                "models": {
                    name: {
                        "type": m.model_type,
                        "trained_at": m.trained_at.isoformat() if m.trained_at else None,
                        "samples": m.training_samples,
                        "is_r2": m.in_sample_r2,
                        "oos_r2": m.oos_r2,
                        "top_features": sorted(
                            m.feature_importance.items(),
                            key=lambda x: x[1],
                            reverse=True
                        )[:5]
                    }
                    for name, m in self.models.items()
                }
            }


# Singleton
_predictor: Optional[EnsemblePredictor] = None


def get_ensemble_predictor() -> EnsemblePredictor:
    """Get or create the Ensemble Predictor."""
    global _predictor
    if _predictor is None:
        _predictor = EnsemblePredictor()
    return _predictor
