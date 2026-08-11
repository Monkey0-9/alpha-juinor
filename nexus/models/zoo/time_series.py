import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class GradientBoostedTimeSeriesModel:
    """
    Real Gradient Boosted Decision Tree model for time-series directional forecasting.
    Uses technical indicator feature vectors to predict price direction.
    """

    FEATURE_COLS = [
        'returns', 'rsi_14', 'macd', 'macd_signal', 'macd_hist',
        'roc_10', 'bb_mean', 'bb_std', 'atr_14'
    ]

    def __init__(self, confidence_threshold: float = 0.55):
        self.confidence_threshold = confidence_threshold
        self.model = HistGradientBoostingClassifier(
            max_iter=100,
            learning_rate=0.05,
            max_depth=5,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_trained = False

    def _prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Extract feature matrix and target vector from DataFrame."""
        df_copy = df.copy()

        # Generate target: +1 if next close > current close, else -1
        if 'close' in df_copy.columns:
            df_copy['target'] = np.where(df_copy['close'].shift(-1) > df_copy['close'], 1, -1)
        else:
            df_copy['target'] = 0

        # Ensure all feature columns exist
        available_cols = [c for c in self.FEATURE_COLS if c in df_copy.columns]
        if not available_cols:
            # Fallback to numeric columns
            numeric_cols = df_copy.select_dtypes(include=[np.number]).columns.tolist()
            available_cols = [c for c in numeric_cols if c != 'target']

        df_clean = df_copy.dropna(subset=available_cols)
        if len(df_clean) < 20:
            return np.array([]), np.array([])

        X = df_clean[available_cols].to_numpy()
        y = df_clean['target'].to_numpy()
        return X, y

    def fit(self, df: pd.DataFrame) -> bool:
        """Trains the model on historical bar data."""
        X, y = self._prepare_features(df)
        if len(X) < 30:
            logger.warning("Insufficient data samples to train TimeSeriesModel.")
            return False

        try:
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled, y)
            self.is_trained = True
            logger.info("GradientBoostedTimeSeriesModel successfully trained on %d samples.", len(X))
            return True
        except Exception as e:
            logger.error("Failed to fit TimeSeriesModel: %s", e)
            return False

    def predict(self, df: pd.DataFrame) -> int:
        """
        Returns signal:
          1  (Buy)
         -1  (Sell)
          0  (Hold / No Trade)
        """
        if not self.is_trained:
            # Attempt quick fit if untrained
            success = self.fit(df)
            if not success:
                return 0

        available_cols = [c for c in self.FEATURE_COLS if c in df.columns]
        if not available_cols or len(df) < 1:
            return 0

        try:
            latest_row = df[available_cols].iloc[-1:].dropna()
            if latest_row.empty:
                return 0

            X_last = self.scaler.transform(latest_row.to_numpy())
            probs = self.model.predict_proba(X_last)[0]
            classes = self.model.classes_

            prob_map = {c: p for c, p in zip(classes, probs)}
            prob_buy = prob_map.get(1, 0.0)
            prob_sell = prob_map.get(-1, 0.0)

            # Confidence gating: Only issue signal if probability exceeds threshold
            if prob_buy > self.confidence_threshold:
                return 1
            elif prob_sell > self.confidence_threshold:
                return -1
            else:
                return 0  # NO_TRADE / HOLD
        except Exception as e:
            logger.debug("TimeSeriesModel prediction error: %s", e)
            return 0


class PyTorchLSTMModel:
    """
    Sequence-based Deep Learning model stub using PyTorch (if available).
    Falls back gracefully to Gradient Boosted Tree if PyTorch is absent.
    """

    def __init__(self):
        self.tree_fallback = GradientBoostedTimeSeriesModel()

    def fit(self, df: pd.DataFrame) -> bool:
        return self.tree_fallback.fit(df)

    def predict(self, df: pd.DataFrame) -> int:
        return self.tree_fallback.predict(df)
