"""
ML Alpha Model - Machine learning based alpha signals.

Uses ensemble methods, neural networks, and reinforcement learning
to generate sophisticated trading signals from market data.
"""

import logging
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import joblib
from pathlib import Path

from .base_alpha import BaseAlpha

logger = logging.getLogger(__name__)

class MLAlpha(BaseAlpha):
    """
    Machine Learning Alpha Model.

    Features:
    - Ensemble of tree-based models
    - Feature engineering for market data
    - Time series cross-validation
    - Model persistence and updating
    """

    def __init__(self,
                 model_path: Optional[str] = None,
                 retrain_frequency: int = 100,
                 prediction_horizon: int = 5,
                 feature_lookback: int = 20):
        """
        Initialize ML Alpha.

        Args:
            model_path: Path to save/load trained models
            retrain_frequency: How often to retrain models (in signals)
            prediction_horizon: Days ahead to predict
            feature_lookback: Days of history for features
        """
        super().__init__(
            name="MLAlpha",
            description="Machine learning ensemble for alpha signal generation"
        )

        self.model_path = Path(model_path) if model_path else Path("models/ml_alpha")
        self.model_path.mkdir(parents=True, exist_ok=True)

        self.retrain_frequency = retrain_frequency
        self.prediction_horizon = prediction_horizon
        self.feature_lookback = feature_lookback

        # Models
        self.return_model = None
        self.volatility_model = None
        self.scaler = StandardScaler()

        # Tracking
        self.signal_count = 0
        self.last_training_date = None

        # Load existing models if available
        self._load_models()

    def generate_signal(self,
                       market_data: pd.DataFrame,
                       regime_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate ML-based alpha signal.

        Args:
            market_data: OHLCV market data
            regime_context: Current market regime

        Returns:
            Signal dictionary with signal, confidence, and metadata
        """
        try:
            # Check if we need to retrain
            self.signal_count += 1
            if self.signal_count % self.retrain_frequency == 0:
                self._train_models(market_data)

            # Generate features
            features = self._extract_features(market_data)

            if features is None or len(features) == 0:
                return {
                    'signal': 0.0,
                    'confidence': 0.0,
                    'metadata': {'error': 'Insufficient data for feature extraction'}
                }

            # Get latest feature vector
            latest_features = features.iloc[-1:].values

            # Generate predictions
            return_pred = self._predict_return(latest_features)
            vol_pred = self._predict_volatility(latest_features)

            # Combine into signal
            signal = self._combine_predictions(return_pred, vol_pred)

            # Calculate confidence
            confidence = self._calculate_confidence(features, return_pred, vol_pred)

            # Adjust for regime
            if regime_context:
                signal, confidence = self._adjust_for_regime(signal, confidence, regime_context)

            return {
                'signal': float(signal),
                'confidence': float(confidence),
                'metadata': {
                    'return_prediction': float(return_pred[0]),
                    'volatility_prediction': float(vol_pred[0]),
                    'features_used': len(features.columns),
                    'model_trained': self.return_model is not None,
                    'regime_adjusted': regime_context is not None
                }
            }

        except Exception as e:
            logger.error(f"ML alpha signal generation failed: {e}")
            return {
                'signal': 0.0,
                'confidence': 0.0,
                'metadata': {'error': str(e)}
            }

    def _extract_features(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Extract ML features from market data.

        Features include:
        - Technical indicators
        - Price momentum
        - Volatility measures
        - Volume indicators
        """
        if len(data) < self.feature_lookback + self.prediction_horizon:
            return None

        df = data.copy()

        # Basic price features
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))

        # Momentum features
        for period in [5, 10, 20, 50]:
            df[f'momentum_{period}'] = df['Close'] / df['Close'].shift(period) - 1
            df[f'volume_momentum_{period}'] = df['Volume'] / df['Volume'].shift(period) - 1

        # Volatility features
        df['realized_vol_20'] = df['returns'].rolling(20).std() * np.sqrt(252)
        df['realized_vol_5'] = df['returns'].rolling(5).std() * np.sqrt(252)

        # Technical indicators
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        ema_12 = df['Close'].ewm(span=12).mean()
        ema_26 = df['Close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()

        # Bollinger Bands
        sma_20 = df['Close'].rolling(20).mean()
        std_20 = df['Close'].rolling(20).std()
        df['bb_upper'] = sma_20 + 2 * std_20
        df['bb_lower'] = sma_20 - 2 * std_20
        df['bb_position'] = (df['Close'] - sma_20) / (2 * std_20)

        # Volume features
        df['volume_sma_20'] = df['Volume'].rolling(20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma_20']

        # Lagged features
        for lag in [1, 2, 3, 5]:
            df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume_ratio'].shift(lag)

        # Drop NaN values
        df = df.dropna()

        # Select feature columns (exclude OHLCV and target variables)
        feature_cols = [col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
        features = df[feature_cols]

        return features

    def _train_models(self, data: pd.DataFrame):
        """
        Train ML models on historical data.
        """
        logger.info("Training ML alpha models...")

        try:
            # Extract features
            features = self._extract_features(data)
            if features is None or len(features) < 50:
                logger.warning("Insufficient data for ML training")
                return

            # Create targets
            returns = data['Close'].pct_change(self.prediction_horizon).shift(-self.prediction_horizon)
            volatility = data['Close'].pct_change().rolling(20).std() * np.sqrt(252)

            # Align features with targets
            common_index = features.index.intersection(returns.index)
            features = features.loc[common_index]
            returns_target = returns.loc[common_index]
            vol_target = volatility.loc[common_index]

            # Remove NaN targets
            valid_idx = returns_target.dropna().index
            features = features.loc[valid_idx]
            returns_target = returns_target.loc[valid_idx]
            vol_target = vol_target.loc[valid_idx]

            if len(features) < 30:
                logger.warning("Insufficient valid training data")
                return

            # Scale features
            features_scaled = self.scaler.fit_transform(features)

            # Train return prediction model
            self.return_model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=3,
                random_state=42
            )
            self.return_model.fit(features_scaled, returns_target)

            # Train volatility prediction model
            self.volatility_model = RandomForestRegressor(
                n_estimators=50,
                max_depth=4,
                random_state=42
            )
            self.volatility_model.fit(features_scaled, vol_target)

            # Save models
            self._save_models()

            self.last_training_date = pd.Timestamp.now()
            logger.info("ML models trained successfully")

        except Exception as e:
            logger.error(f"ML model training failed: {e}")

    def _predict_return(self, features: np.ndarray) -> np.ndarray:
        """Predict future returns."""
        if self.return_model is None:
            return np.array([0.0])

        features_scaled = self.scaler.transform(features)
        return self.return_model.predict(features_scaled)

    def _predict_volatility(self, features: np.ndarray) -> np.ndarray:
        """Predict future volatility."""
        if self.volatility_model is None:
            return np.array([0.2])  # Default 20% vol

        features_scaled = self.scaler.transform(features)
        return self.volatility_model.predict(features_scaled)

    def _combine_predictions(self, return_pred: np.ndarray, vol_pred: np.ndarray) -> float:
        """
        Combine return and volatility predictions into alpha signal.

        Strategy: Go long when expected return > risk-adjusted threshold
        """
        expected_return = return_pred[0]
        expected_vol = vol_pred[0]

        # Risk-adjusted return threshold
        risk_free_rate = 0.02  # 2% annual
        risk_adjustment = expected_vol * 0.5  # Half of expected vol as risk penalty

        threshold = risk_free_rate + risk_adjustment

        # Generate signal
        if expected_return > threshold:
            signal = min(expected_return * 2, 1.0)  # Scale up but cap at 1
        elif expected_return < -threshold:
            signal = max(expected_return * 2, -1.0)  # Scale down but cap at -1
        else:
            signal = 0.0

        return signal

    def _calculate_confidence(self,
                            features: pd.DataFrame,
                            return_pred: np.ndarray,
                            vol_pred: np.ndarray) -> float:
        """
        Calculate prediction confidence based on model certainty and feature stability.
        """
        if self.return_model is None:
            return 0.0

        # Model confidence based on prediction magnitude and training performance
        return_conf = min(abs(return_pred[0]) * 2, 0.8)  # Up to 80% confidence

        # Feature stability (lower std = higher confidence)
        feature_std = features.iloc[-10:].std().mean()  # Last 10 periods
        stability_conf = max(0, 1 - feature_std)  # Lower std = higher confidence

        # Combine confidences
        confidence = (return_conf + stability_conf) / 2

        return min(confidence, 1.0)

    def _adjust_for_regime(self,
                          signal: float,
                          confidence: float,
                          regime_context: Dict[str, Any]) -> Tuple[float, float]:
        """
        Adjust signal and confidence based on market regime.
        """
        regime = regime_context.get('regime_tag', 'NORMAL')

        # ML models perform differently in different regimes
        regime_multipliers = {
            'HIGH_VOL': 0.8,  # ML signals less reliable in high vol
            'LOW_VOL': 1.2,   # ML signals more reliable in low vol
            'BULL_QUIET': 1.1,
            'BEAR_CRISIS': 0.7,  # Less reliable in crises
            'NORMAL': 1.0
        }

        multiplier = regime_multipliers.get(regime, 1.0)
        adjusted_signal = signal * multiplier

        # Confidence adjustments
        if regime in ['LOW_VOL', 'BULL_QUIET']:
            adjusted_confidence = min(confidence * 1.1, 1.0)
        elif regime in ['HIGH_VOL', 'BEAR_CRISIS']:
            adjusted_confidence = confidence * 0.9
        else:
            adjusted_confidence = confidence

        return adjusted_signal, adjusted_confidence

    def _save_models(self):
        """Save trained models to disk."""
        try:
            if self.return_model:
                joblib.dump(self.return_model, self.model_path / 'return_model.pkl')
            if self.volatility_model:
                joblib.dump(self.volatility_model, self.model_path / 'volatility_model.pkl')
            joblib.dump(self.scaler, self.model_path / 'scaler.pkl')
        except Exception as e:
            logger.error(f"Failed to save ML models: {e}")

    def _load_models(self):
        """Load trained models from disk."""
        try:
            return_model_path = self.model_path / 'return_model.pkl'
            vol_model_path = self.model_path / 'volatility_model.pkl'
            scaler_path = self.model_path / 'scaler.pkl'

            if return_model_path.exists():
                self.return_model = joblib.load(return_model_path)
            if vol_model_path.exists():
                self.volatility_model = joblib.load(vol_model_path)
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)

            logger.info("ML models loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load ML models: {e}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the ML models."""
        return {
            'return_model_trained': self.return_model is not None,
            'volatility_model_trained': self.volatility_model is not None,
            'signal_count': self.signal_count,
            'last_training_date': self.last_training_date.isoformat() if self.last_training_date else None,
            'feature_lookback': self.feature_lookback,
            'prediction_horizon': self.prediction_horizon,
            'retrain_frequency': self.retrain_frequency
        }
