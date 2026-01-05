import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import joblib
from pathlib import Path
import os

# Machine Learning imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression

# Deep Learning imports (optional)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
    from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Input
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    from tensorflow.keras.optimizers import Adam
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    DEEP_LEARNING_AVAILABLE = False
    logger.warning("TensorFlow not available. Deep learning features disabled.")

logger = logging.getLogger(__name__)

class InstitutionalMLEngine:
    """
    INSTITUTIONAL-GRADE MACHINE LEARNING ENGINE
    Advanced ensemble methods and deep learning for alpha generation.
    Supports both classification (directional) and regression (magnitude) predictions.
    """

    def __init__(self, model_dir: str = "models", use_deep_learning: bool = True):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.use_deep_learning = use_deep_learning and DEEP_LEARNING_AVAILABLE

        # Model configurations
        self.ensemble_configs = {
            'classification': {
                'rf': {'n_estimators': 200, 'max_depth': 10, 'min_samples_split': 50},
                'gb': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 5},
                'weights': [0.4, 0.6]  # RF: 40%, GB: 60%
            },
            'regression': {
                'rf': {'n_estimators': 200, 'max_depth': 8, 'min_samples_split': 30},
                'gb': {'n_estimators': 100, 'learning_rate': 0.05, 'max_depth': 4},
                'weights': [0.3, 0.7]
            }
        }

        # Feature engineering parameters
        self.feature_configs = {
            'technical_indicators': True,
            'price_patterns': True,
            'volume_analysis': True,
            'sentiment_features': True,
            'macro_indicators': True,
            'order_flow_features': True
        }

        # Model storage
        self.models = {}
        self.scalers = {}
        self.feature_selectors = {}

        # Performance tracking
        self.model_performance = {}

        logger.info(f"Institutional ML Engine initialized (Deep Learning: {self.use_deep_learning})")

    def train_ensemble_model(self, ticker: str, X_train: pd.DataFrame, y_train: pd.Series,
                           model_type: str = 'classification', cv_folds: int = 5) -> Dict[str, Any]:
        """
        Train an ensemble model for a specific ticker.
        Supports both classification (directional) and regression (return magnitude) tasks.
        """
        try:
            logger.info(f"Training {model_type} ensemble model for {ticker}")

            # 1. Feature engineering and selection
            X_processed, feature_names = self._engineer_features(X_train, ticker)
            X_selected = self._select_features(X_processed, y_train, model_type)

            # 2. Scale features
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X_selected)

            # 3. Create ensemble model
            if model_type == 'classification':
                ensemble_model = self._create_classification_ensemble()
            else:
                ensemble_model = self._create_regression_ensemble()

            # 4. Cross-validation
            cv_scores = self._cross_validate_model(ensemble_model, X_scaled, y_train, cv_folds)

            # 5. Train final model
            ensemble_model.fit(X_scaled, y_train)

            # 6. Store model and metadata
            model_key = f"{ticker}_{model_type}"
            self.models[model_key] = ensemble_model
            self.scalers[model_key] = scaler
            self.feature_selectors[model_key] = {
                'selected_features': feature_names,
                'selector': None  # Could store feature selector here
            }

            # 7. Calculate performance metrics
            performance = self._calculate_model_performance(ensemble_model, X_scaled, y_train, cv_scores)

            # 8. Save model
            self._save_model(model_key, ensemble_model, scaler, performance)

            logger.info(f"Ensemble model trained for {ticker} with CV score: {cv_scores['mean']:.4f}")

            return {
                'model_key': model_key,
                'performance': performance,
                'feature_count': X_selected.shape[1],
                'training_samples': len(X_train)
            }

        except Exception as e:
            logger.error(f"Failed to train ensemble model for {ticker}: {e}")
            return {}

    def predict_signal(self, ticker: str, features: pd.DataFrame,
                      model_type: str = 'classification') -> Dict[str, Any]:
        """
        Generate trading signal using trained ensemble model.
        """
        model_key = f"{ticker}_{model_type}"

        if model_key not in self.models:
            logger.warning(f"No trained model found for {model_key}")
            return {'signal': 0.0, 'confidence': 0.0, 'prediction': None}

        try:
            # Load model if not in memory
            if not hasattr(self.models[model_key], 'predict'):
                self._load_model(model_key)

            model = self.models[model_key]
            scaler = self.scalers[model_key]

            # Process features
            X_processed, _ = self._engineer_features(features, ticker)
            X_selected = self._select_features_inference(X_processed, model_key)
            X_scaled = scaler.transform(X_selected)

            # Generate prediction
            if model_type == 'classification':
                prediction_proba = model.predict_proba(X_scaled)[0]
                prediction = model.predict(X_scaled)[0]

                # Convert to signal (-1, 0, 1)
                if prediction == 0:  # Sell signal
                    signal = -1.0
                    confidence = prediction_proba[0]
                elif prediction == 1:  # Buy signal
                    signal = 1.0
                    confidence = prediction_proba[1]
                else:  # Hold
                    signal = 0.0
                    confidence = max(prediction_proba[0], prediction_proba[1], prediction_proba[2])

            else:  # Regression
                prediction = model.predict(X_scaled)[0]
                signal = np.clip(prediction, -1, 1)  # Normalize to [-1, 1]
                confidence = min(abs(prediction), 1.0)  # Confidence based on magnitude

            return {
                'signal': signal,
                'confidence': confidence,
                'prediction': prediction,
                'model_type': model_type
            }

        except Exception as e:
            logger.error(f"Prediction failed for {ticker}: {e}")
            return {'signal': 0.0, 'confidence': 0.0, 'prediction': None}

    def train_deep_learning_model(self, ticker: str, X_train: pd.DataFrame, y_train: pd.Series,
                                sequence_length: int = 60, model_type: str = 'lstm') -> Dict[str, Any]:
        """
        Train deep learning model for time series prediction.
        Supports LSTM, CNN, and hybrid architectures.
        """
        if not self.use_deep_learning:
            logger.warning("Deep learning not available")
            return {}

        try:
            logger.info(f"Training {model_type} deep learning model for {ticker}")

            # 1. Prepare sequential data
            X_seq, y_seq = self._prepare_sequential_data(X_train, y_train, sequence_length)

            # 2. Create model architecture
            if model_type == 'lstm':
                model = self._create_lstm_model(X_seq.shape[1:])
            elif model_type == 'cnn':
                model = self._create_cnn_model(X_seq.shape[1:])
            elif model_type == 'hybrid':
                model = self._create_hybrid_model(X_seq.shape[1:])
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            # 3. Train model
            history = self._train_deep_model(model, X_seq, y_seq, ticker, model_type)

            # 4. Evaluate model
            performance = self._evaluate_deep_model(model, X_seq, y_seq)

            # 5. Store model
            model_key = f"{ticker}_{model_type}_deep"
            self.models[model_key] = model

            # 6. Save model
            model_path = self.model_dir / f"{model_key}.h5"
            model.save(model_path)

            logger.info(f"Deep learning model trained for {ticker} with validation loss: {performance['val_loss']:.4f}")

            return {
                'model_key': model_key,
                'performance': performance,
                'architecture': model_type,
                'sequence_length': sequence_length
            }

        except Exception as e:
            logger.error(f"Failed to train deep learning model for {ticker}: {e}")
            return {}

    def _create_classification_ensemble(self) -> VotingClassifier:
        """Create ensemble classifier with Random Forest and Gradient Boosting."""
        config = self.ensemble_configs['classification']

        rf = RandomForestClassifier(**config['rf'], random_state=42)
        gb = GradientBoostingClassifier(**config['gb'], random_state=42)

        ensemble = VotingClassifier(
            estimators=[('rf', rf), ('gb', gb)],
            voting='soft',
            weights=config['weights']
        )

        return ensemble

    def _create_regression_ensemble(self) -> VotingRegressor:
        """Create ensemble regressor with Random Forest and Gradient Boosting."""
        config = self.ensemble_configs['regression']

        rf = RandomForestRegressor(**config['rf'], random_state=42)
        gb = GradientBoostingRegressor(**config['gb'], random_state=42)

        ensemble = VotingRegressor(
            estimators=[('rf', rf), ('gb', gb)],
            weights=config['weights']
        )

        return ensemble

    def _engineer_features(self, X: pd.DataFrame, ticker: str) -> Tuple[pd.DataFrame, List[str]]:
        """Advanced feature engineering for institutional alpha generation."""
        features = X.copy()
        feature_names = []

        # Technical indicators
        if self.feature_configs['technical_indicators']:
            features, tech_names = self._add_technical_features(features)
            feature_names.extend(tech_names)

        # Price patterns
        if self.feature_configs['price_patterns']:
            features, pattern_names = self._add_pattern_features(features)
            feature_names.extend(pattern_names)

        # Volume analysis
        if self.feature_configs['volume_analysis']:
            features, volume_names = self._add_volume_features(features)
            feature_names.extend(volume_names)

        # Sentiment features (placeholder for integration)
        if self.feature_configs['sentiment_features']:
            features['sentiment_score'] = 0.0  # Would integrate with sentiment provider
            feature_names.append('sentiment_score')

        # Macro indicators (placeholder)
        if self.feature_configs['macro_indicators']:
            features['vix'] = 20.0  # Would integrate with macro data
            features['yield_curve'] = 0.5
            feature_names.extend(['vix', 'yield_curve'])

        # Order flow features (placeholder)
        if self.feature_configs['order_flow_features']:
            features['order_imbalance'] = 0.0  # Would integrate with order flow provider
            feature_names.append('order_imbalance')

        return features.fillna(0), feature_names

    def _add_technical_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Add comprehensive technical indicators."""
        features = df.copy()
        names = []

        # Moving averages
        for period in [5, 10, 20, 50]:
            features[f'sma_{period}'] = features['Close'].rolling(period).mean()
            features[f'ema_{period}'] = features['Close'].ewm(span=period).mean()
            names.extend([f'sma_{period}', f'ema_{period}'])

        # Momentum indicators
        features['rsi'] = self._calculate_rsi(features['Close'])
        features['macd'] = self._calculate_macd(features['Close'])
        features['stoch_k'] = self._calculate_stochastic(features)
        names.extend(['rsi', 'macd', 'stoch_k'])

        # Volatility
        features['bb_upper'], features['bb_lower'] = self._calculate_bollinger_bands(features['Close'])
        features['atr'] = self._calculate_atr(features)
        names.extend(['bb_upper', 'bb_lower', 'atr'])

        return features, names

    def _add_pattern_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Add price pattern recognition features."""
        features = df.copy()
        names = []

        # Candlestick patterns (simplified)
        features['doji'] = ((abs(features['Open'] - features['Close']) / (features['High'] - features['Low'])) < 0.1).astype(int)
        features['hammer'] = ((features['Low'] < features['Open']) & (features['Close'] > (features['High'] + features['Low'])/2)).astype(int)
        names.extend(['doji', 'hammer'])

        # Trend features
        features['trend_strength'] = features['Close'].rolling(20).std() / features['Close'].rolling(20).mean()
        features['momentum'] = features['Close'] / features['Close'].shift(10) - 1
        names.extend(['trend_strength', 'momentum'])

        return features, names

    def _add_volume_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Add volume-based features."""
        features = df.copy()
        names = []

        # Volume indicators
        features['volume_sma'] = features['Volume'].rolling(20).mean()
        features['volume_ratio'] = features['Volume'] / features['volume_sma']
        features['vwap'] = (features['Close'] * features['Volume']).cumsum() / features['Volume'].cumsum()
        names.extend(['volume_sma', 'volume_ratio', 'vwap'])

        return features, names

    def _select_features(self, X: pd.DataFrame, y: pd.Series, model_type: str) -> pd.DataFrame:
        """Select most important features using statistical methods."""
        try:
            if model_type == 'classification':
                # Use mutual information for classification
                selector = SelectKBest(mutual_info_regression, k=min(50, X.shape[1]))
            else:
                # Use f-regression for regression
                selector = SelectKBest(f_regression, k=min(50, X.shape[1]))

            X_selected = selector.fit_transform(X, y)
            selected_features = X.columns[selector.get_support()].tolist()

            return pd.DataFrame(X_selected, columns=selected_features, index=X.index)

        except Exception as e:
            logger.warning(f"Feature selection failed: {e}. Using all features.")
            return X

    def _select_features_inference(self, X: pd.DataFrame, model_key: str) -> pd.DataFrame:
        """Select features for inference using stored feature information."""
        if model_key in self.feature_selectors:
            selected_features = self.feature_selectors[model_key]['selected_features']
            available_features = [f for f in selected_features if f in X.columns]
            return X[available_features]
        else:
            return X

    def _cross_validate_model(self, model, X: np.ndarray, y: pd.Series, cv_folds: int) -> Dict[str, float]:
        """Perform time series cross-validation."""
        try:
            tscv = TimeSeriesSplit(n_splits=cv_folds)

            if hasattr(model, 'predict_proba'):  # Classification
                scores = cross_val_score(model, X, y, cv=tscv, scoring='f1_macro')
            else:  # Regression
                scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_squared_error')
                scores = -scores  # Convert to positive MSE

            return {
                'mean': scores.mean(),
                'std': scores.std(),
                'scores': scores.tolist()
            }

        except Exception as e:
            logger.warning(f"Cross-validation failed: {e}")
            return {'mean': 0.0, 'std': 0.0, 'scores': []}

    def _calculate_model_performance(self, model, X: np.ndarray, y: pd.Series, cv_scores: Dict) -> Dict[str, float]:
        """Calculate comprehensive model performance metrics."""
        try:
            y_pred = model.predict(X)

            if hasattr(model, 'predict_proba'):  # Classification
                performance = {
                    'accuracy': accuracy_score(y, y_pred),
                    'precision': precision_score(y, y_pred, average='macro', zero_division=0),
                    'recall': recall_score(y, y_pred, average='macro', zero_division=0),
                    'f1_score': f1_score(y, y_pred, average='macro', zero_division=0),
                    'cv_mean': cv_scores['mean'],
                    'cv_std': cv_scores['std']
                }
            else:  # Regression
                performance = {
                    'mse': mean_squared_error(y, y_pred),
                    'mae': mean_absolute_error(y, y_pred),
                    'r2_score': r2_score(y, y_pred),
                    'cv_mean': cv_scores['mean'],
                    'cv_std': cv_scores['std']
                }

            return performance

        except Exception as e:
            logger.error(f"Performance calculation failed: {e}")
            return {}

    def _prepare_sequential_data(self, X: pd.DataFrame, y: pd.Series, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for sequential deep learning models."""
        X_seq = []
        y_seq = []

        for i in range(sequence_length, len(X)):
            X_seq.append(X.iloc[i-sequence_length:i].values)
            y_seq.append(y.iloc[i])

        return np.array(X_seq), np.array(y_seq)

    def _create_lstm_model(self, input_shape: Tuple[int, int]) -> Model:
        """Create LSTM-based neural network."""
        model = Sequential([
            LSTM(64, input_shape=input_shape, return_sequences=True),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='linear')  # Regression output
        ])

        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model

    def _create_cnn_model(self, input_shape: Tuple[int, int]) -> Model:
        """Create CNN-based neural network for time series."""
        model = Sequential([
            Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
            MaxPooling1D(pool_size=2),
            Conv1D(32, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(32, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='linear')
        ])

        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model

    def _create_hybrid_model(self, input_shape: Tuple[int, int]) -> Model:
        """Create hybrid CNN-LSTM model."""
        inputs = Input(shape=input_shape)

        # CNN layers
        conv1 = Conv1D(64, kernel_size=3, activation='relu')(inputs)
        pool1 = MaxPooling1D(pool_size=2)(conv1)

        # LSTM layers
        lstm1 = LSTM(64, return_sequences=True)(pool1)
        lstm2 = LSTM(32, return_sequences=False)(lstm1)

        # Dense layers
        dense1 = Dense(32, activation='relu')(lstm2)
        dropout1 = Dropout(0.3)(dense1)
        outputs = Dense(1, activation='linear')(dropout1)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

        return model

    def _train_deep_model(self, model: Model, X_train: np.ndarray, y_train: np.ndarray,
                         ticker: str, model_type: str) -> Dict:
        """Train deep learning model with callbacks."""
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ModelCheckpoint(
                filepath=str(self.model_dir / f"{ticker}_{model_type}_checkpoint.h5"),
                monitor='val_loss',
                save_best_only=True
            )
        ]

        history = model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=0
        )

        return history.history

    def _evaluate_deep_model(self, model: Model, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate deep learning model performance."""
        loss, mae = model.evaluate(X, y, verbose=0)
        y_pred = model.predict(X, verbose=0).flatten()

        return {
            'loss': loss,
            'mae': mae,
            'mse': mean_squared_error(y, y_pred),
            'r2_score': r2_score(y, y_pred)
        }

    def _save_model(self, model_key: str, model, scaler, performance: Dict):
        """Save trained model and metadata."""
        try:
            model_path = self.model_dir / f"{model_key}.joblib"
            joblib.dump({
                'model': model,
                'scaler': scaler,
                'performance': performance,
                'timestamp': datetime.now()
            }, model_path)

        except Exception as e:
            logger.error(f"Failed to save model {model_key}: {e}")

    def _load_model(self, model_key: str):
        """Load trained model from disk."""
        try:
            model_path = self.model_dir / f"{model_key}.joblib"
            if model_path.exists():
                saved_data = joblib.load(model_path)
                self.models[model_key] = saved_data['model']
                self.scalers[model_key] = saved_data['scaler']
                self.model_performance[model_key] = saved_data['performance']
            else:
                logger.warning(f"Model file not found: {model_path}")

        except Exception as e:
            logger.error(f"Failed to load model {model_key}: {e}")

    # Technical indicator calculations
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        """Calculate MACD indicator."""
        fast_ema = prices.ewm(span=fast).mean()
        slow_ema = prices.ewm(span=slow).mean()
        return fast_ema - slow_ema

    def _calculate_stochastic(self, df: pd.DataFrame, k_period: int = 14) -> pd.Series:
        """Calculate Stochastic Oscillator."""
        low_min = df['Low'].rolling(window=k_period).min()
        high_max = df['High'].rolling(window=k_period).max()
        return 100 * (df['Close'] - low_min) / (high_max - low_min)

    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, lower_band

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
