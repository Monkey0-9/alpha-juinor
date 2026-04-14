import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
from collections import defaultdict
import warnings

# ML imports
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import joblib

logger = logging.getLogger(__name__)

class AdaptiveLearningMode(Enum):
    ONLINE = "online"      # Continuous learning with each new data point
    BATCH = "batch"        # Periodic retraining on accumulated data
    HYBRID = "hybrid"      # Combination of online and batch learning

class MarketRegime(Enum):
    BULL = "bull_market"
    BEAR = "bear_market"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    CRISIS = "crisis"
    RECOVERY = "recovery"

@dataclass
class AdaptiveModel:
    """Represents an adaptive learning model."""
    model_id: str
    model_type: str
    created_at: datetime
    last_updated: datetime
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    market_regime: Optional[MarketRegime] = None
    is_active: bool = True
    adaptation_count: int = 0
    confidence_score: float = 0.0

@dataclass
class MarketCondition:
    """Represents current market conditions."""
    regime: MarketRegime
    volatility: float
    trend_strength: float
    volume_trend: float
    momentum: float
    detected_at: datetime
    confidence: float

class InstitutionalAdaptiveEngine:
    """
    INSTITUTIONAL-GRADE ADAPTIVE LEARNING ENGINE
    Dynamic model adaptation, market regime detection, and continuous learning.
    Automatically adjusts strategies based on changing market conditions.
    """

    def __init__(self, model_dir: str = "models/adaptive", cache_dir: str = "data/cache/adaptive"):
        self.model_dir = Path(model_dir)
        self.cache_dir = Path(cache_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Model registry
        self.models: Dict[str, AdaptiveModel] = {}
        self.active_models: Dict[str, Any] = {}

        # Market regime detection
        self.regime_detector = self._initialize_regime_detector()
        self.current_regime: Optional[MarketCondition] = None

        # Performance tracking
        self.performance_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.regime_performance: Dict[MarketRegime, Dict[str, float]] = {}

        # Adaptation parameters
        self.adaptation_threshold = 0.1  # Performance degradation threshold
        self.min_samples_for_adaptation = 100
        self.max_models_per_regime = 3

        # Learning mode
        self.learning_mode = AdaptiveLearningMode.HYBRID

        # Feature engineering
        self.feature_scaler = StandardScaler()
        self.feature_columns = [
            'returns', 'volatility', 'volume', 'momentum', 'trend_strength',
            'rsi', 'macd', 'bb_position', 'volume_ratio'
        ]

        logger.info("Institutional Adaptive Engine initialized")

    def adapt_strategy(self, symbol: str, market_data: pd.DataFrame,
                      current_positions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adapt trading strategy based on current market conditions and performance.
        Returns adaptation recommendations.
        """
        try:
            # 1. Detect current market regime
            regime = self._detect_market_regime(market_data)

            # 2. Evaluate current model performance
            performance_metrics = self._evaluate_model_performance(symbol, market_data)

            # 3. Check if adaptation is needed
            adaptation_needed = self._check_adaptation_needed(performance_metrics, regime)

            if not adaptation_needed:
                return {
                    'adaptation_needed': False,
                    'current_regime': regime.regime.value,
                    'reason': 'Performance within acceptable range'
                }

            # 4. Generate adaptation recommendations
            recommendations = self._generate_adaptation_recommendations(
                symbol, regime, performance_metrics, market_data, current_positions
            )

            # 5. Execute adaptation if confidence is high enough
            if recommendations['confidence'] > 0.7:
                self._execute_adaptation(symbol, recommendations)

            return recommendations

        except Exception as e:
            logger.error(f"Strategy adaptation failed for {symbol}: {e}")
            return {
                'adaptation_needed': False,
                'error': str(e),
                'fallback_action': 'maintain_current_strategy'
            }

    def online_learning_update(self, symbol: str, new_data: pd.DataFrame,
                             prediction_error: float) -> Dict[str, Any]:
        """
        Perform online learning update based on new market data and prediction errors.
        """
        try:
            if self.learning_mode not in [AdaptiveLearningMode.ONLINE, AdaptiveLearningMode.HYBRID]:
                return {'updated': False, 'reason': 'Online learning not enabled'}

            # Get current model
            model_key = f"{symbol}_online"
            if model_key not in self.active_models:
                return {'updated': False, 'reason': 'No active model found'}

            model = self.active_models[model_key]

            # Prepare features
            features = self._extract_features(new_data)

            # Online learning update (simplified - in production use proper online learning algorithms)
            if hasattr(model, 'partial_fit'):
                # For models that support online learning
                target = new_data['target'].values if 'target' in new_data.columns else np.zeros(len(features))
                model.partial_fit(features, target)

                # Update model metadata
                if model_key in self.models:
                    self.models[model_key].last_updated = datetime.utcnow()
                    self.models[model_key].adaptation_count += 1

                return {
                    'updated': True,
                    'model_key': model_key,
                    'update_type': 'online_learning',
                    'samples_processed': len(features)
                }
            else:
                return {'updated': False, 'reason': 'Model does not support online learning'}

        except Exception as e:
            logger.error(f"Online learning update failed for {symbol}: {e}")
            return {'updated': False, 'error': str(e)}

    def train_regime_specific_model(self, symbol: str, market_data: pd.DataFrame,
                                  regime: MarketRegime, target_horizon: int = 1) -> str:
        """
        Train a model specifically for a market regime.
        Returns model ID.
        """
        try:
            # Prepare training data
            X, y = self._prepare_training_data(market_data, target_horizon)

            # Create regime-specific model
            model = self._create_regime_model(regime)

            # Train model
            model.fit(X, y)

            # Evaluate performance
            performance = self._evaluate_model(model, X, y)

            # Create model metadata
            model_id = f"{symbol}_{regime.value}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

            adaptive_model = AdaptiveModel(
                model_id=model_id,
                model_type=f"{regime.value}_regressor",
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow(),
                performance_metrics=performance,
                market_regime=regime,
                confidence_score=performance.get('r2_score', 0)
            )

            # Store model
            self.models[model_id] = adaptive_model
            self.active_models[model_id] = model

            # Save model
            self._save_model(model_id, model, adaptive_model)

            logger.info(f"Trained regime-specific model: {model_id} with R²={performance.get('r2_score', 0):.3f}")

            return model_id

        except Exception as e:
            logger.error(f"Failed to train regime-specific model for {symbol}: {e}")
            return ""

    def get_optimal_model(self, symbol: str, market_data: pd.DataFrame) -> Tuple[str, float]:
        """
        Get the optimal model for current market conditions.
        Returns (model_id, confidence_score).
        """
        try:
            # Detect current regime
            regime = self._detect_market_regime(market_data)

            # Find models for this regime
            regime_models = [
                (model_id, model) for model_id, model in self.models.items()
                if model.market_regime == regime.regime and model.is_active
            ]

            if not regime_models:
                # Fallback to general models
                general_models = [
                    (model_id, model) for model_id, model in self.models.items()
                    if model.market_regime is None and model.is_active
                ]
                if general_models:
                    # Return best performing general model
                    best_model = max(general_models, key=lambda x: x[1].confidence_score)
                    return best_model[0], best_model[1].confidence_score

                return "", 0.0

            # Return best performing regime-specific model
            best_model = max(regime_models, key=lambda x: x[1].confidence_score)
            return best_model[0], best_model[1].confidence_score

        except Exception as e:
            logger.error(f"Failed to get optimal model for {symbol}: {e}")
            return "", 0.0

    def _detect_market_regime(self, market_data: pd.DataFrame) -> MarketCondition:
        """
        Detect current market regime using multiple indicators.
        """
        try:
            if len(market_data) < 50:
                return MarketCondition(
                    regime=MarketRegime.SIDEWAYS,
                    volatility=0.02,
                    trend_strength=0.0,
                    volume_trend=0.0,
                    momentum=0.0,
                    detected_at=datetime.utcnow(),
                    confidence=0.5
                )

            # Calculate trend strength
            recent_data = market_data.tail(50)
            returns = recent_data['Close'].pct_change(fill_method=None).dropna()

            # Linear regression for trend
            X = np.arange(len(recent_data)).reshape(-1, 1)
            y = recent_data['Close'].values
            trend_model = LinearRegression()
            trend_model.fit(X, y)
            trend_slope = trend_model.coef_[0]
            trend_strength = abs(trend_slope) / recent_data['Close'].mean()

            # Volatility calculation
            volatility = returns.std() * np.sqrt(252)  # Annualized

            # Volume trend
            volume_trend = recent_data['Volume'].pct_change().rolling(10).mean().iloc[-1]

            # Momentum (ROC)
            momentum = (recent_data['Close'].iloc[-1] / recent_data['Close'].iloc[-20] - 1)

            # Classify regime
            regime, confidence = self._classify_regime(trend_slope, volatility, momentum, volume_trend)

            condition = MarketCondition(
                regime=regime,
                volatility=volatility,
                trend_strength=trend_strength,
                volume_trend=volume_trend,
                momentum=momentum,
                detected_at=datetime.utcnow(),
                confidence=confidence
            )

            self.current_regime = condition
            return condition

        except Exception as e:
            logger.error(f"Market regime detection failed: {e}")
            return MarketCondition(
                regime=MarketRegime.SIDEWAYS,
                volatility=0.02,
                trend_strength=0.0,
                volume_trend=0.0,
                momentum=0.0,
                detected_at=datetime.utcnow(),
                confidence=0.3
            )

    def _classify_regime(self, trend_slope: float, volatility: float,
                        momentum: float, volume_trend: float) -> Tuple[MarketRegime, float]:
        """Classify market regime based on indicators."""
        # Trend-based classification
        if trend_slope > 0.001:  # Bull market
            if momentum > 0.05:
                regime = MarketRegime.BULL
                confidence = min(abs(trend_slope) * 1000, 1.0)
            else:
                regime = MarketRegime.RECOVERY
                confidence = 0.6
        elif trend_slope < -0.001:  # Bear market
            if momentum < -0.05:
                regime = MarketRegime.BEAR
                confidence = min(abs(trend_slope) * 1000, 1.0)
            else:
                regime = MarketRegime.CRISIS
                confidence = 0.7
        else:  # Sideways
            regime = MarketRegime.SIDEWAYS
            confidence = 0.5

        # Volatility adjustment
        if volatility > 0.04:  # High volatility
            if regime == MarketRegime.BEAR:
                regime = MarketRegime.CRISIS
                confidence = max(confidence, 0.8)
            else:
                regime = MarketRegime.HIGH_VOLATILITY
                confidence = min(confidence + 0.2, 1.0)
        elif volatility < 0.015:  # Low volatility
            regime = MarketRegime.LOW_VOLATILITY
            confidence = min(confidence + 0.1, 1.0)

        return regime, confidence

    def _evaluate_model_performance(self, symbol: str, market_data: pd.DataFrame) -> Dict[str, float]:
        """Evaluate current model performance."""
        try:
            # Get recent predictions and actuals
            recent_data = market_data.tail(100)

            if len(recent_data) < 20:
                return {'sharpe_ratio': 0, 'max_drawdown': 0, 'win_rate': 0.5}

            returns = recent_data['Close'].pct_change(fill_method=None).dropna()

            # Calculate performance metrics
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

            # Max drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()

            # Win rate (simplified)
            win_rate = (returns > 0).mean()

            return {
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'total_return': (recent_data['Close'].iloc[-1] / recent_data['Close'].iloc[0] - 1),
                'volatility': returns.std() * np.sqrt(252)
            }

        except Exception as e:
            logger.error(f"Model performance evaluation failed: {e}")
            return {'sharpe_ratio': 0, 'max_drawdown': 0, 'win_rate': 0.5}

    def _check_adaptation_needed(self, performance: Dict[str, float],
                               regime: MarketCondition) -> bool:
        """Check if strategy adaptation is needed."""
        # Check performance degradation
        sharpe_ratio = performance.get('sharpe_ratio', 0)
        max_drawdown = performance.get('max_drawdown', 0)

        # Adaptation triggers
        poor_performance = sharpe_ratio < 0.5
        excessive_drawdown = max_drawdown < -0.15
        regime_change = (self.current_regime and
                        self.current_regime.regime != regime.regime)

        return poor_performance or excessive_drawdown or regime_change

    def _generate_adaptation_recommendations(self, symbol: str, regime: MarketCondition,
                                           performance: Dict[str, float], market_data: pd.DataFrame,
                                           current_positions: Dict[str, Any]) -> Dict[str, Any]:
        """Generate adaptation recommendations."""
        recommendations = {
            'symbol': symbol,
            'current_regime': regime.regime.value,
            'adaptation_needed': True,
            'recommended_actions': [],
            'confidence': 0.0,
            'expected_improvement': 0.0,
            'risk_assessment': 'medium'
        }

        # Analyze current issues
        issues = []
        if performance.get('sharpe_ratio', 0) < 0.5:
            issues.append('poor_risk_adjusted_returns')
        if performance.get('max_drawdown', 0) < -0.15:
            issues.append('excessive_drawdown')
        if regime.confidence > 0.8:
            issues.append('regime_change')

        # Generate specific recommendations
        if 'regime_change' in issues:
            recommendations['recommended_actions'].append({
                'action': 'switch_model',
                'reason': f'Market regime changed to {regime.regime.value}',
                'new_model_type': f'{regime.regime.value}_optimized'
            })

        if 'poor_risk_adjusted_returns' in issues:
            recommendations['recommended_actions'].append({
                'action': 'reduce_position_size',
                'reason': 'Poor risk-adjusted performance',
                'new_position_size': 0.5  # 50% reduction
            })

        if 'excessive_drawdown' in issues:
            recommendations['recommended_actions'].append({
                'action': 'implement_stop_loss',
                'reason': 'Excessive drawdown detected',
                'stop_loss_level': 0.05  # 5% stop loss
            })

        # Calculate confidence and expected improvement
        recommendations['confidence'] = min(len(issues) * 0.3, 0.9)
        recommendations['expected_improvement'] = len(issues) * 0.1  # 10% per issue addressed

        # Risk assessment
        if regime.regime in [MarketRegime.CRISIS, MarketRegime.HIGH_VOLATILITY]:
            recommendations['risk_assessment'] = 'high'

        return recommendations

    def _execute_adaptation(self, symbol: str, recommendations: Dict[str, Any]):
        """Execute the recommended adaptations."""
        try:
            for action in recommendations.get('recommended_actions', []):
                action_type = action.get('action')

                if action_type == 'switch_model':
                    # Train new regime-specific model
                    market_data = self._get_recent_market_data(symbol)
                    if market_data is not None:
                        model_id = self.train_regime_specific_model(
                            symbol, market_data, self.current_regime.regime
                        )
                        logger.info(f"Switched to new model: {model_id}")

                elif action_type == 'reduce_position_size':
                    # This would integrate with portfolio manager
                    logger.info(f"Position size reduction recommended for {symbol}")

                elif action_type == 'implement_stop_loss':
                    # This would integrate with risk manager
                    logger.info(f"Stop loss implementation recommended for {symbol}")

        except Exception as e:
            logger.error(f"Adaptation execution failed for {symbol}: {e}")

    def _extract_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract features for model training."""
        try:
            features = []

            # Price-based features
            returns = data['Close'].pct_change()
            features.append(returns.fillna(0))

            # Volatility
            volatility = returns.rolling(20).std()
            features.append(volatility.fillna(0))

            # Volume
            volume_ratio = data['Volume'] / data['Volume'].rolling(20).mean()
            features.append(volume_ratio.fillna(1))

            # Momentum
            momentum = data['Close'] / data['Close'].shift(10) - 1
            features.append(momentum.fillna(0))

            # Trend strength (simplified)
            trend = np.sign(data['Close'] - data['Close'].shift(20))
            features.append(trend.fillna(0))

            # Combine features
            feature_matrix = np.column_stack(features)

            # Scale features
            if hasattr(self, 'feature_scaler') and self.feature_scaler:
                feature_matrix = self.feature_scaler.fit_transform(feature_matrix)

            return feature_matrix

        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return np.array([])

    def _prepare_training_data(self, market_data: pd.DataFrame, target_horizon: int) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data with features and targets."""
        # Extract features
        X = self._extract_features(market_data)

        # Create target (future returns)
        returns = market_data['Close'].pct_change(target_horizon).shift(-target_horizon)
        y = returns.fillna(0).values

        # Remove NaN values
        valid_idx = ~np.isnan(y)
        X = X[valid_idx]
        y = y[valid_idx]

        return X, y

    def _create_regime_model(self, regime: MarketRegime) -> Any:
        """Create a model suitable for the given market regime."""
        if regime in [MarketRegime.HIGH_VOLATILITY, MarketRegime.CRISIS]:
            # Use robust models for volatile regimes
            return GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=3,
                random_state=42
            )
        elif regime in [MarketRegime.BULL, MarketRegime.BEAR]:
            # Use complex models for trending regimes
            return RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_split=50,
                random_state=42
            )
        else:
            # Default model for sideways/low volatility
            return LinearRegression()

    def _evaluate_model(self, model: Any, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance."""
        try:
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []

            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                model_cv = model.__class__(**model.get_params())
                model_cv.fit(X_train, y_train)
                y_pred = model_cv.predict(X_test)

                mse = mean_squared_error(y_test, y_pred)
                scores.append(mse)

            # Final fit for metrics
            model.fit(X, y)
            y_pred = model.predict(X)

            return {
                'mse': mean_squared_error(y, y_pred),
                'rmse': np.sqrt(mean_squared_error(y, y_pred)),
                'r2_score': r2_score(y, y_pred),
                'cv_mse_mean': np.mean(scores),
                'cv_mse_std': np.std(scores)
            }

        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            return {'mse': float('inf'), 'r2_score': 0}

    def _save_model(self, model_id: str, model: Any, metadata: AdaptiveModel):
        """Save model and metadata."""
        try:
            model_path = self.model_dir / f"{model_id}.joblib"
            metadata_path = self.model_dir / f"{model_id}_metadata.json"

            # Save model
            joblib.dump(model, model_path)

            # Save metadata
            with open(metadata_path, 'w') as f:
                json.dump({
                    'model_id': metadata.model_id,
                    'model_type': metadata.model_type,
                    'created_at': metadata.created_at.isoformat(),
                    'last_updated': metadata.last_updated.isoformat(),
                    'performance_metrics': metadata.performance_metrics,
                    'market_regime': metadata.market_regime.value if metadata.market_regime else None,
                    'adaptation_count': metadata.adaptation_count,
                    'confidence_score': metadata.confidence_score
                }, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save model {model_id}: {e}")

    def _load_model(self, model_id: str) -> Tuple[Any, AdaptiveModel]:
        """Load model and metadata."""
        try:
            model_path = self.model_dir / f"{model_id}.joblib"
            metadata_path = self.model_dir / f"{model_id}_metadata.json"

            if not model_path.exists() or not metadata_path.exists():
                return None, None

            # Load model
            model = joblib.load(model_path)

            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata_dict = json.load(f)

            metadata = AdaptiveModel(
                model_id=metadata_dict['model_id'],
                model_type=metadata_dict['model_type'],
                created_at=datetime.fromisoformat(metadata_dict['created_at']),
                last_updated=datetime.fromisoformat(metadata_dict['last_updated']),
                performance_metrics=metadata_dict['performance_metrics'],
                market_regime=MarketRegime(metadata_dict['market_regime']) if metadata_dict['market_regime'] else None,
                adaptation_count=metadata_dict['adaptation_count'],
                confidence_score=metadata_dict['confidence_score']
            )

            return model, metadata

        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            return None, None

    def _get_recent_market_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get recent market data for a symbol (placeholder)."""
        # This would integrate with data router
        # For now, return None
        return None

    def _initialize_regime_detector(self) -> Any:
        """Initialize market regime detection model."""
        # Placeholder - in production would train a regime detection model
        return None
