"""
Mixture of Experts (MoE) - Regime-Specific Model Routing
===========================================================

Intelligent model selection based on market regime.

Each "expert" specializes in a specific market condition:
- Trending markets → Momentum expert
- Mean-reverting markets → Statistical arbitrage expert
- High volatility → Risk-averse expert
- Low liquidity → Patient execution expert

A gating network learns which expert to trust in each regime.

References:
- Jacobs, R. A., et al. (1991). "Adaptive mixtures of local experts"
- Shazeer, N., et al. (2017). "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer"
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classifications."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    MEAN_REVERTING = "mean_reverting"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    HIGH_LIQUIDITY = "high_liquidity"
    LOW_LIQUIDITY = "low_liquidity"
    CRISIS = "crisis"
    NORMAL = "normal"


@dataclass
class MoEPrediction:
    """Mixture of Experts prediction."""
    prediction: float
    confidence: float
    active_regime: MarketRegime
    expert_predictions: Dict[str, float]
    expert_weights: Dict[str, float]
    gating_scores: Dict[str, float]


class BaseExpert:
    """Base class for expert models."""

    def __init__(self, name: str):
        self.name = name
        self.performance_history = []

    def predict(self, features: np.ndarray, regime: MarketRegime) -> float:
        """Generate prediction for given features and regime."""
        raise NotImplementedError

    def update(self, actual: float, predicted: float):
        """Update expert based on prediction error."""
        error = abs(actual - predicted)
        self.performance_history.append(error)
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]

    def get_performance_score(self) -> float:
        """Get recent performance score (lower error = higher score)."""
        if len(self.performance_history) == 0:
            return 0.5
        avg_error = np.mean(self.performance_history)
        return 1.0 / (1.0 + avg_error)


class MomentumExpert(BaseExpert):
    """Expert for trending markets."""

    def __init__(self):
        super().__init__("momentum")
        self.lookback = 20

    def predict(self, features: np.ndarray, regime: MarketRegime) -> float:
        """Momentum-based prediction."""
        if len(features) < self.lookback:
            return 0.0

        # Calculate momentum
        recent_returns = features[-self.lookback:, 0] if features.ndim > 1 else features[-self.lookback:]
        momentum = np.mean(recent_returns)

        # Scale by regime confidence
        if regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]:
            scale = 1.5  # High confidence in trends
        else:
            scale = 0.5  # Low confidence in non-trending regimes

        return float(np.clip(momentum * scale, -1, 1))


class MeanReversionExpert(BaseExpert):
    """Expert for mean-reverting markets."""

    def __init__(self):
        super().__init__("mean_reversion")
        self.lookback = 20

    def predict(self, features: np.ndarray, regime: MarketRegime) -> float:
        """Mean reversion prediction."""
        if len(features) < self.lookback:
            return 0.0

        # Calculate z-score
        recent_prices = features[-self.lookback:, 0] if features.ndim > 1 else features[-self.lookback:]
        mean = np.mean(recent_prices)
        std = np.std(recent_prices)

        if std < 1e-10:
            return 0.0

        current_price = recent_prices[-1]
        z_score = (current_price - mean) / std

        # Mean reversion signal (negative z-score → buy, positive → sell)
        signal = -z_score * 0.3

        # Scale by regime
        if regime == MarketRegime.MEAN_REVERTING:
            scale = 1.5
        else:
            scale = 0.5

        return float(np.clip(signal * scale, -1, 1))


class VolatilityAdaptiveExpert(BaseExpert):
    """Expert for high volatility regimes."""

    def __init__(self):
        super().__init__("volatility_adaptive")
        self.lookback = 20

    def predict(self, features: np.ndarray, regime: MarketRegime) -> float:
        """Volatility-adjusted prediction."""
        if len(features) < self.lookback:
            return 0.0

        recent_returns = features[-self.lookback:, 0] if features.ndim > 1 else features[-self.lookback:]

        # Calculate realized volatility
        vol = np.std(recent_returns)

        # In high vol, be more conservative
        if regime == MarketRegime.HIGH_VOLATILITY:
            # Reduce position size, focus on safe assets
            mean_return = np.mean(recent_returns)
            signal = mean_return * 0.3  # Conservative
        else:
            # Low vol → can be more aggressive
            mean_return = np.mean(recent_returns)
            signal = mean_return * 1.2

        return float(np.clip(signal, -1, 1))


class LiquidityExpert(BaseExpert):
    """Expert for liquidity-constrained environments."""

    def __init__(self):
        super().__init__("liquidity")
        self.lookback = 10

    def predict(self, features: np.ndarray, regime: MarketRegime) -> float:
        """Liquidity-aware prediction."""
        if len(features) < self.lookback:
            return 0.0

        # Simple volume-based signal
        if features.ndim > 1 and features.shape[1] > 1:
            recent_volume = features[-self.lookback:, 1]  # Assuming column 1 is volume
            avg_volume = np.mean(recent_volume)
            current_volume = recent_volume[-1]

            volume_ratio = current_volume / (avg_volume + 1e-10)
        else:
            volume_ratio = 1.0

        # In low liquidity, be very conservative
        if regime == MarketRegime.LOW_LIQUIDITY:
            signal = 0.1  # Almost flat
        else:
            # High liquidity → normal trading
            recent_returns = features[-self.lookback:, 0] if features.ndim > 1 else features[-self.lookback:]
            signal = np.mean(recent_returns)

        return float(np.clip(signal, -1, 1))


class GatingNetwork:
    """
    Gating network that selects which experts to use.

    Learns from regime features to predict expert performance.
    """

    def __init__(self, n_experts: int):
        self.n_experts = n_experts

        # Simple linear gating (can be upgraded to neural network)
        self.weights = np.ones(n_experts) / n_experts
        self.learning_rate = 0.01

    def compute_gates(self, regime_features: np.ndarray, expert_scores: np.ndarray) -> np.ndarray:
        """
        Compute gating weights for each expert.

        Args:
            regime_features: Features describing current regime
            expert_scores: Historical performance scores of experts

        Returns:
            Gating weights (sum to 1)
        """
        # Combine static weights with performance scores
        gates = self.weights * expert_scores

        # Softmax normalization
        gates = np.exp(gates - np.max(gates))
        gates = gates / (np.sum(gates) + 1e-10)

        return gates

    def update(self, expert_errors: np.ndarray):
        """
        Update gating weights based on expert errors.

        Uses gradient descent to favor low-error experts.
        """
        # Inverse error = performance
        performance = 1.0 / (expert_errors + 0.01)

        # Gradient update
        self.weights += self.learning_rate * (performance - self.weights)

        # Normalize
        self.weights = np.clip(self.weights, 0.01, 10.0)
        self.weights /= np.sum(self.weights)


class MixtureOfExperts:
    """
    Mixture of Experts system for regime-aware trading.

    Automatically routes predictions to the most appropriate expert
    based on detected market regime.
    """

    def __init__(self):
        # Initialize experts
        self.experts = {
            'momentum': MomentumExpert(),
            'mean_reversion': MeanReversionExpert(),
            'volatility_adaptive': VolatilityAdaptiveExpert(),
            'liquidity': LiquidityExpert()
        }

        # Gating network
        self.gating = GatingNetwork(len(self.experts))

        # Regime detector (simplified)
        self.regime_detector = SimpleRegimeDetector()

        logger.info(f"MixtureOfExperts initialized with {len(self.experts)} experts")

    def predict(self, features: np.ndarray) -> MoEPrediction:
        """
        Generate prediction using mixture of experts.

        Args:
            features: Input features [timesteps, n_features]

        Returns:
            MoEPrediction with weighted expert predictions
        """
        # Detect regime
        regime = self.regime_detector.detect(features)

        # Get expert predictions
        expert_predictions = {}
        for name, expert in self.experts.items():
            expert_predictions[name] = expert.predict(features, regime)

        # Get expert performance scores
        expert_scores = np.array([
            expert.get_performance_score()
            for expert in self.experts.values()
        ])

        # Compute gating weights
        regime_features = self.regime_detector.get_regime_features(features)
        gates = self.gating.compute_gates(regime_features, expert_scores)

        # Weighted prediction
        pred_values = np.array(list(expert_predictions.values()))
        final_prediction = float(np.dot(gates, pred_values))

        # Confidence based on expert agreement
        pred_std = np.std(pred_values)
        confidence = float(1.0 - np.clip(pred_std, 0, 1))

        # Create expert weights dict
        expert_weights = {
            name: float(gates[i])
            for i, name in enumerate(self.experts.keys())
        }

        gating_scores = {
            name: float(expert_scores[i])
            for i, name in enumerate(self.experts.keys())
        }

        return MoEPrediction(
            prediction=final_prediction,
            confidence=confidence,
            active_regime=regime,
            expert_predictions=expert_predictions,
            expert_weights=expert_weights,
            gating_scores=gating_scores
        )

    def update(self, actual: float, prediction: MoEPrediction):
        """
        Update experts and gating based on actual outcome.

        Args:
            actual: Actual return/outcome
            prediction: Previous MoEPrediction
        """
        # Update each expert
        expert_errors = []
        for name, expert in self.experts.items():
            pred = prediction.expert_predictions[name]
            expert.update(actual, pred)
            error = abs(actual - pred)
            expert_errors.append(error)

        # Update gating network
        self.gating.update(np.array(expert_errors))

    def get_dominant_expert(self, prediction: MoEPrediction) -> str:
        """Get the dominant expert for current prediction."""
        return max(prediction.expert_weights, key=prediction.expert_weights.get)


class SimpleRegimeDetector:
    """Simple regime detector using statistical features."""

    def __init__(self, lookback: int = 60):
        self.lookback = lookback

    def detect(self, features: np.ndarray) -> MarketRegime:
        """Detect current market regime."""
        if len(features) < self.lookback:
            return MarketRegime.NORMAL

        recent_returns = features[-self.lookback:, 0] if features.ndim > 1 else features[-self.lookback:]

        # Calculate regime indicators
        mean_return = np.mean(recent_returns)
        vol = np.std(recent_returns)

        # Hurst exponent (simplified)
        hurst = self._compute_hurst(recent_returns)

        # Regime classification
        if vol > 0.03:  # High vol threshold
            if mean_return < -0.02:
                return MarketRegime.CRISIS
            else:
                return MarketRegime.HIGH_VOLATILITY
        elif vol < 0.01:
            return MarketRegime.LOW_VOLATILITY
        elif hurst > 0.6:  # Trending
            if mean_return > 0:
                return MarketRegime.TRENDING_UP
            else:
                return MarketRegime.TRENDING_DOWN
        elif hurst < 0.4:  # Mean reverting
            return MarketRegime.MEAN_REVERTING
        else:
            return MarketRegime.NORMAL

    def get_regime_features(self, features: np.ndarray) -> np.ndarray:
        """Extract regime features for gating network."""
        if len(features) < self.lookback:
            return np.zeros(4)

        recent_returns = features[-self.lookback:, 0] if features.ndim > 1 else features[-self.lookback:]

        regime_features = np.array([
            np.mean(recent_returns),  # Trend
            np.std(recent_returns),   # Volatility
            self._compute_hurst(recent_returns),  # Persistence
            np.mean(np.abs(np.diff(recent_returns)))  # Choppiness
        ])

        return regime_features

    def _compute_hurst(self, returns: np.ndarray, max_lag: int = 20) -> float:
        """Simplified Hurst exponent."""
        try:
            lags = range(2, min(max_lag, len(returns) // 2))
            tau = []

            for lag in lags:
                # R/S statistic
                chunks = len(returns) // lag
                rs_values = []

                for i in range(chunks):
                    chunk = returns[i*lag:(i+1)*lag]
                    if len(chunk) > 1:
                        mean_adj = chunk - chunk.mean()
                        cum_sum = np.cumsum(mean_adj)
                        R = cum_sum.max() - cum_sum.min()
                        S = chunk.std()
                        if S > 0:
                            rs_values.append(R / S)

                if rs_values:
                    tau.append(np.mean(rs_values))

            if len(tau) > 1:
                log_lags = np.log(list(lags[:len(tau)]))
                log_tau = np.log(tau)
                slope = np.polyfit(log_lags, log_tau, 1)[0]
                return float(np.clip(slope, 0, 1))
        except:
            pass

        return 0.5


# Global singleton
_moe: Optional[MixtureOfExperts] = None


def get_mixture_of_experts() -> MixtureOfExperts:
    """Get or create global MoE instance."""
    global _moe
    if _moe is None:
        _moe = MixtureOfExperts()
    return _moe
