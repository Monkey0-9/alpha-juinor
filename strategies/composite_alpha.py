"""
Composite Alpha Strategy - Intelligent combination of multiple alpha models.

Combines signals from various alpha families with regime-aware weighting,
risk-adjusted scaling, and performance-based adaptation.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import pandas as pd
import numpy as np

from risk.engine import RiskManager
from alpha_families.registry import get_alpha_families
from alpha_families.base_alpha import BaseAlpha

logger = logging.getLogger(__name__)

@dataclass
class CompositeSignal:
    """Container for composite alpha signals."""
    combined_signal: float
    confidence: float
    alpha_contributions: Dict[str, float] = field(default_factory=dict)
    regime_adjusted: bool = False
    risk_adjusted: bool = False
    timestamp: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class CompositeAlphaStrategy:
    """
    Intelligent alpha combination strategy.

    Features:
    - Dynamic alpha weighting based on market regimes
    - Risk-adjusted signal scaling
    - Performance-based alpha adaptation
    - Institutional-grade signal combination
    """

    def __init__(
        self,
        risk_manager: RiskManager,
        regime_model: Optional[Any] = None,
        max_alpha_weight: float = 0.25,
        min_alpha_weight: float = 0.02,
        adaptation_window: int = 100
    ):
        """
        Initialize composite alpha strategy.

        Args:
            risk_manager: Risk management engine
            regime_model: Optional market regime detection model
            max_alpha_weight: Maximum weight for any single alpha
            min_alpha_weight: Minimum weight for any single alpha
            adaptation_window: Lookback window for performance adaptation
        """
        self.risk_manager = risk_manager
        self.regime_model = regime_model
        self.max_alpha_weight = max_alpha_weight
        self.min_alpha_weight = min_alpha_weight
        self.adaptation_window = adaptation_window

        # Initialize alpha families
        self.alpha_families = get_alpha_families()
        self.alpha_weights = self._initialize_base_weights()

        # Regime-based weight multipliers
        self.regime_multipliers = {
            'BULL_QUIET': {
                'MomentumTS': 1.2,
                'FundamentalAlpha': 1.1,
                'StatisticalAlpha': 0.9,
                'AlternativeAlpha': 0.8,
                'MLAlpha': 1.0
            },
            'BULL_VOLATILE': {
                'MomentumTS': 0.8,
                'FundamentalAlpha': 0.9,
                'StatisticalAlpha': 1.2,
                'AlternativeAlpha': 1.1,
                'MLAlpha': 1.0
            },
            'BEAR_QUIET': {
                'MomentumTS': 0.5,
                'FundamentalAlpha': 1.3,
                'StatisticalAlpha': 1.1,
                'AlternativeAlpha': 0.9,
                'MLAlpha': 0.8
            },
            'BEAR_CRISIS': {
                'MomentumTS': 0.3,
                'FundamentalAlpha': 1.4,
                'StatisticalAlpha': 0.8,
                'AlternativeAlpha': 0.7,
                'MLAlpha': 0.6
            },
            'HIGH_VOL': {
                'MomentumTS': 0.6,
                'FundamentalAlpha': 1.2,
                'StatisticalAlpha': 1.3,
                'AlternativeAlpha': 0.8,
                'MLAlpha': 1.1
            },
            'LOW_VOL': {
                'MomentumTS': 1.1,
                'FundamentalAlpha': 1.0,
                'StatisticalAlpha': 0.9,
                'AlternativeAlpha': 1.2,
                'MLAlpha': 1.0
            }
        }

        # Performance tracking
        self.signal_history: List[CompositeSignal] = []
        self.alpha_performance: Dict[str, List[float]] = {}

        logger.info(f"Initialized CompositeAlphaStrategy with {len(self.alpha_families)} alpha families")

    def _initialize_base_weights(self) -> Dict[str, float]:
        """Initialize base weights for alpha families."""
        base_weights = {
            'MomentumTS': 0.15,
            'MeanReversionAlpha': 0.12,
            'VolatilityCarry': 0.10,
            'TrendStrength': 0.13,
            'FundamentalAlpha': 0.11,
            'StatisticalAlpha': 0.12,
            'AlternativeAlpha': 0.09,
            'MLAlpha': 0.10,
            'SentimentAlpha': 0.08  # When implemented
        }

        # Filter to only include available alphas
        available_names = [type(alpha).__name__ for alpha in self.alpha_families]
        return {name: base_weights.get(name, 0.1) for name in available_names}

    def _calculate_regime_weights(self, regime_context: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate regime-adjusted alpha weights.

        Args:
            regime_context: Current market regime information

        Returns:
            Dict of alpha name to adjusted weight
        """
        regime_tag = regime_context.get('regime_tag', 'NORMAL')
        multipliers = self.regime_multipliers.get(regime_tag, {})

        adjusted_weights = {}
        for alpha_name, base_weight in self.alpha_weights.items():
            multiplier = multipliers.get(alpha_name, 1.0)
            adjusted_weight = base_weight * multiplier

            # Clip to bounds
            adjusted_weight = np.clip(adjusted_weight, self.min_alpha_weight, self.max_alpha_weight)
            adjusted_weights[alpha_name] = adjusted_weight

        # Renormalize to sum to 1.0
        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:
            adjusted_weights = {k: v / total_weight for k, v in adjusted_weights.items()}

        return adjusted_weights

    def _combine_signals(
        self,
        alpha_signals: Dict[str, Dict[str, Any]],
        regime_weights: Dict[str, float]
    ) -> CompositeSignal:
        """
        Combine individual alpha signals into composite signal.

        Args:
            alpha_signals: Dict of alpha name to signal dict
            regime_weights: Regime-adjusted weights

        Returns:
            CompositeSignal object
        """
        weighted_signals = []
        total_weight = 0.0
        contributions = {}

        for alpha_name, signal_data in alpha_signals.items():
            if alpha_name not in regime_weights:
                continue

            signal = signal_data.get('signal', 0.0)
            confidence = signal_data.get('confidence', 0.5)
            weight = regime_weights[alpha_name]

            # Weight by both regime weight and confidence
            effective_weight = weight * confidence
            weighted_signals.append(signal * effective_weight)
            total_weight += effective_weight
            contributions[alpha_name] = signal

        # Calculate combined signal
        if total_weight > 0:
            combined_signal = sum(weighted_signals) / total_weight
        else:
            combined_signal = 0.0

        # Calculate overall confidence
        confidences = [sig.get('confidence', 0.5) for sig in alpha_signals.values()]
        confidence = np.mean(confidences) if confidences else 0.5

        return CompositeSignal(
            combined_signal=combined_signal,
            confidence=confidence,
            alpha_contributions=contributions,
            regime_adjusted=True
        )

    def _apply_risk_adjustment(self, signal: CompositeSignal, current_risk: float) -> CompositeSignal:
        """
        Apply risk-based signal adjustment.

        Args:
            signal: Original composite signal
            current_risk: Current portfolio risk level

        Returns:
            Risk-adjusted signal
        """
        # Scale down signals when risk is high
        risk_thresholds = [0.05, 0.10, 0.15, 0.20]  # Risk levels
        scale_factors = [1.0, 0.8, 0.6, 0.4]  # Corresponding scale factors

        scale_factor = 1.0
        for threshold, factor in zip(risk_thresholds, scale_factors):
            if current_risk >= threshold:
                scale_factor = factor
            else:
                break

        signal.combined_signal *= scale_factor
        signal.confidence *= scale_factor
        signal.risk_adjusted = True
        signal.metadata['risk_adjustment'] = scale_factor

        return signal

    def _update_alpha_weights(self, performance_scores: Dict[str, float]):
        """
        Update alpha weights based on recent performance.

        Args:
            performance_scores: Dict of alpha name to performance score
        """
        adaptation_rate = 0.1  # How much to adapt weights

        for alpha_name, score in performance_scores.items():
            if alpha_name in self.alpha_weights:
                current_weight = self.alpha_weights[alpha_name]

                # Increase weight for good performers, decrease for poor
                if score > 0:
                    new_weight = current_weight * (1 + adaptation_rate * score)
                else:
                    new_weight = current_weight * (1 + adaptation_rate * score)

                # Clip to bounds
                new_weight = np.clip(new_weight, self.min_alpha_weight, self.max_alpha_weight)
                self.alpha_weights[alpha_name] = new_weight

        # Renormalize
        total_weight = sum(self.alpha_weights.values())
        if total_weight > 0:
            self.alpha_weights = {k: v / total_weight for k, v in self.alpha_weights.items()}

    def generate_signal(
        self,
        market_data: pd.DataFrame,
        regime_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate composite alpha signal.

        Args:
            market_data: OHLCV market data
            regime_context: Current market regime context

        Returns:
            Dict containing signal, confidence, and metadata
        """
        regime_context = regime_context or {}

        try:
            # Calculate regime-adjusted weights
            regime_weights = self._calculate_regime_weights(regime_context)

            # Generate signals from all alphas
            alpha_signals = {}
            for alpha in self.alpha_families:
                try:
                    signal_result = alpha.generate_signal(market_data, regime_context)
                    alpha_signals[type(alpha).__name__] = signal_result
                except Exception as e:
                    logger.warning(f"Alpha {type(alpha).__name__} failed: {e}")
                    alpha_signals[type(alpha).__name__] = {'signal': 0.0, 'confidence': 0.0}

            # Combine signals
            composite_signal = self._combine_signals(alpha_signals, regime_weights)

            # Apply risk adjustment
            current_risk = self._estimate_current_risk()
            composite_signal = self._apply_risk_adjustment(composite_signal, current_risk)

            # Store in history
            composite_signal.timestamp = pd.Timestamp.now().timestamp()
            self.signal_history.append(composite_signal)

            # Keep history bounded
            if len(self.signal_history) > self.adaptation_window:
                self.signal_history = self.signal_history[-self.adaptation_window:]

            return {
                'signal': composite_signal.combined_signal,
                'confidence': composite_signal.confidence,
                'metadata': {
                    'alpha_contributions': composite_signal.alpha_contributions,
                    'regime_adjusted': composite_signal.regime_adjusted,
                    'risk_adjusted': composite_signal.risk_adjusted,
                    'regime_weights': regime_weights,
                    'timestamp': composite_signal.timestamp
                }
            }

        except Exception as e:
            logger.error(f"Composite alpha signal generation failed: {e}")
            return {
                'signal': 0.0,
                'confidence': 0.0,
                'metadata': {'error': str(e)}
            }

    def _estimate_current_risk(self) -> float:
        """Estimate current portfolio risk level."""
        # Simplified risk estimation - in practice would use VaR, CVaR, etc.
        try:
            if hasattr(self.risk_manager, '_current_var'):
                return self.risk_manager._current_var
            else:
                return 0.05  # Default 5% risk
        except:
            return 0.05

    def get_alpha_performance(self) -> Dict[str, Dict[str, float]]:
        """Get performance statistics for each alpha."""
        performance_stats = {}

        for alpha_name in self.alpha_weights.keys():
            signals = [s.alpha_contributions.get(alpha_name, 0) for s in self.signal_history[-50:]]
            if signals:
                performance_stats[alpha_name] = {
                    'mean_signal': np.mean(signals),
                    'signal_std': np.std(signals),
                    'signal_count': len(signals),
                    'current_weight': self.alpha_weights[alpha_name]
                }
            else:
                performance_stats[alpha_name] = {
                    'mean_signal': 0.0,
                    'signal_std': 0.0,
                    'signal_count': 0,
                    'current_weight': self.alpha_weights[alpha_name]
                }

        return performance_stats
