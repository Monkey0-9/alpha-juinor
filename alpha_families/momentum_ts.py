"""
Momentum Time Series Alpha Family.

Generates signals based on time-series momentum: price trends over multiple horizons.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from .base_alpha import BaseAlpha


class MomentumTS(BaseAlpha):
    """
    Time-series momentum alpha: long/short based on recent price trends.
    """

    def __init__(self, horizons: list = [1, 3, 5, 10, 20]):
        super().__init__()
        self.horizons = horizons

    def generate_signal(self, data: pd.DataFrame, regime_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate momentum signal.

        Args:
            data: OHLCV data
            regime_context: Current market regime

        Returns:
            Signal dict with weights and confidence
        """
        if len(data) < max(self.horizons) + 1:
            return {'signal': 0.0, 'confidence': 0.0, 'weights': {}}

        # Calculate momentum for each horizon
        momentum_signals = {}
        for h in self.horizons:
            returns = data['Close'].pct_change(h).iloc[-1]
            momentum_signals[f'momentum_{h}d'] = returns

        # Ensemble: weighted average of horizons
        weights = np.array([1.0 / len(self.horizons)] * len(self.horizons))
        ensemble_signal = np.average(list(momentum_signals.values()), weights=weights)

        # Confidence based on signal consistency
        signal_std = np.std(list(momentum_signals.values()))
        confidence = max(0, 1.0 - signal_std)  # Higher confidence if signals agree

        # Regime adjustment: stronger in trend regimes
        regime_multiplier = 1.0
        if regime_context and regime_context.get('regime_tag') == 'TREND':
            regime_multiplier = 1.5
        elif regime_context and regime_context.get('regime_tag') == 'MEAN_REVERSION':
            regime_multiplier = 0.5

        final_signal = ensemble_signal * regime_multiplier

        return {
            'signal': np.clip(final_signal, -1.0, 1.0),
            'confidence': confidence,
            'weights': momentum_signals
        }
