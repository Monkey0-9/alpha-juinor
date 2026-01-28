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

    def generate_signal(self, data: pd.DataFrame, regime_context: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
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
            returns = data['Close'].pct_change(h, fill_method=None).iloc[-1]
            momentum_signals[f'momentum_{h}d'] = returns

        # Ensemble: weighted average of horizons
        weights = np.array([1.0 / len(self.horizons)] * len(self.horizons))
        ensemble_signal = np.average(list(momentum_signals.values()), weights=weights)

        # Confidence based on signal consistency
        signal_std = np.std(list(momentum_signals.values()))
        # Raw confidence from consistency
        raw_confidence = max(0, 1.0 - signal_std)

        # NORMALIZE USING HISTORY
        # We need historical ensemble signals to normalize properly.
        # Computing history on the fly for ensemble is expensive.
        # Simplified approach: Use the single-series normalization on the *longest* horizon as a proxy,
        # OR just normalize the current scalar assuming a distribution if we lack history.
        # BETTER: Compute one primary horizon's rolling Z and use that.

        # Let's use longest horizon (20d) for normalization reference or re-compute ensemble history?
        # Re-computing ensemble history is best.

        # History of ensemble:
        # We need to compute momentum for all horizons over history.
        # Vectorized approach:
        window_size = 252
        closes = data['Close'].iloc[-(window_size+max(self.horizons)):]

        hist_signals = []
        for h in self.horizons:
             # Rolling pct change
             ret_h = closes.pct_change(h)
             # Fill na?
             ret_h = ret_h.fillna(0)
             hist_signals.append(ret_h.values)

        # Average across horizons (axis 0)
        ensemble_history = np.mean(hist_signals, axis=0)
        # Trim to valid window
        valid_ensemble_history = ensemble_history[max(self.horizons):]
        ensemble_series = pd.Series(valid_ensemble_history)

        current_val = ensemble_signal

        from alpha_families.normalization import AlphaNormalizer
        norm = AlphaNormalizer()

        z, conf = norm.normalize_signal(current_val, ensemble_series, data_confidence=1.0)

        # Combine with signal consistency confidence
        final_conf = conf * raw_confidence

        # Construct
        vol = data['Close'].pct_change().std() * np.sqrt(252)
        dist = norm.construct_distribution(z, final_conf, vol)

        return {
            'mu': dist['mu'],
            'sigma': dist['sigma'],
            'confidence': dist['confidence'],
            'p_loss': dist['p_loss'],
            'cvar_95': dist['cvar_95'],
            'weights': momentum_signals
        }
