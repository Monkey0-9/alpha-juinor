"""
Volatility Carry Alpha Family.

Generates signals based on volatility carry: selling volatility in low-vol regimes, buying in high-vol.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from .base_alpha import BaseAlpha


class VolatilityCarry(BaseAlpha):
    """
    Volatility carry alpha: long volatility when cheap, short when expensive.
    """

    def __init__(self, vol_lookback: int = 20, carry_threshold: float = 0.1):
        super().__init__()
        self.vol_lookback = vol_lookback
        self.carry_threshold = carry_threshold

    def generate_signal(self, data: pd.DataFrame, regime_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate volatility carry signal.

        Args:
            data: OHLCV data
            regime_context: Current market regime

        Returns:
            Signal dict with weights and confidence
        """
        if len(data) < self.vol_lookback + 1:
            return {'signal': 0.0, 'confidence': 0.0, 'weights': {}}

        # Calculate realized volatility
        returns = data['Close'].pct_change().dropna()
        realized_vol = returns.rolling(self.vol_lookback).std() * np.sqrt(252)

        # Implied vol proxy (assume VIX or similar, use realized as proxy)
        implied_vol = realized_vol  # In practice, use VIX

        # Carry signal: positive when realized > implied (vol cheap), negative when realized < implied (vol expensive)
        carry_signal = (realized_vol - implied_vol) / implied_vol

        # Normalize and threshold
        signal = np.clip(carry_signal.iloc[-1], -self.carry_threshold, self.carry_threshold) / self.carry_threshold

        # Confidence based on vol stability
        vol_stability = 1.0 / (1.0 + realized_vol.std())
        confidence = min(vol_stability, 1.0)

        # Regime adjustment: stronger in vol compression regimes
        regime_multiplier = 1.0
        if regime_context and regime_context.get('regime_tag') == 'VOL_COMPRESSION':
            regime_multiplier = 1.5
        elif regime_context and regime_context.get('regime_tag') == 'VOL_EXPANSION':
            regime_multiplier = 0.5

        final_signal = signal * regime_multiplier

        return {
            'signal': np.clip(final_signal, -1.0, 1.0),
            'confidence': confidence,
            'weights': {'vol_carry': final_signal}
        }
