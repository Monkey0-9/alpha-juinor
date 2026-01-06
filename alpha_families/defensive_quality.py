"""
Defensive Quality Alpha Family.

Generates signals based on quality factors: low volatility, high dividend yield, stable earnings.
Focuses on defensive stocks with quality characteristics.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from .base_alpha import BaseAlpha


class DefensiveQuality(BaseAlpha):
    """
    Defensive quality alpha: long low-vol, high-quality stocks.
    """

    def __init__(self, vol_lookback: int = 60, quality_threshold: float = 0.7):
        super().__init__()
        self.vol_lookback = vol_lookback
        self.quality_threshold = quality_threshold

    def generate_signal(self, data: pd.DataFrame, regime_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate defensive quality signal.

        Args:
            data: OHLCV data
            regime_context: Current market regime

        Returns:
            Signal dict with weights and confidence
        """
        if len(data) < self.vol_lookback + 1:
            return {'signal': 0.0, 'confidence': 0.0, 'weights': {}}

        # Calculate volatility (lower is better for defensive)
        returns = data['Close'].pct_change().dropna()
        volatility = returns.rolling(self.vol_lookback).std() * np.sqrt(252)

        # Estimate dividend yield proxy (using volume/price stability)
        volume_stability = data['Volume'].rolling(20).std() / data['Volume'].rolling(20).mean()
        price_stability = data['Close'].rolling(20).std() / data['Close'].rolling(20).mean()
        dividend_proxy = 1.0 / (1.0 + volume_stability + price_stability)  # Higher stability = higher yield proxy

        # Quality score: combination of low vol and stability
        quality_score = (1.0 - volatility / volatility.max()) * 0.6 + dividend_proxy * 0.4

        # Signal: stronger for higher quality scores
        signal = (quality_score - self.quality_threshold) / (1.0 - self.quality_threshold)
        signal = np.clip(signal, 0.0, 1.0)  # Only positive signals (long defensive)

        # Confidence based on quality score stability
        quality_stability = 1.0 - quality_score.rolling(10).std()
        confidence = quality_stability.iloc[-1]

        # Regime adjustment: stronger in risk-off regimes
        regime_multiplier = 1.0
        if regime_context and regime_context.get('regime_tag') == 'RISK_OFF':
            regime_multiplier = 1.5
        elif regime_context and regime_context.get('regime_tag') == 'RISK_ON':
            regime_multiplier = 0.7

        final_signal = signal.iloc[-1] * regime_multiplier

        return {
            'signal': np.clip(final_signal, -1.0, 1.0),
            'confidence': confidence,
            'weights': {
                'volatility': 1.0 - volatility.iloc[-1] / volatility.max(),
                'stability': dividend_proxy.iloc[-1]
            }
        }
