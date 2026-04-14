"""
Mean Reversion Alpha Family.

Generates signals based on mean reversion: RSI, Bollinger Bands, z-score from mean.
Regime gated: stronger in mean-reversion regimes.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from .base_alpha import BaseAlpha


class MeanReversionAlpha(BaseAlpha):
    """
    Mean reversion alpha: long oversold, short overbought.
    """

    def __init__(self, rsi_period: int = 14, bb_period: int = 20, bb_std: float = 2.0, z_threshold: float = 2.0):
        super().__init__()
        self.rsi_period = rsi_period
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.z_threshold = z_threshold

    def generate_signal(self, data: pd.DataFrame, regime_context: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        """
        Generate mean reversion signal.

        Args:
            data: OHLCV data
            regime_context: Current market regime

        Returns:
            Signal dict with weights and confidence
        """
        if len(data) < max(self.rsi_period, self.bb_period) + 1:
            return {'signal': 0.0, 'confidence': 0.0, 'weights': {}}

        # Calculate RSI
        rsi = self._calculate_rsi(data, self.rsi_period)

        # Calculate Bollinger Bands
        bb_upper, bb_lower, bb_middle = self._calculate_bollinger_bands(data, self.bb_period, self.bb_std)

        # Calculate z-score from mean
        close = data['Close']
        z_score = (close - close.rolling(self.bb_period).mean()) / close.rolling(self.bb_period).std()

        # Mean reversion signal: positive when oversold, negative when overbought
        rsi_signal = (30 - rsi) / 30.0  # 0 when RSI=30, 1 when RSI=0
        bb_signal = (bb_middle - close) / (bb_upper - bb_middle)  # Normalized distance from middle
        z_signal = -z_score / self.z_threshold  # Negative z-score = buy signal

        # Combine signals
        combined_signal = (rsi_signal + bb_signal + z_signal) / 3.0

        # Confidence based on signal agreement and volatility
        signal_agreement = 1.0 - np.std([rsi_signal.iloc[-1], bb_signal.iloc[-1], z_signal.iloc[-1]])
        volatility = close.pct_change(fill_method=None).rolling(20).std().iloc[-1]
        confidence = signal_agreement * (1.0 - min(volatility * 10, 1.0))  # Lower confidence in high vol

        # Regime adjustment: stronger in mean-reversion regimes
        regime_multiplier = 1.0
        if regime_context and regime_context.get('regime_tag') == 'MEAN_REVERSION':
            regime_multiplier = 1.5
        elif regime_context and regime_context.get('regime_tag') == 'TREND':
            regime_multiplier = 0.5

        final_signal = combined_signal.iloc[-1] * regime_multiplier

        return {
            'signal': np.clip(final_signal, -1.0, 1.0),
            'confidence': confidence,
            'weights': {
                'rsi': rsi_signal.iloc[-1],
                'bb_position': bb_signal.iloc[-1],
                'z_score': z_signal.iloc[-1]
            }
        }

    def _calculate_rsi(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate RSI."""
        close = data['Close']
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_bollinger_bands(self, data: pd.DataFrame, period: int, std_dev: float) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        close = data['Close']
        middle = close.rolling(period).mean()
        std = close.rolling(period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, lower, middle
