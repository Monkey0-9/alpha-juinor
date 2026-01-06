"""
Trend Strength Alpha Family.

Generates signals based on trend strength: ADX, MACD, moving average crossovers.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from .base_alpha import BaseAlpha


class TrendStrength(BaseAlpha):
    """
    Trend strength alpha: long in strong uptrends, short in strong downtrends.
    """

    def __init__(self, adx_period: int = 14, macd_fast: int = 12, macd_slow: int = 26, macd_signal: int = 9):
        super().__init__()
        self.adx_period = adx_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal

    def generate_signal(self, data: pd.DataFrame, regime_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate trend strength signal.

        Args:
            data: OHLCV data
            regime_context: Current market regime

        Returns:
            Signal dict with weights and confidence
        """
        if len(data) < max(self.adx_period, self.macd_slow + self.macd_signal) + 1:
            return {'signal': 0.0, 'confidence': 0.0, 'weights': {}}

        # Calculate ADX
        adx = self._calculate_adx(data, self.adx_period)

        # Calculate MACD
        macd_line, signal_line, histogram = self._calculate_macd(data, self.macd_fast, self.macd_slow, self.macd_signal)

        # Trend strength signal: ADX * MACD direction
        macd_signal = np.sign(macd_line.iloc[-1] - signal_line.iloc[-1])
        trend_signal = adx.iloc[-1] * macd_signal / 100.0  # Normalize ADX

        # Confidence based on ADX strength
        confidence = min(adx.iloc[-1] / 50.0, 1.0)  # Higher ADX = higher confidence

        # Regime adjustment: stronger in trend regimes
        regime_multiplier = 1.0
        if regime_context and regime_context.get('regime_tag') == 'TREND':
            regime_multiplier = 1.5
        elif regime_context and regime_context.get('regime_tag') == 'MEAN_REVERSION':
            regime_multiplier = 0.5

        final_signal = trend_signal * regime_multiplier

        return {
            'signal': np.clip(final_signal, -1.0, 1.0),
            'confidence': confidence,
            'weights': {'adx': adx.iloc[-1], 'macd_signal': macd_signal}
        }

    def _calculate_adx(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average Directional Index (ADX)."""
        high = data['High']
        low = data['Low']
        close = data['Close']

        # True Range
        tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)

        # Directional Movement
        dm_plus = pd.Series(np.where((high - high.shift(1)) > (low.shift(1) - low), high - high.shift(1), 0), index=data.index)
        dm_minus = pd.Series(np.where((low.shift(1) - low) > (high - high.shift(1)), low.shift(1) - low, 0), index=data.index)

        # Smoothed averages
        atr = tr.rolling(period).mean()
        di_plus = 100 * (dm_plus.rolling(period).mean() / atr)
        di_minus = 100 * (dm_minus.rolling(period).mean() / atr)

        # ADX
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = dx.rolling(period).mean()

        return adx.fillna(0)

    def _calculate_macd(self, data: pd.DataFrame, fast: int, slow: int, signal: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD."""
        close = data['Close']
        ema_fast = close.ewm(span=fast).mean()
        ema_slow = close.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram
