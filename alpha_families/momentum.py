# alpha_families/momentum.py
from alpha_families.base_alpha import BaseAlpha
import pandas as pd
from typing import Dict, Any


class MomentumAlpha(BaseAlpha):
    def __init__(self):
        super().__init__()

    def generate_signal(self, data: pd.DataFrame, regime_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate momentum-based alpha signal.

        Args:
            data: OHLCV data
            regime_context: Current market regime info

        Returns:
            Dict with signal, confidence, and metadata
        """
        if not self.validate_data(data):
            return {'signal': 0.0, 'confidence': 0.0, 'metadata': {'error': 'Invalid data'}}

        # Simple momentum calculation: recent return over lookback period
        lookback = 20  # 20-day momentum
        if len(data) < lookback:
            return {'signal': 0.0, 'confidence': 0.0, 'metadata': {'error': 'Insufficient data'}}

        # Calculate momentum signal
        current_price = data['Close'].iloc[-1]
        past_price = data['Close'].iloc[-lookback]
        momentum = (current_price - past_price) / past_price

        # Normalize signal to [-1, 1]
        signal = self.normalize_signal(momentum * 2)  # Scale up for better signal strength

        # Calculate confidence based on volatility
        returns = data['Close'].pct_change().dropna()
        volatility = returns.std()
        confidence = min(1.0, max(0.1, 1.0 - volatility))  # Higher confidence with lower volatility

        return {
            'signal': signal,
            'confidence': confidence,
            'metadata': {
                'momentum': momentum,
                'lookback': lookback,
                'volatility': volatility,
                'regime': regime_context.get('regime', 'unknown') if regime_context else 'unknown'
            }
        }
