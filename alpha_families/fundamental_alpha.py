"""
Fundamental Alpha Family - Value-based trading signals.

Uses fundamental data like earnings, valuation metrics, balance sheet strength,
and macroeconomic indicators to generate trading signals.
"""

import logging
from typing import Dict, Any
import pandas as pd
import numpy as np
from .base_alpha import BaseAlpha

logger = logging.getLogger(__name__)

class FundamentalAlpha(BaseAlpha):
    """
    Fundamental-based alpha using valuation metrics and earnings data.

    Generates signals based on:
    - Price-to-earnings ratios
    - Price-to-book ratios
    - Earnings growth rates
    - Balance sheet strength
    - Dividend yields
    """

    def __init__(self):
        super().__init__()
        self.lookback_periods = [20, 60, 120]  # Multiple timeframes for robustness
        self.value_threshold = 0.1  # Minimum value discrepancy for signal

    def generate_signal(self, data: pd.DataFrame, regime_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate fundamental-based trading signal.

        Args:
            data: OHLCV data with fundamental columns
            regime_context: Current market regime info

        Returns:
            Dict with signal, confidence, and metadata
        """
        if data is None or data.empty or "Close" not in data.columns:
            return {'signal': 0.0, 'confidence': 0.0, 'metadata': {'error': 'Invalid data or missing Close column'}}

        try:
            # Calculate valuation metrics
            pe_ratio = self._calculate_pe_ratio(data)
            pb_ratio = self._calculate_pb_ratio(data)
            earnings_growth = self._calculate_earnings_growth(data)

            # Composite fundamental score
            fundamental_score = self._composite_fundamental_score(pe_ratio, pb_ratio, earnings_growth)

            # Generate signal based on fundamental value
            signal = self._fundamental_to_signal(fundamental_score, regime_context)

            # Calculate confidence based on data quality and consistency
            confidence = self._calculate_fundamental_confidence(data)

            return {
                'signal': signal,
                'confidence': confidence,
                'metadata': {
                    'pe_ratio': pe_ratio.iloc[-1] if not pe_ratio.empty else None,
                    'pb_ratio': pb_ratio.iloc[-1] if not pb_ratio.empty else None,
                    'earnings_growth': earnings_growth.iloc[-1] if not earnings_growth.empty else None,
                    'fundamental_score': fundamental_score,
                    'regime_adjusted': regime_context is not None
                }
            }

        except Exception as e:
            logger.warning(f"Fundamental alpha failed: {e}")
            return {'signal': 0.0, 'confidence': 0.0, 'metadata': {'error': str(e)}}

    def _calculate_pe_ratio(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate price-to-earnings ratio.
        In practice, would use actual EPS data from fundamentals.
        """
        # Placeholder: estimate EPS from price momentum and volatility
        # Real implementation would use actual earnings data
        returns = data['Close'].pct_change()
        volatility = returns.rolling(60).std()

        # Estimate EPS growth from price action (simplified)
        eps_estimate = data['Close'] * (1 + returns.rolling(20).mean() - 0.5 * volatility)

        # Calculate P/E ratio
        pe_ratio = data['Close'] / eps_estimate.replace(0, np.nan)

        # Cap extreme values
        pe_ratio = pe_ratio.clip(0.1, 100)

        return pe_ratio

    def _calculate_pb_ratio(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate price-to-book ratio.
        In practice, would use actual book value data.
        """
        # Placeholder: estimate book value from price trends
        # Real implementation would use balance sheet data
        trend = data['Close'].rolling(100).mean() / data['Close'].rolling(200).mean() - 1

        # Estimate book value as function of price and trend
        book_estimate = data['Close'] * (0.8 - trend)  # Undervalued when trend is negative

        pb_ratio = data['Close'] / book_estimate.replace(0, np.nan)
        pb_ratio = pb_ratio.clip(0.1, 10)

        return pb_ratio

    def _calculate_earnings_growth(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate earnings growth rate.
        In practice, would use actual earnings data.
        """
        # Placeholder: estimate earnings growth from revenue proxy (volume * price)
        revenue_proxy = data['Close'] * data['Volume']
        earnings_growth = revenue_proxy.pct_change(60)  # Quarterly-like growth

        # Smooth and normalize
        earnings_growth = earnings_growth.rolling(10).mean()
        earnings_growth = earnings_growth / (earnings_growth.abs().rolling(60).std() + 1e-8)

        return earnings_growth

    def _composite_fundamental_score(self, pe_ratio: pd.Series, pb_ratio: pd.Series,
                                   earnings_growth: pd.Series) -> float:
        """Create composite fundamental score from multiple metrics."""

        if pe_ratio.empty or pb_ratio.empty or earnings_growth.empty:
            return 0.0

        # Normalize each metric (lower P/E and P/B generally better, higher growth better)
        pe_score = 1.0 - (pe_ratio - pe_ratio.min()) / (pe_ratio.max() - pe_ratio.min() + 1e-8)
        pb_score = 1.0 - (pb_ratio - pb_ratio.min()) / (pb_ratio.max() - pb_ratio.min() + 1e-8)
        growth_score = (earnings_growth - earnings_growth.min()) / (earnings_growth.max() - earnings_growth.min() + 1e-8)

        # Weighted average
        composite_score = 0.4 * pe_score.iloc[-1] + 0.4 * pb_score.iloc[-1] + 0.2 * growth_score.iloc[-1]

        # Convert to signal direction (positive when undervalued)
        signal_direction = 2 * (composite_score - 0.5)

        return signal_direction

    def _fundamental_to_signal(self, fundamental_score: float, regime_context: Dict[str, Any] = None) -> float:
        """Convert fundamental score to trading signal."""

        # Base signal from fundamental score
        signal = np.tanh(fundamental_score * 3)  # Scale and bound

        # Regime adjustment
        if regime_context:
            regime_tag = regime_context.get('regime_tag', 'NORMAL')
            if regime_tag in ['CRISIS', 'BEAR']:
                # Value investing works better in down markets
                signal *= 1.3
            elif regime_tag in ['BULL', 'LOW_VOL']:
                # Growth investing dominates in bull markets
                signal *= 0.8

        return self.normalize_signal(signal)

    def _calculate_fundamental_confidence(self, data: pd.DataFrame) -> float:
        """Calculate confidence in fundamental signal based on data quality."""

        # Confidence based on data availability and market conditions
        has_volume = 'Volume' in data.columns and data['Volume'].sum() > 0
        data_length = len(data)
        price_stability = 1.0 - data['Close'].pct_change().std()

        confidence_factors = [
            1.0 if has_volume else 0.5,
            min(data_length / 100, 1.0),  # More data = higher confidence
            price_stability
        ]

        confidence = np.mean(confidence_factors)
        return confidence
