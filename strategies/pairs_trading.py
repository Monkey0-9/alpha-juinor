"""
Pairs Trading Engine - Statistical Arbitrage.

Features:
- Cointegration testing
- Z-score mean reversion
- Dynamic hedge ratio
- Spread monitoring
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PairSignal:
    """Pairs trading signal."""
    stock_a: str
    stock_b: str
    spread: float
    z_score: float
    hedge_ratio: float
    signal: str  # "LONG_SPREAD", "SHORT_SPREAD", "NEUTRAL"
    confidence: float


@dataclass
class CointegratedPair:
    """Cointegrated pair result."""
    stock_a: str
    stock_b: str
    coint_pvalue: float
    hedge_ratio: float
    half_life: float
    is_cointegrated: bool


class PairsTradingEngine:
    """
    Pairs Trading / Statistical Arbitrage Engine.

    Methods:
    - Cointegration test (Engle-Granger)
    - Dynamic hedge ratio (rolling regression)
    - Z-score trading signals
    """

    def __init__(
        self,
        coint_threshold: float = 0.05,
        z_entry: float = 2.0,
        z_exit: float = 0.5,
        lookback: int = 60
    ):
        self.coint_threshold = coint_threshold
        self.z_entry = z_entry
        self.z_exit = z_exit
        self.lookback = lookback

        # Active pairs
        self.pairs: Dict[str, CointegratedPair] = {}

        # Spread history
        self.spread_history: Dict[str, List[float]] = {}

    def test_cointegration(
        self,
        prices_a: pd.Series,
        prices_b: pd.Series,
        stock_a: str,
        stock_b: str
    ) -> CointegratedPair:
        """
        Test for cointegration using Engle-Granger method.

        Step 1: Regress A on B to get hedge ratio
        Step 2: Test residuals for stationarity (ADF)
        """
        # Simple OLS for hedge ratio
        # A = beta * B + residual
        X = prices_b.values.reshape(-1, 1)
        y = prices_a.values

        # Manual OLS
        X_mean = np.mean(X)
        y_mean = np.mean(y)

        numerator = np.sum((X.flatten() - X_mean) * (y - y_mean))
        denominator = np.sum((X.flatten() - X_mean) ** 2)

        hedge_ratio = numerator / denominator if denominator != 0 else 1.0

        # Calculate spread (residual)
        spread = y - hedge_ratio * X.flatten()

        # ADF-like test (simplified)
        # Test if spread is mean-reverting
        spread_diff = np.diff(spread)
        spread_lag = spread[:-1]

        if len(spread_lag) < 10:
            pvalue = 1.0
        else:
            # Regression: diff = alpha * lag + error
            beta_adf = np.sum(spread_lag * spread_diff) / np.sum(spread_lag ** 2)

            # t-statistic (simplified)
            residuals = spread_diff - beta_adf * spread_lag
            se = np.sqrt(np.sum(residuals ** 2) / (len(residuals) - 1))
            t_stat = beta_adf / (se / np.sqrt(np.sum(spread_lag ** 2)))

            # Approximate p-value (ADF critical values)
            if t_stat < -3.5:
                pvalue = 0.01
            elif t_stat < -2.9:
                pvalue = 0.05
            elif t_stat < -2.6:
                pvalue = 0.10
            else:
                pvalue = 0.50

        # Calculate half-life
        if len(spread) > 20:
            half_life = self._calculate_half_life(spread)
        else:
            half_life = 100.0

        pair = CointegratedPair(
            stock_a=stock_a,
            stock_b=stock_b,
            coint_pvalue=pvalue,
            hedge_ratio=float(hedge_ratio),
            half_life=half_life,
            is_cointegrated=pvalue < self.coint_threshold
        )

        if pair.is_cointegrated:
            key = f"{stock_a}_{stock_b}"
            self.pairs[key] = pair

        return pair

    def _calculate_half_life(self, spread: np.ndarray) -> float:
        """Calculate mean reversion half-life."""
        spread_lag = spread[:-1]
        spread_diff = np.diff(spread)

        if len(spread_lag) < 5:
            return 100.0

        # AR(1) regression
        theta = np.sum(spread_lag * spread_diff) / np.sum(spread_lag ** 2)

        if theta >= 0:
            return 100.0  # Not mean reverting

        half_life = -np.log(2) / theta
        return float(np.clip(half_life, 1, 100))

    def calculate_z_score(
        self,
        prices_a: pd.Series,
        prices_b: pd.Series,
        hedge_ratio: float
    ) -> float:
        """Calculate current z-score of the spread."""
        spread = prices_a - hedge_ratio * prices_b

        if len(spread) < self.lookback:
            return 0.0

        recent = spread.iloc[-self.lookback:]
        mean = recent.mean()
        std = recent.std()

        if std < 1e-10:
            return 0.0

        z = (spread.iloc[-1] - mean) / std
        return float(z)

    def generate_signal(
        self,
        stock_a: str,
        stock_b: str,
        prices_a: pd.Series,
        prices_b: pd.Series
    ) -> PairSignal:
        """Generate trading signal for a pair."""
        key = f"{stock_a}_{stock_b}"

        # Check if pair is in our cointegrated list
        if key not in self.pairs:
            # Test cointegration
            pair = self.test_cointegration(prices_a, prices_b, stock_a, stock_b)
            if not pair.is_cointegrated:
                return PairSignal(
                    stock_a=stock_a,
                    stock_b=stock_b,
                    spread=0.0,
                    z_score=0.0,
                    hedge_ratio=1.0,
                    signal="NEUTRAL",
                    confidence=0.0
                )

        pair = self.pairs[key]

        # Calculate current z-score
        z = self.calculate_z_score(prices_a, prices_b, pair.hedge_ratio)

        # Store spread
        spread = float(prices_a.iloc[-1] - pair.hedge_ratio * prices_b.iloc[-1])
        if key not in self.spread_history:
            self.spread_history[key] = []
        self.spread_history[key].append(spread)

        # Generate signal
        if z > self.z_entry:
            signal = "SHORT_SPREAD"  # Short A, Long B
            confidence = min(1.0, (z - self.z_entry) / 2)
        elif z < -self.z_entry:
            signal = "LONG_SPREAD"  # Long A, Short B
            confidence = min(1.0, (-z - self.z_entry) / 2)
        elif abs(z) < self.z_exit:
            signal = "NEUTRAL"  # Close position
            confidence = 0.5
        else:
            signal = "HOLD"
            confidence = 0.3

        return PairSignal(
            stock_a=stock_a,
            stock_b=stock_b,
            spread=spread,
            z_score=z,
            hedge_ratio=pair.hedge_ratio,
            signal=signal,
            confidence=confidence
        )

    def find_pairs(
        self,
        prices_df: pd.DataFrame,
        min_correlation: float = 0.7
    ) -> List[CointegratedPair]:
        """
        Find cointegrated pairs from a universe.
        """
        symbols = prices_df.columns.tolist()
        pairs = []

        for i, sym_a in enumerate(symbols):
            for sym_b in symbols[i+1:]:
                # Quick correlation filter
                corr = prices_df[sym_a].corr(prices_df[sym_b])

                if abs(corr) < min_correlation:
                    continue

                # Test cointegration
                pair = self.test_cointegration(
                    prices_df[sym_a],
                    prices_df[sym_b],
                    sym_a,
                    sym_b
                )

                if pair.is_cointegrated:
                    pairs.append(pair)

        return pairs


# Global singleton
_pairs_engine: Optional[PairsTradingEngine] = None


def get_pairs_engine() -> PairsTradingEngine:
    """Get or create global pairs trading engine."""
    global _pairs_engine
    if _pairs_engine is None:
        _pairs_engine = PairsTradingEngine()
    return _pairs_engine
