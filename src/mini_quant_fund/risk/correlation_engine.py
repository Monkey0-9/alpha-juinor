"""
Cross-Asset Correlation Engine - Multi-Asset Regime Detection.

Features:
- Real-time correlation tracking
- Regime shifts via correlation changes
- Cross-asset arbitrage signals
- Risk-off/Risk-on detection
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class CorrelationRegime:
    """Correlation regime detection result."""
    regime: str  # "RISK_ON", "RISK_OFF", "DECORRELATING", "NORMAL"
    avg_correlation: float
    max_correlation: float
    correlation_change: float
    signal: str


class CrossAssetCorrelationEngine:
    """
    Track and analyze cross-asset correlations.

    Multi-asset relationships:
    - Equity-bond (typically negative in risk-off)
    - Equity-gold (hedge relationship)
    - VIX-equity (inverse)
    - Currency-equity (carry trade)
    """

    def __init__(
        self,
        lookback_short: int = 20,
        lookback_long: int = 60,
        correlation_threshold: float = 0.7
    ):
        self.lookback_short = lookback_short
        self.lookback_long = lookback_long
        self.correlation_threshold = correlation_threshold

        # Price history
        self.price_history: Dict[str, deque] = {}

        # Correlation history
        self.correlation_history: List[Dict[str, float]] = []

        # Asset class mappings
        self.asset_classes = {
            "SPY": "equity",
            "TLT": "bond",
            "GLD": "gold",
            "UUP": "usd",
            "VIX": "volatility"
        }

    def update_price(self, symbol: str, price: float):
        """Update price for a symbol."""
        if symbol not in self.price_history:
            self.price_history[symbol] = deque(maxlen=self.lookback_long * 2)
        self.price_history[symbol].append(price)

    def calculate_returns(self, symbol: str) -> np.ndarray:
        """Calculate returns for a symbol."""
        if symbol not in self.price_history:
            return np.array([])

        prices = list(self.price_history[symbol])
        if len(prices) < 2:
            return np.array([])

        returns = np.diff(prices) / prices[:-1]
        return returns

    def calculate_correlation(
        self,
        symbol1: str,
        symbol2: str,
        lookback: int = None
    ) -> float:
        """Calculate correlation between two symbols."""
        lookback = lookback or self.lookback_short

        returns1 = self.calculate_returns(symbol1)
        returns2 = self.calculate_returns(symbol2)

        min_len = min(len(returns1), len(returns2), lookback)

        if min_len < 5:
            return 0.0

        r1 = returns1[-min_len:]
        r2 = returns2[-min_len:]

        corr = np.corrcoef(r1, r2)[0, 1]

        if np.isnan(corr):
            return 0.0

        return float(corr)

    def get_correlation_matrix(
        self,
        symbols: List[str],
        lookback: int = None
    ) -> pd.DataFrame:
        """Get correlation matrix for symbols."""
        n = len(symbols)
        matrix = np.eye(n)

        for i in range(n):
            for j in range(i + 1, n):
                corr = self.calculate_correlation(symbols[i], symbols[j], lookback)
                matrix[i, j] = corr
                matrix[j, i] = corr

        return pd.DataFrame(matrix, index=symbols, columns=symbols)

    def detect_regime(
        self,
        equity_symbol: str = "SPY",
        bond_symbol: str = "TLT",
        gold_symbol: str = "GLD"
    ) -> CorrelationRegime:
        """
        Detect correlation regime.

        Risk-off: equity-bond correlation rises, gold rallies
        Risk-on: equity-bond correlation falls/negative
        """
        # Calculate correlations
        equity_bond_short = self.calculate_correlation(
            equity_symbol, bond_symbol, self.lookback_short
        )
        equity_bond_long = self.calculate_correlation(
            equity_symbol, bond_symbol, self.lookback_long
        )

        equity_gold = self.calculate_correlation(
            equity_symbol, gold_symbol, self.lookback_short
        )

        # Correlation change
        corr_change = equity_bond_short - equity_bond_long

        # Store history
        self.correlation_history.append({
            "equity_bond": equity_bond_short,
            "equity_gold": equity_gold,
            "timestamp": pd.Timestamp.utcnow()
        })
        if len(self.correlation_history) > 252:
            self.correlation_history = self.correlation_history[-252:]

        # Detect regime
        if equity_bond_short > 0.3:
            # Positive correlation = risk-off (flight to quality)
            regime = "RISK_OFF"
            signal = "REDUCE_EQUITY"
        elif equity_bond_short < -0.3:
            # Negative correlation = normal risk-on
            regime = "RISK_ON"
            signal = "INCREASE_EQUITY"
        elif abs(corr_change) > 0.3:
            # Large correlation shift
            regime = "DECORRELATING"
            signal = "REDUCE_ALL"
        else:
            regime = "NORMAL"
            signal = "MAINTAIN"

        return CorrelationRegime(
            regime=regime,
            avg_correlation=equity_bond_short,
            max_correlation=max(abs(equity_bond_short), abs(equity_gold)),
            correlation_change=corr_change,
            signal=signal
        )

    def get_diversification_ratio(self, symbols: List[str]) -> float:
        """
        Calculate portfolio diversification ratio.

        DR = weighted avg volatility / portfolio volatility
        Higher is better (more diversification benefit)
        """
        if len(symbols) < 2:
            return 1.0

        corr_matrix = self.get_correlation_matrix(symbols)

        # Assume equal weights
        weights = np.ones(len(symbols)) / len(symbols)

        # Calculate portfolio correlation
        avg_corr = 0
        n = len(symbols)
        for i in range(n):
            for j in range(i + 1, n):
                avg_corr += corr_matrix.iloc[i, j]

        avg_corr = avg_corr / (n * (n - 1) / 2) if n > 1 else 0

        # Diversification ratio approximation
        dr = 1 / np.sqrt(1 / n + (1 - 1 / n) * avg_corr)

        return float(dr)


# Global singleton
_correlation_engine: Optional[CrossAssetCorrelationEngine] = None


def get_correlation_engine() -> CrossAssetCorrelationEngine:
    """Get or create global correlation engine."""
    global _correlation_engine
    if _correlation_engine is None:
        _correlation_engine = CrossAssetCorrelationEngine()
    return _correlation_engine
