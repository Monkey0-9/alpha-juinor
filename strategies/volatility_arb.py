"""
Volatility Arbitrage - Implied vs Realized Vol Spread.

Strategies:
- Long/short volatility based on IV-RV spread
- VIX term structure trades
- Volatility mean reversion
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class VolatilitySignal:
    """Volatility trading signal."""
    symbol: str
    implied_vol: float
    realized_vol: float
    spread: float  # IV - RV
    z_score: float
    signal: str  # "LONG_VOL", "SHORT_VOL", "NEUTRAL"
    confidence: float


class VolatilityArbitrage:
    """
    Volatility arbitrage strategy engine.

    Trades based on:
    - IV-RV spread (sell vol when IV >> RV)
    - VIX term structure (contango/backwardation)
    - Mean reversion in volatility
    """

    def __init__(
        self,
        rv_window: int = 20,
        z_threshold: float = 1.5,
        mean_reversion_half_life: int = 10
    ):
        self.rv_window = rv_window
        self.z_threshold = z_threshold
        self.mean_reversion_half_life = mean_reversion_half_life

        # Historical spreads for z-score
        self.spread_history: Dict[str, List[float]] = {}

    def calculate_realized_vol(
        self,
        prices: pd.Series,
        window: int = None
    ) -> float:
        """Calculate realized volatility."""
        window = window or self.rv_window

        if len(prices) < window + 1:
            return np.nan

        returns = prices.pct_change().dropna().iloc[-window:]
        return float(returns.std() * np.sqrt(252))

    def calculate_iv_rv_spread(
        self,
        implied_vol: float,
        realized_vol: float
    ) -> float:
        """Calculate IV-RV spread."""
        return implied_vol - realized_vol

    def get_spread_z_score(
        self,
        symbol: str,
        current_spread: float
    ) -> float:
        """Calculate z-score of current spread vs history."""
        if symbol not in self.spread_history:
            self.spread_history[symbol] = []

        history = self.spread_history[symbol]
        history.append(current_spread)

        # Keep last 252 observations
        if len(history) > 252:
            self.spread_history[symbol] = history[-252:]

        if len(history) < 20:
            return 0.0

        mean = np.mean(history)
        std = np.std(history)

        if std < 0.001:
            return 0.0

        return (current_spread - mean) / std

    def generate_signal(
        self,
        symbol: str,
        prices: pd.Series,
        implied_vol: float
    ) -> VolatilitySignal:
        """
        Generate volatility trading signal.

        Logic:
        - If IV >> RV (high z-score): Sell volatility
        - If IV << RV (low z-score): Buy volatility
        """
        realized_vol = self.calculate_realized_vol(prices)
        spread = self.calculate_iv_rv_spread(implied_vol, realized_vol)
        z_score = self.get_spread_z_score(symbol, spread)

        # Determine signal
        if z_score > self.z_threshold:
            signal = "SHORT_VOL"
            confidence = min(1.0, (z_score - self.z_threshold) / 2)
        elif z_score < -self.z_threshold:
            signal = "LONG_VOL"
            confidence = min(1.0, (-z_score - self.z_threshold) / 2)
        else:
            signal = "NEUTRAL"
            confidence = 0.0

        return VolatilitySignal(
            symbol=symbol,
            implied_vol=implied_vol,
            realized_vol=realized_vol,
            spread=spread,
            z_score=z_score,
            signal=signal,
            confidence=confidence
        )

    def vix_term_structure_signal(
        self,
        vix_spot: float,
        vix_1m: float,
        vix_3m: float
    ) -> Tuple[str, float]:
        """
        Signal from VIX term structure.

        Contango (VIX_future > VIX_spot): Short VIX
        Backwardation: Long VIX
        """
        front_spread = vix_1m - vix_spot
        term_spread = vix_3m - vix_1m

        if front_spread > 0 and term_spread > 0:
            # Strong contango - short VIX
            return "SHORT_VIX", min(1.0, front_spread / 5)
        elif front_spread < 0 and term_spread < 0:
            # Backwardation - long VIX
            return "LONG_VIX", min(1.0, -front_spread / 5)
        else:
            return "NEUTRAL", 0.0


# Global singleton
_vol_arb: Optional[VolatilityArbitrage] = None


def get_volatility_arb() -> VolatilityArbitrage:
    """Get or create global volatility arbitrage engine."""
    global _vol_arb
    if _vol_arb is None:
        _vol_arb = VolatilityArbitrage()
    return _vol_arb
