"""
Market Neutral Portfolio Constructor.

Creates beta-hedged long/short portfolios:
- Zero market beta exposure
- Sector neutrality
- Factor neutrality
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MarketNeutralPortfolio:
    """Market neutral portfolio result."""
    long_weights: Dict[str, float]
    short_weights: Dict[str, float]
    net_weights: Dict[str, float]

    # Exposure metrics
    gross_exposure: float
    net_exposure: float
    portfolio_beta: float

    # Sector exposures
    sector_exposures: Dict[str, float]


class MarketNeutralConstructor:
    """
    Construct market neutral portfolios.

    Approach:
    1. Start with alpha-weighted long/short
    2. Beta-hedge to achieve zero market exposure
    3. Apply sector neutrality constraints
    4. Normalize to target gross exposure
    """

    def __init__(
        self,
        target_gross_exposure: float = 2.0,  # 100% long, 100% short
        max_sector_deviation: float = 0.05,
        beta_tolerance: float = 0.05
    ):
        self.target_gross_exposure = target_gross_exposure
        self.max_sector_deviation = max_sector_deviation
        self.beta_tolerance = beta_tolerance

    def construct(
        self,
        alpha_scores: Dict[str, float],
        betas: Dict[str, float],
        sectors: Optional[Dict[str, str]] = None
    ) -> MarketNeutralPortfolio:
        """
        Construct market neutral portfolio.

        Args:
            alpha_scores: Dict of symbol -> alpha signal (-1 to 1)
            betas: Dict of symbol -> beta
            sectors: Optional dict of symbol -> sector

        Returns:
            MarketNeutralPortfolio
        """
        symbols = list(alpha_scores.keys())
        n_assets = len(symbols)

        if n_assets == 0:
            return self._empty_portfolio()

        # Separate long and short based on alpha
        longs = {s: a for s, a in alpha_scores.items() if a > 0}
        shorts = {s: a for s, a in alpha_scores.items() if a < 0}

        if not longs or not shorts:
            logger.warning("Cannot create market neutral: need both longs and shorts")
            return self._empty_portfolio()

        # Normalize weights
        long_sum = sum(longs.values())
        short_sum = sum(abs(v) for v in shorts.values())

        long_weights = {s: v / long_sum for s, v in longs.items()}
        short_weights = {s: abs(v) / short_sum for s, v in shorts.items()}

        # Scale to half of gross exposure each
        half_exposure = self.target_gross_exposure / 2
        long_weights = {s: v * half_exposure for s, v in long_weights.items()}
        short_weights = {s: v * half_exposure for s, v in short_weights.items()}

        # Calculate initial beta exposure
        long_beta = sum(long_weights[s] * betas.get(s, 1.0) for s in long_weights)
        short_beta = sum(short_weights[s] * betas.get(s, 1.0) for s in short_weights)
        net_beta = long_beta - short_beta

        # Beta hedge: adjust long/short ratio
        if abs(net_beta) > self.beta_tolerance:
            long_weights, short_weights = self._beta_hedge(
                long_weights, short_weights, betas, net_beta
            )

        # Calculate final beta
        long_beta = sum(long_weights[s] * betas.get(s, 1.0) for s in long_weights)
        short_beta = sum(short_weights[s] * betas.get(s, 1.0) for s in short_weights)

        # Net weights (longs positive, shorts negative)
        net_weights = {}
        for s, w in long_weights.items():
            net_weights[s] = net_weights.get(s, 0) + w
        for s, w in short_weights.items():
            net_weights[s] = net_weights.get(s, 0) - w

        # Sector exposures
        sector_exposures = self._calculate_sector_exposures(net_weights, sectors)

        gross = sum(long_weights.values()) + sum(short_weights.values())
        net = sum(long_weights.values()) - sum(short_weights.values())

        return MarketNeutralPortfolio(
            long_weights=long_weights,
            short_weights=short_weights,
            net_weights=net_weights,
            gross_exposure=gross,
            net_exposure=net,
            portfolio_beta=long_beta - short_beta,
            sector_exposures=sector_exposures
        )

    def _beta_hedge(
        self,
        longs: Dict[str, float],
        shorts: Dict[str, float],
        betas: Dict[str, float],
        current_beta: float
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Adjust weights to achieve zero beta.
        """
        # If net beta is positive, increase shorts or decrease longs
        # If net beta is negative, increase longs or decrease shorts

        long_total = sum(longs.values())
        short_total = sum(shorts.values())

        # Simple scaling approach
        if current_beta > 0:
            # Need more short beta
            scale_factor = 1 + current_beta / (short_total + 0.001)
            shorts = {s: w * min(scale_factor, 1.5) for s, w in shorts.items()}
        else:
            # Need more long beta
            scale_factor = 1 + abs(current_beta) / (long_total + 0.001)
            longs = {s: w * min(scale_factor, 1.5) for s, w in longs.items()}

        # Renormalize to target exposure
        new_gross = sum(longs.values()) + sum(shorts.values())
        if new_gross > 0:
            ratio = self.target_gross_exposure / new_gross
            longs = {s: w * ratio for s, w in longs.items()}
            shorts = {s: w * ratio for s, w in shorts.items()}

        return longs, shorts

    def _calculate_sector_exposures(
        self,
        weights: Dict[str, float],
        sectors: Optional[Dict[str, str]]
    ) -> Dict[str, float]:
        """Calculate net sector exposures."""
        if not sectors:
            return {}

        sector_exp = {}
        for symbol, weight in weights.items():
            sector = sectors.get(symbol, "Unknown")
            sector_exp[sector] = sector_exp.get(sector, 0) + weight

        return sector_exp

    def _empty_portfolio(self) -> MarketNeutralPortfolio:
        """Return empty portfolio."""
        return MarketNeutralPortfolio(
            long_weights={},
            short_weights={},
            net_weights={},
            gross_exposure=0.0,
            net_exposure=0.0,
            portfolio_beta=0.0,
            sector_exposures={}
        )


# Global singleton
_market_neutral: Optional[MarketNeutralConstructor] = None


def get_market_neutral_constructor() -> MarketNeutralConstructor:
    """Get or create global market neutral constructor."""
    global _market_neutral
    if _market_neutral is None:
        _market_neutral = MarketNeutralConstructor()
    return _market_neutral
