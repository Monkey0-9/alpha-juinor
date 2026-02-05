"""
P&L Attribution - Factor-Based Performance Decomposition.

Decomposes returns into:
- Alpha (skill)
- Factor exposures (market, value, momentum, etc.)
- Residual (unexplained)
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class PnLAttribution:
    """P&L attribution breakdown."""
    date: str
    total_return: float

    # Factor contributions
    alpha: float
    market_beta: float
    value_factor: float
    momentum_factor: float
    size_factor: float
    volatility_factor: float

    # Residual
    residual: float

    # Portfolio metrics
    gross_exposure: float
    net_exposure: float


@dataclass
class PositionAttribution:
    """Per-position P&L attribution."""
    symbol: str
    weight: float
    return_pct: float
    contribution: float  # weight * return

    # Factor loadings
    beta: float
    factor_exposures: Dict[str, float] = field(default_factory=dict)


class PnLAttributor:
    """
    Real-time P&L attribution engine.

    Uses factor model to decompose returns:
    R = alpha + beta * R_market + sum(factor_exposures * factor_returns) + residual
    """

    FACTOR_NAMES = [
        "market",
        "value",
        "momentum",
        "size",
        "volatility",
        "quality"
    ]

    def __init__(self):
        self.history: List[PnLAttribution] = []
        self.position_history: Dict[str, List[PositionAttribution]] = {}

        # Factor returns (would be fetched from data provider)
        self.factor_returns: Dict[str, List[float]] = {
            name: [] for name in self.FACTOR_NAMES
        }

    def attribute_daily(
        self,
        portfolio_return: float,
        position_returns: Dict[str, float],
        position_weights: Dict[str, float],
        factor_exposures: Dict[str, Dict[str, float]],
        factor_returns: Dict[str, float]
    ) -> PnLAttribution:
        """
        Perform daily P&L attribution.

        Args:
            portfolio_return: Total portfolio return
            position_returns: Dict of symbol -> return
            position_weights: Dict of symbol -> weight
            factor_exposures: Dict of symbol -> {factor -> loading}
            factor_returns: Dict of factor -> factor return

        Returns:
            PnLAttribution breakdown
        """
        date = datetime.utcnow().strftime("%Y-%m-%d")

        # Calculate factor contributions
        factor_contribs = {}
        total_factor_contrib = 0.0

        for factor in self.FACTOR_NAMES:
            factor_ret = factor_returns.get(factor, 0.0)

            # Portfolio exposure to this factor
            portfolio_exposure = sum(
                position_weights.get(sym, 0) *
                factor_exposures.get(sym, {}).get(factor, 0)
                for sym in position_weights
            )

            contrib = portfolio_exposure * factor_ret
            factor_contribs[factor] = contrib
            total_factor_contrib += contrib

        # Alpha is residual after factor attribution
        alpha = portfolio_return - total_factor_contrib

        # Calculate exposures
        gross_exposure = sum(abs(w) for w in position_weights.values())
        net_exposure = sum(position_weights.values())

        attribution = PnLAttribution(
            date=date,
            total_return=portfolio_return,
            alpha=alpha,
            market_beta=factor_contribs.get("market", 0),
            value_factor=factor_contribs.get("value", 0),
            momentum_factor=factor_contribs.get("momentum", 0),
            size_factor=factor_contribs.get("size", 0),
            volatility_factor=factor_contribs.get("volatility", 0),
            residual=alpha,  # In simple model, alpha = residual
            gross_exposure=gross_exposure,
            net_exposure=net_exposure
        )

        self.history.append(attribution)

        # Keep last 252 days
        if len(self.history) > 252:
            self.history = self.history[-252:]

        logger.info(
            f"PnL Attribution: Total={portfolio_return:.2%}, "
            f"Alpha={alpha:.2%}, Market={factor_contribs.get('market', 0):.2%}"
        )

        return attribution

    def attribute_position(
        self,
        symbol: str,
        weight: float,
        return_pct: float,
        beta: float,
        factor_exposures: Dict[str, float]
    ) -> PositionAttribution:
        """Attribute P&L for a single position."""
        contribution = weight * return_pct

        attr = PositionAttribution(
            symbol=symbol,
            weight=weight,
            return_pct=return_pct,
            contribution=contribution,
            beta=beta,
            factor_exposures=factor_exposures
        )

        if symbol not in self.position_history:
            self.position_history[symbol] = []
        self.position_history[symbol].append(attr)

        return attr

    def get_cumulative_attribution(self) -> Dict[str, float]:
        """Get cumulative attribution over history."""
        if not self.history:
            return {}

        return {
            "total_return": sum(a.total_return for a in self.history),
            "alpha": sum(a.alpha for a in self.history),
            "market_beta": sum(a.market_beta for a in self.history),
            "value_factor": sum(a.value_factor for a in self.history),
            "momentum_factor": sum(a.momentum_factor for a in self.history),
            "size_factor": sum(a.size_factor for a in self.history),
            "volatility_factor": sum(a.volatility_factor for a in self.history),
        }

    def get_top_contributors(
        self, limit: int = 10
    ) -> List[Tuple[str, float]]:
        """Get top contributing positions."""
        contributions = {}

        for symbol, attrs in self.position_history.items():
            total_contrib = sum(a.contribution for a in attrs)
            contributions[symbol] = total_contrib

        sorted_contribs = sorted(
            contributions.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return sorted_contribs[:limit]

    def get_top_detractors(
        self, limit: int = 10
    ) -> List[Tuple[str, float]]:
        """Get top detracting positions."""
        contributions = {}

        for symbol, attrs in self.position_history.items():
            total_contrib = sum(a.contribution for a in attrs)
            contributions[symbol] = total_contrib

        sorted_contribs = sorted(
            contributions.items(),
            key=lambda x: x[1]
        )

        return sorted_contribs[:limit]

    def get_summary(self) -> Dict[str, Any]:
        """Get attribution summary."""
        cumulative = self.get_cumulative_attribution()

        # Calculate information ratio
        if self.history:
            alphas = [a.alpha for a in self.history]
            ir = (np.mean(alphas) * 252) / (np.std(alphas) * np.sqrt(252)) if np.std(alphas) > 0 else 0
        else:
            ir = 0

        return {
            "cumulative": cumulative,
            "information_ratio": round(ir, 2),
            "alpha_pct_of_return": (
                cumulative.get("alpha", 0) / cumulative.get("total_return", 1) * 100
                if cumulative.get("total_return", 0) != 0 else 0
            ),
            "days_tracked": len(self.history),
            "top_contributors": self.get_top_contributors(5),
            "top_detractors": self.get_top_detractors(5)
        }


# Global singleton
_attributor: Optional[PnLAttributor] = None


def get_pnl_attributor() -> PnLAttributor:
    """Get or create global PnLAttributor."""
    global _attributor
    if _attributor is None:
        _attributor = PnLAttributor()
    return _attributor
