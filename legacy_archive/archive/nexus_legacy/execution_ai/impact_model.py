"""
Market Impact Model (Almgren-Chriss Style).
Estimates permanent and temporary impact of trades.
"""

import logging
import numpy as np
from typing import Dict, Any

logger = logging.getLogger(__name__)


class ImpactModel:
    """
    Almgren-Chriss style market impact estimator.
    Separates permanent and temporary impact.
    """

    def __init__(
        self,
        permanent_impact_coef: float = 0.1,
        temporary_impact_coef: float = 0.5,
        volatility_scaling: float = 1.0
    ):
        """
        Args:
            permanent_impact_coef: Coefficient for permanent impact (default 0.1)
            temporary_impact_coef: Coefficient for temporary impact (default 0.5)
            volatility_scaling: Scale impact by volatility (default 1.0)
        """
        self.perm_coef = permanent_impact_coef
        self.temp_coef = temporary_impact_coef
        self.vol_scaling = volatility_scaling

    def estimate_impact(
        self,
        symbol: str,
        order_size_usd: float,
        adv_usd: float,
        volatility: float = 0.02
    ) -> Dict[str, float]:
        """
        Estimate market impact for an order.

        Args:
            symbol: Symbol being traded
            order_size_usd: Order size in USD
            adv_usd: Average daily volume in USD
            volatility: Daily volatility (default 2%)

        Returns:
            Dict with {
                'permanent_impact': float (as fraction),
                'temporary_impact': float (as fraction),
                'total_bps': float (total impact in basis points),
                'participation_rate': float (order_size / ADV)
            }
        """
        if adv_usd <= 0:
            logger.warning(f"ImpactModel: ADV is zero for {symbol}, assuming high impact")
            return {
                'permanent_impact': 0.01,  # 1% impact
                'temporary_impact': 0.01,
                'total_bps': 200.0,
                'participation_rate': 1.0
            }

        # Participation rate (what fraction of daily volume is this order?)
        participation_rate = order_size_usd / adv_usd

        # Permanent impact: proportional to sqrt(participation_rate) * volatility
        # Formula: perm_impact = k * sqrt(order_size / ADV) * sigma
        permanent_impact = self.perm_coef * np.sqrt(participation_rate) * volatility * self.vol_scaling

        # Temporary impact: proportional to participation_rate * volatility
        # Formula: temp_impact = k * (order_size / ADV) * sigma
        temporary_impact = self.temp_coef * participation_rate * volatility * self.vol_scaling

        # Total impact (as fraction)
        total_impact = permanent_impact + temporary_impact

        # Convert to basis points
        total_bps = total_impact * 10000

        logger.debug(f"ImpactModel: {symbol} order_size=${order_size_usd:,.0f}, ADV=${adv_usd:,.0f}, "
                    f"participation={participation_rate:.2%}, "
                    f"perm_impact={permanent_impact:.4f}, temp_impact={temporary_impact:.4f}, "
                    f"total={total_bps:.1f}bps")

        return {
            'permanent_impact': permanent_impact,
            'temporary_impact': temporary_impact,
            'total_bps': total_bps,
            'participation_rate': participation_rate
        }

    def is_acceptable(
        self,
        impact_result: Dict[str, float],
        max_impact_bps: float = 50.0,
        max_participation: float = 0.10
    ) -> tuple[bool, str]:
        """
        Check if impact is acceptable.

        Args:
            impact_result: Result from estimate_impact()
            max_impact_bps: Maximum acceptable impact in bps (default 50)
            max_participation: Maximum acceptable participation rate (default 10%)

        Returns:
            (is_acceptable, reason)
        """
        if impact_result['total_bps'] > max_impact_bps:
            return False, f"IMPACT_TOO_HIGH: {impact_result['total_bps']:.1f}bps > {max_impact_bps}bps"

        if impact_result['participation_rate'] > max_participation:
            return False, f"PARTICIPATION_TOO_HIGH: {impact_result['participation_rate']:.1%} > {max_participation:.1%}"

        return True, "IMPACT_OK"
