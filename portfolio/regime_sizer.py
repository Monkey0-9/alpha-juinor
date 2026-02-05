"""
Regime-Adaptive Position Sizer - Dynamic Risk Allocation.

Features:
- Adjust position sizes based on market regime
- VIX-scaled sizing
- Correlation-adjusted limits
- Kelly-constrained positions
"""

import logging
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class RegimeType(Enum):
    BULL_LOW_VOL = "BULL_LOW_VOL"
    BULL_HIGH_VOL = "BULL_HIGH_VOL"
    BEAR_LOW_VOL = "BEAR_LOW_VOL"
    BEAR_HIGH_VOL = "BEAR_HIGH_VOL"
    CRISIS = "CRISIS"
    TRANSITION = "TRANSITION"


@dataclass
class PositionSize:
    """Position sizing result."""
    symbol: str
    base_size: float
    adjusted_size: float
    regime: RegimeType
    adjustments: Dict[str, float]
    max_loss: float


class RegimePositionSizer:
    """
    Regime-adaptive position sizing.

    Sizing factors:
    - Regime type (bull/bear, vol level)
    - VIX level
    - Asset correlation
    - Kelly fraction
    - Max loss constraint
    """

    def __init__(
        self,
        base_risk_per_trade: float = 0.02,  # 2% risk per trade
        max_position: float = 0.15,  # Max 15% per position
        max_portfolio_heat: float = 0.25  # Max 25% total risk
    ):
        self.base_risk_per_trade = base_risk_per_trade
        self.max_position = max_position
        self.max_portfolio_heat = max_portfolio_heat

        # Regime multipliers
        self.regime_multipliers = {
            RegimeType.BULL_LOW_VOL: 1.5,
            RegimeType.BULL_HIGH_VOL: 0.8,
            RegimeType.BEAR_LOW_VOL: 0.6,
            RegimeType.BEAR_HIGH_VOL: 0.4,
            RegimeType.CRISIS: 0.2,
            RegimeType.TRANSITION: 0.5
        }

        # VIX scaling thresholds
        self.vix_thresholds = {
            12: 1.3,   # Very low vol - increase size
            15: 1.0,   # Normal
            20: 0.8,
            25: 0.6,
            30: 0.4,
            40: 0.2    # High vol - reduce size
        }

    def detect_regime(
        self,
        market_return: float,
        volatility: float,
        vix: float
    ) -> RegimeType:
        """Detect current market regime."""
        is_bull = market_return > 0
        is_high_vol = volatility > 0.015 or vix > 20

        if vix > 35:
            return RegimeType.CRISIS

        if is_bull:
            if is_high_vol:
                return RegimeType.BULL_HIGH_VOL
            else:
                return RegimeType.BULL_LOW_VOL
        else:
            if is_high_vol:
                return RegimeType.BEAR_HIGH_VOL
            else:
                return RegimeType.BEAR_LOW_VOL

    def get_vix_multiplier(self, vix: float) -> float:
        """Get position multiplier based on VIX."""
        for threshold, multiplier in sorted(self.vix_thresholds.items()):
            if vix <= threshold:
                return multiplier
        return 0.1  # Extreme VIX

    def calculate_correlation_adjustment(
        self,
        symbol: str,
        correlations: Dict[str, float]
    ) -> float:
        """
        Adjust size based on correlation to existing positions.

        High correlation = smaller position to avoid concentration.
        """
        if not correlations:
            return 1.0

        avg_correlation = np.mean(list(correlations.values()))

        # Reduce size for highly correlated positions
        if avg_correlation > 0.7:
            return 0.5
        elif avg_correlation > 0.5:
            return 0.7
        elif avg_correlation > 0.3:
            return 0.85
        else:
            return 1.0

    def size_position(
        self,
        symbol: str,
        portfolio_value: float,
        entry_price: float,
        stop_loss: float,
        regime: RegimeType,
        vix: float,
        kelly_fraction: float = 0.5,
        correlations: Optional[Dict[str, float]] = None
    ) -> PositionSize:
        """
        Calculate position size with all adjustments.
        """
        adjustments = {}

        # Base position from risk
        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share == 0:
            risk_per_share = entry_price * 0.05  # Default 5% stop

        risk_amount = portfolio_value * self.base_risk_per_trade
        base_shares = risk_amount / risk_per_share
        base_value = base_shares * entry_price
        base_size = base_value / portfolio_value

        # Apply regime adjustment
        regime_mult = self.regime_multipliers.get(regime, 1.0)
        adjustments["regime"] = regime_mult

        # Apply VIX adjustment
        vix_mult = self.get_vix_multiplier(vix)
        adjustments["vix"] = vix_mult

        # Apply correlation adjustment
        corr_mult = self.calculate_correlation_adjustment(symbol, correlations or {})
        adjustments["correlation"] = corr_mult

        # Apply Kelly constraint
        kelly_mult = min(1.0, kelly_fraction * 2)  # Scale Kelly
        adjustments["kelly"] = kelly_mult

        # Calculate adjusted size
        total_mult = regime_mult * vix_mult * corr_mult * kelly_mult
        adjusted_size = base_size * total_mult

        # Apply limits
        adjusted_size = min(adjusted_size, self.max_position)

        # Calculate max loss
        max_loss = adjusted_size * (risk_per_share / entry_price)

        return PositionSize(
            symbol=symbol,
            base_size=base_size,
            adjusted_size=adjusted_size,
            regime=regime,
            adjustments=adjustments,
            max_loss=max_loss
        )

    def check_portfolio_heat(
        self,
        current_positions: Dict[str, float],
        new_position: PositionSize
    ) -> Tuple[bool, float]:
        """
        Check if adding position exceeds portfolio heat.
        """
        current_heat = sum(current_positions.values())
        new_heat = current_heat + new_position.max_loss

        allowed = new_heat <= self.max_portfolio_heat

        return allowed, new_heat


# Global singleton
_position_sizer: Optional[RegimePositionSizer] = None


def get_position_sizer() -> RegimePositionSizer:
    """Get or create global position sizer."""
    global _position_sizer
    if _position_sizer is None:
        _position_sizer = RegimePositionSizer()
    return _position_sizer
