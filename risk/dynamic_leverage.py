"""
Dynamic Leverage Engine - Regime-Adaptive Position Sizing.

Elite hedge funds adjust leverage based on:
- Market regime (bull/bear/crisis)
- Volatility (VIX level)
- Drawdown state
- Correlation breakdown

This engine scales exposure from 0x (cash) to 2x (max leverage).
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class LeverageRegime(Enum):
    """Leverage regime categories."""
    AGGRESSIVE = "AGGRESSIVE"  # 1.5-2.0x
    NORMAL = "NORMAL"  # 1.0x
    DEFENSIVE = "DEFENSIVE"  # 0.5-0.7x
    CRISIS = "CRISIS"  # 0.0-0.3x


@dataclass
class LeverageDecision:
    """Leverage decision with full rationale."""
    target_leverage: float
    regime: LeverageRegime

    # Contributing factors
    regime_score: float  # -1 (crisis) to 1 (bull)
    vix_adjustment: float
    drawdown_adjustment: float
    correlation_adjustment: float

    # Rationale
    reasons: list

    # Action
    action: str  # "INCREASE", "DECREASE", "HOLD"


class DynamicLeverageEngine:
    """
    Compute optimal leverage based on market conditions.

    Approach:
    1. Base leverage from market regime (Markov HMM)
    2. Adjust for VIX level (fear indicator)
    3. Adjust for current drawdown
    4. Adjust for correlation breakdown (diversification loss)

    Output: 0.0 to 2.0x leverage target
    """

    # Regime -> base leverage mapping
    REGIME_LEVERAGE = {
        "BULL_QUIET": 1.5,
        "BULL_VOLATILE": 1.0,
        "NEUTRAL": 0.8,
        "BEAR_QUIET": 0.5,
        "BEAR_VOLATILE": 0.3,
        "CRISIS": 0.0,
    }

    def __init__(
        self,
        max_leverage: float = 2.0,
        min_leverage: float = 0.0,
        vix_neutral: float = 20.0,
        vix_panic: float = 35.0,
        max_drawdown: float = 0.15,
        correlation_threshold: float = 0.8
    ):
        self.max_leverage = max_leverage
        self.min_leverage = min_leverage
        self.vix_neutral = vix_neutral
        self.vix_panic = vix_panic
        self.max_drawdown = max_drawdown
        self.correlation_threshold = correlation_threshold

        self.current_leverage = 1.0
        self.leverage_history: list = []

    def compute_leverage(
        self,
        regime: str,
        vix: float,
        current_drawdown: float,
        avg_correlation: float = 0.5,
        current_equity: float = 100000,
        peak_equity: float = 100000
    ) -> LeverageDecision:
        """
        Compute optimal leverage for current conditions.

        Args:
            regime: Market regime string
            vix: Current VIX level
            current_drawdown: Current portfolio drawdown (0 to 1)
            avg_correlation: Average pairwise correlation

        Returns:
            LeverageDecision with target leverage and rationale
        """
        reasons = []

        # 1. Base leverage from regime
        base_leverage = self.REGIME_LEVERAGE.get(regime, 0.8)
        regime_score = self._regime_to_score(regime)
        reasons.append(f"Regime {regime} -> base {base_leverage:.1f}x")

        # 2. VIX adjustment
        vix_adj = self._vix_adjustment(vix)
        reasons.append(f"VIX {vix:.1f} -> adjustment {vix_adj:.2f}")

        # 3. Drawdown adjustment
        dd_adj = self._drawdown_adjustment(current_drawdown)
        reasons.append(f"Drawdown {current_drawdown:.1%} -> adjustment {dd_adj:.2f}")

        # 4. Correlation adjustment
        corr_adj = self._correlation_adjustment(avg_correlation)
        reasons.append(f"Avg correlation {avg_correlation:.2f} -> adjustment {corr_adj:.2f}")

        # Combine adjustments
        target = base_leverage * vix_adj * dd_adj * corr_adj
        target = float(np.clip(target, self.min_leverage, self.max_leverage))

        # Determine regime category
        if target >= 1.3:
            leverage_regime = LeverageRegime.AGGRESSIVE
        elif target >= 0.8:
            leverage_regime = LeverageRegime.NORMAL
        elif target >= 0.3:
            leverage_regime = LeverageRegime.DEFENSIVE
        else:
            leverage_regime = LeverageRegime.CRISIS

        # Determine action
        if target > self.current_leverage * 1.1:
            action = "INCREASE"
        elif target < self.current_leverage * 0.9:
            action = "DECREASE"
        else:
            action = "HOLD"

        decision = LeverageDecision(
            target_leverage=round(target, 2),
            regime=leverage_regime,
            regime_score=regime_score,
            vix_adjustment=vix_adj,
            drawdown_adjustment=dd_adj,
            correlation_adjustment=corr_adj,
            reasons=reasons,
            action=action
        )

        # Update state
        self.current_leverage = target
        self.leverage_history.append({
            "leverage": target,
            "regime": regime,
            "vix": vix
        })

        # Keep last 100 records
        if len(self.leverage_history) > 100:
            self.leverage_history = self.leverage_history[-100:]

        logger.info(
            f"DynamicLeverage: {regime} -> {target:.2f}x ({leverage_regime.value})"
        )

        return decision

    def _regime_to_score(self, regime: str) -> float:
        """Convert regime to -1 to 1 score."""
        scores = {
            "BULL_QUIET": 1.0,
            "BULL_VOLATILE": 0.5,
            "NEUTRAL": 0.0,
            "BEAR_QUIET": -0.3,
            "BEAR_VOLATILE": -0.7,
            "CRISIS": -1.0,
        }
        return scores.get(regime, 0.0)

    def _vix_adjustment(self, vix: float) -> float:
        """
        Adjust leverage based on VIX.

        VIX < 15: 1.2x (calm markets, add leverage)
        VIX 15-25: 1.0x (normal)
        VIX 25-35: 0.7x (elevated fear)
        VIX > 35: 0.4x (panic)
        """
        if vix < 15:
            return 1.2
        elif vix < 20:
            return 1.1
        elif vix < 25:
            return 1.0
        elif vix < 30:
            return 0.8
        elif vix < 35:
            return 0.6
        else:
            return 0.4

    def _drawdown_adjustment(self, drawdown: float) -> float:
        """
        Adjust leverage based on current drawdown.

        Drawdown < 5%: Full leverage
        Drawdown 5-10%: 80% leverage
        Drawdown 10-15%: 50% leverage
        Drawdown > 15%: 25% leverage (capital preservation mode)
        """
        if drawdown < 0.05:
            return 1.0
        elif drawdown < 0.10:
            return 0.8
        elif drawdown < 0.15:
            return 0.5
        else:
            return 0.25

    def _correlation_adjustment(self, avg_correlation: float) -> float:
        """
        Adjust leverage based on correlation.

        High correlation = less diversification benefit = reduce leverage
        """
        if avg_correlation < 0.3:
            return 1.1  # Good diversification
        elif avg_correlation < 0.5:
            return 1.0  # Normal
        elif avg_correlation < 0.7:
            return 0.9  # Elevated
        else:
            return 0.7  # Correlation breakdown

    def get_position_scalar(self, base_weight: float) -> float:
        """
        Scale position weight by current leverage target.

        Args:
            base_weight: Original position weight (0 to 1)

        Returns:
            Scaled weight
        """
        return base_weight * self.current_leverage

    def suggest_rebalance(self, current_weights: Dict[str, float]) -> Dict[str, float]:
        """
        Scale all weights by leverage target.
        """
        return {
            k: v * self.current_leverage
            for k, v in current_weights.items()
        }

    def get_risk_budget(self) -> float:
        """
        Get current risk budget as a percentage.

        At 2x leverage, risk budget = 200%
        At 0.5x leverage, risk budget = 50%
        """
        return self.current_leverage * 100


# Global singleton
_leverage_engine: Optional[DynamicLeverageEngine] = None


def get_leverage_engine() -> DynamicLeverageEngine:
    """Get or create global DynamicLeverageEngine."""
    global _leverage_engine
    if _leverage_engine is None:
        _leverage_engine = DynamicLeverageEngine()
    return _leverage_engine
