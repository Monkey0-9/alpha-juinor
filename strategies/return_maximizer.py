"""
Return Maximizer - Aggressive Portfolio Optimization.

Targets 60-70% annual returns through:
- Dynamic Kelly Criterion sizing
- Concentration in high-conviction bets
- Trend following overlay
- Regime-aware allocation
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class OpportunityScore:
    """Scoring for return potential."""
    symbol: str
    expected_return: float
    volatility: float
    sharpe_ratio: float
    conviction: float
    kelly_fraction: float
    recommended_weight: float


class ReturnMaximizer:
    """
    Aggressive return optimization engine.

    Strategies:
    1. Kelly Criterion-based sizing
    2. Concentration in top ideas
    3. Trend following overlay
    4. Volatility-scaled positions
    """

    def __init__(
        self,
        target_return: float = 0.65,  # 65% target
        max_concentration: float = 0.25,
        min_positions: int = 5,
        max_positions: int = 20,
        kelly_fraction: float = 0.5,  # Half-Kelly for safety
        trend_weight: float = 0.3
    ):
        self.target_return = target_return
        self.max_concentration = max_concentration
        self.min_positions = min_positions
        self.max_positions = max_positions
        self.kelly_fraction = kelly_fraction
        self.trend_weight = trend_weight

    def calculate_kelly(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float
    ) -> float:
        """
        Calculate Kelly Criterion fraction.

        f* = (p * b - q) / b
        where p = win rate, q = 1-p, b = win/loss ratio
        """
        if avg_loss == 0 or win_rate <= 0:
            return 0.0

        p = win_rate
        q = 1 - p
        b = avg_win / avg_loss

        kelly = (p * b - q) / b

        # Apply fraction for safety
        kelly = kelly * self.kelly_fraction

        # Clamp to reasonable range
        return float(np.clip(kelly, 0, self.max_concentration))

    def score_opportunity(
        self,
        symbol: str,
        expected_return: float,
        volatility: float,
        win_rate: float = 0.55,
        conviction: float = 0.5
    ) -> OpportunityScore:
        """Score a trading opportunity."""
        # Sharpe ratio (assume risk-free = 0)
        sharpe = expected_return / volatility if volatility > 0 else 0

        # Kelly fraction
        avg_win = expected_return * 2  # Assume 2:1 win/loss
        avg_loss = expected_return
        kelly = self.calculate_kelly(win_rate, avg_win, avg_loss)

        # Recommended weight (scaled by conviction)
        rec_weight = kelly * conviction

        return OpportunityScore(
            symbol=symbol,
            expected_return=expected_return,
            volatility=volatility,
            sharpe_ratio=sharpe,
            conviction=conviction,
            kelly_fraction=kelly,
            recommended_weight=rec_weight
        )

    def optimize_portfolio(
        self,
        opportunities: List[OpportunityScore],
        current_regime: str = "NEUTRAL"
    ) -> Dict[str, float]:
        """
        Optimize portfolio for maximum return.

        Returns: Dict of symbol -> weight
        """
        if not opportunities:
            return {}

        # Sort by expected Sharpe * conviction
        ranked = sorted(
            opportunities,
            key=lambda x: x.sharpe_ratio * x.conviction,
            reverse=True
        )

        # Select top opportunities
        n_positions = min(len(ranked), self.max_positions)
        n_positions = max(n_positions, self.min_positions)
        selected = ranked[:n_positions]

        # Regime adjustment
        regime_scale = self._get_regime_scale(current_regime)

        # Allocate weights
        weights = {}
        total_score = sum(o.sharpe_ratio * o.conviction for o in selected)

        for opp in selected:
            if total_score > 0:
                base_weight = (opp.sharpe_ratio * opp.conviction) / total_score
            else:
                base_weight = 1.0 / n_positions

            # Apply Kelly and concentration limits
            weight = min(
                base_weight * regime_scale,
                opp.recommended_weight,
                self.max_concentration
            )
            weights[opp.symbol] = weight

        # Normalize to sum to 1 (or less if conservative)
        total_weight = sum(weights.values())
        if total_weight > 1.0:
            weights = {k: v / total_weight for k, v in weights.items()}

        return weights

    def _get_regime_scale(self, regime: str) -> float:
        """Get scaling factor based on market regime."""
        scales = {
            "BULL_QUIET": 1.2,
            "BULL_VOLATILE": 1.0,
            "NEUTRAL": 0.9,
            "BEAR_QUIET": 0.7,
            "BEAR_VOLATILE": 0.5,
            "CRISIS": 0.3,
        }
        return scales.get(regime, 0.8)

    def apply_trend_overlay(
        self,
        base_weights: Dict[str, float],
        trend_signals: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Apply trend following overlay to base weights.

        trend_signals: Dict of symbol -> trend score (-1 to 1)
        """
        adjusted = {}

        for symbol, base in base_weights.items():
            trend = trend_signals.get(symbol, 0.0)

            # Boost aligned trends, reduce counter-trend
            if trend > 0:
                adjustment = 1 + (trend * self.trend_weight)
            else:
                adjustment = 1 + (trend * self.trend_weight * 0.5)

            adjusted[symbol] = base * adjustment

        # Re-normalize
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {k: v / total for k, v in adjusted.items()}

        return adjusted

    def estimate_portfolio_return(
        self,
        weights: Dict[str, float],
        expected_returns: Dict[str, float]
    ) -> float:
        """Estimate expected portfolio return."""
        portfolio_return = 0.0
        for symbol, weight in weights.items():
            ret = expected_returns.get(symbol, 0.0)
            portfolio_return += weight * ret
        return portfolio_return

    def check_target_achievable(
        self,
        weights: Dict[str, float],
        expected_returns: Dict[str, float]
    ) -> Dict[str, Any]:
        """Check if target return is achievable with current setup."""
        est_return = self.estimate_portfolio_return(weights, expected_returns)

        achievable = est_return >= self.target_return * 0.8  # 80% of target

        gap = self.target_return - est_return
        concentration = max(weights.values()) if weights else 0

        return {
            "estimated_return": est_return,
            "target_return": self.target_return,
            "gap": gap,
            "achievable": achievable,
            "max_concentration": concentration,
            "positions": len(weights),
            "recommendation": self._get_recommendation(est_return, gap)
        }

    def _get_recommendation(
        self, est_return: float, gap: float
    ) -> str:
        """Generate recommendation based on return gap."""
        if gap <= 0:
            return "ON_TARGET: Current opportunities meet return goal"
        elif gap < 0.10:
            return "CLOSE: Slightly below target, consider increasing conviction bets"
        elif gap < 0.25:
            return "GAP: Seek higher-conviction opportunities or increase position sizes"
        else:
            return "CHALLENGING: Significant gap to target, review opportunity set"


# Global instance
_maximizer: Optional[ReturnMaximizer] = None


def get_return_maximizer() -> ReturnMaximizer:
    """Get or create global ReturnMaximizer."""
    global _maximizer
    if _maximizer is None:
        _maximizer = ReturnMaximizer()
    return _maximizer
