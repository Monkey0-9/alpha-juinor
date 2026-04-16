"""
Kelly Criterion Optimizer - Optimal Position Sizing.

Multi-asset Kelly formula for maximizing long-term growth:
- Full Kelly for maximum growth
- Half-Kelly for reduced variance
- Drawdown-adjusted Kelly for safety
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class KellyAllocation:
    """Kelly-optimal allocation result."""
    weights: Dict[str, float]
    full_kelly: Dict[str, float]
    fractional_kelly: float  # Fraction used
    expected_growth: float
    expected_variance: float
    leverage: float


class KellyOptimizer:
    """
    Multi-asset Kelly Criterion optimizer.

    Kelly formula for single asset:
    f* = (p * b - q) / b
    where p = win probability, q = 1-p, b = win/loss ratio

    Multi-asset Kelly:
    f* = Sigma^(-1) * mu
    """

    def __init__(
        self,
        kelly_fraction: float = 0.5,  # Half-Kelly for safety
        max_leverage: float = 2.0,
        max_position: float = 0.25,  # Max 25% per position
        drawdown_adjustment: bool = True,
        max_drawdown_trigger: float = 0.10
    ):
        self.kelly_fraction = kelly_fraction
        self.max_leverage = max_leverage
        self.max_position = max_position
        self.drawdown_adjustment = drawdown_adjustment
        self.max_drawdown_trigger = max_drawdown_trigger

        self.current_drawdown = 0.0

    def compute_full_kelly(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Compute full Kelly weights.

        f* = Sigma^(-1) * mu
        """
        try:
            # Regularize covariance for numerical stability
            cov_reg = covariance_matrix + np.eye(len(expected_returns)) * 1e-6

            # Compute Kelly weights
            cov_inv = np.linalg.inv(cov_reg)
            kelly_weights = cov_inv @ expected_returns

            return kelly_weights

        except np.linalg.LinAlgError:
            logger.warning("Covariance matrix singular, using diagonal approximation")
            variances = np.diag(covariance_matrix) + 1e-6
            return expected_returns / variances

    def adjust_for_drawdown(self, weights: np.ndarray) -> np.ndarray:
        """
        Reduce Kelly fraction during drawdowns.

        At 10% drawdown, reduce to 50% of target Kelly
        At 15% drawdown, reduce to 25% of target Kelly
        """
        if not self.drawdown_adjustment:
            return weights

        if self.current_drawdown < 0.05:
            adjustment = 1.0
        elif self.current_drawdown < 0.10:
            adjustment = 0.75
        elif self.current_drawdown < 0.15:
            adjustment = 0.50
        else:
            adjustment = 0.25

        return weights * adjustment

    def apply_constraints(
        self,
        weights: np.ndarray,
        symbols: List[str]
    ) -> np.ndarray:
        """Apply position limits and leverage constraints."""
        # Clip individual positions
        weights = np.clip(weights, -self.max_position, self.max_position)

        # Apply leverage constraint
        total_exposure = np.sum(np.abs(weights))
        if total_exposure > self.max_leverage:
            weights = weights * (self.max_leverage / total_exposure)

        return weights

    def optimize(
        self,
        expected_returns: Dict[str, float],
        covariance_matrix: pd.DataFrame,
        current_drawdown: float = 0.0
    ) -> KellyAllocation:
        """
        Compute Kelly-optimal allocation.

        Args:
            expected_returns: Dict of symbol -> expected return
            covariance_matrix: DataFrame with covariance
            current_drawdown: Current portfolio drawdown

        Returns:
            KellyAllocation with weights and metadata
        """
        self.current_drawdown = current_drawdown

        symbols = list(expected_returns.keys())
        n_assets = len(symbols)

        # Convert to arrays
        mu = np.array([expected_returns[s] for s in symbols])
        cov = covariance_matrix.loc[symbols, symbols].values

        # Compute full Kelly
        full_kelly = self.compute_full_kelly(mu, cov)

        # Apply fractional Kelly
        fractional = full_kelly * self.kelly_fraction

        # Adjust for drawdown
        adjusted = self.adjust_for_drawdown(fractional)

        # Apply constraints
        constrained = self.apply_constraints(adjusted, symbols)

        # Compute expected growth (Kelly criterion)
        expected_growth = mu @ constrained - 0.5 * constrained @ cov @ constrained
        expected_variance = constrained @ cov @ constrained

        leverage = float(np.sum(np.abs(constrained)))

        return KellyAllocation(
            weights={s: float(constrained[i]) for i, s in enumerate(symbols)},
            full_kelly={s: float(full_kelly[i]) for i, s in enumerate(symbols)},
            fractional_kelly=self.kelly_fraction,
            expected_growth=float(expected_growth),
            expected_variance=float(expected_variance),
            leverage=leverage
        )

    def single_asset_kelly(
        self,
        win_probability: float,
        win_loss_ratio: float
    ) -> float:
        """
        Classic single-asset Kelly formula.

        f* = (p * b - q) / b
        """
        p = win_probability
        q = 1 - p
        b = win_loss_ratio

        kelly = (p * b - q) / b
        return max(0, kelly * self.kelly_fraction)

    def update_drawdown(self, current_equity: float, peak_equity: float):
        """Update current drawdown for adjustment."""
        self.current_drawdown = (peak_equity - current_equity) / peak_equity


# Global singleton
_kelly: Optional[KellyOptimizer] = None


def get_kelly_optimizer() -> KellyOptimizer:
    """Get or create global Kelly optimizer."""
    global _kelly
    if _kelly is None:
        _kelly = KellyOptimizer()
    return _kelly
