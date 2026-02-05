"""
Risk Parity - Bridgewater-style Equal Risk Contribution.

Allocates based on risk contribution, not capital:
- Each asset contributes equal volatility
- Inverse volatility weighting
- Leveraged for target return
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


@dataclass
class RiskParityAllocation:
    """Risk parity allocation result."""
    weights: Dict[str, float]
    risk_contributions: Dict[str, float]
    portfolio_volatility: float
    leverage: float
    diversification_ratio: float


class RiskParityOptimizer:
    """
    Risk Parity (Equal Risk Contribution) optimizer.

    Bridgewater's All Weather approach:
    - Each asset contributes equal risk
    - Weights inversely proportional to volatility
    - Leverage to achieve target return

    Objective: Minimize sum((w_i * sigma_i / sigma_p - 1/N)^2)
    """

    def __init__(
        self,
        target_volatility: float = 0.10,  # 10% annual vol
        max_leverage: float = 3.0,
        min_weight: float = 0.01
    ):
        self.target_volatility = target_volatility
        self.max_leverage = max_leverage
        self.min_weight = min_weight

    def inverse_volatility_weights(
        self,
        volatilities: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Simple inverse volatility weighting.

        w_i = (1/sigma_i) / sum(1/sigma_j)
        """
        symbols = list(volatilities.keys())
        vols = np.array([volatilities[s] for s in symbols])

        # Avoid division by zero
        vols = np.maximum(vols, 0.001)

        inv_vols = 1.0 / vols
        weights = inv_vols / inv_vols.sum()

        return {s: float(weights[i]) for i, s in enumerate(symbols)}

    def equal_risk_contribution(
        self,
        covariance_matrix: pd.DataFrame
    ) -> Dict[str, float]:
        """
        True equal risk contribution optimization.

        Solve: min sum((RC_i - RC_target)^2)
        where RC_i = w_i * (Sigma @ w)_i / sigma_portfolio
        """
        symbols = list(covariance_matrix.columns)
        n = len(symbols)
        cov = covariance_matrix.values

        def risk_contribution(weights):
            """Calculate risk contribution for each asset."""
            weights = np.array(weights)
            portfolio_vol = np.sqrt(weights @ cov @ weights)

            if portfolio_vol < 1e-10:
                return np.zeros(n)

            marginal_risk = cov @ weights
            risk_contrib = weights * marginal_risk / portfolio_vol
            return risk_contrib

        def objective(weights):
            """Minimize deviation from equal risk contribution."""
            rc = risk_contribution(weights)
            target_rc = 1.0 / n
            return np.sum((rc - target_rc) ** 2)

        # Initial guess: equal weights
        x0 = np.ones(n) / n

        # Constraints: weights sum to 1, all positive
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        ]
        bounds = [(self.min_weight, 1.0) for _ in range(n)]

        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000}
        )

        if result.success:
            weights = result.x
        else:
            logger.warning("Risk parity optimization failed, using inverse vol")
            vols = np.sqrt(np.diag(cov))
            weights = (1.0 / vols) / (1.0 / vols).sum()

        return {s: float(weights[i]) for i, s in enumerate(symbols)}

    def optimize(
        self,
        covariance_matrix: pd.DataFrame,
        expected_returns: Optional[Dict[str, float]] = None
    ) -> RiskParityAllocation:
        """
        Compute risk parity allocation.

        Args:
            covariance_matrix: Covariance matrix DataFrame
            expected_returns: Optional expected returns for Sharpe optimization

        Returns:
            RiskParityAllocation
        """
        symbols = list(covariance_matrix.columns)
        n = len(symbols)

        # Get equal risk contribution weights
        weights = self.equal_risk_contribution(covariance_matrix)

        # Convert to arrays
        w = np.array([weights[s] for s in symbols])
        cov = covariance_matrix.values

        # Calculate portfolio volatility
        port_vol = np.sqrt(w @ cov @ w)

        # Calculate risk contributions
        marginal_risk = cov @ w
        risk_contribs = w * marginal_risk / max(port_vol, 1e-10)

        # Calculate leverage for target volatility
        if port_vol > 0:
            leverage = min(self.target_volatility / port_vol, self.max_leverage)
        else:
            leverage = 1.0

        # Apply leverage
        leveraged_weights = {s: weights[s] * leverage for s in symbols}

        # Diversification ratio
        individual_vols = np.sqrt(np.diag(cov))
        weighted_avg_vol = w @ individual_vols
        diversification_ratio = weighted_avg_vol / max(port_vol, 1e-10)

        return RiskParityAllocation(
            weights=leveraged_weights,
            risk_contributions={
                s: float(risk_contribs[i]) for i, s in enumerate(symbols)
            },
            portfolio_volatility=float(port_vol * leverage),
            leverage=leverage,
            diversification_ratio=float(diversification_ratio)
        )


# Global singleton
_risk_parity: Optional[RiskParityOptimizer] = None


def get_risk_parity_optimizer() -> RiskParityOptimizer:
    """Get or create global risk parity optimizer."""
    global _risk_parity
    if _risk_parity is None:
        _risk_parity = RiskParityOptimizer()
    return _risk_parity
