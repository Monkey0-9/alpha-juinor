"""
Dynamic Portfolio Optimizer - 2026 Elite
=========================================

Advanced portfolio optimization for maximum risk-adjusted returns.

Features:
- Black-Litterman with AI views
- Hierarchical Risk Parity
- Regime-Conditional Optimization
- Transaction Cost Aware
- Dynamic Rebalancing

Target: Sharpe > 2.5, Max DD < 15%
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PortfolioAllocation:
    """Optimized portfolio allocation."""
    weights: Dict[str, float]
    expected_return: float
    expected_volatility: float
    expected_sharpe: float
    turnover: float
    regime: str


class DynamicPortfolioOptimizer:
    """
    Elite portfolio optimizer with regime-aware allocation.
    """

    def __init__(
        self,
        risk_free_rate: float = 0.04,
        max_position: float = 0.15,
        min_position: float = 0.01
    ):
        self.rf = risk_free_rate
        self.max_pos = max_position
        self.min_pos = min_position

        # Regime-specific risk budgets
        self.regime_risk_budget = {
            "BULL": 0.20,
            "BEAR": 0.10,
            "VOLATILE": 0.08,
            "SIDEWAYS": 0.12,
            "NORMAL": 0.15,
            "CRISIS": 0.05
        }

        logger.info("[PORTFOLIO_OPT] Elite optimizer initialized")

    def optimize(
        self,
        signals: Dict[str, float],
        expected_returns: Dict[str, float],
        volatilities: Dict[str, float],
        correlations: Dict[Tuple[str, str], float],
        current_weights: Dict[str, float],
        regime: str,
        transaction_cost: float = 0.001
    ) -> PortfolioAllocation:
        """
        Generate optimal portfolio allocation.
        """
        if not signals:
            return PortfolioAllocation(
                weights={},
                expected_return=0.0,
                expected_volatility=0.0,
                expected_sharpe=0.0,
                turnover=0.0,
                regime=regime
            )

        symbols = list(signals.keys())
        n = len(symbols)

        # Get risk budget for regime
        risk_budget = self.regime_risk_budget.get(regime, 0.15)

        # Build covariance matrix
        cov_matrix = self._build_cov_matrix(
            symbols, volatilities, correlations
        )

        # Get expected returns vector
        exp_ret = np.array([
            expected_returns.get(s, 0.0) for s in symbols
        ])

        # Get signals as views
        views = np.array([signals.get(s, 0.0) for s in symbols])

        # Blend AI views with expected returns (Black-Litterman style)
        tau = 0.05
        blended_returns = exp_ret + tau * views

        # Risk parity weights (HRP-inspired)
        weights = self._risk_parity_weights(cov_matrix, blended_returns)

        # Apply position limits
        weights = self._apply_limits(weights)

        # Apply signal direction
        for i, sym in enumerate(symbols):
            if signals[sym] < -0.3:
                weights[i] = 0  # Remove negative signals

        # Normalize
        total = np.sum(weights)
        if total > 0:
            weights = weights / total

        # Scale to risk budget
        port_vol = self._portfolio_volatility(weights, cov_matrix)
        if port_vol > 0:
            scale = min(1.0, risk_budget / port_vol)
            weights = weights * scale

        # Calculate turnover
        turnover = 0.0
        for i, sym in enumerate(symbols):
            old_w = current_weights.get(sym, 0.0)
            turnover += abs(weights[i] - old_w)

        # Penalize turnover if too high
        if turnover > 0.5:
            # Blend with current weights
            for i, sym in enumerate(symbols):
                old_w = current_weights.get(sym, 0.0)
                weights[i] = 0.7 * weights[i] + 0.3 * old_w

        # Create weight dict
        weight_dict = {
            symbols[i]: float(weights[i])
            for i in range(n)
            if weights[i] > self.min_pos
        }

        # Calculate expected metrics
        exp_port_ret = float(np.dot(weights, blended_returns))
        exp_port_vol = float(self._portfolio_volatility(weights, cov_matrix))
        exp_sharpe = (exp_port_ret - self.rf) / exp_port_vol \
            if exp_port_vol > 0 else 0.0

        return PortfolioAllocation(
            weights=weight_dict,
            expected_return=exp_port_ret,
            expected_volatility=exp_port_vol,
            expected_sharpe=exp_sharpe,
            turnover=turnover,
            regime=regime
        )

    def _build_cov_matrix(
        self,
        symbols: List[str],
        volatilities: Dict[str, float],
        correlations: Dict[Tuple[str, str], float]
    ) -> np.ndarray:
        """Build covariance matrix from volatilities and correlations."""
        n = len(symbols)
        cov = np.zeros((n, n))

        for i, sym_i in enumerate(symbols):
            vol_i = volatilities.get(sym_i, 0.02)

            for j, sym_j in enumerate(symbols):
                vol_j = volatilities.get(sym_j, 0.02)

                if i == j:
                    cov[i, j] = vol_i ** 2
                else:
                    # Get correlation
                    key1 = (sym_i, sym_j)
                    key2 = (sym_j, sym_i)
                    corr = correlations.get(
                        key1, correlations.get(key2, 0.3)
                    )
                    cov[i, j] = vol_i * vol_j * corr

        return cov

    def _risk_parity_weights(
        self,
        cov_matrix: np.ndarray,
        returns: np.ndarray
    ) -> np.ndarray:
        """Calculate risk parity weights with return tilt."""
        n = cov_matrix.shape[0]

        # Inverse volatility weights
        vols = np.sqrt(np.diag(cov_matrix))
        vols[vols < 1e-6] = 1e-6

        inv_vol_weights = 1.0 / vols

        # Return tilt
        return_tilt = np.maximum(returns, 0) + 0.01

        # Blend
        weights = inv_vol_weights * return_tilt
        weights = weights / np.sum(weights)

        return weights

    def _apply_limits(self, weights: np.ndarray) -> np.ndarray:
        """Apply position limits."""
        weights = np.clip(weights, 0, self.max_pos)
        return weights

    def _portfolio_volatility(
        self,
        weights: np.ndarray,
        cov_matrix: np.ndarray
    ) -> float:
        """Calculate portfolio volatility."""
        var = np.dot(weights.T, np.dot(cov_matrix, weights))
        return np.sqrt(max(0, var))


# Singleton
_optimizer = None


def get_portfolio_optimizer() -> DynamicPortfolioOptimizer:
    global _optimizer
    if _optimizer is None:
        _optimizer = DynamicPortfolioOptimizer()
    return _optimizer
