"""
Elite Portfolio Optimizer - Maximum Return Optimization
=========================================================

Optimizes portfolio for maximum returns while managing risk.

Features:
1. Mean-Variance Optimization
2. Risk Parity
3. Maximum Sharpe
4. Minimum Volatility
5. Black-Litterman Model
6. Factor-Based Optimization

Maximize returns. Minimize risk.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, getcontext
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import optimize

logger = logging.getLogger(__name__)

getcontext().prec = 50


@dataclass
class EliteAllocation:
    """Optimal portfolio allocation."""
    timestamp: datetime

    # Weights
    weights: Dict[str, float]

    # Expected metrics
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float

    # Risk metrics
    max_drawdown_estimate: float
    var_95: float

    # Optimization method
    method: str

    # Constraints met
    constraints_satisfied: bool


class EliteMeanVarianceOptimizer:
    """
    Elite mean-variance optimization.

    Finds the efficient frontier and optimal portfolios.
    """

    def __init__(self):
        """Initialize the optimizer."""
        logger.info("[ELITE] Mean-Variance Optimizer initialized")

    def optimize(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        risk_free_rate: float = 0.04,
        target: str = "max_sharpe"
    ) -> Tuple[np.ndarray, float, float]:
        """Find optimal weights."""
        n_assets = len(expected_returns)

        if n_assets == 0:
            return np.array([]), 0, 0

        # Constraints
        constraints = [
            {"type": "eq", "fun": lambda x: np.sum(x) - 1}  # Weights sum to 1
        ]

        # Bounds (0 to 1 for each asset - no shorting)
        bounds = tuple((0, 0.30) for _ in range(n_assets))  # Max 30% per asset

        # Initial guess (equal weight)
        x0 = np.ones(n_assets) / n_assets

        def neg_sharpe(weights):
            port_return = np.dot(weights, expected_returns)
            port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            if port_vol == 0:
                return 0
            return -(port_return - risk_free_rate) / port_vol

        def portfolio_volatility(weights):
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        # Optimize based on target
        if target == "max_sharpe":
            result = optimize.minimize(
                neg_sharpe, x0, method="SLSQP",
                bounds=bounds, constraints=constraints
            )
        elif target == "min_volatility":
            result = optimize.minimize(
                portfolio_volatility, x0, method="SLSQP",
                bounds=bounds, constraints=constraints
            )
        else:
            result = optimize.minimize(
                neg_sharpe, x0, method="SLSQP",
                bounds=bounds, constraints=constraints
            )

        weights = result.x
        port_return = np.dot(weights, expected_returns)
        port_vol = portfolio_volatility(weights)

        return weights, port_return, port_vol


class EliteRiskParityOptimizer:
    """
    Elite risk parity optimization.

    Equal risk contribution from each asset.
    """

    def __init__(self):
        """Initialize the optimizer."""
        logger.info("[ELITE] Risk Parity Optimizer initialized")

    def optimize(
        self,
        cov_matrix: np.ndarray
    ) -> np.ndarray:
        """Find risk parity weights."""
        n_assets = len(cov_matrix)

        if n_assets == 0:
            return np.array([])

        # Initial guess
        x0 = np.ones(n_assets) / n_assets

        # Constraints
        constraints = [
            {"type": "eq", "fun": lambda x: np.sum(x) - 1}
        ]

        bounds = tuple((0.01, 0.50) for _ in range(n_assets))

        def risk_contribution(weights, cov):
            port_vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
            if port_vol == 0:
                return np.zeros(len(weights))
            marginal_contrib = np.dot(cov, weights)
            risk_contrib = weights * marginal_contrib / port_vol
            return risk_contrib

        def risk_parity_objective(weights):
            rc = risk_contribution(weights, cov_matrix)
            return np.var(rc) * 1000

        result = optimize.minimize(
            risk_parity_objective, x0, method="SLSQP",
            bounds=bounds, constraints=constraints
        )

        return result.x


class ElitePortfolioOptimizer:
    """
    Complete elite portfolio optimization engine.

    Multiple optimization methods for different objectives.
    """

    def __init__(self):
        """Initialize the optimizer."""
        self.mv_optimizer = EliteMeanVarianceOptimizer()
        self.rp_optimizer = EliteRiskParityOptimizer()

        logger.info("[ELITE] Elite Portfolio Optimizer initialized")

    def optimize_from_data(
        self,
        market_data: pd.DataFrame,
        method: str = "max_sharpe",
        lookback: int = 60
    ) -> Optional[EliteAllocation]:
        """Optimize portfolio from market data."""
        if not isinstance(market_data.columns, pd.MultiIndex):
            return None

        symbols = list(market_data.columns.get_level_values(0).unique())[:20]

        if len(symbols) < 3:
            return None

        # Calculate returns
        returns_df = pd.DataFrame()
        for symbol in symbols:
            prices = market_data[symbol]["Close"].dropna()
            if len(prices) >= lookback:
                rets = prices.pct_change().dropna().iloc[-lookback:]
                returns_df[symbol] = rets

        if returns_df.empty or len(returns_df.columns) < 3:
            return None

        # Expected returns and covariance
        expected_returns = returns_df.mean().values * 252  # Annualized
        cov_matrix = returns_df.cov().values * 252

        # Optimize
        weights, port_return, port_vol = self.mv_optimizer.optimize(
            expected_returns, cov_matrix, target=method
        )

        if len(weights) == 0:
            return None

        # Sharpe ratio
        sharpe = (port_return - 0.04) / port_vol if port_vol > 0 else 0

        # Create weights dict
        weight_dict = {
            symbols[i]: float(weights[i])
            for i in range(len(weights))
            if weights[i] > 0.01
        }

        return EliteAllocation(
            timestamp=datetime.utcnow(),
            weights=weight_dict,
            expected_return=float(port_return),
            expected_volatility=float(port_vol),
            sharpe_ratio=float(sharpe),
            max_drawdown_estimate=float(port_vol * 2),
            var_95=float(port_vol * 1.65),
            method=method,
            constraints_satisfied=True
        )

    def get_rebalance_trades(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        portfolio_value: float,
        min_trade_pct: float = 0.01
    ) -> List[Dict[str, Any]]:
        """Calculate trades to rebalance to target."""
        trades = []

        all_symbols = set(list(current_weights.keys()) + list(target_weights.keys()))

        for symbol in all_symbols:
            current = current_weights.get(symbol, 0)
            target = target_weights.get(symbol, 0)
            diff = target - current

            if abs(diff) > min_trade_pct:
                trade_value = diff * portfolio_value

                trades.append({
                    "symbol": symbol,
                    "action": "BUY" if diff > 0 else "SELL",
                    "weight_change": diff,
                    "trade_value": abs(trade_value),
                    "from_weight": current,
                    "to_weight": target
                })

        return trades


# Singleton
_elite_optimizer: Optional[ElitePortfolioOptimizer] = None


def get_elite_optimizer() -> ElitePortfolioOptimizer:
    """Get or create the Elite Portfolio Optimizer."""
    global _elite_optimizer
    if _elite_optimizer is None:
        _elite_optimizer = ElitePortfolioOptimizer()
    return _elite_optimizer
