"""
execution/strategies/almgren_chriss.py

Almgren-Chriss Optimal Execution Model
Implements optimal trading trajectories for executing large orders
while minimizing market impact.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ACMeta:
    """Metadata for Almgren-Chriss optimization."""

    def __init__(self, symbol: str, shares: float, duration_hours: float):
        self.symbol = symbol
        self.shares = shares
        self.duration_hours = duration_hours


class AlmgrenChrissOptimizer:
    """
    Almgren-Chriss Optimal Execution algorithm.

    Minimizes expected cost + risk of executing a large order by
    choosing an optimal execution trajectory.

    Parameters:
    - alpha: Permanent market impact coefficient
    - beta: Temporary market impact coefficient
    - gamma: Volatility parameter
    """

    def __init__(
        self,
        alpha: float = 0.05,
        beta: float = 0.1,
        gamma: float = 0.5,
    ):
        self.alpha = alpha  # Permanent impact
        self.beta = beta  # Temporary impact
        self.gamma = gamma  # Volatility
        logger.info(
            f"[AlmgrenChriss] Initialized: alpha={alpha}, beta={beta}, gamma={gamma}"
        )

    def optimize(
        self,
        symbol: str,
        total_shares: float,
        duration_hours: float,
        lambda_param: float = 0.5,
    ) -> Dict[str, float]:
        """
        Compute optimal execution schedule.

        Args:
            symbol: Trading symbol
            total_shares: Total shares to execute
            duration_hours: Execution time horizon
            lambda_param: Risk aversion parameter (0=min cost, 1=min volume)

        Returns:
            Dictionary with execution parameters and trajectory
        """
        # Number of time intervals
        n_intervals = max(int(duration_hours * 4), 1)  # 15-min buckets
        dt = duration_hours / n_intervals

        # Theoretical shares per interval
        shares_per_interval = total_shares / n_intervals

        # Adjust for costs using Almgren-Chriss formula
        # Simple linear approximation
        k = np.sqrt(self.alpha * self.gamma * lambda_param)

        trajectory = []
        cumulative_shares = 0

        for i in range(n_intervals):
            # Hyperbolic tangent trajectory
            t_ratio = (i + 1) / n_intervals
            execution_rate = shares_per_interval / (dt if dt > 0 else 1)

            # With cost minimization
            adjusted_rate = execution_rate * np.tanh(k * t_ratio)

            shares_in_window = min(adjusted_rate * dt, total_shares - cumulative_shares)

            trajectory.append(
                {
                    "interval": i + 1,
                    "shares": shares_in_window,
                    "cumulative": cumulative_shares + shares_in_window,
                    "rate": shares_in_window / (dt if dt > 0 else 1),
                }
            )

            cumulative_shares += shares_in_window

        return {
            "symbol": symbol,
            "total_shares": total_shares,
            "duration_hours": duration_hours,
            "n_intervals": n_intervals,
            "suggested_lambda": lambda_param,
            "trajectory": trajectory,
            "final_cumulative": cumulative_shares,
        }

    def estimate_cost(
        self,
        trajectory: List[Dict],
        price: float,
        volatility: float,
    ) -> Tuple[float, float]:
        """
        Estimate implementation cost and risk.

        Returns:
            (expected_cost_bps, standard_deviation_bps)
        """
        total_shares = sum(t["shares"] for t in trajectory)

        # Temporary impact cost
        temp_cost = self.beta * sum(
            (t["shares"] ** 2) / total_shares for t in trajectory
        )

        # Permanent impact cost
        perm_cost = self.alpha * total_shares

        total_impact_bps = (temp_cost + perm_cost) * 10000 / price if price > 0 else 0

        # Risk from volatility and execution profile
        risk_bps = (
            volatility
            * np.sqrt(sum((t["shares"] / total_shares) ** 2 for t in trajectory))
            * 10000
        )

        return total_impact_bps, risk_bps
