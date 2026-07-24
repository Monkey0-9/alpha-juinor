import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


class ExecutionAgent:
    """
    Institutional Execution Agent with VWAP, TWAP, and Almgren-Chriss Implementation Shortfall models.

    Actions:
        0 — Wait / Defer Execution
        1 — Small / Conservative VWAP Order
        2 — Aggressive Immediate Order
        3 — TWAP Sliced Execution
    """

    def __init__(self) -> None:
        self.cumulative_reward: float = 0.0
        self.decision_count: int = 0

    def get_action(self, market_state: np.ndarray[Any, Any]) -> int:
        """Determine optimal execution tactic based on spread, volume ratio, and volatility."""
        if market_state is None or len(market_state) < 4:
            return 1  # Conservative default

        spread_bps = float(market_state[0]) if len(market_state) > 0 else 0.0
        volatility = float(market_state[2]) if len(market_state) > 2 else 0.0
        momentum = float(market_state[3]) if len(market_state) > 3 else 0.0

        # High volatility or wide spread -> VWAP/TWAP slicing
        if volatility > 0.03 or spread_bps > 15.0:
            return 3  # TWAP Slices

        # High volatility spike -> Defer execution
        if volatility > 0.05:
            return 0  # Wait

        # Strong directional momentum -> Aggressive execution
        if abs(momentum) > 0.02:
            return 2  # Aggressive

        return 1  # Conservative VWAP

    def calculate_vwap_schedule(
        self, total_quantity: int, volume_profile: List[float], n_slices: int = 10
    ) -> List[int]:
        """
        Volume-Weighted Average Price (VWAP) Execution Schedule.
        Distributes shares across execution windows based on intraday volume distribution.
        """
        if not volume_profile or sum(volume_profile) <= 0:
            # Uniform fallback
            slice_qty = int(total_quantity / max(1, n_slices))
            slices = [slice_qty] * n_slices
            slices[-1] += total_quantity - sum(slices)
            return slices

        profile = np.array(volume_profile[:n_slices])
        profile = profile / np.sum(profile)
        raw_slices = np.round(profile * total_quantity).astype(int)
        raw_slices[-1] += total_quantity - np.sum(raw_slices)
        return list(raw_slices)

    def calculate_twap_schedule(self, total_quantity: int, n_slices: int = 10) -> List[int]:
        """
        Time-Weighted Average Price (TWAP) Execution Schedule.
        Equal volume distribution over uniform time intervals.
        """
        slice_size = int(total_quantity // max(1, n_slices))
        remainder = total_quantity % max(1, n_slices)
        slices = [slice_size] * n_slices
        slices[-1] += remainder
        return slices

    def almgren_chriss_schedule(
        self,
        total_shares: int,
        n_periods: int = 10,
        volatility: float = 0.02,
        gamma: float = 1e-6,
        eta: float = 1e-5,
        risk_aversion: float = 1e-5
    ) -> Dict[str, Any]:
        """
        Almgren-Chriss Optimal Execution / Implementation Shortfall Model.
        Calculates optimal trajectory balancing market impact vs. volatility risk.
        """
        if total_shares <= 0 or n_periods <= 0:
            return {"trajectory": [], "expected_shortfall": 0.0}

        tau = 1.0 / n_periods
        kappa_sq = (risk_aversion * volatility**2) / (eta * (1 + 0.5 * gamma * tau))
        kappa = np.sqrt(max(1e-8, kappa_sq))

        trajectory = []
        for j in range(n_periods + 1):
            t_j = j * tau
            shares_remaining = total_shares * (np.sinh(kappa * (1.0 - t_j)) / np.sinh(kappa))
            trajectory.append(float(max(0.0, shares_remaining)))

        trades_per_period = [trajectory[i] - trajectory[i + 1] for i in range(n_periods)]

        # Expected Implementation Shortfall Cost
        expected_shortfall = 0.5 * gamma * (total_shares**2) + eta * sum(t**2 / tau for t in trades_per_period)

        return {
            "trajectory": trajectory,
            "trades_per_period": trades_per_period,
            "expected_shortfall": float(expected_shortfall),
            "half_life_periods": float(np.log(2) / max(1e-4, kappa))
        }

    def learn(self, reward: float) -> None:
        """Record execution feedback reward."""
        self.cumulative_reward += reward
        self.decision_count += 1
        avg = self.cumulative_reward / self.decision_count if self.decision_count else 0.0
        logger.debug(
            f"Execution feedback — Reward: {reward:.4f}, Cumulative avg: {avg:.4f}, Decisions: {self.decision_count}"
        )
