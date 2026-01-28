"""
risk/capital_stress.py

Capital Scaling Realism.
Models what breaks first at 10x/50x capital:
1. Liquidity exhaustion (% ADV limits)
2. Non-linear market impact (square-root law)
3. Capacity limits per strategy
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger("CAPITAL_STRESS")

@dataclass
class StressTestResult:
    aum_multiple: float  # 1x, 10x, 50x
    total_aum: float
    feasible: bool
    sharpe_degradation: float  # % drop from baseline
    max_position_breach_count: int
    liquidity_violations: List[str]
    estimated_capacity: float
    impact_cost_bps: float  # Total impact cost in bps

class CapitalStressTester:
    def __init__(self,
                 max_adv_participation: float = 0.05,  # 5% of ADV
                 impact_coefficient: float = 0.1,  # Square-root impact constant
                 sharpe_decay_rate: float = 0.15):  # Sharpe decay per doubling of AUM
        """
        Initialize Capital Stress Tester.

        Args:
            max_adv_participation: Maximum % of ADV we can trade
            impact_coefficient: Market impact coefficient (higher = more impact)
            sharpe_decay_rate: Sharpe ratio decay per 2x AUM increase
        """
        self.max_adv_participation = max_adv_participation
        self.impact_coefficient = impact_coefficient
        self.sharpe_decay_rate = sharpe_decay_rate

    def compute_market_impact(self,
                             order_size: float,
                             avg_daily_volume: float) -> float:
        """
        Compute market impact using square-root law.

        Impact (bps) = k * sqrt(order_size / ADV)

        Args:
            order_size: Size of order in dollars
            avg_daily_volume: Average daily volume in dollars

        Returns:
            Impact cost in basis points
        """
        if avg_daily_volume <= 0:
            return 999.0  # Infinite impact

        participation_rate = order_size / avg_daily_volume

        # Square-root impact model
        impact_bps = self.impact_coefficient * np.sqrt(participation_rate) * 10000

        return impact_bps

    def check_liquidity_feasibility(self,
                                   position_size: float,
                                   avg_daily_volume: float,
                                   symbol: str) -> Tuple[bool, Optional[str]]:
        """
        Check if position size violates liquidity constraints.

        Args:
            position_size: Target position size in dollars
            avg_daily_volume: Average daily volume in dollars
            symbol: Symbol identifier

        Returns:
            (is_feasible, violation_message)
        """
        if avg_daily_volume <= 0:
            return False, f"{symbol}: Zero ADV"

        participation = position_size / avg_daily_volume

        if participation > self.max_adv_participation:
            return False, f"{symbol}: {participation:.1%} of ADV exceeds {self.max_adv_participation:.1%} limit"

        return True, None

    def estimate_sharpe_degradation(self,
                                   baseline_sharpe: float,
                                   aum_multiple: float) -> float:
        """
        Estimate Sharpe ratio degradation due to capacity constraints.

        Sharpe degrades logarithmically with AUM:
        Sharpe(AUM) = Sharpe_0 * (1 - decay_rate * log2(AUM_multiple))

        Args:
            baseline_sharpe: Baseline Sharpe at 1x AUM
            aum_multiple: AUM scaling factor (e.g., 10 for 10x)

        Returns:
            Degraded Sharpe ratio
        """
        if aum_multiple <= 1.0:
            return baseline_sharpe

        # Logarithmic decay
        log_factor = np.log2(aum_multiple)
        degradation_factor = 1.0 - (self.sharpe_decay_rate * log_factor)
        degradation_factor = max(0.0, degradation_factor)  # Floor at 0

        degraded_sharpe = baseline_sharpe * degradation_factor

        return degraded_sharpe

    def simulate_scaling(self,
                        portfolio: Dict[str, Dict],
                        aum_multiple: float,
                        baseline_sharpe: float = 1.5) -> StressTestResult:
        """
        Simulate portfolio at scaled AUM.

        Args:
            portfolio: Dict of {symbol: {'weight': float, 'adv': float}}
            aum_multiple: AUM scaling factor
            baseline_sharpe: Current Sharpe ratio

        Returns:
            StressTestResult
        """
        base_aum = sum(p['weight'] for p in portfolio.values())
        scaled_aum = base_aum * aum_multiple

        violations = []
        breach_count = 0
        total_impact_bps = 0.0

        for symbol, data in portfolio.items():
            scaled_position = data['weight'] * aum_multiple
            adv = data.get('adv', 1e9)  # Default to large ADV if missing

            # Check liquidity
            is_feasible, msg = self.check_liquidity_feasibility(scaled_position, adv, symbol)
            if not is_feasible:
                violations.append(msg)
                breach_count += 1

            # Compute impact
            impact = self.compute_market_impact(scaled_position, adv)
            total_impact_bps += impact * (data['weight'] / base_aum)  # Weighted average

        # Estimate Sharpe degradation
        degraded_sharpe = self.estimate_sharpe_degradation(baseline_sharpe, aum_multiple)
        sharpe_degradation_pct = (baseline_sharpe - degraded_sharpe) / baseline_sharpe * 100

        # Estimate capacity (AUM at which Sharpe drops to 50% of baseline)
        # Solve: Sharpe_0 * (1 - decay * log2(x)) = 0.5 * Sharpe_0
        # log2(x) = 0.5 / decay
        capacity_multiple = 2 ** (0.5 / self.sharpe_decay_rate) if self.sharpe_decay_rate > 0 else 999
        estimated_capacity = base_aum * capacity_multiple

        feasible = len(violations) == 0

        return StressTestResult(
            aum_multiple=aum_multiple,
            total_aum=scaled_aum,
            feasible=feasible,
            sharpe_degradation=sharpe_degradation_pct,
            max_position_breach_count=breach_count,
            liquidity_violations=violations,
            estimated_capacity=estimated_capacity,
            impact_cost_bps=total_impact_bps
        )

    def find_capacity_limit(self,
                           portfolio: Dict[str, Dict],
                           baseline_sharpe: float = 1.5,
                           sharpe_threshold: float = 0.75) -> float:
        """
        Binary search to find maximum AUM before Sharpe drops below threshold.

        Args:
            portfolio: Portfolio definition
            baseline_sharpe: Current Sharpe
            sharpe_threshold: Minimum acceptable Sharpe

        Returns:
            Maximum AUM multiple
        """
        low, high = 1.0, 100.0
        tolerance = 0.1

        while high - low > tolerance:
            mid = (low + high) / 2
            result = self.simulate_scaling(portfolio, mid, baseline_sharpe)

            degraded_sharpe = self.estimate_sharpe_degradation(baseline_sharpe, mid)

            if degraded_sharpe >= sharpe_threshold and result.feasible:
                low = mid  # Can scale more
            else:
                high = mid  # Hit limit

        return low

    def run_stress_suite(self,
                        portfolio: Dict[str, Dict],
                        baseline_sharpe: float = 1.5) -> Dict[str, StressTestResult]:
        """
        Run standard stress test suite: 1x, 10x, 50x AUM.

        Args:
            portfolio: Portfolio definition
            baseline_sharpe: Current Sharpe ratio

        Returns:
            Dict of {scenario: StressTestResult}
        """
        scenarios = {
            "1x_baseline": 1.0,
            "10x_growth": 10.0,
            "50x_institutional": 50.0
        }

        results = {}
        for name, multiple in scenarios.items():
            result = self.simulate_scaling(portfolio, multiple, baseline_sharpe)
            results[name] = result

            logger.info(
                f"[STRESS] {name}: AUM=${result.total_aum:,.0f}, "
                f"Feasible={result.feasible}, Sharpe Degradation={result.sharpe_degradation:.1f}%, "
                f"Impact={result.impact_cost_bps:.1f}bps"
            )

        return results
