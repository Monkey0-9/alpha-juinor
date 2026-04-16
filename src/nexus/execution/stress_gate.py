"""
execution/stress_gate.py

Stress-First Simulation Gate
Runs synthetic stress before allowing execution.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

class StressSimulationGate:
    """
    Pre-execution stress testing with multiple shock scenarios.

    Scenarios:
    1. Correlation spike (all correlations → 0.9)
    2. Liquidity evaporation (spreads 10x)
    3. -5σ market move
    """

    def __init__(self, max_stress_loss: float = 0.10):
        """
        Args:
            max_stress_loss: Maximum acceptable loss in stress (10% default)
        """
        self.max_stress_loss = max_stress_loss

    def stress_test_portfolio(self,
                             proposed_trades: Dict[str, float],
                             current_positions: Dict[str, float],
                             alphas: Dict[str, 'AlphaDistribution']) -> Dict:
        """
        Simulate portfolio under stress scenarios.

        Returns:
            {
                "pass": bool,
                "max_drawdown": float,
                "worst_scenario": str,
                "details": Dict
            }
        """
        # Build test portfolio
        test_portfolio = current_positions.copy()
        test_portfolio.update(proposed_trades)

        # Run stress scenarios
        scenario_results = {}

        scenario_results["correlation_spike"] = self._simulate_correlation_spike(
            test_portfolio, alphas
        )

        scenario_results["liquidity_shock"] = self._simulate_liquidity_shock(
            test_portfolio, alphas
        )

        scenario_results["tail_event"] = self._simulate_tail_event(
            test_portfolio, alphas
        )

        # Find worst scenario
        worst_loss = min(scenario_results.values())
        worst_scenario = min(scenario_results, key=scenario_results.get)

        # Pass/Fail decision
        passed = worst_loss > -self.max_stress_loss

        result = {
            "pass": passed,
            "max_drawdown": worst_loss,
            "worst_scenario": worst_scenario,
            "details": scenario_results
        }

        if not passed:
            logger.error(
                f"[STRESS_GATE] FAILED: {worst_scenario} loss={worst_loss:.2%} "
                f"exceeds limit={-self.max_stress_loss:.2%}"
            )
        else:
            logger.info(
                f"[STRESS_GATE] PASSED: Worst case={worst_loss:.2%} "
                f"(scenario: {worst_scenario})"
            )

        return result

    def _simulate_correlation_spike(self,
                                    portfolio: Dict[str, float],
                                    alphas: Dict[str, 'AlphaDistribution']) -> float:
        """
        Scenario 1: All correlations spike to 0.9.
        Simulates market panic/correlation breakdown.
        """
        if not portfolio:
            return 0.0

        symbols = list(portfolio.keys())
        weights = np.array([portfolio[s] for s in symbols])

        # Extract volatilities
        sigmas = np.array([alphas[s].sigma for s in symbols])

        # Build correlation matrix (0.9 off-diagonal)
        n = len(symbols)
        corr_matrix = np.ones((n, n)) * 0.9
        np.fill_diagonal(corr_matrix, 1.0)

        # Covariance matrix
        cov_matrix = np.outer(sigmas, sigmas) * corr_matrix

        # Portfolio variance
        portfolio_var = np.dot(weights, np.dot(cov_matrix, weights))
        portfolio_sigma = np.sqrt(portfolio_var)

        # Assume -3σ move under correlation spike
        loss = -3.0 * portfolio_sigma

        logger.debug(f"[STRESS] Correlation spike scenario: {loss:.2%}")
        return loss

    def _simulate_liquidity_shock(self,
                                  portfolio: Dict[str, float],
                                  alphas: Dict[str, 'AlphaDistribution']) -> float:
        """
        Scenario 2: Liquidity evaporates, spreads widen 10x.
        Simulates inability to exit at fair prices.
        """
        if not portfolio:
            return 0.0

        # Assume 0.5% normal spread, 5% in crisis
        normal_spread = 0.005
        crisis_spread = 0.05

        # Total notional
        total_notional = sum(abs(w) for w in portfolio.values())

        # Slippage cost = (crisis_spread - normal_spread) * notional
        slippage_cost = (crisis_spread - normal_spread) * total_notional

        loss = -slippage_cost

        logger.debug(f"[STRESS] Liquidity shock scenario: {loss:.2%}")
        return loss

    def _simulate_tail_event(self,
                            portfolio: Dict[str, float],
                            alphas: Dict[str, 'AlphaDistribution']) -> float:
        """
        Scenario 3: -5σ market move.
        Simulates extreme tail event.
        """
        if not portfolio:
            return 0.0

        symbols = list(portfolio.keys())
        weights = np.array([portfolio[s] for s in symbols])

        # Use CVaR directly from alphas
        cvars = np.array([alphas[s].cvar_95 for s in symbols])

        # Portfolio CVaR (weighted average - simplified)
        portfolio_cvar = np.dot(weights, cvars)

        # Scale to -5σ equivalent
        loss = portfolio_cvar * 1.5  # CVaR at 95% ~= -2σ, scale to -5σ

        logger.debug(f"[STRESS] Tail event scenario: {loss:.2%}")
        return loss

    def get_scale_down_factor(self, stress_loss: float) -> float:
        """
        Compute position size scale-down to pass stress test.

        If stress loss = -15% and limit = -10%:
        Scale factor = 10 / 15 = 0.67 (reduce by 33%)
        """
        if stress_loss >= -self.max_stress_loss:
            return 1.0  # No scaling needed

        scale_factor = abs(self.max_stress_loss / stress_loss)

        logger.warning(
            f"[STRESS_GATE] Scaling positions by {scale_factor:.2f} "
            f"to meet stress limit"
        )

        return scale_factor
