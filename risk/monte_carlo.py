"""
Monte Carlo Risk Simulator - Scenario Analysis.

Features:
- Portfolio stress testing
- VaR/CVaR calculation
- Tail risk analysis
- Path-dependent simulations
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MonteCarloResult:
    """Monte Carlo simulation result."""
    mean_return: float
    median_return: float
    std_dev: float

    # Risk metrics
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float

    # Tail metrics
    max_drawdown_mean: float
    max_drawdown_worst: float
    prob_loss: float
    prob_large_loss: float

    # Distribution
    percentiles: Dict[int, float]


@dataclass
class StressScenario:
    """Stress test scenario."""
    name: str
    market_shock: float
    vol_multiplier: float
    correlation_shock: float
    result: Optional[float] = None


class MonteCarloSimulator:
    """
    Monte Carlo simulation for portfolio risk.

    Features:
    - Geometric Brownian Motion
    - Correlated returns
    - Jump diffusion (fat tails)
    - Path-dependent metrics
    """

    def __init__(
        self,
        n_simulations: int = 10000,
        n_days: int = 252,
        seed: int = 42
    ):
        self.n_simulations = n_simulations
        self.n_days = n_days
        np.random.seed(seed)

        # Standard stress scenarios
        self.stress_scenarios = [
            StressScenario("Black Monday", -0.20, 3.0, 0.5),
            StressScenario("2008 Crisis", -0.40, 4.0, 0.7),
            StressScenario("COVID Crash", -0.35, 5.0, 0.6),
            StressScenario("Flash Crash", -0.10, 2.0, 0.3),
            StressScenario("Gradual Bear", -0.25, 2.0, 0.4),
        ]

    def simulate_gbm(
        self,
        initial_value: float,
        mu: float,
        sigma: float,
        n_paths: int = None,
        n_steps: int = None
    ) -> np.ndarray:
        """
        Simulate Geometric Brownian Motion paths.

        dS = mu*S*dt + sigma*S*dW
        """
        n_paths = n_paths or self.n_simulations
        n_steps = n_steps or self.n_days

        dt = 1.0 / 252  # Daily

        # Generate random shocks
        Z = np.random.standard_normal((n_paths, n_steps))

        # Calculate returns
        returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z

        # Calculate prices
        log_prices = np.cumsum(returns, axis=1)
        prices = initial_value * np.exp(log_prices)

        return prices

    def simulate_with_jumps(
        self,
        initial_value: float,
        mu: float,
        sigma: float,
        jump_intensity: float = 0.1,
        jump_mean: float = -0.05,
        jump_std: float = 0.10
    ) -> np.ndarray:
        """
        Simulate with jump diffusion (Merton model).

        More realistic for tail events.
        """
        n_paths = self.n_simulations
        n_steps = self.n_days
        dt = 1.0 / 252

        # GBM component
        Z = np.random.standard_normal((n_paths, n_steps))

        # Jump component
        n_jumps = np.random.poisson(jump_intensity * dt, (n_paths, n_steps))
        jump_sizes = np.random.normal(jump_mean, jump_std, (n_paths, n_steps)) * n_jumps

        # Combined returns
        returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z + jump_sizes

        log_prices = np.cumsum(returns, axis=1)
        prices = initial_value * np.exp(log_prices)

        return prices

    def calculate_var(
        self,
        returns: np.ndarray,
        confidence: float = 0.95
    ) -> float:
        """Calculate Value at Risk."""
        return float(-np.percentile(returns, (1 - confidence) * 100))

    def calculate_cvar(
        self,
        returns: np.ndarray,
        confidence: float = 0.95
    ) -> float:
        """Calculate Conditional VaR (Expected Shortfall)."""
        var = self.calculate_var(returns, confidence)
        tail_returns = returns[returns <= -var]

        if len(tail_returns) > 0:
            return float(-np.mean(tail_returns))
        return var

    def calculate_max_drawdown(self, prices: np.ndarray) -> np.ndarray:
        """Calculate max drawdown for each path."""
        peak = np.maximum.accumulate(prices, axis=1)
        drawdown = (peak - prices) / peak
        max_dd = np.max(drawdown, axis=1)
        return max_dd

    def run_simulation(
        self,
        portfolio_value: float,
        expected_return: float,
        volatility: float,
        use_jumps: bool = True
    ) -> MonteCarloResult:
        """
        Run full Monte Carlo simulation.
        """
        # Simulate paths
        if use_jumps:
            prices = self.simulate_with_jumps(
                portfolio_value,
                expected_return,
                volatility
            )
        else:
            prices = self.simulate_gbm(
                portfolio_value,
                expected_return,
                volatility
            )

        # Calculate final returns
        final_returns = (prices[:, -1] - portfolio_value) / portfolio_value

        # Calculate metrics
        mean_ret = np.mean(final_returns)
        median_ret = np.median(final_returns)
        std_ret = np.std(final_returns)

        # VaR and CVaR
        var_95 = self.calculate_var(final_returns, 0.95)
        var_99 = self.calculate_var(final_returns, 0.99)
        cvar_95 = self.calculate_cvar(final_returns, 0.95)
        cvar_99 = self.calculate_cvar(final_returns, 0.99)

        # Max drawdown
        max_dd = self.calculate_max_drawdown(prices)

        # Probability metrics
        prob_loss = np.mean(final_returns < 0)
        prob_large_loss = np.mean(final_returns < -0.10)

        # Percentiles
        percentiles = {
            1: float(np.percentile(final_returns, 1)),
            5: float(np.percentile(final_returns, 5)),
            10: float(np.percentile(final_returns, 10)),
            25: float(np.percentile(final_returns, 25)),
            50: float(np.percentile(final_returns, 50)),
            75: float(np.percentile(final_returns, 75)),
            90: float(np.percentile(final_returns, 90)),
            95: float(np.percentile(final_returns, 95)),
            99: float(np.percentile(final_returns, 99))
        }

        return MonteCarloResult(
            mean_return=float(mean_ret),
            median_return=float(median_ret),
            std_dev=float(std_ret),
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            max_drawdown_mean=float(np.mean(max_dd)),
            max_drawdown_worst=float(np.max(max_dd)),
            prob_loss=float(prob_loss),
            prob_large_loss=float(prob_large_loss),
            percentiles=percentiles
        )

    def run_stress_test(
        self,
        portfolio_value: float,
        positions: Dict[str, float],
        betas: Dict[str, float]
    ) -> List[StressScenario]:
        """Run stress test scenarios."""
        results = []

        for scenario in self.stress_scenarios:
            # Calculate portfolio impact
            portfolio_impact = 0.0

            for symbol, weight in positions.items():
                beta = betas.get(symbol, 1.0)
                impact = weight * beta * scenario.market_shock

                # Add vol impact
                impact *= scenario.vol_multiplier ** 0.5

                portfolio_impact += impact

            # Correlation shock increases impact
            portfolio_impact *= (1 + scenario.correlation_shock)

            scenario_result = StressScenario(
                name=scenario.name,
                market_shock=scenario.market_shock,
                vol_multiplier=scenario.vol_multiplier,
                correlation_shock=scenario.correlation_shock,
                result=portfolio_value * portfolio_impact
            )
            results.append(scenario_result)

        return results


# Global singleton
_mc_simulator: Optional[MonteCarloSimulator] = None


def get_monte_carlo() -> MonteCarloSimulator:
    """Get or create global Monte Carlo simulator."""
    global _mc_simulator
    if _mc_simulator is None:
        _mc_simulator = MonteCarloSimulator()
    return _mc_simulator
