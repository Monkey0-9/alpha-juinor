import numpy as np
import pandas as pd
from typing import Dict


class PortfolioOptimizer:
    """Mean-Variance Optimization and sizing logic."""

    def optimize_weights(
        self, symbols: list[str], signals: list[float]
    ) -> Dict[str, float]:
        """Calculate optimal weights based on alpha signals."""
        if not symbols:
            return {}

        abs_signals = np.abs(signals)
        total = np.sum(abs_signals)
        if total == 0:
            return {s: 1.0 / len(symbols) for s in symbols}

        weights = {
            s: float(v / total)
            for s, v in zip(symbols, abs_signals, strict=True)
        }
        return weights


class MultiFactorEngine:
    """Ranks assets based on multiple alpha factors."""

    def rank_assets(
        self,
        signals: Dict[str, float],
        historical_data: Dict[str, pd.DataFrame],
    ) -> Dict[str, float]:
        """Rank assets by alpha + momentum - volatility."""
        rankings = {}
        for symbol, alpha in signals.items():
            if symbol in historical_data:
                data = historical_data[symbol]
                close = data["close"]
                momentum = (close.iloc[-1] / close.iloc[0]) - 1
                vol = close.pct_change().std()
                rankings[symbol] = (
                    alpha + 0.2 * momentum - 0.5 * vol
                )
            else:
                rankings[symbol] = alpha

        return dict(
            sorted(
                rankings.items(), key=lambda x: x[1], reverse=True
            )
        )


class MonteCarloSimulator:
    """Portfolio-level Monte Carlo survival analysis.

    Runs N simulated random walks using bootstrapped daily returns
    to estimate the probability that the portfolio survives
    (i.e. does not breach the ruin threshold) over a given horizon.
    """

    def run_survival_analysis(
        self,
        initial_capital: float,
        daily_returns: np.ndarray,
        days: int = 252,
        n_simulations: int = 1000,
        ruin_threshold: float = 0.5,
    ) -> float:
        """Estimate probability of survival over N days.

        Parameters
        ----------
        initial_capital : float
            Starting portfolio value.
        daily_returns : np.ndarray
            Historical daily return samples to bootstrap from.
        days : int
            Simulation horizon in trading days.
        n_simulations : int
            Number of Monte Carlo paths.
        ruin_threshold : float
            Fraction of capital lost that constitutes ruin
            (0.5 = 50% drawdown).

        Returns
        -------
        float
            Probability of survival (0.0 to 1.0).
        """
        if (
            len(daily_returns) < 2
            or initial_capital <= 0
            or days <= 0
        ):
            return 0.5  # Insufficient data

        mu = float(np.mean(daily_returns))
        sigma = float(np.std(daily_returns))
        if sigma == 0:
            return 1.0  # No volatility = no ruin

        ruin_level = initial_capital * (1 - ruin_threshold)
        survived = 0

        rng = np.random.default_rng(42)  # Reproducible

        for _ in range(n_simulations):
            # Bootstrap path from observed return distribution
            path_returns = rng.normal(mu, sigma, days)
            prices = initial_capital * np.cumprod(1 + path_returns)

            if np.min(prices) > ruin_level:
                survived += 1

        return survived / n_simulations
