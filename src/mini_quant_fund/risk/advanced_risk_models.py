"""
Advanced Risk Modeling
======================

Systemic risk, network contagion, and stress testing.

Models:
- Network contagion analysis (using network graphs)
- Cross-asset spillover effects (Diebold-Yilmaz)
- Extreme value theory for tail risk
- Monte Carlo stress testing
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class ContagionRisk:
    """Contagion risk assessment."""

    source_entity: str
    affected_entities: List[str]
    spillover_probabilities: List[float]
    expected_loss: float


class NetworkContagionModel:
    """
    Models financial contagion through network effects.

    Based on Eisenberg-Noe model for systemic risk.
    """

    def __init__(self):
        self.entities: List[str] = []
        self.exposures: np.ndarray = None  # Exposure matrix
        self.capital_buffers: Dict[str, float] = {}

    def build_network(
        self,
        entities: List[str],
        exposures: np.ndarray,
        capital_buffers: Dict[str, float],
    ):
        """
        Build financial network.

        Args:
            entities: List of entities (banks, funds, etc.)
            exposures: Exposure matrix [N x N] where exposure[i,j] is i's exposure to j
            capital_buffers: Capital buffer for each entity
        """
        self.entities = entities
        self.exposures = exposures
        self.capital_buffers = capital_buffers

        logger.info(f"Built network with {len(entities)} entities")

    def simulate_default_cascade(
        self, initial_defaults: List[str], recovery_rate: float = 0.4
    ) -> Dict[str, bool]:
        """
        Simulate default cascade from initial defaults.

        Args:
            initial_defaults: Initially defaulted entities
            recovery_rate: Recovery rate on defaulted exposures

        Returns:
            Dictionary {entity: is_defaulted}
        """
        n = len(self.entities)
        entity_index = {name: i for i, name in enumerate(self.entities)}

        # Track default status
        is_defaulted = {entity: False for entity in self.entities}
        for entity in initial_defaults:
            is_defaulted[entity] = True

        # Iterative contagion
        changed = True
        iteration = 0
        max_iterations = 100

        while changed and iteration < max_iterations:
            changed = False
            iteration += 1

            for entity in self.entities:
                if is_defaulted[entity]:
                    continue

                # Calculate loss from defaults
                entity_idx = entity_index[entity]
                loss = 0

                for other in self.entities:
                    if is_defaulted[other]:
                        other_idx = entity_index[other]
                        exposure = self.exposures[entity_idx, other_idx]
                        loss += exposure * (1 - recovery_rate)

                # Check if capital buffer is breached
                capital = self.capital_buffers.get(entity, 0)
                if loss > capital:
                    is_defaulted[entity] = True
                    changed = True
                    logger.info(
                        f"Iteration {iteration}: {entity} defaults (loss={loss:.2f}, capital={capital:.2f})"
                    )

        return is_defaulted

    def compute_contagion_index(self, entity: str) -> float:
        """
        Compute systemic importance index for entity.

        Higher = more systemically important.

        Args:
            entity: Entity name

        Returns:
            Contagion index [0, 1]
        """
        # Simulate default of this entity
        defaults = self.simulate_default_cascade([entity])

        # Count how many others default
        num_defaults = sum(1 for d in defaults.values() if d)

        # Normalize by total entities
        index = (num_defaults - 1) / max(1, len(self.entities) - 1)

        return index


class CrossAssetSpilloverModel:
    """
    Models spillover effects across asset classes.

    Based on Diebold-Yilmaz connectedness framework.
    """

    def __init__(self, lookback_days: int = 252):
        self.lookback_days = lookback_days
        self.spillover_matrix: Optional[np.ndarray] = None
        self.assets: List[str] = []

    def estimate_spillover(
        self, returns: pd.DataFrame
    ) -> Tuple[np.ndarray, float]:
        """
        Estimate spillover matrix from returns.

        Args:
            returns: DataFrame of asset returns [dates x assets]

        Returns:
            (spillover_matrix, total_spillover_index)
            spillover_matrix[i,j] = spillover from j to i
        """
        self.assets = list(returns.columns)
        n = len(self.assets)

        # Compute variance-covariance matrix
        cov_matrix = returns.cov().values

        # Forecast error variance decomposition (simplified)
        # In full implementation, would use VAR model
        variance_decomp = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i == j:
                    # Own variance contribution
                    variance_decomp[i, j] = cov_matrix[i, i]
                else:
                    # Cross variance contribution
                    variance_decomp[i, j] = abs(cov_matrix[i, j])

        # Normalize rows
        row_sums = variance_decomp.sum(axis=1, keepdims=True)
        spillover_matrix = variance_decomp / (row_sums + 1e-8)

        # Total spillover index = average off-diagonal
        mask = ~np.eye(n, dtype=bool)
        total_spillover = spillover_matrix[mask].mean()

        self.spillover_matrix = spillover_matrix

        return spillover_matrix, total_spillover

    def get_directional_spillover(
        self, from_asset: str, to_asset: str
    ) -> float:
        """Get spillover from one asset to another."""
        if self.spillover_matrix is None:
            return 0.0

        from_idx = self.assets.index(from_asset)
        to_idx = self.assets.index(to_asset)

        return self.spillover_matrix[to_idx, from_idx]


class ExtremeValueTheory:
    """
    Tail risk modeling using Extreme Value Theory (EVT).

    Models extreme losses beyond historical observations.
    """

    def __init__(self):
        self.params: Dict[str, Tuple[float, float, float]] = {}

    def fit_gpd(self, losses: np.ndarray, threshold: float) -> Tuple[float, float]:
        """
        Fit Generalized Pareto Distribution to exceedances.

        Args:
            losses: Loss data
            threshold: Threshold for exceedances

        Returns:
            (shape, scale) parameters
        """
        exceedances = losses[losses > threshold] - threshold

        if len(exceedances) < 10:
            logger.warning("Too few exceedances for reliable GPD fit")
            return (0.1, 1.0)

        # Fit GPD using MLE
        shape, loc, scale = stats.genpareto.fit(exceedances, floc=0)

        return (shape, scale)

    def estimate_var(
        self, losses: np.ndarray, confidence: float = 0.99, threshold_quantile: float = 0.95
    ) -> float:
        """
        Estimate Value-at-Risk using EVT.

        Args:
            losses: Historical losses
            confidence: Confidence level (e.g., 0.99)
            threshold_quantile: Quantile for threshold

        Returns:
            VaR estimate
        """
        threshold = np.quantile(losses, threshold_quantile)
        shape, scale = self.fit_gpd(losses, threshold)

        # Number of exceedances
        n_exceed = np.sum(losses > threshold)
        n_total = len(losses)
        exceed_prob = n_exceed / n_total

        # EVT-based VaR
        q = (1 - confidence) / (1 - threshold_quantile)

        if abs(shape) > 1e-6:
            var = threshold + (scale / shape) * ((q / exceed_prob) ** (-shape) - 1)
        else:
            var = threshold - scale * np.log(q / exceed_prob)

        return var

    def estimate_cvar(
        self, losses: np.ndarray, confidence: float = 0.99
    ) -> float:
        """
        Estimate Conditional Value-at-Risk (Expected Shortfall).

        Args:
            losses: Historical losses
            confidence: Confidence level

        Returns:
            CVaR estimate
        """
        var = self.estimate_var(losses, confidence)

        # CVaR = average loss beyond VaR
        tail_losses = losses[losses > var]

        if len(tail_losses) > 0:
            cvar = tail_losses.mean()
        else:
            # Use EVT extrapolation
            threshold = np.quantile(losses, 0.95)
            shape, scale = self.fit_gpd(losses, threshold)

            if shape < 1:
                cvar = var + (scale + shape * (var - threshold)) / (1 - shape)
            else:
                cvar = var * 1.2  # Conservative estimate

        return cvar


class StressTestingFramework:
    """
    Monte Carlo-based stress testing framework.

    Scenarios:
    - Market crashes
    - Credit events
    - Liquidity crises
    """

    def __init__(self):
        self.scenarios: Dict[str, Dict] = {}

    def define_scenario(
        self,
        name: str,
        asset_shocks: Dict[str, float],
        correlation_shock: float = 1.5,
    ):
        """
        Define stress scenario.

        Args:
            name: Scenario name
            asset_shocks: Dictionary {asset: shock_pct}
            correlation_shock: Correlation multiplier
        """
        self.scenarios[name] = {
            "asset_shocks": asset_shocks,
            "correlation_shock": correlation_shock,
        }

    def run_scenario(
        self,
        scenario_name: str,
        portfolio: Dict[str, float],
        base_prices: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Run stress scenario.

        Args:
            scenario_name: Name of scenario
            portfolio: Portfolio holdings {asset: quantity}
            base_prices: Current prices {asset: price}

        Returns:
            Dictionary with stress results
        """
        if scenario_name not in self.scenarios:
            raise ValueError(f"Unknown scenario: {scenario_name}")

        scenario = self.scenarios[scenario_name]
        asset_shocks = scenario["asset_shocks"]

        # Calculate shocked prices
        shocked_prices = {}
        for asset, price in base_prices.items():
            shock = asset_shocks.get(asset, 0)
            shocked_prices[asset] = price * (1 + shock / 100)

        # Calculate P&L
        base_value = sum(
            portfolio.get(asset, 0) * base_prices.get(asset, 0)
            for asset in set(portfolio.keys()) | set(base_prices.keys())
        )

        stressed_value = sum(
            portfolio.get(asset, 0) * shocked_prices.get(asset, 0)
            for asset in set(portfolio.keys()) | set(shocked_prices.keys())
        )

        pnl = stressed_value - base_value
        pnl_pct = (pnl / base_value * 100) if base_value != 0 else 0

        return {
            "scenario": scenario_name,
            "base_value": base_value,
            "stressed_value": stressed_value,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
        }
