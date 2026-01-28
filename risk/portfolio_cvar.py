"""
risk/portfolio_cvar.py

Portfolio-Level CVaR Attribution
Calculates marginal CVaR contribution of positions.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional, Tuple
from scipy import stats

logger = logging.getLogger(__name__)

class PortfolioCVaR:
    """
    Computes portfolio-level Conditional Value at Risk.

    Features:
    - Marginal CVaR contribution
    - Correlation-aware risk attribution
    - Tail risk decomposition
    """

    def __init__(self, confidence_level: float = 0.95):
        """
        Args:
            confidence_level: CVaR confidence level (default 95%)
        """
        self.confidence_level = confidence_level
        self.alpha = 1.0 - confidence_level

    def compute_portfolio_cvar(self,
                               positions: Dict[str, float],
                               alphas: Dict[str, 'AlphaDistribution'],
                               correlation_matrix: Optional[np.ndarray] = None) -> float:
        """
        Compute portfolio CVaR using parametric method.

        Args:
            positions: Dict[symbol, weight]
            alphas: Dict[symbol, AlphaDistribution]
            correlation_matrix: Optional correlation matrix

        Returns:
            Portfolio CVaR (negative value, e.g., -0.05 = -5%)
        """
        if not positions:
            return 0.0

        symbols = list(positions.keys())
        weights = np.array([positions[s] for s in symbols])

        # Extract distributional parameters
        mus = np.array([alphas[s].mu for s in symbols])
        sigmas = np.array([alphas[s].sigma for s in symbols])

        # Portfolio mean and variance
        portfolio_mu = np.dot(weights, mus)

        if correlation_matrix is None:
            # Assume independence
            portfolio_var = np.sum((weights * sigmas) ** 2)
        else:
            # Use correlation structure
            cov_matrix = np.outer(sigmas, sigmas) * correlation_matrix
            portfolio_var = np.dot(weights, np.dot(cov_matrix, weights))

        portfolio_sigma = np.sqrt(portfolio_var)

        # CVaR calculation (parametric normal assumption)
        # CVaR = μ - σ * φ(Φ^(-1)(α)) / α
        # Where φ is normal PDF, Φ is normal CDF

        z_alpha = stats.norm.ppf(self.alpha)
        phi_z = stats.norm.pdf(z_alpha)

        cvar = portfolio_mu - portfolio_sigma * (phi_z / self.alpha)

        return cvar

    def compute_marginal_cvar(self,
                             new_symbol: str,
                             new_alpha: 'AlphaDistribution',
                             new_weight: float,
                             current_positions: Dict[str, float],
                             current_alphas: Dict[str, 'AlphaDistribution'],
                             correlation_matrix: Optional[np.ndarray] = None) -> float:
        """
        Compute marginal CVaR contribution of adding new position.

        Returns:
            Change in portfolio CVaR (negative = worsening risk)
        """
        # Current portfolio CVaR
        current_cvar = self.compute_portfolio_cvar(
            current_positions,
            current_alphas,
            correlation_matrix
        )

        # Hypothetical portfolio with new position
        test_positions = current_positions.copy()
        test_positions[new_symbol] = new_weight

        test_alphas = current_alphas.copy()
        test_alphas[new_symbol] = new_alpha

        new_cvar = self.compute_portfolio_cvar(
            test_positions,
            test_alphas,
            correlation_matrix
        )

        # Marginal contribution (negative = risk increased)
        marginal = new_cvar - current_cvar

        logger.info(f"[CVAR] {new_symbol}: current={current_cvar:.4f}, new={new_cvar:.4f}, marginal={marginal:.4f}")

        return marginal

    def should_reject_trade(self,
                           marginal_cvar: float,
                           cvar_tolerance: float = -0.01) -> bool:
        """
        Decide if trade should be rejected based on marginal CVaR.

        Args:
            marginal_cvar: Change in portfolio CVaR
            cvar_tolerance: Maximum acceptable CVaR degradation (e.g., -1%)

        Returns:
            True if trade worsens risk beyond tolerance
        """
        if marginal_cvar < cvar_tolerance:
            logger.warning(f"[CVAR_REJECT] Marginal CVaR {marginal_cvar:.4f} exceeds tolerance {cvar_tolerance:.4f}")
            return True

        return False

    def decompose_risk_sources(self,
                               positions: Dict[str, float],
                               alphas: Dict[str, 'AlphaDistribution']) -> Dict[str, float]:
        """
        Decompose portfolio risk by position.

        Returns:
            Dict[symbol, risk_contribution]
        """
        contributions = {}

        # Compute portfolio CVaR
        portfolio_cvar = self.compute_portfolio_cvar(positions, alphas)

        # Compute contribution of each position
        for symbol in positions:
            # Remove position and recompute
            test_positions = positions.copy()
            del test_positions[symbol]

            test_alphas = alphas.copy()
            del test_alphas[symbol]

            cvar_without = self.compute_portfolio_cvar(test_positions, test_alphas)

            # Contribution = difference
            contribution = portfolio_cvar - cvar_without
            contributions[symbol] = contribution

        return contributions

    def get_correlation_estimate(self,
                                 symbols: list,
                                 market_data: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Estimate correlation matrix from market data.
        Falls back to assumed correlation if no data available.
        """
        n = len(symbols)

        if market_data is not None and not market_data.empty:
            # Compute realized correlation from returns
            returns_data = []
            for symbol in symbols:
                if symbol in market_data.columns:
                    returns = market_data[symbol]['Close'].pct_change().dropna()
                    returns_data.append(returns)

            if len(returns_data) == n:
                returns_df = pd.concat(returns_data, axis=1, keys=symbols)
                correlation_matrix = returns_df.corr().values
                return correlation_matrix

        # Fallback: Assume moderate correlation
        correlation_matrix = np.eye(n) * 0.7 + np.ones((n, n)) * 0.3
        logger.warning(f"[CVAR] Using assumed correlation matrix (0.3 off-diagonal)")

        return correlation_matrix
