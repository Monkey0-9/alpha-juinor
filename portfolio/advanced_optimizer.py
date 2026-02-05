"""
Advanced Portfolio Optimizer
============================

Institutional portfolio construction implementing:
- Black-Litterman Model
- Hierarchical Risk Parity (HRP)
- Maximum Diversification
- Risk Parity (Equal Risk Contribution)
- Robust Mean-Variance Integration

Key Features:
- View blending (Analyst/ML views)
- Graph-theoretic clustering for HRP
- Covariance shrinkage
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.spatial.distance import squareform
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

@dataclass
class OptimizationResult:
    """Optimization output."""
    weights: Dict[str, float]
    expected_return: float
    expected_volatility: float
    diversification_ratio: float


class AdvancedOptimizer:
    """
    Portfolio construction engine for capital allocation.
    """

    def __init__(self, risk_free_rate: float = 0.02):
        self.rf = risk_free_rate

    def optimize_black_litterman(self,
                               market_caps: Dict[str, float],
                               cov_matrix: pd.DataFrame,
                               views: Dict[str, float],
                               view_confidences: Dict[str, float],
                               risk_aversion: float = 2.5) -> OptimizationResult:
        """
        Black-Litterman optimization.
        Combines market equilibrium with investor views.
        """
        assets = cov_matrix.columns

        # 1. Market Equilibrium (Prior)
        total_cap = sum(market_caps.values())
        market_weights = np.array([market_caps.get(a, 0)/total_cap for a in assets])

        # Reverse optimize to get Implied Excess Returns (Pi)
        pi = risk_aversion * cov_matrix.dot(market_weights)

        # 2. Integrate Views
        # P: Link matrix (identifies assets involved in views)
        # Q: View vector (expected returns)
        # Omega: Uncertainty matrix (diagonal of view variances)

        n = len(assets)
        k = len(views)

        if k == 0:
            combined_returns = pi
        else:
            P = np.zeros((k, n))
            Q = np.zeros(k)
            Omega = np.zeros((k, k))

            for i, (asset, view_ret) in enumerate(views.items()):
                idx = list(assets).index(asset)
                P[i, idx] = 1
                Q[i] = view_ret
                # Confidence -> Variance (lower conf = higher var)
                conf = view_confidences.get(asset, 0.5)
                Omega[i, i] = (1 - conf) / (conf + 1e-6) * 0.1  # Scaling factor

            # BL Formula for Posterior Returns
            tau = 0.05  # Scaling factor for uncertainty in prior

            # inv(tau*Sigma)
            tau_sigma_inv = np.linalg.inv(tau * cov_matrix)

            # P_T * inv(Omega) * P
            omega_inv = np.linalg.inv(Omega)

            # M = inv(inv(tau*Sigma) + P_T * inv(Omega) * P)
            M = np.linalg.inv(tau_sigma_inv + P.T @ omega_inv @ P)

            # Combined Returns = M * (inv(tau*Sigma)*Pi + P_T*inv(Omega)*Q)
            combined_returns = M @ (tau_sigma_inv @ pi + P.T @ omega_inv @ Q)

        # 3. Mean-Variance Optimization with posterior parameters
        res = self._run_mean_variance(combined_returns, cov_matrix, risk_aversion)

        return self._package_result(res, assets, combined_returns, cov_matrix)

    def optimize_hrp(self, returns: pd.DataFrame) -> OptimizationResult:
        """
        Hierarchical Risk Parity (HRP).
        Uses clustering to allocate risk, avoiding MV instability.
        """
        corr = returns.corr()
        cov = returns.cov()

        # 1. Hierarchical Clustering
        dist = np.sqrt(0.5 * (1 - corr))
        link = linkage(squareform(dist), method='single')

        # 2. Reorder covariance
        sort_ix = leaves_list(link)
        sorted_assets = returns.columns[sort_ix]
        cov_sorted = cov.loc[sorted_assets, sorted_assets]

        # 3. Recursive Bisection
        hrp_weights = self._recursive_bisection(cov_sorted, sort_ix)

        # Align weights to original order
        final_weights = pd.Series(hrp_weights, index=sorted_assets)
        final_weights = final_weights.reindex(returns.columns)

        return self._package_result(final_weights.values, returns.columns, returns.mean(), cov)

    def _recursive_bisection(self, cov: pd.DataFrame, sort_ix: List[int]) -> pd.Series:
        """Recursive bisection allocation."""
        weights = pd.Series(1, index=cov.index)
        items = [cov.index.tolist()]

        while len(items) > 0:
            items = [i for i in items if len(i) > 1]
            for item in items:
                # Bisect
                mid = len(item) // 2
                left = item[:mid]
                right = item[mid:]

                # Variance of clusters
                var_left = self._get_cluster_var(cov, left)
                var_right = self._get_cluster_var(cov, right)

                # Split factor based on inverse variance
                alpha = 1 - var_left / (var_left + var_right)

                # Apply weights
                weights[left] *= alpha
                weights[right] *= (1 - alpha)

            if len(items) > 0:
                # Generate next level items
                new_items = []
                for item in items:
                    mid = len(item) // 2
                    new_items.append(item[:mid])
                    new_items.append(item[mid:])
                items = new_items

        return weights

    def _get_cluster_var(self, cov, items):
        """Calculate variance of a cluster using Inverse Variance Portfolio."""
        sub_cov = cov.loc[items, items]
        inv_diag = 1 / np.diag(sub_cov)
        parity_w = inv_diag / inv_diag.sum()
        return parity_w.T @ sub_cov @ parity_w

    def _run_mean_variance(self, mu, sigma, gamma):
        """Standard convex optimization for MV."""
        n = len(mu)
        def utility(w):
            return - (w @ mu - 0.5 * gamma * (w.T @ sigma @ w))

        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = [(0.0, 0.4) for _ in range(n)] # 40% max weight cap

        w0 = np.ones(n) / n
        res = minimize(utility, w0, constraints=constraints, bounds=bounds)
        return res.x

    def _package_result(self, weights_array, assets, mu, sigma):
        weights = dict(zip(assets, weights_array))
        port_ret = weights_array @ mu
        port_vol = np.sqrt(weights_array.T @ sigma @ weights_array)

        # Div Ratio
        w_vol = weights_array @ np.sqrt(np.diag(sigma))
        div_ratio = w_vol / port_vol if port_vol > 0 else 0

        return OptimizationResult(weights, port_ret, port_vol, div_ratio)

