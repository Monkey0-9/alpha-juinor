# portfolio/optimizer.py
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from typing import Dict, List, Optional

class MeanVarianceOptimizer:
    """
    Institutional Mean-Variance Optimization (Markowitz) with concentration constraints.
    """

    def __init__(self, risk_free_rate: float = 0.02, max_weight: float = 0.30, min_assets: int = 3):
        self.rf = risk_free_rate
        self.max_weight = max_weight
        self.min_assets = min_assets

    def optimize(self, expected_returns: pd.Series, covariate_matrix: pd.DataFrame, current_weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Optimize weights to maximize Sharpe Ratio with concentration caps.
        Includes Robust Covariance Shrinkage & Turnover Awareness.
        """
        tickers = expected_returns.index.tolist()
        n = len(tickers)
        
        if n == 0:
            return {}

        # 1. Robust Covariance Estimation (Manual Shrinkage)
        # Blend Sample Covariance with Diagonal (Variance-Target)
        # Prevents extreme corner solutions driven by noise.
        # Target: 0.3 Shrinkage (Institutional Standard for small interactions)
        Sigma_sample = covariate_matrix.values
        Sigma_diag = np.diag(np.diag(Sigma_sample))
        shrinkage = 0.30
        Sigma = (1.0 - shrinkage) * Sigma_sample + shrinkage * Sigma_diag
        
        # Prepare Current Weights Vector for Turnover Penalty
        if current_weights:
            # Align current weights to the ticker list
            w_current = np.array([current_weights.get(t, 0.0) for t in tickers])
        else:
            w_current = np.zeros(n)

        # Initial guess: equal weights
        w0 = np.array([1/n] * n)
        mu = expected_returns.values
        
        # Transaction Cost Constant (e.g., 20bps impact estimate)
        TC_PENALTY = 0.002

        def neg_sharpe(weights):
            p_ret = np.dot(weights, mu)
            p_vol = np.sqrt(np.dot(weights.T, np.dot(Sigma, weights)))
            
            # Turnover Penalty: Reduce utility by transaction costs
            turnover = np.sum(np.abs(weights - w_current))
            
            # Adjusted Utility: Sharpe - Cost Impact
            # Note: We penalize the numerator (Return) effectively
            # Heuristic: (Ret - RF - Cost) / Vol
            
            adj_ret = p_ret - self.rf - (turnover * TC_PENALTY)
            
            if p_vol < 1e-6: p_vol = 1e-6
            
            return - (adj_ret / p_vol)

        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}
        ]
        
        # Bounds enforce the concentration cap per asset
        # Fix #1: Concentration Risk (Enforce max position cap)
        bounds = tuple((0.0, self.max_weight) for _ in range(n))
        
        # Adjustment: If n * max_weight < 1.0, the 'eq' constraint is impossible.
        # This happens if tickers are too few. We relax the sum=1 constraint to sum <= 1
        # and treat the remainder as "Not allocated" (Cash).
        if n * self.max_weight < 1.0:
             constraints = [{'type': 'ineq', 'fun': lambda x: 1.0 - np.sum(x)}]
             # In this case, we prefer to fill up to max_weight
        
        try:
            result = minimize(neg_sharpe, w0, method='SLSQP', bounds=bounds, constraints=constraints)
            if not result.success:
                # Fallback: Diversified Equal Weight (Capped)
                w_eq = np.array([min(1/n, self.max_weight)] * n)
                w_eq = w_eq / w_eq.sum() if w_eq.sum() > 0 else w_eq
                return {t: w for t, w in zip(tickers, w_eq)}
            
            weights = result.x
            weights[weights < 0.001] = 0.0
            
            # Final check: diversity
            pos_count = np.sum(weights > 0.01)
            if pos_count < self.min_assets and n >= self.min_assets:
                 # If MVO concentrated too much, force more assets by slightly smoothing
                 # This is a heuristic to satisfy "Enforce minimum positions"
                 w_smooth = weights * 0.7 + (w0 * 0.3)
                 weights = w_smooth

            return {t: float(w) for t, w in zip(tickers, weights)}
            
        except Exception as e:
            return {t: 1.0/n for t in tickers}
