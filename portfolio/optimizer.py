# portfolio/optimizer.py
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from typing import Dict, List, Optional

class MeanVarianceOptimizer:
    """
    Classic Mean-Variance Optimization (Markowitz).
    """

    def __init__(self, risk_free_rate: float = 0.02):
        self.rf = risk_free_rate

    def optimize(self, expected_returns: pd.Series, covariate_matrix: pd.DataFrame) -> Dict[str, float]:
        """
        Optimize weights to maximize Sharpe Ratio.
        """
        tickers = expected_returns.index.tolist()
        n = len(tickers)
        
        if n == 0:
            return {}
        if n == 1:
            return {tickers[0]: 1.0}

        # Initial guess: equal weights
        w0 = np.array([1/n] * n)
        
        # Convert inputs to numpy
        mu = expected_returns.values
        Sigma = covariate_matrix.values
        
        # Objective: Minimize negative Sharpe Ratio
        def neg_sharpe(weights):
            p_ret = np.dot(weights, mu)
            p_vol = np.sqrt(np.dot(weights.T, np.dot(Sigma, weights)))
            # Annualize if inputs are daily? 
            # Assuming inputs are annualized for simplicity, or consistency doesn't matter for weights
            # Let's assume inputs are annualized parameters
            return - (p_ret - self.rf) / (p_vol + 1e-6)

        # Constraints
        # 1. Sum of weights = 1.0 (Fully invested) - user might want cash, but MVO usually assumes 100% aloc
        # 2. Weights between 0 and 1 (Long only)
        
        constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}
        )
        bounds = tuple((0.0, 1.0) for _ in range(n))
        
        try:
            result = minimize(neg_sharpe, w0, method='SLSQP', bounds=bounds, constraints=constraints)
            if not result.success:
                print(f"[Optimizer] Optimization failed: {result.message}. Using equal weights.")
                return {t: 1.0/n for t in tickers}
            
            weights = result.x
            # limit tiny weights
            weights[weights < 0.01] = 0.0
            weights = weights / weights.sum()
            
            return {t: w for t, w in zip(tickers, weights)}
            
        except Exception as e:
            print(f"[Optimizer] Error: {e}. Using equal weights.")
            return {t: 1.0/n for t in tickers}
