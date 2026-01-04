
import pandas as pd
import numpy as np
from typing import Dict

def fit_pot_tail(returns: pd.Series, threshold_quantile: float = 0.95) -> Dict[str, float]:
    """
    EVT / POT tail detector.
    Uses simplified Peak-Over-Threshold logic via Generalized Pareto Distribution (GPD) proxy (Hill Estimator / Moments).
    Returns {'shape_xi': float, 'tail_risk_score': 0.0-1.0}
    """
    try:
        clean_rets = returns.replace([np.inf, -np.inf], np.nan).dropna()
        # We only care about LOSS tail -> invert negative returns
        losses = -clean_rets[clean_rets < 0]
        
        if len(losses) < 50:
            return {'shape_xi': 0.0, 'tail_risk_score': 0.0}
            
        u = np.percentile(losses, threshold_quantile * 100)
        exceedances = losses[losses > u] - u
        
        if len(exceedances) < 10:
             return {'shape_xi': 0.0, 'tail_risk_score': 0.0}
             
        # Estimate Shape Parameter (Xi) using Method of Moments for GPD
        # mean_exc = exceedances.mean()
        # var_exc = exceedances.var()
        # xi = 0.5 * (1 - (mean_exc**2 / var_exc)) # Rough estimator
        
        # Simplified Hill Estimator for Tail Index (valid for heavy tails)
        # alpha = 1 / xi
        xi = (np.log(exceedances / u)).mean()
        
        # Threshold: Xi > 0.2 implies Heavy Tail (Infinite Variance if Xi > 0.5)
        # Score 0.0 (Gaussian) to 1.0 (Cauchy/Infinite)
        score = np.clip(xi / 0.5, 0.0, 1.0)
        
        return {'shape_xi': float(xi), 'tail_risk_score': float(score)}
        
    except Exception:
        return {'shape_xi': 0.0, 'tail_risk_score': 0.0}
