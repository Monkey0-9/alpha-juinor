import numpy as np
import pandas as pd
from typing import Dict, Tuple

def compute_tail_risk_metrics(returns: pd.Series, threshold_quantile: float = 0.90) -> Dict[str, float]:
    """
    Implements Extreme Value Theory (EVT) using Peaks Over Threshold (POT) approach
    to estimate tail risk beyond standard Gaussian assumptions.
    
    Returns:
    - cvar: Conditional Value at Risk (Expected Shortfall) at 99%
    - tail_index: The shape parameter 'xi' (measure of fat-tailedness)
    - evtrisk_score: Normalized 0-1 score of tail danger
    """
    if len(returns) < 60:
        return {"cvar": 0.0, "tail_index": 0.0, "evtrisk_score": 0.0}

    # 1. Isolate Tail Losses (Positive values for loss calculations)
    losses = -returns[returns < 0]
    if len(losses) < 20:
        return {"cvar": 0.0, "tail_index": 0.0, "evtrisk_score": 0.0}
        
    # 2. Determine Threshold (u)
    u = losses.quantile(threshold_quantile)
    exceedances = losses[losses > u] - u
    
    if len(exceedances) < 5:
        # Fallback to Historical CVaR if strictly insufficient data for GPD
        var_99 = -returns.quantile(0.01)
        cvar_99 = -returns[returns <= -var_99].mean()
        return {
            "cvar": cvar_99 if not np.isnan(cvar_99) else 0.0,
            "tail_index": 0.0, 
            "evtrisk_score": 0.0  
        }

    # 3. Method of Moments Estimation for GPD parameters (Robust & Fast)
    # Shape (xi) and Scale (sigma)
    # E[X] = sigma / (1 - xi)
    # Var(X) = sigma^2 / ((1-xi)^2 * (1-2xi))
    
    mean_excess = exceedances.mean()
    var_excess = exceedances.var()
    
    # xi estimate using moments
    if var_excess > mean_excess**2:
        xi = 0.5 * (1 - (mean_excess**2 / var_excess))
    else:
        xi = 0.0 # Thin tail or insufficient variance
        
    # Sigma estimate
    sigma = mean_excess * (1 - xi)
    
    # 4. Calculate EVT-based VaR and CVaR (99%)
    # Formula: VaR_p = u + (sigma/xi) * (( (n/Nu) * (1-p) )^(-xi) - 1)
    
    confidence = 0.99
    n = len(returns)
    Nu = len(exceedances)
    prob_exceed = Nu / n
    
    # Safety clamp for xi to prevent explosion
    xi = np.clip(xi, -0.5, 0.5)
    
    try:
        if abs(xi) < 0.001:
            # Limit case (Exponential)
            var_evt = u - sigma * np.log( (n/Nu) * (1-confidence) )
        else:
            term =  (( (n/Nu) * (1-confidence) ) ** (-xi)) - 1
            var_evt = u + (sigma / xi) * term
            
        # CVaR (Expected Shortfall)
        # ES = VaR + (sigma + xi*(VaR - u)) / (1 - xi)
        cvar_evt = var_evt + (sigma + xi * (var_evt - u)) / (1 - xi)
        
    except:
        cvar_evt = 0.0
        
    # 5. Risk Score
    # Tail Index > 0.2 implies very fat tails (Dangerous)
    evtrisk_score = np.clip(xi / 0.4, 0.0, 1.0)
    
    return {
        "cvar": float(cvar_evt),
        "tail_index": float(xi),
        "evtrisk_score": float(evtrisk_score)
    }
