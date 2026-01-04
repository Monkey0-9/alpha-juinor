
import pandas as pd
import numpy as np

def compute_cvar(returns: pd.Series, alpha: float = 0.95) -> float:
    """
    Return CVaR (negative = loss). If insufficient data -> return 0.0 (neutral).
    Formula: CVaR_α = returns[returns <= VaR_α].mean()
    """
    try:
        # PANDAS HYGIENE: Explicitly handle infs/nans
        clean_rets = returns.replace([np.inf, -np.inf], np.nan).dropna()
        if len(clean_rets) < 20: # Minimal samples
            return 0.0
            
        # VaR cutoff
        var_cutoff = np.percentile(clean_rets, 100 * (1 - alpha))
        
        # Tail losses
        tail_losses = clean_rets[clean_rets <= var_cutoff]
        if tail_losses.empty:
            return 0.0
            
        cvar = tail_losses.mean()
        
        # Return negative float (e.g., -0.05 for 5% loss)
        return float(cvar)
        
    except Exception:
        return 0.0
