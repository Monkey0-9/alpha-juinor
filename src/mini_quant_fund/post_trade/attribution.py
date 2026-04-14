from typing import List, Dict
import pandas as pd
import numpy as np

class PnLAttributionEngine:
    """Decompose P&L into factors (Scaffold)"""
    
    def attribute_pnl(self, daily_returns: pd.Series, factor_returns: pd.DataFrame) -> Dict[str, float]:
        """
        Decompose returns into Beta, Alpha, and Factor exposures.
        """
        # Simple regression
        A = np.vstack([factor_returns.values.T, np.ones(len(factor_returns))]).T
        betas, residuals, rank, s = np.linalg.lstsq(A, daily_returns.values, rcond=None)
        
        exposures = betas[:-1]
        alpha = betas[-1]
        
        attribution = {factor: float(exposures[i] * factor_returns[factor].mean()) 
                       for i, factor in enumerate(factor_returns.columns)}
        attribution["alpha"] = float(alpha)
        
        return attribution
