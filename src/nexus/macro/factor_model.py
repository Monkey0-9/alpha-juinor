import pandas as pd
import numpy as np
from typing import Dict

class MacroFactorModel:
    """Portfolio sensitivity to macro factors (Scaffold)"""
    
    def calculate_factor_exposure(self, returns: pd.Series, factors: pd.DataFrame) -> pd.Series:
        """
        Estimate factor betas via regression.
        """
        # returns = beta * factors + alpha
        # Simplified: using numpy lstsq
        A = np.vstack([factors.values.T, np.ones(len(factors))]).T
        betas, residuals, rank, s = np.linalg.lstsq(A, returns.values, rcond=None)
        
        return pd.Series(betas[:-1], index=factors.columns)
