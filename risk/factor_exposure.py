# risk/factor_exposure.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from risk.factor_model import StatisticalRiskModel

class FactorExposureEngine:
    """
    Institutional Factor Exposure Tracking.
    Decomposes portfolio weights into systematic factor exposures.
    """
    
    def __init__(self, risk_model: StatisticalRiskModel):
        self.risk_model = risk_model
        
    def calculate_exposures(self, weights: Dict[str, float], returns_history: pd.DataFrame) -> pd.Series:
        """
        Calculate portfolio-level exposure to statistical factors.
        Returns a Series of factor loadings.
        """
        if not weights or returns_history.empty:
            return pd.Series(dtype=float)
            
        # 1. Align tickers
        tickers = [t for t in weights.keys() if t in returns_history.columns]
        if not tickers:
            return pd.Series(dtype=float)
            
        w_vec = pd.Series({t: weights[t] for t in tickers})
        
        # 2. Extract Factor Loadings (B) from Risk Model
        # This requires fitting the model if not already done, or reusing components
        # For simplicity, we re-run the PCA logic to get current loadings
        rets = returns_history[tickers].fillna(0.0)
        
        # We use the PCA components from the risk model logic
        # Standardize n_components to avoid ValueError
        n_comps = min(self.risk_model.n_components, rets.shape[1] - 1)
        if n_comps < 1:
            return pd.Series(0.0, index=["PC1"])
            
        self.risk_model.pca.n_components = n_comps
        self.risk_model.pca.fit(rets)
        # B shape: (n_assets, n_factors)
        B = self.risk_model.pca.components_.T
        
        # 3. Portfolio Exposure = w.T @ B
        # Shape: (1, n_assets) @ (n_assets, n_factors) -> (1, n_factors)
        exposures = w_vec.values @ B
        
        factor_names = [f"PC{i+1}" for i in range(B.shape[1])]
        return pd.Series(exposures, index=factor_names)

    def check_exposure_limits(self, exposures: pd.Series, limit: float = 0.40) -> List[str]:
        """
        Identify any factor exposures exceeding institutional limits.
        """
        violations = []
        for factor, value in exposures.items():
            if abs(value) > limit:
                violations.append(f"Factor {factor} Exposure {value:.2f} exceeds limit {limit}")
        return violations
