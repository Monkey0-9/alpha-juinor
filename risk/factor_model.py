# risk/factor_model.py
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from typing import Tuple, Dict

class StatisticalRiskModel:
    """
    Statistical Risk Model using PCA (Principal Component Analysis).
    Decomposes returns into:
    R = B * F + epsilon
    
    Where:
    - B (Beta): Factor Loadings (Exposures)
    - F (Factors): Factor Returns (Systematic Risk)
    - epsilon: Idiosyncratic Returns (Specific Risk)
    
    Covariance Matrix V = B * Sigma_f * B.T + D
    where Sigma_f is Factor Covariance, D is diagonal Specific Variances.
    """

    def __init__(self, n_components: int = 3, lookback: int = 252):
        self.n_components = n_components
        self.lookback = lookback
        self.pca = PCA(n_components=n_components)

    def compute_covariance(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Compute robust covariance matrix using Factor Model.
        Input: DataFrame of returns (not prices).
        """
        if returns.empty or len(returns) < self.lookback // 4:
            # Fallback to simple covariance if insufficient data
            return returns.cov() * 252

        # 1. Prepare Data
        # Fill NaNs (Risk models hate NaNs)
        X = returns.fillna(0.0)
        
        # Verify we have enough columns for components
        n_assets = X.shape[1]
        n_comps = min(self.n_components, n_assets - 1)
        if n_comps < 1:
            return X.cov() * 252
            
        # 2. Fit PCA
        # Sklearn PCA centers data automatically
        self.pca.n_components = n_comps
        self.pca.fit(X)
        
        # 3. Extract Factor Loadings (B)
        # Components_ shape: (n_components, n_features)
        # B shape: (n_features, n_components)
        B = self.pca.components_.T
        
        # 4. Factor Covariance (Sigma_f)
        # Factor returns F = X @ B (if B is projection)
        # Actually PCA transform gives factors
        F = self.pca.transform(X)
        Sigma_f = np.cov(F.T)
        
        # Reshape Sigma_f to 2D if scalar (1 component)
        if n_comps == 1:
            Sigma_f = np.array([[Sigma_f]])
            
        # 5. Specific Variance (D)
        # Reproduced returns = F @ B.T + Mean
        # Residuals = Original - Reproduced
        X_reconstructed = self.pca.inverse_transform(F)
        residuals = X - X_reconstructed
        
        # Diagonal specific variance
        specific_vars = np.var(residuals, axis=0)
        D = np.diag(specific_vars)
        
        # 6. Reconstruct Robust Covariance (Annualized)
        # V = (B * Sigma_f * B.T + D) * 252
        
        # B is (N, K), Sigma_f is (K, K) -> (N, K)
        systematic_cov = B @ Sigma_f @ B.T
        robust_cov = (systematic_cov + D) * 252
        
        return pd.DataFrame(robust_cov, index=returns.columns, columns=returns.columns)
