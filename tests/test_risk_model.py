
import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from risk.factor_model import StatisticalRiskModel

def test_pca_risk_model():
    # 1. Create correlated synthetic data
    np.random.seed(42)
    n_obs = 500
    n_assets = 4
    
    # 2 common factors
    F = np.random.normal(0, 0.01, size=(n_obs, 2))
    # Loadings
    B = np.random.normal(0.5, 0.2, size=(n_assets, 2))
    # Idiosyncratic
    epsilon = np.random.normal(0, 0.005, size=(n_obs, n_assets))
    
    # R = F * B' + e
    Returns = F @ B.T + epsilon
    
    df = pd.DataFrame(Returns, columns=["A", "B", "C", "D"])
    
    # 2. Initialize Model
    model = StatisticalRiskModel(n_components=2)
    
    # 3. Compute Covariance
    cov = model.compute_covariance(df)
    
    # 4. Checks
    assert cov.shape == (n_assets, n_assets)
    # Symmetry
    assert np.allclose(cov, cov.T, atol=1e-8)
    # Diagonals should be positive
    assert np.all(np.diag(cov) > 0)
    
    print("Robust Covariance:\n", cov)
