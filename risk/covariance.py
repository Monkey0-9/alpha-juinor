"""
Covariance Estimation with Shrinkage

Implements Ledoit-Wolf shrinkage for robust covariance estimation.
Critical for stable VaR/CVaR and portfolio optimization.
"""
import numpy as np
import pandas as pd
from typing import Union, Tuple
from sklearn.covariance import LedoitWolf, OAS
import logging

logger = logging.getLogger(__name__)


def ledoit_wolf_covariance(returns: pd.DataFrame, assume_centered: bool = False) -> Tuple[np.ndarray, float]:
    """
    Compute Ledoit-Wolf shrinkage covariance estimator.
    
    Args:
        returns: DataFrame of asset returns (rows=dates, cols=assets)
        assume_centered: If True, data will not be centered before computation
    
    Returns:
        Tuple of (covariance_matrix, shrinkage_coefficient)
    """
    if returns.empty or len(returns) < 2:
        raise ValueError("Insufficient data for covariance estimation")
    
    # Remove NaN
    returns_clean = returns.dropna()
    if len(returns_clean) < 2:
        raise ValueError("Insufficient non-NaN data for covariance estimation")
    
    # Ledoit-Wolf estimator
    lw = LedoitWolf(assume_centered=assume_centered)
    lw.fit(returns_clean.values)
    
    cov_matrix = lw.covariance_
    shrinkage = lw.shrinkage_
    
    logger.debug(f"Ledoit-Wolf covariance: shrinkage={shrinkage:.4f}")
    
    return cov_matrix, shrinkage


def oracle_approximating_shrinkage_covariance(returns: pd.DataFrame, assume_centered: bool = False) -> Tuple[np.ndarray, float]:
    """
    Compute Oracle Approximating Shrinkage (OAS) covariance estimator.
    
    Similar to Ledoit-Wolf but uses a different shrinkage formula.
    
    Args:
        returns: DataFrame of asset returns
        assume_centered: If True, data will not be centered
    
    Returns:
        Tuple of (covariance_matrix, shrinkage_coefficient)
    """
    if returns.empty or len(returns) < 2:
        raise ValueError("Insufficient data for covariance estimation")
    
    returns_clean = returns.dropna()
    if len(returns_clean) < 2:
        raise ValueError("Insufficient non-NaN data for covariance estimation")
    
    oas = OAS(assume_centered=assume_centered)
    oas.fit(returns_clean.values)
    
    cov_matrix = oas.covariance_
    shrinkage = oas.shrinkage_
    
    logger.debug(f"OAS covariance: shrinkage={shrinkage:.4f}")
    
    return cov_matrix, shrinkage


def robust_covariance(returns: pd.DataFrame, method: str = "ledoit_wolf") -> np.ndarray:
    """
    Compute robust covariance matrix using shrinkage.
    
    Args:
        returns: DataFrame of asset returns
        method: "ledoit_wolf" or "oas"
    
    Returns:
        Covariance matrix
    """
    if method == "ledoit_wolf":
        cov, _ = ledoit_wolf_covariance(returns)
    elif method == "oas":
        cov, _ = oracle_approximating_shrinkage_covariance(returns)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return cov


def exponentially_weighted_covariance(returns: pd.DataFrame, halflife: int = 60) -> np.ndarray:
    """
    Compute exponentially weighted covariance matrix.
    
    Gives more weight to recent observations.
    
    Args:
        returns: DataFrame of asset returns
        halflife: Half-life for exponential weighting (in days)
    
    Returns:
        Covariance matrix
    """
    if returns.empty:
        raise ValueError("Empty returns data")
    
    # Use pandas ewm for convenience
    ewm_cov = returns.ewm(halflife=halflife).cov()
    
    # Extract the latest covariance matrix
    # ewm_cov is a MultiIndex DataFrame, we want the last "date" level
    latest_cov = ewm_cov.xs(returns.index[-1], level=0)
    
    return latest_cov.values


def hybrid_covariance(returns: pd.DataFrame, 
                     shrinkage_weight: float = 0.7,
                     ewm_halflife: int = 60) -> np.ndarray:
    """
    Combine shrinkage and exponentially weighted covariance.
    
    Args:
        returns: DataFrame of asset returns
        shrinkage_weight: Weight for shrinkage cov (1 - weight for EWM)
        ewm_halflife: Half-life for EWM component
    
    Returns:
        Hybrid covariance matrix
    """
    cov_shrink = robust_covariance(returns, method="ledoit_wolf")
    cov_ewm = exponentially_weighted_covariance(returns, halflife=ewm_halflife)
    
    # Weighted average
    hybrid_cov = shrinkage_weight * cov_shrink + (1 - shrinkage_weight) * cov_ewm
    
    return hybrid_cov
