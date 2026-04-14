
import numpy as np
import pandas as pd
from typing import List, Union

def calculate_mu_hat(alphas: List[float], weights: List[float] = None) -> float:
    """
    Weighted average of alpha signals.
    mu_hat_i = sum_k w_k * alpha_i_k
    """
    if not alphas:
        return 0.0
    if weights is None:
        return np.mean(alphas)
    return np.average(alphas, weights=weights)

def robust_z_score(val: float, population: List[float]) -> float:
    """
    MAD-based Robust Z-Score.
    z_i = (val - median) / MAD
    """
    if not population or len(population) < 2:
        return 0.0

    med = np.median(population)
    mad = np.median(np.abs(np.array(population) - med))

    if mad == 0:
        return 0.0

    return (val - med) / (mad * 1.4826) # 1.4826 scales MAD to Sigma for normal dist

def risk_adjusted_score(mu: float, sigma: float) -> float:
    """S_i = mu_i / sigma_i"""
    if sigma <= 0: return 0.0
    return mu / sigma

def fractional_kelly(mu: float, sigma: float, gamma: float = 0.2, max_leverage: float = 2.0) -> float:
    """
    f_i = gamma * mu_i / sigma_i^2
    """
    if sigma <= 0: return 0.0

    f = gamma * mu / (sigma ** 2)
    return np.clip(f, -max_leverage, max_leverage)

def estimate_impact(size: float, adv: float, volatility: float) -> float:
    """
    Square-root impact law (Almgren-Chriss approximation).
    Impact ~ sigma * sqrt(size / adv)
    """
    if adv <= 0: return float('inf')
    return volatility * np.sqrt(abs(size) / adv)
