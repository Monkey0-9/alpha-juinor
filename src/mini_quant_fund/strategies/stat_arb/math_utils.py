"""
Statistical Arbitrage Math Utilities
====================================

Core mathematical functions for Pairs Trading and Mean Reversion.

Includes:
1. Augmented Dickey-Fuller (ADF) Test for cointegration/stationarity.
2. Ornstein-Uhlenbeck (OU) Process fitting for spread modeling.
"""

import numpy as np
import pandas as pd
try:
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import adfuller
except ImportError as e:
    raise RuntimeError(
        "statsmodels is required for StatArbEngine. "
        "Install with: pip install statsmodels"
    ) from e
from typing import Tuple, Dict


def check_cointegration(series_a: pd.Series, series_b: pd.Series) -> Tuple[bool, float, float]:
    """
    Perform Engle-Granger two-step cointegration test.

    Step 1: Regress log(A) on log(B) to find spread.
    Step 2: Run ADF on residuals (spread).

    Returns:
        (is_cointegrated, p_value, hedge_ratio)
    """
    # Ensure inputs are valid
    if len(series_a) != len(series_b) or len(series_a) < 30:
        return False, 1.0, 0.0

    # Log prices
    log_a = np.log(series_a)
    log_b = np.log(series_b)

    # Linear Regression (OLS)
    # log_a = beta * log_b + c + epsilon
    X = sm.add_constant(log_b)
    model = sm.OLS(log_a, X).fit()

    hedge_ratio = model.params.iloc[1]
    spread = log_a - (hedge_ratio * log_b) - model.params.iloc[0]

    # Run ADF on residuals
    # Null Hypothesis: Non-stationary (Unit Root)
    # Low p-value (<0.05) => Stationary => Cointegrated
    adf_result = adfuller(spread)
    p_value = adf_result[1]

    is_cointegrated = p_value < 0.05

    return is_cointegrated, p_value, hedge_ratio


def fit_ou_process(spread: pd.Series, dt: float = 1/252) -> Dict[str, float]:
    """
    Fit an Ornstein-Uhlenbeck process to the spread.

    dX_t = theta * (mu - X_t) * dt + sigma * dW_t

    Discretized (AR(1)):
    X_{t+1} = X_t * e^{-theta*dt} + mu * (1 - e^{-theta*dt}) + epsilon

    Returns:
        {theta, mu, sigma, half_life}
    """
    X_t = spread[:-1].values
    X_tp1 = spread[1:].values

    # Linear Regression: X_{t+1} = a * X_t + b + error
    # Slope (a) = exp(-theta * dt)
    # Intercept (b) = mu * (1 - exp(-theta * dt))

    model = sm.OLS(X_tp1, sm.add_constant(X_t)).fit()
    a = model.params[1]
    b = model.params[0]

    # Recover parameters
    # theta = -ln(a) / dt
    if a <= 0 or a >= 1:
        # Non-mean-reverting or explosive
        return {"theta": 0.0, "mu": 0.0, "sigma": 0.0, "half_life": float('inf')}

    theta = -np.log(a) / dt
    mu = b / (1 - a)

    # Sigma from residuals
    residuals = model.resid
    sigma_epsilon = np.std(residuals)
    # Variance of shock: sigma^2 * (1 - exp(-2*theta*dt)) / (2*theta) -> Approx for discrete
    # Simplification: sigma = std(resid) * sqrt(2*theta / (1-a^2))
    # Or just sigma_eq = sigma_epsilon / sqrt( (1-exp(-2*theta*dt)) / (2*theta) )

    # Common approx: sigma_residual = sigma * sqrt(dt)
    sigma = sigma_epsilon / np.sqrt(dt)

    half_life = np.log(2) / theta

    return {
        "theta": theta,
        "mu": mu,
        "sigma": sigma,
        "half_life": half_life
    }


def calculate_ou_score(spread_val: float, params: Dict[str, float]) -> float:
    """
    Calculate Z-score based on OU stationary distribution.

    Z = (X_t - mu) / sigma_eq
    sigma_eq = sigma / sqrt(2*theta)
    """
    if params['theta'] <= 0 or params['sigma'] <= 0:
        return 0.0

    mu = params['mu']
    sigma = params['sigma']
    theta = params['theta']

    sigma_eq = sigma / np.sqrt(2 * theta)

    z_score = (spread_val - mu) / sigma_eq
    return z_score
