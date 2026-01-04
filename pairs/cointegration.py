
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from typing import Tuple, Dict, Any

def engle_granger_test(x: pd.Series, y: pd.Series) -> Tuple[bool, float]:
    """Return (cointegrated, beta). If N<min_samples -> (False, None)"""
    try:
        clean_x = x.replace([np.inf, -np.inf], np.nan).dropna()
        clean_y = y.replace([np.inf, -np.inf], np.nan).dropna()
        
        # Align indices
        common_idx = clean_x.index.intersection(clean_y.index)
        if len(common_idx) < 30: return (False, 0.0)
        
        x_a = clean_x.loc[common_idx]
        y_a = clean_y.loc[common_idx]
        
        # OLS y = beta*x + c
        # Add constant
        X = sm.add_constant(x_a)
        model = sm.OLS(y_a, X).fit()
        beta = model.params.iloc[1]
        residuals = model.resid
        
        # ADF Test on residuals
        # H0: Non-stationary (Unit Root)
        adf_res = adfuller(residuals, maxlag=1, autolag=None)
        p_val = adf_res[1]
        
        is_coint = p_val < 0.05
        return (is_coint, float(beta))
        
    except Exception:
        return (False, 0.0)

def estimate_ou_params(spread: pd.Series) -> Dict[str, float]:
    """
    Return {'mu':..., 'theta':..., 'sigma':..., 'half_life':...} or {}
    OU: dX = -theta(X - mu)dt + sigma dW
    Discrete: X_t+1 = X_t - theta*mu*dt + theta*X_t*dt + ...
    Regress dx on x
    """
    try:
        clean_s = spread.replace([np.inf, -np.inf], np.nan).dropna()
        if len(clean_s) < 30: return {}
        
        # dt = 1 (daily)
        dx = clean_s.diff().dropna()
        x_lag = clean_s.shift(1).loc[dx.index]
        
        # Regress dx ~ constant + x_lag
        # dx = alpha + beta * x_lag
        # beta = -theta * dt  => theta = -beta/dt
        # alpha = theta * mu * dt => mu = alpha / (theta*dt)
        
        df_reg = pd.DataFrame({'dx': dx, 'x_lag': x_lag})
        df_reg = sm.add_constant(df_reg)
        
        res = sm.OLS(df_reg['dx'], df_reg[['const', 'x_lag']]).fit()
        alpha = res.params['const']
        beta = res.params['x_lag']
        
        dt = 1.0 # Assume daily steps
        theta = -beta / dt
        
        if abs(theta) < 1e-9: return {}
        
        mu = alpha / (theta * dt)
        sigma = res.resid.std()
        
        half_life = np.log(2) / theta if theta > 0 else -1.0
        
        return {
            'mu': float(mu),
            'theta': float(theta),
            'sigma': float(sigma),
            'half_life': float(half_life)
        }
        
    except Exception:
        return {}
