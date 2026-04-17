import numpy as np
import pandas as pd

TRADING_DAYS = 252

def daily_returns(prices: pd.Series) -> pd.Series:
    returns = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    return returns.fillna(0)

def annualized_return(returns: pd.Series) -> float:
    compounded = (1 + returns).prod()
    years = len(returns) / TRADING_DAYS
    return compounded ** (1 / years) - 1

def annualized_volatility(returns: pd.Series) -> float:
    if returns.empty or returns.isna().all():
        return 0.0
    return returns.std() * np.sqrt(TRADING_DAYS)

def sharpe_ratio(returns: pd.Series, risk_free_rate=0.0) -> float:
    if returns.empty or returns.isna().all():
        return np.nan
    excess = returns - risk_free_rate / TRADING_DAYS
    vol = annualized_volatility(excess)
    if vol == 0:
        return np.nan
    return annualized_return(excess) / vol

def max_drawdown(equity_curve: pd.Series):
    if equity_curve.empty or equity_curve.isna().all():
        return 0.0
    peak = equity_curve.cummax()
    drawdown = equity_curve / peak - 1
    return drawdown.min()
