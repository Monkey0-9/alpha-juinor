import pandas as pd
import numpy as np
from typing import List, Dict

class DistributedBacktestEngine:
    """Institutional-grade backtesting with transaction costs and slippage"""
    
    def __init__(self, commission_bps: float = 1.0, slippage_model: str = "linear"):
        self.commission = commission_bps / 10000
        self.slippage_model = slippage_model
        
    def run_backtest(self, alpha_values: pd.Series, price_data: pd.DataFrame) -> Dict:
        """
        Run backtest with cost modeling.
        alpha_values: target weights (summing to 1.0)
        """
        returns = price_data["close"].pct_change().shift(-1)
        
        # Calculate turnover (for commission calculation)
        turnover = alpha_values.diff().abs().fillna(alpha_values.iloc[0])
        transaction_costs = turnover * self.commission
        
        # Simple slippage: 10% of volatility
        vol = returns.rolling(20).std().fillna(0)
        slippage = turnover * vol * 0.1
        
        raw_strategy_returns = alpha_values * returns
        net_strategy_returns = raw_strategy_returns - transaction_costs - slippage
        
        cumulative_returns = (1 + net_strategy_returns).cumprod()
        
        # Annualized Metrics
        mean_ret = net_strategy_returns.mean()
        std_ret = net_strategy_returns.std()
        sharpe = np.sqrt(252) * mean_ret / std_ret if std_ret > 0 else 0
        
        drawdown = (cumulative_returns / cumulative_returns.cummax() - 1)
        
        return {
            "sharpe_ratio": float(sharpe),
            "total_return_net": float(cumulative_returns.iloc[-2]),
            "max_drawdown": float(drawdown.min()),
            "avg_annual_return": float(mean_ret * 252),
            "volatility": float(std_ret * np.sqrt(252)),
            "turnover_annual": float(turnover.mean() * 252)
        }
