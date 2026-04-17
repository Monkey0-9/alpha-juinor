import numpy as np
import pandas as pd
from typing import Dict, Any, List

class PerformanceStats:
    """
    Computes institutional performance metrics from equity history.
    """
    @staticmethod
    def calculate_metrics(equity_history: List[Dict[str, Any]], risk_free_rate: float = 0.0) -> Dict[str, Any]:
        if not equity_history:
            return {}

        df = pd.DataFrame(equity_history)
        df.set_index('timestamp', inplace=True)
        
        # 1. Returns
        df['returns'] = df['equity'].pct_change().fillna(0)
        
        # 2. Cumulative Return
        total_return = (df['equity'].iloc[-1] / df['equity'].iloc[0]) - 1
        
        # 3. Volatility (Annualized)
        vol = df['returns'].std() * np.sqrt(252)
        
        # 4. Sharpe Ratio
        mean_return = df['returns'].mean() * 252
        sharpe = (mean_return - risk_free_rate) / vol if vol != 0 else 0
        
        # 5. Sortino Ratio (Downside deviation)
        downside_returns = df[df['returns'] < 0]['returns']
        downside_vol = downside_returns.std() * np.sqrt(252)
        sortino = (mean_return - risk_free_rate) / downside_vol if downside_vol != 0 else 0
        
        # 6. Drawdown
        df['peak'] = df['equity'].cummax()
        df['drawdown'] = (df['equity'] - df['peak']) / df['peak']
        max_drawdown = df['drawdown'].min()
        
        # 7. Calmar Ratio
        calmar = mean_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # 8. Win Rate (of individual entries, but here we just show pos returns days)
        win_rate = len(df[df['returns'] > 0]) / len(df) if len(df) > 0 else 0

        return {
            "total_return": total_return,
            "annualized_return": mean_return,
            "annualized_vol": vol,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_drawdown,
            "calmar_ratio": calmar,
            "win_rate": win_rate,
            "days_tracked": len(df)
        }
