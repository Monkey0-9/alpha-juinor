# analytics/metrics.py
import pandas as pd
import numpy as np
from typing import Dict, Optional
import json

class PerformanceAnalyzer:
    """
    Calculate institutional-grade performance metrics.
    """
    
    def __init__(self, equity_curve: pd.Series, trades: Optional[pd.DataFrame] = None, risk_free_rate: float = 0.02):
        self.equity = equity_curve
        self.trades = trades
        self.rf = risk_free_rate
        
    def calculate_returns(self) -> pd.Series:
        """Daily returns from equity curve."""
        return self.equity.pct_change().fillna(0)
    
    def sharpe_ratio(self) -> float:
        """Annualized Sharpe Ratio."""
        returns = self.calculate_returns()
        if returns.std() == 0:
            return 0.0
        excess_return = returns.mean() - self.rf / 252
        return (excess_return / returns.std()) * np.sqrt(252)
    
    def sortino_ratio(self) -> float:
        """Annualized Sortino Ratio (downside deviation)."""
        returns = self.calculate_returns()
        downside = returns[returns < 0]
        if len(downside) == 0 or downside.std() == 0:
            return 0.0
        excess_return = returns.mean() - self.rf / 252
        return (excess_return / downside.std()) * np.sqrt(252)
    
    def max_drawdown(self) -> float:
        """Maximum Drawdown (%)."""
        cummax = self.equity.cummax()
        drawdown = (self.equity - cummax) / cummax
        return drawdown.min()
    
    def calmar_ratio(self) -> float:
        """Calmar Ratio (Annual Return / Max Drawdown)."""
        annual_return = self.annualized_return()
        max_dd = abs(self.max_drawdown())
        if max_dd == 0:
            return 0.0
        return annual_return / max_dd
    
    def annualized_return(self) -> float:
        """Annualized Return (%)."""
        if len(self.equity) < 2:
            return 0.0
        total_return = (self.equity.iloc[-1] / self.equity.iloc[0]) - 1
        years = len(self.equity) / 252
        if years == 0:
            return 0.0
        return (1 + total_return) ** (1 / years) - 1
    
    def annualized_volatility(self) -> float:
        """Annualized Volatility (%)."""
        returns = self.calculate_returns()
        return returns.std() * np.sqrt(252)
    
    def win_rate(self) -> float:
        """Win rate from trades (%)."""
        if self.trades is None or self.trades.empty:
            return 0.0
        if "pnl" not in self.trades.columns:
            return 0.0
        wins = (self.trades["pnl"] > 0).sum()
        total = len(self.trades)
        return wins / total if total > 0 else 0.0
    
    def avg_trade_pnl(self) -> float:
        """Average trade P&L."""
        if self.trades is None or self.trades.empty:
            return 0.0
        if "pnl" not in self.trades.columns:
            return 0.0
        return self.trades["pnl"].mean()
    
    def generate_report(self) -> Dict:
        """Generate comprehensive performance report."""
        report = {
            "total_return": float((self.equity.iloc[-1] / self.equity.iloc[0]) - 1) if len(self.equity) > 0 else 0.0,
            "annualized_return": float(self.annualized_return()),
            "annualized_volatility": float(self.annualized_volatility()),
            "sharpe_ratio": float(self.sharpe_ratio()),
            "sortino_ratio": float(self.sortino_ratio()),
            "max_drawdown": float(self.max_drawdown()),
            "calmar_ratio": float(self.calmar_ratio()),
            "win_rate": float(self.win_rate()),
            "avg_trade_pnl": float(self.avg_trade_pnl()),
            "total_trades": int(len(self.trades)) if self.trades is not None else 0,
            "final_equity": float(self.equity.iloc[-1]) if len(self.equity) > 0 else 0.0
        }
        return report
    
    def print_summary(self):
        """Print formatted summary to console."""
        report = self.generate_report()
        print("\n" + "="*60)
        print("PERFORMANCE SUMMARY")
        print("="*60)
        print(f"Total Return:          {report['total_return']*100:>8.2f}%")
        print(f"Annualized Return:     {report['annualized_return']*100:>8.2f}%")
        print(f"Annualized Volatility: {report['annualized_volatility']*100:>8.2f}%")
        print(f"Sharpe Ratio:          {report['sharpe_ratio']:>8.2f}")
        print(f"Sortino Ratio:         {report['sortino_ratio']:>8.2f}")
        print(f"Max Drawdown:          {report['max_drawdown']*100:>8.2f}%")
        print(f"Calmar Ratio:          {report['calmar_ratio']:>8.2f}")
        print(f"Win Rate:              {report['win_rate']*100:>8.2f}%")
        print(f"Avg Trade P&L:         ${report['avg_trade_pnl']:>8.2f}")
        print(f"Total Trades:          {report['total_trades']:>8}")
        print(f"Final Equity:          ${report['final_equity']:>8,.2f}")
        print("="*60 + "\n")
    
    def save_report(self, filepath: str):
        """Save report as JSON."""
        report = self.generate_report()
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
