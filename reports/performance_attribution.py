# reports/performance_attribution.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

class PerformanceAnalyzer:
    """
    Institutional Performance Attribution & Analytics.
    
    Provides:
    - Risk-adjusted metrics (Sharpe, Sortino, Calmar)
    - Drawdown analysis
    - Execution drag (Slippage vs Alpha)
    - Risk contribution by asset
    """
    
    def __init__(self, equity_series: pd.Series, trades_df: pd.DataFrame, risk_free_rate: float = 0.02):
        self.equity = equity_series
        self.trades = trades_df
        self.rfr = risk_free_rate
        self.returns = equity_series.pct_change().dropna()
        
    def calculate_metrics(self) -> Dict[str, float]:
        """Core risk-adjusted performance metrics."""
        if self.returns.empty:
            return {}
            
        total_return = (self.equity.iloc[-1] / self.equity.iloc[0]) - 1
        ann_return = (1 + total_return) ** (252 / len(self.equity)) - 1
        
        vol = self.returns.std() * np.sqrt(252)
        sharpe = (ann_return - self.rfr) / vol if vol > 0 else 0.0
        
        downside_rets = self.returns[self.returns < 0]
        downside_vol = downside_rets.std() * np.sqrt(252)
        sortino = (ann_return - self.rfr) / downside_vol if downside_vol > 0 else 0.0
        
        # Max Drawdown
        cum_rets = (1 + self.returns).cumprod()
        running_max = cum_rets.cummax()
        drawdowns = (cum_rets - running_max) / running_max
        max_dd = drawdowns.min()
        
        metrics = {
            "Total Return": total_return,
            "Annualized Return": ann_return,
            "Annualized Vol": vol,
            "Sharpe Ratio": sharpe,
            "Sortino Ratio": sortino,
            "Max Drawdown": max_dd,
            "Trade Count": len(self.trades)
        }
        
        return metrics

    def analyze_execution(self) -> Dict[str, Any]:
        """Evaluate execution efficiency and drag."""
        if self.trades.empty:
            return {"Total Cost": 0.0, "Execution Drag (bps)": 0.0}
            
        total_commission = self.trades["commission"].sum()
        total_slippage = self.trades["slippage"].sum()
        total_cost = self.trades["cost"].sum()
        
        # Drag in basis points relative to average equity
        avg_equity = self.equity.mean()
        drag_bps = (total_cost / avg_equity) * 10000 if avg_equity > 0 else 0.0
        
        return {
            "Total Commission": total_commission,
            "Total Slippage": total_slippage,
            "Total Cost": total_cost,
            "Execution Drag (bps)": drag_bps,
            "Avg Slippage per Trade": total_slippage / len(self.trades)
        }

    def analyze_attribution(self, benchmark_returns: pd.Series) -> Dict[str, float]:
        """Decompose performance into Alpha and Beta relative to benchmark."""
        common_idx = self.returns.index.intersection(benchmark_returns.index)
        if len(common_idx) < 20:
            return {"Alpha (Ann)": 0.0, "Beta": 1.0, "Correlation": 0.0}
            
        y = self.returns.loc[common_idx]
        X = benchmark_returns.loc[common_idx]
        
        # OLS beta
        beta = np.cov(y, X)[0, 1] / np.var(X)
        alpha_ann = (y.mean() - beta * X.mean()) * 252
        corr = np.corrcoef(y, X)[0, 1]
        
        return {
            "Alpha (Ann)": float(alpha_ann),
            "Beta": float(beta),
            "Correlation": float(corr),
            "Active Return": float((y.mean() - X.mean()) * 252)
        }

    def generate_report(self, benchmark_returns: Optional[pd.Series] = None) -> str:
        """String representation of the performance report."""
        m = self.calculate_metrics()
        e = self.analyze_execution()
        
        report = []
        report.append("="*40)
        report.append("   INSTITUTIONAL PERFORMANCE REPORT")
        report.append("="*40)
        
        report.append("\n[RISK/RETURN METRICS]")
        report.append(f"Total Return:         {m.get('Total Return', 0):.2%}")
        report.append(f"Annualized Return:    {m.get('Annualized Return', 0):.2%}")
        report.append(f"Annualized Vol:       {m.get('Annualized Vol', 0):.2%}")
        report.append(f"Sharpe Ratio:         {m.get('Sharpe Ratio', 0):.2f}")
        report.append(f"Sortino Ratio:        {m.get('Sortino Ratio', 0):.2f}")
        report.append(f"Max Drawdown:         {m.get('Max Drawdown', 0):.2%}")
        
        report.append("\n[EXECUTION ANALYSIS]")
        report.append(f"Total Trades:         {m.get('Trade Count', 0)}")
        report.append(f"Total Cost:           ${e.get('Total Cost', 0):,.2f}")
        report.append(f"Execution Drag:       {e.get('Execution Drag (bps)', 0):.1f} bps")
        report.append(f"Avg Slippage/Trade:   ${e.get('Avg Slippage per Trade', 0):.2f}")
        
        if benchmark_returns is not None:
            attr = self.analyze_attribution(benchmark_returns)
            report.append("\n[PERFORMANCE ATTRIBUTION]")
            report.append(f"Alpha (Annualized):   {attr.get('Alpha (Ann)', 0):.2%}")
            report.append(f"Beta (vs Bench):      {attr.get('Beta', 0):.2f}")
            report.append(f"Correlation:          {attr.get('Correlation', 0):.2f}")
            report.append(f"Active Return:        {attr.get('Active Return', 0):.2%}")

        report.append("="*40)
        return "\n".join(report)
