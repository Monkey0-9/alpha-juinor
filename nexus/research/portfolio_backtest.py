import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class PortfolioBacktestResult:
    equity_curve: pd.Series
    drawdown_curve: pd.Series
    returns_series: pd.Series
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float

class PortfolioBacktester:
    """Institutional Portfolio-level Backtest Engine.
    
    Simulates multi-asset trading with proportional sizing, 
    commission modeling, and cross-asset correlation impact.
    """

    def __init__(
        self, 
        initial_capital: float = 1_000_000.0, 
        commission_bps: float = 4.0
    ):
        self.initial_capital = initial_capital
        self.commission = commission_bps / 10000.0

    def run(
        self, 
        price_data: Dict[str, pd.Series], 
        signal_data: Dict[str, pd.Series],
        max_positions: int = 12
    ) -> PortfolioBacktestResult:
        """Execute a vectorized portfolio backtest."""
        # Align all data to a common index
        all_prices = pd.DataFrame(price_data).dropna()
        all_signals = pd.DataFrame(signal_data).reindex(all_prices.index).fillna(0)
        
        # 1. Strategy Logic: Rank signals and select top N
        ranked_signals = all_signals.rank(axis=1, ascending=False)
        target_weights = (ranked_signals <= max_positions).astype(float)
        
        # Normalize weights so they sum to 1.0 (or less if fewer symbols)
        row_sums = target_weights.sum(axis=1)
        weights = target_weights.div(row_sums, axis=0).fillna(0)
        
        # 2. Performance Logic
        asset_returns = all_prices.pct_change().fillna(0)
        
        # Simple weighted sum of returns (assuming daily rebalancing)
        portfolio_returns = (weights.shift(1) * asset_returns).sum(axis=1)
        
        # 3. Transaction Costs Estimation
        # (Weight change * commission)
        weight_diff = weights.diff().abs().sum(axis=1)
        costs = weight_diff * self.commission
        
        net_returns = portfolio_returns - costs
        
        # 4. Equity Curve
        equity_curve = self.initial_capital * (1 + net_returns).cumprod()
        
        # 5. Metrics
        total_return = (equity_curve.iloc[-1] / self.initial_capital) - 1
        vol = net_returns.std() * np.sqrt(252)
        sharpe = (net_returns.mean() * 252) / (vol if vol > 0 else 1e-9)
        
        rolling_max = equity_curve.cummax()
        drawdowns = (equity_curve - rolling_max) / rolling_max
        max_dd = drawdowns.min()
        
        win_rate = (net_returns > 0).mean()

        return PortfolioBacktestResult(
            equity_curve=equity_curve,
            drawdown_curve=drawdowns,
            returns_series=net_returns,
            total_return=float(total_return),
            sharpe_ratio=float(sharpe),
            max_drawdown=float(max_dd),
            win_rate=float(win_rate)
        )
