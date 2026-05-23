import logging
import numpy as np
import pandas as pd
from typing import Any, Dict
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

    def run_walkforward(
        self,
        price_data: Dict[str, pd.Series],
        signal_data: Dict[str, pd.Series],
        train_window: int = 63,
        test_window: int = 21,
        step: int = 21,
        max_positions: int = 12,
    ) -> Dict[str, Any]:
        """Execute walk-forward backtesting using a rolling train/test split."""
        prices = pd.DataFrame(price_data).dropna()
        signals = pd.DataFrame(signal_data).reindex(prices.index).fillna(0)

        if prices.empty or signals.empty:
            raise ValueError("Price and signal data must not be empty for walk-forward validation.")

        fold_results = []
        start = 0
        while start + train_window + test_window <= len(prices):
            test_index = prices.index[start + train_window : start + train_window + test_window]

            test_prices = prices.loc[test_index]
            test_signals = signals.loc[test_index]

            # Use the same ordering logic on the test segment.
            result = self.run(
                {c: test_prices[c] for c in test_prices.columns},
                {c: test_signals[c] for c in test_signals.columns},
                max_positions=max_positions,
            )

            fold_results.append(result)
            start += step

        if not fold_results:
            raise ValueError("Not enough data for walk-forward validation.")

        all_equity = pd.concat([fold.equity_curve for fold in fold_results])
        total_return = (all_equity.iloc[-1] / self.initial_capital) - 1
        avg_sharpe = float(np.mean([fold.sharpe_ratio for fold in fold_results]))
        avg_max_dd = float(np.mean([fold.max_drawdown for fold in fold_results]))

        return {
            "num_folds": len(fold_results),
            "total_return": total_return,
            "average_sharpe": avg_sharpe,
            "average_max_drawdown": avg_max_dd,
            "fold_results": fold_results,
        }
