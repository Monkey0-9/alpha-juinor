import logging
import numpy as np
import pandas as pd
from typing import Callable, Dict, Optional

logger = logging.getLogger(__name__)

class BacktestResult:
    def __init__(self, equity_curve: pd.Series, trades: pd.DataFrame, metrics: Dict[str, float]):
        self.equity_curve = equity_curve
        self.trades = trades
        self.metrics = metrics

class BacktestEngine:
    """Simple strategy backtest engine for historical research."""

    def __init__(self, initial_capital: float = 1_000_000.0, commission: float = 0.0004):
        self.initial_capital = initial_capital
        self.commission = commission

    def run_portfolio(
        self, 
        universe_prices: Dict[str, pd.Series], 
        universe_signals: Dict[str, pd.Series], 
        allocation_pct: float = 0.05
    ) -> BacktestResult:
        """Run a portfolio-level backtest across multiple assets."""
        if not universe_prices or not universe_signals:
            raise ValueError("Universe data cannot be empty.")

        # Align all price series to a common timeline
        prices_df = pd.DataFrame(universe_prices).fillna(method="ffill").dropna()
        signals_df = pd.DataFrame(universe_signals).reindex(prices_df.index).fillna(0.0)

        equity = self.initial_capital
        holdings: Dict[str, float] = {s: 0.0 for s in prices_df.columns}
        trades = []
        equity_curve = []

        for timestamp, row in prices_df.iterrows():
            current_signals = signals_df.loc[timestamp]
            
            # Calculate total portfolio value at start of step
            market_value = sum(holdings[s] * row[s] for s in row.index)
            current_total_equity = equity + market_value
            
            # Rebalance logic: simplicity — buy signals > 0.5, sell < -0.5
            for symbol in row.index:
                price = row[symbol]
                signal = current_signals.get(symbol, 0.0)
                
                # Close if signal reverses
                if holdings[symbol] > 0 and signal < -0.1:
                    qty = holdings[symbol]
                    proceeds = qty * price * (1 - self.commission)
                    equity += proceeds
                    trades.append({"time": timestamp, "symbol": symbol, "side": "sell", "qty": qty, "price": price})
                    holdings[symbol] = 0.0
                
                # Open if signal triggers and we have room
                elif holdings[symbol] == 0 and signal > 0.5:
                    target_value = current_total_equity * allocation_pct
                    qty = int(target_value / price)
                    cost = qty * price * (1 + self.commission)
                    if cost <= equity:
                        equity -= cost
                        holdings[symbol] = qty
                        trades.append({"time": timestamp, "symbol": symbol, "side": "buy", "qty": qty, "price": price})

            # Re-calculate market value after trades
            market_value = sum(holdings[s] * row[s] for s in row.index)
            equity_curve.append(equity + market_value)

        equity_series = pd.Series(equity_curve, index=prices_df.index)
        metrics = {
            "return": float(equity_series.iloc[-1] / self.initial_capital - 1),
            "sharpe": float(self._sharpe_ratio(equity_series.pct_change().dropna())),
            "max_drawdown": float(self._max_drawdown(equity_series)),
        }
        return BacktestResult(equity_series, pd.DataFrame(trades), metrics)

    def run(self, prices: pd.Series, signals: pd.Series, entry_size: float = 0.1) -> BacktestResult:
        """Legacy single-symbol run method wrapper."""
        return self.run_portfolio({"BACKTEST": prices}, {"BACKTEST": signals}, allocation_pct=entry_size)

    def _sharpe_ratio(self, returns: pd.Series) -> float:
        if returns.empty:
            return 0.0
        return float(returns.mean() / (returns.std() + 1e-9) * np.sqrt(252))

    def _max_drawdown(self, equity: pd.Series) -> float:
        if equity.empty:
            return 0.0
        rolling_max = equity.cummax()
        drawdowns = (equity - rolling_max) / rolling_max
        return float(drawdowns.min())
