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

    def run(self, prices: pd.DataFrame, signals: pd.Series, entry_size: float = 0.1) -> BacktestResult:
        if prices.empty or signals.empty:
            raise ValueError("Price and signal series cannot be empty.")

        equity = self.initial_capital
        position = 0.0
        trades = []
        equity_curve = []

        for timestamp, price in prices.items():
            raw_signal = signals.get(timestamp, 0.0)
            signal = float(raw_signal) if raw_signal is not None else 0.0
            target_qty = int((equity * entry_size) / max(price, 1))
            if signal > 0.1 and position <= 0:
                qty = target_qty
                cost = qty * price * (1 + self.commission)
                if cost <= equity:
                    position = qty
                    equity -= cost
                    trades.append({"time": timestamp, "symbol": "BACKTEST", "side": "buy", "qty": qty, "price": price})
            elif signal < -0.1 and position >= 0:
                qty = abs(position) if position < 0 else target_qty
                proceeds = qty * price * (1 - self.commission)
                equity += proceeds
                trades.append({"time": timestamp, "symbol": "BACKTEST", "side": "sell", "qty": qty, "price": price})
                position = -qty

            market_value = position * price
            equity_curve.append(equity + market_value)

        equity_series = pd.Series(equity_curve, index=prices.index)
        metrics = {
            "return": float(equity_series.iloc[-1] / self.initial_capital - 1),
            "sharpe": float(self._sharpe_ratio(equity_series.pct_change().dropna())),
            "max_drawdown": float(self._max_drawdown(equity_series)),
        }
        return BacktestResult(equity_series, pd.DataFrame(trades), metrics)

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
