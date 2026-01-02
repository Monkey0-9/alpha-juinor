# backtest/engine.py
"""
Backtest engine compatible with institutional execution handler.

Key compatibility fixes:
- Calls execution_handler.fill_order(...) with signature:
    (order, bar, bar_timestamp, price_history, volume_history)
- Passes bar_timestamp (trade timestamp attribution)
- Passes volume_history (ADV & liquidity estimates)
- Preserves Order.remaining_qty and Order.status mutated in-place by handler
- Applies capital check before applying trade effects
"""

from datetime import datetime
from typing import Callable, Dict, List, Optional

import pandas as pd
import numpy as np
import yfinance as yf

# Optional provider import
try:
    from data.provider import YahooDataProvider  # type: ignore
except Exception:
    YahooDataProvider = None

from backtest.execution import (
    Order,
    Trade,
    TradeBlotter,
    RealisticExecutionHandler,
    OrderStatus,
)


class BacktestEngine:
    """
    Simple event-driven backtest engine.

    strategy_fn signature:
        strategy_fn(timestamp: pd.Timestamp, prices: Dict[str, float], engine: BacktestEngine) -> List[Order]
    """

    def __init__(
        self,
        provider: Optional[object] = None,
        initial_capital: float = 1_000_000.0,
        execution_handler: Optional[RealisticExecutionHandler] = None,
    ):
        self.provider = provider
        self.initial_capital = float(initial_capital)
        self.cash: float = float(initial_capital)
        self.positions: Dict[str, float] = {}
        self.execution_handler = execution_handler or RealisticExecutionHandler()
        self.blotter = TradeBlotter()
        self._equity_series = pd.Series(dtype=float)

    # -------------------------
    # Data loading
    # -------------------------
    def _build_price_panel(self, tickers: List[str], start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Return a DataFrame with MultiIndex columns (ticker, field) containing OHLCV.
        Tries provider, falls back to yfinance.
        """
        if self.provider is not None and hasattr(self.provider, "get_panel"):
            panel = self.provider.get_panel(tickers, start_date, end_date)
            if panel is not None and not panel.empty:
                return panel

        df = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False, group_by="ticker", progress=False)
        if df.empty:
            raise RuntimeError("No market data returned for requested period/tickers.")

        if not isinstance(df.columns, pd.MultiIndex):
            if len(tickers) == 1:
                tk = tickers[0]
                df.columns = pd.MultiIndex.from_product([[tk], df.columns])
            else:
                raise RuntimeError("Ambiguous price panel format: expected MultiIndex columns for multiple tickers.")

        df.index = pd.to_datetime(df.index)
        try:
            df = df.sort_index(axis=1)
        except Exception:
            pass
        return df

    # -------------------------
    # Portfolio valuation
    # -------------------------
    def _portfolio_value(self, close_prices: Dict[str, float]) -> float:
        nav = self.cash
        for tk, qty in self.positions.items():
            price = close_prices.get(tk, np.nan)
            if price is None or (isinstance(price, float) and np.isnan(price)):
                continue
            nav += qty * price
        return float(nav)

    # -------------------------
    # Apply trade to portfolio
    # -------------------------
    def _apply_trade(self, trade: Trade) -> None:
        """
        Apply trade effects: cash and positions.
        trade.quantity is signed (+ buy, - sell).
        trade.cost already includes commission + slippage in currency.
        """
        self.cash -= trade.quantity * trade.fill_price
        self.cash -= trade.cost
        self.positions[trade.ticker] = self.positions.get(trade.ticker, 0.0) + trade.quantity

    # -------------------------
    # Main run loop
    # -------------------------
    def run(self, start_date: str, strategy_fn: Callable, tickers: List[str], end_date: Optional[str] = None) -> None:
        """
        Run the backtest.

        - start_date/end_date: "YYYY-MM-DD"
        - strategy_fn: callable(timestamp, prices, engine) -> list[Order]
        - tickers: list of symbols to fetch and iterate
        """
        if not tickers:
            raise ValueError("tickers must be provided to run()")

        panel = self._build_price_panel(tickers, start_date, end_date)
        if panel is None or panel.empty:
            raise RuntimeError("Price panel empty.")

        equity_records = []

        for timestamp, row in panel.iterrows():
            # Build close price dict for strategy and NAV
            close_prices: Dict[str, float] = {}
            for tk in tickers:
                try:
                    close_prices[tk] = float(row[(tk, "Close")])
                except Exception:
                    close_prices[tk] = np.nan

            # Call strategy
            try:
                orders = strategy_fn(timestamp, close_prices, self) or []
            except Exception as e:
                # Strategy errors do not crash engine; they skip this bar.
                print(f"[ENGINE] Strategy error at {timestamp}: {e}")
                orders = []

            # Process orders returned by strategy
            for order in orders:
                # Record order immediately for audit
                try:
                    self.blotter.record_order(order)
                except Exception:
                    pass

                # Skip if order is already terminal
                if order.status in (OrderStatus.FILLED, OrderStatus.CANCELLED):
                    continue

                # Build bar for the specific ticker (OHLCV)
                try:
                    bar = {
                        "Open": float(row[(order.ticker, "Open")]),
                        "High": float(row[(order.ticker, "High")]),
                        "Low": float(row[(order.ticker, "Low")]),
                        "Close": float(row[(order.ticker, "Close")]),
                        "Volume": float(row[(order.ticker, "Volume")]),
                    }
                except Exception:
                    # Missing OHLCV — cannot execute this order on this bar
                    continue

                # Prepare historical series up to and including this timestamp
                try:
                    price_history = panel.loc[:timestamp, (order.ticker, "Close")].astype(float)
                except Exception:
                    price_history = pd.Series(dtype=float)

                try:
                    volume_history = panel.loc[:timestamp, (order.ticker, "Volume")].astype(float)
                except Exception:
                    volume_history = pd.Series(dtype=float)

                # IMPORTANT: call execution handler with full signature expected by institutional handler
                trade = self.execution_handler.fill_order(
                    order=order,
                    bar=bar,
                    bar_timestamp=timestamp,
                    price_history=price_history,
                    volume_history=volume_history,
                )

                # No fill this bar
                if trade is None:
                    continue

                # Capital constraint: prevent negative cash unless allowed by strategy (here we reject)
                projected_cash = self.cash - (trade.quantity * trade.fill_price + trade.cost)
                if projected_cash < 0:
                    # Cancel order to avoid negative cash; order mutated by handler may be partially filled; we treat this conservatively.
                    order.status = OrderStatus.CANCELLED
                    # Do not apply or record this trade (it was unaffordable)
                    continue

                # Apply trade effects and record it
                self._apply_trade(trade)
                self.blotter.record_trade(trade)

            # End-of-bar: compute NAV using close prices
            nav = self._portfolio_value(close_prices)
            equity_records.append({"timestamp": timestamp, "equity": nav})

        # Store equity series
        if equity_records:
            df_equity = pd.DataFrame(equity_records).set_index("timestamp")
            self._equity_series = df_equity["equity"].astype(float)
        else:
            self._equity_series = pd.Series(dtype=float)

    # -------------------------
    # Accessors
    # -------------------------
    def get_results(self) -> pd.DataFrame:
        return pd.DataFrame({"equity": self._equity_series})

    def get_blotter(self) -> TradeBlotter:
        return self.blotter
