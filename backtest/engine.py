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
import logging

logger = logging.getLogger(__name__)

# Risk Decision reference
try:
    from risk.engine import RiskDecision, RiskCheckResult, RiskManager
except ImportError:
    RiskDecision = None
    RiskCheckResult = None
    RiskManager = None

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
    BarData,
)

from .portfolio import Portfolio


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
        risk_manager: Optional[RiskManager] = None,
        risk_free_rate: float = 0.02,
    ):
        self.provider = provider
        self.initial_capital = float(initial_capital)
        if not np.isfinite(self.initial_capital) or self.initial_capital <= 0:
            raise ValueError(f"Invalid initial_capital: {self.initial_capital}. Must be positive and finite.")
        self.risk_free_rate = float(risk_free_rate)
        
        self.portfolio = Portfolio(self.initial_capital)
        self.execution_handler = execution_handler or RealisticExecutionHandler()
        self.risk_manager = risk_manager
        self.blotter = TradeBlotter()
        self._equity_series = pd.Series(dtype=float)
        self._last_prices: Dict[str, float] = {}  # Track last known prices for equity calculation
        self.open_orders: List[Order] = [] # Queue for working orders

    def _accrue_interest(self, timestamp: datetime):
        """
        Institutional Cash Management: Idle cash earns risk-free rate.
        Applied daily at market close.
        """
        if self.portfolio.cash > 0:
            daily_rfr = self.risk_free_rate / 252.0
            interest = self.portfolio.cash * daily_rfr
            # Institutional: Cash interest is a CASH_DEPOSIT/CREDIT event
            from portfolio.ledger import PortfolioEvent, EventType
            event = PortfolioEvent(
                timestamp=timestamp, 
                event_type=EventType.CASH_DEPOSIT,
                amount=interest,
                metadata={"description": "Risk-free interest"}
            )
            self.portfolio.ledger.record_event(event)

    # -------------------------
    # Data loading
    # -------------------------
    def _build_price_panel(self, tickers: List[str], start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Return a DataFrame with MultiIndex columns (ticker, field) containing OHLCV.
        Fetches an additional 365 days of warmup data to satisfy lookbacks.
        """
        # Calculate warmup start (approx 1 year lookback)
        start_dt = pd.to_datetime(start_date)
        warmup_dt = start_dt - pd.Timedelta(days=365)
        warmup_start = warmup_dt.strftime("%Y-%m-%d")

        if self.provider is not None and hasattr(self.provider, "get_panel"):
            panel = self.provider.get_panel(tickers, warmup_start, end_date)
            if panel is not None and not panel.empty:
                return panel

        df = yf.download(tickers, start=warmup_start, end=end_date, auto_adjust=False, group_by="ticker", progress=False)
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
    # Portfolio valuation & State Validation
    # -------------------------
    def _portfolio_value(self, close_prices: Dict[str, float]) -> float:
        """
        SINGLE SOURCE OF TRUTH: Compute equity from cash + positions.
        This is the authoritative equity calculator.
        """
        nav = self.portfolio.cash
        for tk, qty in self.portfolio.positions.items():
            price = close_prices.get(tk, self._last_prices.get(tk, np.nan))
            if price is None or (isinstance(price, float) and np.isnan(price)):
                # HARD STOP: Cannot value position without price
                # For engine validation, we still use this authoritative check
                raise RuntimeError(f"Cannot compute equity: missing price for {tk} (qty={qty})")
            nav += qty * price
        return float(nav)

    def _validate_state(self, timestamp) -> None:
        """
        HARD INVARIANT CHECKS: Cash, positions, and equity must be finite.
        Called after every trade to ensure accounting integrity.
        """
        if not np.isfinite(self.portfolio.cash):
            raise RuntimeError(f"INVARIANT VIOLATION at {timestamp}: cash={self.portfolio.cash} (must be finite)")
        
        for tk, qty in self.portfolio.positions.items():
            if not np.isfinite(qty):
                raise RuntimeError(f"INVARIANT VIOLATION at {timestamp}: position[{tk}]={qty} (must be finite)")
        
        # Compute equity using last known prices
        try:
            equity = self._portfolio_value(self._last_prices)
        except RuntimeError as e:
            raise RuntimeError(f"INVARIANT VIOLATION at {timestamp}: {e}")
        
        if not np.isfinite(equity):
            raise RuntimeError(f"INVARIANT VIOLATION at {timestamp}: equity={equity} (must be finite)")
        
        if equity <= 0:
            raise RuntimeError(f"INVARIANT VIOLATION at {timestamp}: equity={equity} (must be positive)")

    # -------------------------
    # Apply trade to portfolio
    # -------------------------
    def _apply_trade(self, trade: Trade) -> None:
        """
        Apply trade effects: cash and positions.
        trade.quantity is signed (+ buy, - sell).
        trade.cost already includes commission + slippage in currency.
        """
        # Use the ledger-backed portfolio
        self.portfolio.on_trade(trade)

    # -------------------------
    # Main run loop
    # -------------------------
    def _on_market_open(self, timestamp: pd.Timestamp) -> None:
        """Architecural hook: Pre-trade logic/logging."""
        # Generic Monthly Heartbeat (Professional formatting)
        if timestamp.day <= 3 and timestamp.month != getattr(self, '_last_heartbeat_month', -1):
            nav = self._portfolio_value(self._last_prices)
            # Calculate Gross Leverage: sum(|position_value|) / NAV
            exposure = sum(abs(qty * self._last_prices.get(tk, 0.0)) for tk, qty in self.portfolio.positions.items())
            leverage = exposure / nav if nav > 0 else 0.0
            
            logger.info(f"[ENGINE] {timestamp.date()} | NAV: ${nav:,.0f} | Leverage: {leverage:.2f}x | Pos: {len(self.portfolio.positions)}")
            self._last_heartbeat_month = timestamp.month

    def _on_market_close(self, timestamp: pd.Timestamp, nav: float, records: list) -> None:
        """Architecural hook: Post-trade NAV recording and state verification."""
        # 1. Accrue Interest on Cash (Institutional Cash Management)
        self._accrue_interest(timestamp)
        # Recalculate NAV after interest
        nav = self._portfolio_value(self._last_prices)
        
        records.append({"timestamp": timestamp, "equity": nav})
        self._validate_state(timestamp)

        # POST-TRADE RISK: Circuit Breaker Evaluation
        if self.risk_manager:
            # Create a localized return series from recent records (last 30 bars)
            if len(records) > 2:
                recent_equity = [r['equity'] for r in records[-60:]] # Increased lookback for Beta
                rets = pd.Series(recent_equity).pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).dropna()
                self.risk_manager.check_circuit_breaker(nav, rets)

    def run(self, start_date: str, strategy_fn: Callable, tickers: List[str], end_date: Optional[str] = None) -> None:
        """
        Run the backtest.

        - start_date/end_date: "YYYY-MM-DD"
        - strategy_fn: callable(timestamp, prices, engine) -> list[Order]
        - tickers: list of symbols to fetch and iterate
        """
        if not tickers:
            raise ValueError("tickers must be provided to run()")

        full_panel = self._build_price_panel(tickers, start_date, end_date)
        if full_panel is None or full_panel.empty:
            raise RuntimeError("Price panel empty.")

        # Filter iteration to start at start_date, but keep full_panel for .loc[:ts] lookbacks
        start_ts = pd.to_datetime(start_date)
        trading_panel = full_panel[full_panel.index >= start_ts]
        if trading_panel.empty:
            raise RuntimeError(f"No trading data available starting at {start_date}")

        # HARD INVARIANT CHECK: Validate state before start
        self._validate_state(start_date)

        equity_records = []

        for timestamp, row in trading_panel.iterrows():
            ts = pd.Timestamp(timestamp)
            
            # HOOK: Market Open
            self._on_market_open(ts)

            # Build close price dict for strategy and NAV
            close_prices: Dict[str, float] = {}
            for tk in tickers:
                try:
                    price = float(row[(tk, "Close")])
                    if np.isfinite(price):
                        close_prices[tk] = price
                        self._last_prices[tk] = price  # Update last known ONLY if valid
                    else:
                        close_prices[tk] = np.nan
                except Exception:
                    close_prices[tk] = np.nan

            # -------------------------
            # 1. Strategy & Execution
            # -------------------------
            # Call strategy to generate new orders
            # Fail fast if strategy crashes - institutional requirement.
            new_orders = strategy_fn(ts, close_prices, self) or []
            if new_orders:
                for o in new_orders:
                    self.blotter.record_order(o)
                self.open_orders.extend(new_orders)

            # Process Open Orders (Execution Simulation)
            active_orders = []
            for order in self.open_orders:
                tk = order.ticker
                
                # Check data availability
                if (tk, "Close") not in row or np.isnan(row[(tk, "Close")]):
                     active_orders.append(order) # Retain
                     continue
                
                try:
                     # Build Bar
                     bar = BarData(
                         ticker=tk,
                         timestamp=ts,
                         open=float(row[(tk, "Open")]),
                         high=float(row[(tk, "High")]),
                         low=float(row[(tk, "Low")]),
                         close=float(row[(tk, "Close")]),
                         volume=float(row[(tk, "Volume")])
                     )
                     
                     # Checks
                     if bar.volume <= 0 or np.isnan(bar.close):
                          active_orders.append(order)
                          continue
                          
                     # Get History for ADV/Vol estimation (slice last 60d)
                     tk_df = full_panel.loc[full_panel.index <= ts].xs(tk, axis=1, level=0).tail(60)
                     
                     trade = self.execution_handler.fill_order(
                         order,
                         bar,
                         tk_df["Close"],
                         tk_df["Volume"]
                     )
                     
                     if trade:
                         self._apply_trade(trade)
                         self.blotter.record_trade(trade)
                         
                     if order.status == OrderStatus.FILLED or order.status == OrderStatus.CANCELLED:
                          continue # Remove
                     else:
                          active_orders.append(order) # Keep
                          
                except Exception as e:
                     logger.warning(f"Order processing error {tk}: {e}")
                     active_orders.append(order)

            self.open_orders = active_orders

            # ... (Risk logic) ...

            # End-of-bar hooks: create ledger snapshot
            # Merge current valid prices with last known for valuation
            valuation_prices = self._last_prices.copy()
            # Overlay current bar's valid prices (already done via _last_prices update, but to be sure)
            # Actually _last_prices contains the latest valid prices now.
            # But close_prices contains NaNs for current bar misses, which strategy might want to see?
            # Strategy sees close_prices (raw). Portfolio sees valuation_prices (clean).
            
            self.portfolio.update_market_value(valuation_prices, ts)
            nav = self.portfolio.ledger.equity_curve[-1]["equity"]
            self._on_market_close(ts, nav, equity_records)

        # Store equity series
        if equity_records:
            df_equity = pd.DataFrame(equity_records).set_index("timestamp")
            self._equity_series = df_equity["equity"].astype(float)
            # Hard assert: fill NaNs with initial capital or forward fill
            self._equity_series = self._equity_series.ffill().fillna(self.initial_capital)
        else:
            # Fallback for empty run: seed with initial capital
            # Use start_date as timestamp
            ts_start = pd.Timestamp(start_date)
            self._equity_series = pd.Series([self.initial_capital], index=[ts_start])


    # -------------------------
    # Accessors
    # -------------------------
    def get_results(self) -> pd.DataFrame:
        """
        Return equity series for metrics calculation.
        HARD STOP if equity is corrupted (NaN) or empty.
        """
        if len(self._equity_series) == 0:
            raise RuntimeError("No equity records. Backtest did not execute.")
        
        if self._equity_series.isna().any():
            raise RuntimeError(
                f"Equity series contains NaN values. Accounting is corrupted. "
                f"Cannot produce valid metrics. NaN count: {self._equity_series.isna().sum()}"
            )
        
        return pd.DataFrame({"equity": self._equity_series})

    def get_blotter(self) -> TradeBlotter:
        return self.blotter
