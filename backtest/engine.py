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
    from risk.engine import RiskDecision
except ImportError:
    RiskDecision = None

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
        risk_manager: Optional[object] = None,
        risk_free_rate: float = 0.02,
    ):
        self.provider = provider
        self.initial_capital = float(initial_capital)
        if not np.isfinite(self.initial_capital) or self.initial_capital <= 0:
            raise ValueError(f"Invalid initial_capital: {self.initial_capital}. Must be positive and finite.")
        self.cash: float = self.initial_capital
        self.risk_free_rate = float(risk_free_rate)
        
        self.positions: Dict[str, float] = {}
        self.execution_handler = execution_handler or RealisticExecutionHandler()
        self.risk_manager = risk_manager
        self.blotter = TradeBlotter()
        self._equity_series = pd.Series(dtype=float)
        self._last_prices: Dict[str, float] = {}  # Track last known prices for equity calculation

    def _accrue_interest(self):
        """
        Institutional Cash Management: Idle cash earns risk-free rate.
        Applied daily at market close.
        """
        if self.cash > 0:
            daily_rfr = self.risk_free_rate / 252.0
            interest = self.cash * daily_rfr
            self.cash += interest

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
    # Portfolio valuation & State Validation
    # -------------------------
    def _portfolio_value(self, close_prices: Dict[str, float]) -> float:
        """
        SINGLE SOURCE OF TRUTH: Compute equity from cash + positions.
        This is the authoritative equity calculator.
        """
        nav = self.cash
        for tk, qty in self.positions.items():
            price = close_prices.get(tk, self._last_prices.get(tk, np.nan))
            if price is None or (isinstance(price, float) and np.isnan(price)):
                # HARD STOP: Cannot value position without price
                raise RuntimeError(f"Cannot compute equity: missing price for {tk} (qty={qty})")
            nav += qty * price
        return float(nav)

    def _validate_state(self, timestamp) -> None:
        """
        HARD INVARIANT CHECKS: Cash, positions, and equity must be finite.
        Called after every trade to ensure accounting integrity.
        """
        if not np.isfinite(self.cash):
            raise RuntimeError(f"INVARIANT VIOLATION at {timestamp}: cash={self.cash} (must be finite)")
        
        for tk, qty in self.positions.items():
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
        self.cash -= trade.quantity * trade.fill_price
        self.cash -= trade.cost
        self.positions[trade.ticker] = self.positions.get(trade.ticker, 0.0) + trade.quantity

    # -------------------------
    # Main run loop
    # -------------------------
    def _on_market_open(self, timestamp: pd.Timestamp) -> None:
        """Architecural hook: Pre-trade logic/logging."""
        # Generic Monthly Heartbeat (Professional formatting)
        if timestamp.day <= 3 and timestamp.month != getattr(self, '_last_heartbeat_month', -1):
            nav = self._portfolio_value(self._last_prices)
            # Calculate Gross Leverage: sum(|position_value|) / NAV
            exposure = sum(abs(qty * self._last_prices.get(tk, 0.0)) for tk, qty in self.positions.items())
            leverage = exposure / nav if nav > 0 else 0.0
            
            logger.info(f"[ENGINE] {timestamp.date()} | NAV: ${nav:,.0f} | Leverage: {leverage:.2f}x | Pos: {len(self.positions)}")
            self._last_heartbeat_month = timestamp.month

    def _on_market_close(self, timestamp: pd.Timestamp, nav: float, records: list) -> None:
        """Architecural hook: Post-trade NAV recording and state verification."""
        # 1. Accrue Interest on Cash (Institutional Cash Management)
        self._accrue_interest()
        # Recalculate NAV after interest
        nav = self._portfolio_value(self._last_prices)
        
        records.append({"timestamp": timestamp, "equity": nav})
        self._validate_state(timestamp)

        # POST-TRADE RISK: Circuit Breaker Evaluation
        if self.risk_manager:
            # Create a localized return series from recent records (last 30 bars)
            if len(records) > 2:
                recent_equity = [r['equity'] for r in records[-60:]] # Increased lookback for Beta
                rets = pd.Series(recent_equity).pct_change().dropna()
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

        panel = self._build_price_panel(tickers, start_date, end_date)
        if panel is None or panel.empty:
            raise RuntimeError("Price panel empty.")

        # HARD INVARIANT CHECK: Validate state before start
        self._validate_state(start_date)

        equity_records = []

        for timestamp, row in panel.iterrows():
            ts = pd.Timestamp(timestamp)
            
            # HOOK: Market Open
            self._on_market_open(ts)

            # Build close price dict for strategy and NAV
            close_prices: Dict[str, float] = {}
            for tk in tickers:
                try:
                    price = float(row[(tk, "Close")])
                    close_prices[tk] = price
                    self._last_prices[tk] = price  # Track for state validation
                except Exception:
                    close_prices[tk] = np.nan

            # 1. Market Regime & Risk State (Daily Processing)
            is_risk_on = True
            try:
                if "SPY" in panel.columns.get_level_values(0):
                    spy_hist = panel.loc[:ts, ("SPY", "Close")].tail(201)
                    if len(spy_hist) > 200:
                        ma200 = spy_hist.mean()
                        current_spy = spy_hist.iloc[-1]
                        is_risk_on = current_spy > ma200
            except Exception: pass

            # RISK STATE MANAGEMENT
            if self.risk_manager:
                # Progress timers/phases
                self.risk_manager.process_state_daily(is_risk_on)

                if self.risk_manager.state == RiskDecision.FREEZE:
                    # ELITE ACTION: If frozen, we MUST liquidate to stop the bleed.
                    orders = []
                    from backtest.execution import OrderType
                    for tk, qty in self.positions.items():
                         if abs(qty) > 0.001:
                              orders.append(Order(
                                  ticker=tk, 
                                  quantity=-float(qty), 
                                  order_type=OrderType.MARKET, 
                                  timestamp=ts
                              ))
                else:
                    # ALLOW or RECOVERY (Gradual re-entry)
                    orders = strategy_fn(ts, close_prices, self) or []
            else:
                orders = strategy_fn(ts, close_prices, self) or []

            # RISK OVERLAY ENFORCEMENT
            if self.risk_manager and orders and self.risk_manager.state != RiskDecision.FREEZE:
                # Convert current intent to weights for vetting
                nav = self._portfolio_value(close_prices)
                target_notionals = {tk: qty * close_prices.get(tk, 0.0) for tk, qty in self.positions.items()}
                for o in orders:
                    target_notionals[o.ticker] = target_notionals.get(o.ticker, 0.0) + (o.quantity * close_prices.get(o.ticker, 0.0))
                
                target_weights = {tk: val / nav if nav > 0 else 0.0 for tk, val in target_notionals.items()}
                
                # Fetch returns for VaR/CVaR
                lookback_end = ts
                lookback_start = ts - pd.Timedelta(days=252)
                sub_returns = panel.loc[lookback_start:lookback_end, (slice(None), "Close")].pct_change().dropna()
                sub_returns.columns = sub_returns.columns.get_level_values(0)
                
                risk_res = self.risk_manager.check_pre_trade(
                    target_weights=target_weights, 
                    baskets_returns=sub_returns, 
                    timestamp=ts,
                    current_equity=nav,
                    is_risk_on=is_risk_on
                )
                
                if not risk_res["ok"]:
                    decision = risk_res.get("decision")
                    decision_val = decision.value if hasattr(decision, "value") else str(decision)
                    
                    if decision_val == "SCALE":
                        scale_factor = risk_res.get("scale_factor", 1.0)
                        logger.warning(f"[RISK SCALE] {ts.date()} | Intent scaled by {scale_factor:.2%} | Violations: {risk_res['violations']}")
                        for o in orders:
                            o.quantity *= scale_factor

            # Process orders returned by strategy
            for order in orders:
                # Record order immediately for audit
                try:
                    self.blotter.record_order(order)
                except Exception:
                    pass

                # Skip if order is already terminal or cancelled by risk
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
                    # Validate bar data is finite
                    for k, v in bar.items():
                        if not np.isfinite(v):
                            raise ValueError(f"NaN/Inf in {k} for {order.ticker}")
                except Exception as e:
                    # Missing or invalid OHLCV — cannot execute this order on this bar
                    # In institutional systems, we don't guess prices.
                    logger.warning(f"Skipping trade execution for {order.ticker} due to invalid data: {e}")
                    continue

                # Prepare historical series up to and including this timestamp
                try:
                    price_history = panel.loc[:ts, (order.ticker, "Close")].astype(float)
                except Exception:
                    price_history = pd.Series(dtype=float)

                try:
                    volume_history = panel.loc[:ts, (order.ticker, "Volume")].astype(float)
                except Exception:
                    volume_history = pd.Series(dtype=float)

                # IMPORTANT: call execution handler with full signature expected by institutional handler
                trade = self.execution_handler.fill_order(
                    order=order,
                    bar=bar,
                    bar_timestamp=ts,
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
                
                # HARD INVARIANT CHECK: Validate state after each trade
                self._validate_state(ts)

            # End-of-bar hooks
            nav = self._portfolio_value(close_prices)
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
