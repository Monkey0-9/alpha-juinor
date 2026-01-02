# ============================================================
# MINI QUANT FUND — INSTITUTIONAL DRIVER (v2 — production-ready)
# ============================================================
"""
Main driver for institutional backtests.

Features:
- Uses a Data Provider (e.g., YahooDataProvider) to fetch history.
- Uses an Execution Handler and BacktestEngine (user-provided).
- Uses Alpha pipeline (CompositeAlpha / TrendAlpha / MeanReversionAlpha).
- RiskManager enforces volatility / leverage caps BEFORE orders are sent.
- Monthly rebalance (1st trading day approx) to limit turnover.
- Target allocations are proportional to adjusted conviction (market-driven).
- Defensive fallbacks for provider APIs (tries provider.get_history, provider.get_bars, else yf.download).
"""

import sys
import time
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd

# ---------------------------
# Infrastructure / project imports
# ---------------------------
# These imports assume you have the corresponding modules defined.
# If names differ in your repo, adjust them accordingly.
try:
    from data.provider import YahooDataProvider
    from data.storage import DataStore
    from backtest.engine import BacktestEngine
    from backtest.execution import RealisticExecutionHandler, Order, OrderType
    from backtest.registry import BacktestRegistry
except Exception:
    # If your module layout differs, import errors will be raised here.
    # Keep this so the file still opens and you can adapt names.
    print("Warning: Could not import project modules directly. Ensure module paths are correct.")
    # Re-raise only if you actually intend to run now; otherwise keep going for static inspection.
    # raise

# Analytics & reporting
from engine.analytics import (
    annualized_return,
    annualized_volatility,
    sharpe_ratio,
    max_drawdown,
)
try:
    from reports.reporting import plot_equity_curve, plot_drawdown, allocation_table
except Exception:
    # If plotting/reporting modules missing, we'll provide simple fallbacks later.
    plot_equity_curve = None
    plot_drawdown = None
    allocation_table = None

# Alpha & Risk (if present in your repo)
try:
    from strategies.alpha import TrendAlpha, MeanReversionAlpha, CompositeAlpha
except Exception:
    # If you haven't added strategies/alpha.py yet, we'll fallback to simple placeholders.
    TrendAlpha = None
    MeanReversionAlpha = None
    CompositeAlpha = None

try:
    from risk.engine import RiskManager
except Exception:
    RiskManager = None

# Fallback provider: yfinance (only used if provider doesn't support history fetch)
import yfinance as yf


# ---------------------------
# Configuration
# ---------------------------
TICKERS = ["SPY", "QQQ", "TLT", "GLD"]
START_DATE = "2018-01-01"
INITIAL_CAPITAL = 1_000_000

# Execution / backtest config
COMMISSION_PCT = 0.001  # 0.1%
BASE_SLIPPAGE = 0.0005

# Risk manager defaults (can be made configurable)
DEFAULT_MAX_LEVERAGE = 1.0
DEFAULT_TARGET_VOL = 0.12

# Minimum trade threshold (avoid tiny rebalances)
MIN_TRADE_PCT = 0.02  # only trade if >2% of total equity difference

# ---------------------------
# Helper functions (defensive)
# ---------------------------


def get_price_history(provider, ticker: str, end_dt: datetime, lookback: int = 500) -> pd.Series:
    """
    Fetch price history for 'ticker' up to end_dt.
    Tries common provider methods: get_bars, get_history, history. Falls back to yfinance.
    Returns a pandas Series of adjusted close prices indexed by timestamp.
    """
    # try common adapter methods
    try:
        if hasattr(provider, "get_bars"):
            df = provider.get_bars(ticker, end=end_dt, lookback=lookback)
            # expect df with 'Close' or 'close'
            if "Close" in df.columns:
                return df["Close"].squeeze()
            if "close" in df.columns:
                return df["close"].squeeze()
            # if it's a Series already
            if isinstance(df, pd.Series):
                return df.squeeze()
        if hasattr(provider, "get_history"):
            df = provider.get_history(ticker, end=end_dt, lookback=lookback)
            if isinstance(df, pd.Series):
                return df.squeeze()
            if "Close" in df.columns:
                return df["Close"].squeeze()
    except Exception:
        # provider misbehaved for this ticker/time; fall through to yfinance
        pass

    # Fallback: yfinance fetch for lookback days ending at end_dt
    try:
        # yfinance's period string: f"{lookback}d"
        end_str = end_dt.strftime("%Y-%m-%d")
        df = yf.download(ticker, end=end_str, period=f"{lookback}d", auto_adjust=True, progress=False)
        if df is None or df.empty:
            return pd.Series(dtype=float)
        return df["Close"].squeeze()
    except Exception:
        return pd.Series(dtype=float)


def total_portfolio_value(portfolio) -> float:
    """
    Try to calculate current portfolio equity:
    Prefer portfolio.market_value() or portfolio.get_equity_value()
    Fall back to portfolio.get_equity_curve_df() last market_value
    """
    try:
        if hasattr(portfolio, "market_value"):
            return float(portfolio.market_value())
        if hasattr(portfolio, "get_equity_value"):
            return float(portfolio.get_equity_value())
        # fallback to equity_curve
        if hasattr(portfolio, "get_equity_curve_df"):
            df = portfolio.get_equity_curve_df()
            if df is not None and not df.empty:
                return float(df.iloc[-1]["market_value"])
    except Exception:
        pass
    # last-resort: use portfolio.cash + sum(position valuations) if structure exists
    try:
        cash = float(getattr(portfolio, "cash", 0.0))
        positions = getattr(portfolio, "positions", {})
        total = cash
        for k, qty in positions.items():
            # positions may be (qty,) or simple map; we can't value them precisely here
            total += 0.0
        return total
    except Exception:
        return INITIAL_CAPITAL


def is_first_trading_day_of_month(ts: datetime) -> bool:
    # approximate: treat day <= 3 as first trading day candidate (engine controls actual runs)
    return ts.day <= 3


# ---------------------------
# Main driver
# ---------------------------


def main():
    print("\n================================================")
    print("      MINI QUANT FUND — INSTITUTIONAL DRIVER")
    print("================================================\n")

    # ---------------------------
    # Instantiate infrastructure (user modules)
    # ---------------------------
    provider = None
    try:
        provider = YahooDataProvider()
    except Exception:
        # Fallback provider to None (we'll use yfinance in get_price_history)
        provider = None
        print("Warning: YahooDataProvider not instantiated; using yfinance fallback for history.")

    # Data store (optional)
    try:
        datastore = DataStore()
    except Exception:
        datastore = None

    # Execution handler & backtest engine
    try:
        handler = RealisticExecutionHandler(commission_pct=COMMISSION_PCT, base_slippage=BASE_SLIPPAGE)
    except Exception:
        handler = None
        print("Warning: RealisticExecutionHandler unavailable; ensure backtest.execution module exists.")

    try:
        engine = BacktestEngine(data_provider=provider, initial_capital=INITIAL_CAPITAL, execution_handler=handler)
    except Exception as e:
        engine = None
        print(f"Warning: BacktestEngine unavailable: {e}")

    # registry
    try:
        registry = BacktestRegistry()
    except Exception:
        registry = None

    # ---------------------------
    # Strategy components (alpha + risk)
    # ---------------------------
    # instantiate alphas if available; else fallback to simple momentum placeholder
    if CompositeAlpha is not None:
        trend_alpha = TrendAlpha(short=50, long=200)
        meanrev_alpha = MeanReversionAlpha(short=5, lookback_std=21)
        composite_alpha = CompositeAlpha([trend_alpha, meanrev_alpha])
    else:
        composite_alpha = None
        print("Warning: CompositeAlpha not available. Please add strategies/alpha.py for full behavior.")

    # Risk manager (enforces volatility/leverage caps)
    if RiskManager is not None:
        risk_manager = RiskManager(max_leverage=1.0, target_vol_limit=DEFAULT_TARGET_VOL, min_allowed=0.0)
    else:
        risk_manager = None
        print("Warning: RiskManager not available. Add risk/engine.py for risk enforcement.")

    # ---------------------------
    # Engine setup: add tickers
    # ---------------------------
    if engine is not None:
        try:
            engine.add_tickers(TICKERS)
        except Exception:
            # Some engines accept add_universe or similar
            try:
                engine.set_universe(TICKERS)
            except Exception:
                print("Warning: couldn't add tickers to engine; it may accept tickers at run-time.")

    # ---------------------------
    # last_rebalance_month tracker
    # ---------------------------
    last_rebalance_month = None

    # ---------------------------
    # Strategy function (closure uses provider, composite_alpha, risk_manager)
    # Signature expected by engine: strategy_fn(timestamp: datetime, prices: Dict[str, float], portfolio) -> List[Order]
    # ---------------------------
    def strategy_fn(timestamp: datetime, prices: Dict[str, float], portfolio) -> List:
        """
        Event-driven strategy function invoked by BacktestEngine.
        - We compute per-asset conviction by running composite_alpha on historical prices.
        - RiskManager enforces volatility/leverage caps, returning adjusted conviction.
        - We rebalance monthly (first trading day approx) and create market orders to move to
          target allocations proportional to adjusted conviction.
        """
        nonlocal last_rebalance_month

        # Only rebalance on the first trading day of the month (approx). Skip otherwise.
        if not is_first_trading_day_of_month(timestamp):
            return []

        current_month = timestamp.month
        if last_rebalance_month == current_month:
            # already rebalanced this month
            return []
        last_rebalance_month = current_month

        # gather historical price series for each ticker
        history = {}
        
        # Cache for full history to avoid repeated downloads
        if not hasattr(strategy_fn, "history_cache"):
            strategy_fn.history_cache = {}
            
        for tk in prices.keys():
            # Check cache
            if tk not in strategy_fn.history_cache:
                # Fetch full history once
                # Try provider first
                if provider:
                    full_df = provider.fetch_ohlcv(tk, start_date="2000-01-01")
                    if not full_df.empty:
                        strategy_fn.history_cache[tk] = full_df["Close"]
                    else:
                        strategy_fn.history_cache[tk] = pd.Series(dtype=float)
                else:
                    # Fallback yf
                    try:
                        df = yf.download(tk, start="2000-01-01", progress=False, auto_adjust=True)
                        strategy_fn.history_cache[tk] = df["Close"] if not df.empty else pd.Series(dtype=float)
                    except:
                         strategy_fn.history_cache[tk] = pd.Series(dtype=float)

            # Slice from cache
            full_series = strategy_fn.history_cache[tk]
            # Efficient slicing: all data up to current timestamp
            # slice slightly more than needed then tail
            series = full_series[full_series.index <= timestamp]
            if not series.empty:            
                history[tk] = series.iloc[-500:] # take last 500
            else:
                 # fallback to single point
                 history[tk] = pd.Series([prices[tk]], index=[timestamp])

        # compute raw convictions (0..1) per asset
        raw_conv = {}
        for tk, series in history.items():
            if composite_alpha is not None:
                try:
                    conv_series = composite_alpha.compute(series)
                    raw_conv[tk] = float(conv_series.iloc[-1]) if not conv_series.empty else 0.0
                except Exception as e:
                    print(f"Alpha compute failed for {tk}: {e}")
                    raw_conv[tk] = 0.0
            else:
                # fallback: simple momentum proxy: 200-day return > 0 -> 1 else 0
                try:
                    s = series.dropna()
                    raw_conv[tk] = 1.0 if len(s) > 200 and (s.iloc[-1] / s.iloc[-200] - 1) > 0 else 0.0
                except Exception:
                    raw_conv[tk] = 0.0

        # apply RiskManager per asset to get adjusted conviction and leverage factors
        adjusted_conv = {}
        leverage_factors = {}
        for tk, series in history.items():
            rc = raw_conv.get(tk, 0.0)
            # convert rc to a one-element Series to pass to RiskManager.enforce_limits (which expects Series)
            rc_series = pd.Series([rc], index=[timestamp])
            if risk_manager is not None:
                try:
                    adj_series, lev_series = risk_manager.enforce_limits(rc_series, series)
                    adjusted_conv[tk] = float(adj_series.iloc[-1]) if not adj_series.empty else 0.0
                    leverage_factors[tk] = float(lev_series.iloc[-1]) if not lev_series.empty else 1.0
                except Exception as e:
                    print(f"RiskManager enforcement failed for {tk}: {e}")
                    adjusted_conv[tk] = rc
                    leverage_factors[tk] = 1.0
            else:
                adjusted_conv[tk] = rc
                leverage_factors[tk] = 1.0

        # Compute proportional target allocations based on adjusted conviction
        total_conv = sum(adjusted_conv.values())
        orders = []

        total_equity = total_portfolio_value(portfolio)
        if total_conv <= 0:
            # no conviction -> move to cash (no orders here)
            return []

        # determine target dollar allocations proportional to conviction
        target_allocations = {tk: (adjusted_conv[tk] / total_conv) for tk in adjusted_conv}

        # build orders to rebalance to target allocations
        for tk, target_pct in target_allocations.items():
            target_value = total_equity * target_pct
            current_qty = portfolio.positions.get(tk, 0) if hasattr(portfolio, "positions") else 0
            current_price = prices.get(tk, None)
            if current_price is None or current_price <= 0:
                continue
            current_val = current_qty * current_price
            diff_val = target_value - current_val

            # avoid tiny noise trades
            if abs(diff_val) < total_equity * MIN_TRADE_PCT:
                continue

            diff_qty = diff_val / current_price

            # Create Order object from backtest.execution module
            try:
                order = Order(ticker=tk, quantity=diff_qty, order_type=OrderType.MARKET, timestamp=timestamp)
            except Exception:
                # If Order signature differs, try positional fallback
                try:
                    order = Order(tk, diff_qty, OrderType.MARKET, timestamp)
                except Exception:
                    print("Unable to instantiate Order for", tk)
                    continue

            orders.append(order)

        return orders

    # ---------------------------
    # Run the backtest
    # ---------------------------
    if engine is None:
        print("Backtest engine not available. Exiting.")
        return

    print(f"▶ Starting backtest from {START_DATE} for universe: {TICKERS}")
    # engine.run signature in your code earlier used (start_date, strategy_fn)
    try:
        engine.run(start_date=START_DATE, strategy_fn=strategy_fn)
    except TypeError:
        # if signature differs, try other common signatures
        try:
            engine.run(strategy_fn=strategy_fn, start_date=START_DATE, tickers=TICKERS)
        except Exception as e:
            print("Engine.run failed with unexpected signature:", e)
            return

    # ---------------------------
    # Get results from engine
    # ---------------------------
    try:
        results = engine.get_results()
    except Exception:
        # fallback: engine may expose attribute equity_curve or similar
        try:
            results = getattr(engine, "results", None)
        except Exception:
            results = None

    if results is None or results.empty:
        print("No results produced by backtest engine.")
        return

    # results expected to include equity column; adapt if different
    if "equity" in results.columns:
        equity_curve = results["equity"]
    else:
        # try to find an equity-like column
        possible = [c for c in results.columns if "equity" in c.lower() or "nav" in c.lower()]
        equity_curve = results[possible[0]] if possible else results.iloc[:, 0]

    returns = equity_curve.pct_change().fillna(0)

    # ---------------------------
    # Analytics
    # ---------------------------
    ann_ret = annualized_return(returns)
    ann_vol = annualized_volatility(returns)
    sharpe = sharpe_ratio(returns)
    mdd = max_drawdown(equity_curve)

    print("\n--- BACKTEST METRICS ---")
    print(f"Final Equity: ${equity_curve.iloc[-1]:,.2f}")
    print(f"Annualized Return: {ann_ret:.3f}")
    print(f"Annualized Volatility: {ann_vol:.3f}")
    print(f"Sharpe Ratio: {sharpe:.3f}")
    print(f"Max Drawdown: {mdd:.3%}")

    # ---------------------------
    # Save trades to CSV (New)
    # ---------------------------
    try:
        import os
        os.makedirs("output/backtests", exist_ok=True)
        trades_df = engine.blotter.trades_df()
        if not trades_df.empty:
            trades_df.to_csv("output/backtests/trades.csv", index=False)
            print(f"Trades saved: {len(trades_df)} records to 'output/backtests/trades.csv'")
        else:
            print("No trades recorded to save.")
    except Exception as e:
        print(f"Failed to save trades: {e}")

    # ---------------------------
    # Save run to registry (if available)
    # ---------------------------
    if registry is not None:
        try:
            metrics = {
                "final_equity": float(equity_curve.iloc[-1]),
                "annualized_return": float(ann_ret),
                "annualized_volatility": float(ann_vol),
                "sharpe": float(sharpe),
                "max_drawdown": float(mdd),
            }
            run_id = registry.register_run(
                config={
                    "tickers": TICKERS,
                    "start_date": START_DATE,
                    "strategy": "CompositeAlpha_RiskManager_RebalanceMonthly",
                },
                results_df=results,
                metrics=metrics,
            )
            print(f"Backtest registered: run_id={run_id}")
        except Exception as e:
            print("Registry save failed:", e)

    # ---------------------------
    # Reporting / Visuals (best-effort)
    # ---------------------------
    if allocation_table is not None:
        try:
            # try to compute final weights by looking at last row of results positions if available
            if "positions" in results.columns:
                # not standard; skip
                pass
            print("\n--- FINAL PORTFOLIO ALLOCATION ---")
            # compute a simple allocation snapshot from last positions if possible
            try:
                last_positions = getattr(engine, "positions_snapshot", None)
                if last_positions:
                    # last_positions expected dict ticker->(qty, price) or similar
                    weights_df = pd.Series({k: v[0] * v[1] for k, v in last_positions.items()})
                    weights = weights_df / weights_df.sum()
                    print(allocation_table(weights))
                else:
                    print("Allocation snapshot not available from engine.")
            except Exception:
                print("Unable to print allocation table.")
        except Exception:
            pass

    # plots
    if plot_equity_curve is not None:
        try:
            plot_equity_curve(equity_curve, title="Institutional Backtest Equity Curve")
        except Exception:
            pass
    if plot_drawdown is not None:
        try:
            plot_drawdown(equity_curve, title="Institutional Backtest Drawdown")
        except Exception:
            pass

    print("\nDone.")


if __name__ == "__main__":
    main()
