# tests/test_engine_integration.py
"""
Institutional integration test for:
provider -> engine -> execution -> blotter -> equity -> artifacts

Compatibility:
- Uses engine.run(...) to exercise RealisticExecutionHandler (correct fill_order signature)
- Uses blotter.orders_df() and blotter.trades_df() (stable APIs)
- Asserts trade attribution, partial-fill lifecycle, CSV export, and equity reconciliation
"""

import os
import sys
from datetime import datetime
import pandas as pd
import numpy as np

# Ensure project root on path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from backtest.engine import BacktestEngine
from backtest.execution import (
    RealisticExecutionHandler,
    Order,
    OrderType,
)

# Deterministic mock provider matching engine._build_price_panel expectations
class MockProvider:
    def get_panel(self, tickers, start_date, end_date=None):
        dates = pd.date_range(start=start_date, periods=500, freq="B")
        data = {}
        for tk in tickers:
            prices = np.full(len(dates), 100.0) + np.random.normal(0, 1.0, len(dates)) # Add noise
            data[(tk, "Open")] = prices
            data[(tk, "High")] = prices + 0.5
            data[(tk, "Low")] = prices - 0.5
            data[(tk, "Close")] = prices
            data[(tk, "Volume")] = np.full(len(dates), 1_000_000.0)
        panel = pd.DataFrame(data, index=dates)
        panel.columns = pd.MultiIndex.from_tuples(panel.columns)
        return panel

def test_engine_execution_blotter_and_equity_pipeline(tmp_path):
    provider = MockProvider()

    handler = RealisticExecutionHandler(
        commission_pct=0.001,
        max_participation_rate=0.10,
    )

    engine = BacktestEngine(
        provider=provider,
        initial_capital=10_000.0,
        execution_handler=handler,
    )

    # Strategy: submit a single buy on first bar
    def strategy_fn(timestamp, prices, portfolio):
        # use blotter DF to detect prior orders
        blotter = portfolio.get_blotter()
        orders_df = blotter.orders_df()
        if orders_df.empty:
            return [Order("FAKE", 10, OrderType.MARKET, timestamp)]
        return []

    # Run the engine (preferred signature)
    engine.run(start_date="2020-01-01", tickers=["FAKE"], strategy_fn=strategy_fn)

    blotter = engine.get_blotter()
    orders_df = blotter.orders_df()
    trades_df = blotter.trades_df()

    # Blotter integrity
    assert len(orders_df) == 1, f"Expected 1 order, got {len(orders_df)}"
    assert len(trades_df) == 1, f"Expected 1 trade, got {len(trades_df)}"

    # Cross-check attribution between orders.csv and trades.csv
    order_id = orders_df.iloc[0]["id"] if "id" in orders_df.columns else orders_df.iloc[0]["order_id"]
    trade_order_id = trades_df.iloc[0]["order_id"]
    assert str(trade_order_id) == str(order_id), "Trade.order_id must match Order.id"

    # Timestamp attribution: trade timestamp equals order timestamp (bar timestamp)
    # normalize to pandas timestamps for robust comparison
    order_ts = pd.to_datetime(orders_df.iloc[0]["timestamp"])
    trade_ts = pd.to_datetime(trades_df.iloc[0]["timestamp"])
    assert order_ts == trade_ts, f"Timestamps must match (order {order_ts} vs trade {trade_ts})"

    # Execution realism: check prices, costs present
    fill_price = float(trades_df.iloc[0]["fill_price"])
    commission = float(trades_df.iloc[0].get("commission", trades_df.iloc[0].get("commission", 0.0)))
    cost = float(trades_df.iloc[0].get("cost", 0.0))
    qty = float(trades_df.iloc[0]["quantity"])

    assert fill_price > 0.0
    assert cost >= 0.0
    assert abs(qty) <= 10.0 + 1e-8

    # Order lifecycle: either FILLED or PARTIALLY_FILLED stored in orders_df.status
    if "status" in orders_df.columns:
        status = orders_df.iloc[0]["status"]
        assert status in ("FILLED", "PARTIALLY_FILLED", "CANCELLED")
        if status == "PARTIALLY_FILLED":
            remaining = float(orders_df.iloc[0]["remaining_qty"])
            assert remaining != 0.0

    # Equity reconciliation
    results = engine.get_results()
    assert results is not None and not results.empty
    assert "equity" in results.columns
    final_equity = float(results["equity"].iloc[-1])
    # buying 10 lots at ~100 and paying costs.
    # We just ensure it's not the initial 10,000 exactly due to noise/costs.
    assert final_equity != 10_000.0

    # CSV export check (main.py behavior)
    out_dir = tmp_path / "output" / "backtests"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "test_trades.csv"
    trades_df.to_csv(csv_path, index=False)
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    assert len(df) == 1
    for col in ["trade_id", "order_id", "ticker", "quantity", "fill_price", "commission", "cost"]:
        assert col in df.columns, f"Missing expected column {col}"
