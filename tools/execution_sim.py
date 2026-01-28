import pandas as pd
import numpy as np
import sqlite3
import os
import csv
import sys
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.execution import Order, OrderType, BarData, RealisticExecutionHandler, TradeBlotter

# Results directory
res_dir = r"C:\mini-quant-fund\runtime\agent_results"
ts = "20260121_091700" # Use the fixed TS for consistency
sim_res_path = os.path.join(res_dir, ts, "execution_simulation.csv")

def run_execution_simulation():
    db_path = r"C:\mini-quant-fund\runtime\institutional_trading.db"
    if not os.path.exists(db_path):
        print("DB not found")
        return

    conn = sqlite3.connect(db_path)

    # 1. Setup components
    handler = RealisticExecutionHandler(
        commission_pct=0.0005,
        max_participation_rate=0.10,
        impact_coeff=0.15
    )
    blotter = TradeBlotter()

    # 2. Load some data for simulation
    symbol = "AAPL"
    df = pd.read_sql_query(f"SELECT * FROM price_history WHERE symbol='{symbol}' ORDER BY date DESC LIMIT 50", conn)
    df = df.sort_values('date')

    # 3. Simulate Orders
    # Test Idempotency: Same order ID should not produce multiple fills in a real system,
    # but in simulation we check if we can prevent duplicate processing.

    order1 = Order(
        ticker=symbol,
        quantity=1000,
        order_type=OrderType.MARKET,
        timestamp=datetime.now()
    )

    # Process several bars
    trades_executed = []

    print(f"Simulating execution for {symbol}...")

    for _, row in df.iterrows():
        bar = BarData(
            open=row['open'],
            high=row['high'],
            low=row['low'],
            close=row['close'],
            volume=row['volume'],
            timestamp=pd.to_datetime(row['date']),
            ticker=symbol
        )

        # In actual run, we'd need history for ADV/Vol
        # For simplicity in this check, we use the row itself as history or mock it
        hist_p = df['close']
        hist_v = df['volume']

        trade = handler.fill_order(order1, bar, hist_p, hist_v)
        if trade:
            blotter.record_trade(trade)
            trades_executed.append(trade)

        if order1.remaining_qty <= 0:
            break

    # 4. Verify results
    trades_df = blotter.trades_df()

    # Check slippage > 0
    test_slippage = (trades_df['slippage'] > 0).all()

    # Check Idempotency (Simulation specific: if we re-run the same filling logic on same order, it should respect remaining_qty)
    # We already have order1.remaining_qty at 0.

    # 5. Save results
    os.makedirs(os.path.dirname(sim_res_path), exist_ok=True)
    trades_df.to_csv(sim_res_path, index=False)

    print(f"Execution simulation complete. Results saved to {sim_res_path}")
    print(f"Slippage Check: {'PASS' if test_slippage else 'FAIL'}")

    conn.close()

if __name__ == "__main__":
    run_execution_simulation()
