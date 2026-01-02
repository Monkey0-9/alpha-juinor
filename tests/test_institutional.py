import pandas as pd
from datetime import datetime
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backtest.engine import BacktestEngine
from backtest.execution import RealisticExecutionHandler, Order, OrderType
from data.provider import DataProvider

# Mock Provider
class MockProvider(DataProvider):
    def fetch_ohlcv(self, ticker: str, start_date: str, end_date: str = None) -> pd.DataFrame:
        # Create 10 days of fake data
        dates = pd.date_range(start=start_date, periods=10, freq='B')
        data = {
            "Open": [100.0] * 10,
            "High": [105.0] * 10,
            "Low": [95.0] * 10,
            "Close": [101.0] * 10, # Constant price
            "Volume": [1000000] * 10
        }
        df = pd.DataFrame(data, index=dates)
        return df

def test_trade_recording():
    print("--- Testing Trade Recording & Execution ---")
    provider = MockProvider()
    handler = RealisticExecutionHandler() # Defaults
    engine = BacktestEngine(provider, initial_capital=10000.0, execution_handler=handler)
    engine.add_tickers(["FAKE"])

    def strategy(timestamp, prices, portfolio):
        # Buy on first day
        if len(portfolio.blotter.orders) == 0:
            return [Order("FAKE", 10, OrderType.MARKET, timestamp)]
        return []

    engine.run("2020-01-01", strategy_fn=strategy)

    # Verify Blotter
    orders = engine.blotter.orders
    trades = engine.blotter.trades

    print(f"Orders: {len(orders)}")
    print(f"Trades: {len(trades)}")

    if len(orders) == 1 and len(trades) == 1:
        print("✅ Order and Trade recorded properly.")
        t = trades[0]
        print(f"Trade details: Qty={t.quantity}, Price={t.fill_price}, Cost={t.cost}")
        
        # Verify execution logic basics
        # Price was 101. Buy 10. 
        # Slippage/Comm should apply.
        expected_notional = 10 * 101
        if t.cost > 0:
             print("✅ Commission/Slippage calculated.")
        else:
             print("❌ No cost recorded.")
    else:
        print("❌ Failed to record order/trade.")

    # Verify CSV Export capability (Simulating what main.py does)
    import os
    os.makedirs("output/backtests", exist_ok=True)
    engine.blotter.trades_df().to_csv("output/backtests/test_trades.csv", index=False)
    if os.path.exists("output/backtests/test_trades.csv"):
        print("✅ Trades exported to CSV successfully.")
    else:
        print("❌ Failed to export trades CSV.")

    # Verify Equity Curve
    res = engine.get_results()
    if not res.empty and "market_value" in res.columns:
        print("✅ Results contain market_value.")
        print(res.head(2))
    else:
        print("❌ Results missing or malformed.")

if __name__ == "__main__":
    test_trade_recording()
