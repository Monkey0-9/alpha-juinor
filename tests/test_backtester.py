import unittest
from datetime import datetime
import pandas as pd
from unittest.mock import MagicMock
from backtest.execution import Order, OrderType, SimpleExecutionHandler
from backtest.portfolio import Portfolio
from backtest.engine import BacktestEngine
from data.provider import DataProvider

class TestBacktestComponents(unittest.TestCase):
    
    def test_simple_execution(self):
        handler = SimpleExecutionHandler(commission_pct=0.01, slippage_pct=0.0) # 1% comm for easy math
        order = Order("AAPL", 10, OrderType.MARKET, datetime.now())
        price = 100.0
        
        trade = handler.fill_order(order, price, datetime.now())
        
        self.assertEqual(trade.price, 100.0)
        self.assertEqual(trade.commission, 10.0) # 100 * 10 * 0.01
        self.assertEqual(trade.size, 10)
        
    def test_portfolio_update(self):
        port = Portfolio(initial_capital=1000)
        
        # Buy 10 shares @ 10, comm 0
        trade = MagicMock()
        trade.ticker = "A"
        trade.size = 10
        trade.price = 10.0
        trade.commission = 0.0
        
        port.on_trade(trade)
        
        self.assertEqual(port.cash, 900)
        self.assertEqual(port.positions["A"], 10)
        
        # Sim market update: Price moves to 20
        ts = datetime.now()
        port.update_market_value({"A": 20.0}, ts)
        
        equity = port.get_equity_curve_df().iloc[-1]["equity"]
        self.assertEqual(equity, 1100) # 900 cash + (10 * 20) val
        
class MockDataProvider(DataProvider):
    def fetch_ohlcv(self, ticker, start_date, end_date=None):
        dates = pd.date_range(start="2023-01-01", periods=5)
        data = {"Close": [100, 101, 102, 103, 104], "Open": [100]*5, "High": [105]*5, "Low": [95]*5, "Volume": [1000]*5}
        return pd.DataFrame(data, index=dates)

class TestBacktestEngine(unittest.TestCase):
    def test_engine_run(self):
        provider = MockDataProvider()
        engine = BacktestEngine(provider, initial_capital=10000)
        engine.add_tickers(["SPY"])
        
        # Simple strategy: Buy 1 share every day
        def strat(ts, prices, port):
            return [Order("SPY", 1, OrderType.MARKET, ts)]
            
        engine.run(start_date="2023-01-01", strategy_fn=strat)
        
        res = engine.get_results()
        self.assertEqual(len(res), 5)
        # Check we bought 5 times
        self.assertEqual(engine.portfolio.positions["SPY"], 5)

if __name__ == "__main__":
    unittest.main()
