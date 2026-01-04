
import unittest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from portfolio.ledger import PortfolioLedger, PortfolioEvent, EventType, PositionBook
from backtest.execution import RealisticExecutionHandler, Order, OrderType, BarData, ExecutionError
from risk.engine import RiskManager

class TestCoreFixes(unittest.TestCase):
    
    def test_ledger_strict_accounting(self):
        """Verify strict equity calculation and missing price handling."""
        ledger = PortfolioLedger(initial_capital=10000.0)
        ts = datetime(2025, 1, 1)
        
        # 1. Buy 10 AAPL @ 100
        fill_event = PortfolioEvent(
            timestamp=ts,
            event_type=EventType.ORDER_FILLED,
            ticker="AAPL",
            quantity=10,
            price=100.0,
            commission=1.0
        )
        ledger.record_event(fill_event)
        
        # Cash should be 10000 - 1000 - 1 = 8999
        self.assertEqual(ledger.cash_book.balance, 8999.0)
        self.assertEqual(ledger.position_book.positions["AAPL"], 10.0)
        
        # 2. Snapshot with VALID price
        prices = {"AAPL": 105.0} # +50 profit
        snapshot = ledger.create_snapshot(ts, prices)
        
        # Equity = 8999 (cash) + 1050 (mv) = 10049
        self.assertAlmostEqual(snapshot["equity"], 10049.0)
        
        # 3. Snapshot with MISSING price -> Should RAISE ValueError
        try:
            ledger.create_snapshot(ts, {})
            self.fail("Ledger should have raised ValueError for missing price")
        except ValueError as e:
            self.assertIn("Missing price", str(e))
            
    def test_execution_zero_volume(self):
        """Verify zero volume bars prevent fills."""
        handler = RealisticExecutionHandler()
        
        # Generate sufficient history for Vol/ADV estimation
        # Vol lookback is 21, ADV is 20. Need ~30 points.
        hist_prices = pd.Series([100 + i*0.1 for i in range(30)])
        hist_vols = pd.Series([1000 for _ in range(30)])
        
        order = Order(
            ticker="SPY",
            quantity=100,
            order_type=OrderType.MARKET,
            timestamp=datetime(2025, 1, 1)
        )
        
        # Zero volume bar
        bad_bar = BarData(
            open=100, high=101, low=99, close=100,
            volume=0, 
            timestamp=datetime(2025, 1, 1),
            ticker="SPY"
        )
        
        # Use valid history so it doesn't fail on estimation
        trade = handler.fill_order(
            order, 
            bad_bar, 
            price_history=hist_prices,
            volume_history=hist_vols
        )
        
        self.assertIsNone(trade, "Zero volume bar should not generate a fill")
        
        # Valid volume bar
        good_bar = BarData(
            open=100, high=101, low=99, close=100,
            volume=1000, 
            timestamp=datetime(2025, 1, 1),
            ticker="SPY"
        )
        
        trade_ok = handler.fill_order(
             order,
             good_bar,
             price_history=hist_prices,
             volume_history=hist_vols
        )
        
        self.assertIsNotNone(trade_ok, "Positive volume should fill")
        # Check participation limit (10% of 1000 = 100). Order is 100. Should full fill.
        self.assertEqual(trade_ok.quantity, 100)
        
        # Test Participation Limit
        huge_order = Order(
            ticker="SPY",
            quantity=10000, # 10x volume
            order_type=OrderType.MARKET,
            timestamp=datetime(2025, 1, 1)
        )
        
        trade_partial = handler.fill_order(
             huge_order,
             good_bar,
             price_history=hist_prices,
             volume_history=hist_vols
        )
        
        # Should be capped at 10% of 1000 = 100
        self.assertEqual(trade_partial.quantity, 100)
        self.assertEqual(huge_order.remaining_qty, 9900)

    def test_risk_enforce_limits(self):
        """Verify RiskManager.enforce_limits implements Vol Scaling."""
        # Setup RiskManager with target vol 10%
        rm = RiskManager(target_vol_limit=0.10)
        
        # 1. Low Volatility Case (Realized = 5%)
        # Should scale UP (capped at 1.2 or similar)
        # 30 days of flat-ish returns
        prices_stable = pd.Series([100 * (1.001 ** i) for i in range(60)])
        # Realized vol approx 1.5% (sqrt(252)*0.001) ~ 1.6%
        # Target 10%. Scale = 10/1.6 = > 6. But clipped at 1.2.
        
        conviction = pd.Series([5_000], index=[0]) # $5k intent (Below 10% of $100k ADV)
        
        res_stable = rm.enforce_limits(conviction, prices_stable, pd.Series([1000]*60))
        
        # Expected scale factor 1.2
        scale = res_stable.adjusted_conviction.iloc[-1] / 5_000
        self.assertAlmostEqual(scale, 1.2, places=1)
        
        # 2. High Volatility Case (Realized = 20%)
        # Should scale DOWN to 0.5 (10% / 20%)
        # Generate random high vol
        np.random.seed(42)
        returns_high = np.random.normal(0, 0.02, 60) # 2% daily ~ 32% annual
        prices_vol = pd.Series(100 * (1 + returns_high).cumprod())
        
        # Realized vol ~ 32%
        # Target 10%. Scale ~ 0.31
        
        res_vol = rm.enforce_limits(conviction, prices_vol, pd.Series([1000]*60))
        scale_vol = res_vol.adjusted_conviction.iloc[-1] / 5_000
        
        # Verify it scaled down significantly
        self.assertTrue(scale_vol < 0.5, f"Expected scale < 0.5, got {scale_vol}")
        self.assertTrue(res_vol.estimated_leverage < 500_000)

if __name__ == '__main__':
    unittest.main()
