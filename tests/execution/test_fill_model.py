"""
tests/execution/test_fill_model.py
"""
import unittest
from execution.fill_simulator import FillSimulator
from risk.pnl_attribution import PnLAttribution

class TestExecution(unittest.TestCase):
    def test_simulator_slicing(self):
        sim = FillSimulator(impact_lambda=0.5)
        # Large order -> multiple slices expected
        # Args: order_id, symbol, side, qty, price, adv, urgency
        fills = sim.simulate_order("ORD1", "AAPL", "BUY", 10000, 150.0, 50000, "HIGH")

        self.assertTrue(len(fills) > 1, "Should effect slicing")
        total_filled = sum(f.qty for f in fills)
        self.assertAlmostEqual(total_filled, 10000)

        # Check impact
        prices = [f.price for f in fills]
        self.assertTrue(prices[-1] != 150.0)

    def test_attribution(self):
        attr = PnLAttribution(beta=1.0)

        # Mock fill
        from execution.fill_simulator import Fill
        f = Fill("A", "1", 101.0, 1.0, "", 0.0, 1.0, 0.0, 100.0)

        rec = attr.attribute([f], 100.0, 110.0, 0.05, "BUY")

        self.assertAlmostEqual(rec.total_pnl, 9.0)
        self.assertAlmostEqual(rec.execution_loss, 1.0)
        self.assertAlmostEqual(rec.market_pnl, 5.0)
        self.assertAlmostEqual(rec.alpha_pnl, 5.0)

if __name__ == "__main__":
    unittest.main()
