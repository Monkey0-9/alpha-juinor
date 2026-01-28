"""
tests/stress/test_capital_scaling.py

Stress tests for capital scaling realism.
"""
import unittest
import os
from risk.capital_stress import CapitalStressTester, StressTestResult

class TestCapitalScaling(unittest.TestCase):
    def setUp(self):
        self.tester = CapitalStressTester(
            max_adv_participation=0.05,
            impact_coefficient=0.1,
            sharpe_decay_rate=0.15
        )

        # Sample portfolio
        self.portfolio = {
            "AAPL": {"weight": 100000, "adv": 5000000},
            "MSFT": {"weight": 100000, "adv": 4000000},
            "GOOGL": {"weight": 100000, "adv": 3000000}
        }

    def test_market_impact_calculation(self):
        """Test square-root impact model"""
        # Small order: low impact
        impact_small = self.tester.compute_market_impact(10000, 1000000)
        self.assertLess(impact_small, 5.0)  # < 5 bps

        # Large order: higher impact
        impact_large = self.tester.compute_market_impact(100000, 1000000)
        self.assertGreater(impact_large, impact_small)

    def test_liquidity_feasibility_pass(self):
        """Test liquidity check for feasible position"""
        is_feasible, msg = self.tester.check_liquidity_feasibility(
            position_size=40000,  # 4% of ADV
            avg_daily_volume=1000000,
            symbol="TEST"
        )
        self.assertTrue(is_feasible)
        self.assertIsNone(msg)

    def test_liquidity_feasibility_fail(self):
        """Test liquidity check for infeasible position"""
        is_feasible, msg = self.tester.check_liquidity_feasibility(
            position_size=100000,  # 10% of ADV
            avg_daily_volume=1000000,
            symbol="TEST"
        )
        self.assertFalse(is_feasible)
        self.assertIsNotNone(msg)

    def test_sharpe_degradation(self):
        """Test Sharpe degradation with scaling"""
        baseline = 1.5

        # 1x: no degradation
        sharpe_1x = self.tester.estimate_sharpe_degradation(baseline, 1.0)
        self.assertAlmostEqual(sharpe_1x, baseline, places=2)

        # 10x: some degradation
        sharpe_10x = self.tester.estimate_sharpe_degradation(baseline, 10.0)
        self.assertLess(sharpe_10x, baseline)
        self.assertGreater(sharpe_10x, 0)

        # 50x: significant degradation
        sharpe_50x = self.tester.estimate_sharpe_degradation(baseline, 50.0)
        self.assertLess(sharpe_50x, sharpe_10x)

    def test_1x_baseline_feasible(self):
        """Test that 1x AUM is feasible"""
        result = self.tester.simulate_scaling(self.portfolio, 1.0, baseline_sharpe=1.5)

        self.assertTrue(result.feasible)
        self.assertEqual(result.max_position_breach_count, 0)
        self.assertLess(result.sharpe_degradation, 1.0)  # Minimal degradation

    def test_10x_scaling(self):
        """Test 10x AUM scenario"""
        result = self.tester.simulate_scaling(self.portfolio, 10.0, baseline_sharpe=1.5)

        # Should have some degradation but might still be feasible
        self.assertGreater(result.sharpe_degradation, 0)
        self.assertGreater(result.impact_cost_bps, 0)

    def test_50x_scaling_stress(self):
        """Test 50x AUM extreme scenario"""
        result = self.tester.simulate_scaling(self.portfolio, 50.0, baseline_sharpe=1.5)

        # Likely to have violations or significant degradation
        self.assertGreater(result.sharpe_degradation, 20.0)  # > 20% Sharpe drop

    def test_capacity_limit_finding(self):
        """Test capacity limit discovery"""
        limit = self.tester.find_capacity_limit(self.portfolio, baseline_sharpe=1.5, sharpe_threshold=1.0)

        # Should find some reasonable limit
        self.assertGreater(limit, 1.0)
        self.assertLess(limit, 100.0)

    def test_stress_suite(self):
        """Test full stress suite execution"""
        results = self.tester.run_stress_suite(self.portfolio, baseline_sharpe=1.5)

        # Should have all scenarios
        self.assertIn("1x_baseline", results)
        self.assertIn("10x_growth", results)
        self.assertIn("50x_institutional", results)

        # Degradation should increase with scale
        deg_1x = results["1x_baseline"].sharpe_degradation
        deg_10x = results["10x_growth"].sharpe_degradation
        deg_50x = results["50x_institutional"].sharpe_degradation

        self.assertLessEqual(deg_1x, deg_10x)
        self.assertLessEqual(deg_10x, deg_50x)

if __name__ == "__main__":
    # Create stress directory if needed
    os.makedirs("tests/stress", exist_ok=True)
    unittest.main()
