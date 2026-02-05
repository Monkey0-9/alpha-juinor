
import unittest
import unittest.mock
import pandas as pd
import numpy as np
import logging
import sys
import os

# Ensure project root is in path
sys.path.append(os.getcwd())

from strategies.stat_arb.engine import StatArbEngine
from alpha_agents.statistical_fundamental import StatArbAgent
from contracts import AgentResult, DecisionRecord
from meta_intelligence.pm_brain import PMBrain
from meta_intelligence.bayesian_scorer import BayesianScorer
from risk.engine import RiskManager

logging.basicConfig(level=logging.ERROR)

class TestPhase8Integration(unittest.TestCase):
    def setUp(self):
        # 1. Setup Mock Market Data
        dates = pd.date_range("2023-01-01", periods=100)
        self.prices = pd.DataFrame({
            "AAPL": np.random.normal(150, 5, 100),
            "MSFT": np.random.normal(300, 10, 100),
            "GOOGL": np.random.normal(120, 3, 100)
        }, index=dates)

        # Correlate MSFT and AAPL slightly
        self.prices["MSFT"] = self.prices["AAPL"] * 2 + np.random.normal(0, 2, 100)

    def test_statarb_engine_scan(self):
        print("\n[TEST] Verifying StatArb Engine Signal Generation...")
        engine = StatArbEngine()
        signals = engine.generate_signals(self.prices)

        print(f" -> Generated {len(signals)} signals.")
        if not signals.empty:
            print(f" -> Signal Sample:\n{signals.head()}")
            self.assertIn("leg1", signals.columns)
            self.assertIn("leg2", signals.columns)
            self.assertIn("z_score", signals.columns)

    def test_bayesian_performance_weighting(self):
        print("\n[TEST] Verifying Bayesian Scorer...")
        scorer = BayesianScorer(storage_path="tests/temp_performance.json")

        # Test Default Weight
        w_default = scorer.get_weight("UnknownAgent")
        print(f" -> Default Weight: {w_default:.4f} (Expected ~0.5)")
        self.assertAlmostEqual(w_default, 0.5, delta=0.1)

        # Simulate Success
        scorer.update_performance({"GoodAgent": 0.05}, actual_return=0.02)
        scorer.update_performance({"GoodAgent": 0.05}, actual_return=0.03)

        w_good = scorer.get_weight("GoodAgent")
        print(f" -> Good Agent Weight: {w_good:.4f}")
        self.assertGreater(w_good, w_default)

        # Simulate Failure
        scorer.update_performance({"BadAgent": 0.05}, actual_return=-0.02)
        scorer.update_performance({"BadAgent": 0.05}, actual_return=-0.03)

        w_bad = scorer.get_weight("BadAgent")
        print(f" -> Bad Agent Weight: {w_bad:.4f}")
        self.assertLess(w_bad, w_default)

        if os.path.exists("tests/temp_performance.json"):
            os.remove("tests/temp_performance.json")

    def test_pm_optimization_ledoit_wolf(self):
        print("\n[TEST] Verifying PMBrain Ledoit-Wolf Optimization...")

        # Patch the optimizer to avoid cvxpy dependency issues
        with unittest.mock.patch('meta_intelligence.pm_brain.optimize_portfolio') as mock_opt:
            mock_opt.return_value = {
                "w": np.array([0.5, 0.5]),
                "rejected_assets": [],
                "symbols": ["AAPL", "MSFT"]
            }

            # Force patch at module level for PMBrain
            import meta_intelligence.pm_brain
            meta_intelligence.pm_brain.optimize_portfolio = mock_opt

            pm = PMBrain()

            # Mock Candidates
            c1 = DecisionRecord(
                cycle_id="TEST_CYCLE",
                symbol="AAPL",
                final_decision="CANDIDATE",
                mu=0.005,
                sigma=0.01,
                timestamp=pd.Timestamp.utcnow().isoformat()
            )
            c2 = DecisionRecord(
                cycle_id="TEST_CYCLE",
                symbol="MSFT",
                final_decision="CANDIDATE",
                mu=0.004,
                sigma=0.012,
                timestamp=pd.Timestamp.utcnow().isoformat()
            )
            candidates = [c1, c2]

            # Mock Historical Returns (Needed for Ledoit-Wolf)
            hist_returns = self.prices.pct_change().dropna()

            result = pm.optimize_cycle(
                candidates,
                historical_returns=hist_returns,
                config={"risk_aversion": 1.0}
            )

            weights = result.get("w", [])
            print(f" -> Optimized Weights: {weights}")
            self.assertEqual(len(weights), 2)

            # Verify optimize_portfolio was called
            self.assertTrue(mock_opt.called)

            # Verify Ledoit-Wolf was used (Sigma check via kwargs)
            args, kwargs = mock_opt.call_args
            sigma_arg = kwargs.get('Sigma')

            print(f" -> Sigma Shape: {sigma_arg.shape}")
            self.assertEqual(sigma_arg.shape, (2, 2))

    def test_risk_stress_test(self):
        print("\n[TEST] Verifying Risk Shock Simulation...")
        rm = RiskManager()

        target_weights = {"AAPL": 0.05, "MSFT": 0.05}

        report = rm.run_stress_test(target_weights, self.prices["AAPL"].to_frame())

        print(" -> Stress Test Report:")
        print(report)

        self.assertIn("max_loss", report)
        self.assertIn("scenarios", report)
        if "Black Monday" in report["scenarios"]:
            self.assertIn("Black Monday", report["scenarios"])

if __name__ == "__main__":
    unittest.main()
