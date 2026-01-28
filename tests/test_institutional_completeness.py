
import unittest
import pandas as pd
import numpy as np
import shutil
import os
import json
from unittest.mock import MagicMock
from orchestration.cycle_orchestrator import CycleOrchestrator
from meta_intelligence.pm_brain import PMBrain
from data_intelligence.confidence_agent import ConfidenceAgent
from data_intelligence.quality_agent import QualityAgent
from risk.cvar import CVaRGate

class TestInstitutionalCompleteness(unittest.TestCase):
    def setUp(self):
        # Setup temporary audit file
        self.test_audit_path = "runtime/test_audit.log"
        if os.path.exists(self.test_audit_path):
             os.remove(self.test_audit_path)

    def test_completeness_no_drops(self):
        """
        Criteria: Cycle Decision Coverage = 100% (No dropped symbols).
        """
        orchestrator = CycleOrchestrator(mode="paper")
        # Ensure we don't accidentally pull real data
        orchestrator.universe_manager = MagicMock()
        orchestrator.universe_manager.get_active_universe.return_value = ["AAPL", "GOOG", "MSFT", "BAD_TICKER", "NULL"]

        # Mock DataRouter to fail for BAD_TICKER
        def mock_get_price(symbol, *args, **kwargs):
            if symbol == "BAD_TICKER":
                return pd.DataFrame() # Empty
            dates = pd.date_range("2023-01-01", periods=100)
            return pd.DataFrame({
                "Close": np.random.randn(100) + 100,
                "High": np.random.randn(100) + 105,
                "Low": np.random.randn(100) + 95,
                "Volume": np.random.randn(100) * 1000
            }, index=dates)

        orchestrator.data_router.get_price_history = MagicMock(side_effect=mock_get_price)

        # Run Cycle (using ThreadPoolExecutor logic in `run_cycle`)
        # Wait, CycleOrchestrator.run_cycle needs to be called.
        # We need to ensure we can run it in test mode involving the worker.

        # Stub worker execution for speed (or just run it if fast enough)
        # 5 symbols should be fast.

        decisions = orchestrator.run_cycle()

        # Assertions
        processed_symbols = {d.symbol for d in decisions}
        expected_symbols = {"AAPL", "GOOG", "MSFT", "BAD_TICKER", "NULL"}

        self.assertEqual(processed_symbols, expected_symbols, "Must process ALL symbols, even bad/empty ones.")

        # Check BAD_TICKER decision
        bad_decision = next(d for d in decisions if d.symbol == "BAD_TICKER")
        self.assertEqual(bad_decision.final_decision.value, "REJECT")
        self.assertIn("NO_DATA", bad_decision.reason_codes)

    def test_pm_brain_v2_logic(self):
        """
        Criteria: Opportunity Cost and Disagreement logic works.
        """
        brain = PMBrain(config={"pm_threshold": 0.5, "oc_penalty": 0.2})

        # Case 1: High Agreement
        high_agree = [MagicMock(mu=0.05, sigma=0.01, confidence=0.9, name=f"A{i}") for i in range(5)]
        decision = brain.aggregate("TEST1", high_agree)
        # Should have reasonable score
        self.assertTrue(decision.metadata['pm_score'] != 0)

        # Case 2: High Disagreement
        high_disagree = [MagicMock(mu=0.10, sigma=0.01, confidence=0.9, name="Optimist"),
                         MagicMock(mu=-0.10, sigma=0.01, confidence=0.9, name="Pessimist")]
        decision_bad = brain.aggregate("TEST2", high_disagree)

        # Disagreement penalty should lower the score significantly
        # Variance of (0.1, -0.1) is 0.01.
        # Penalty = exp(-5 * (1/(1+0.01))) = exp(-5 * 0.99) = exp(-4.95) ~= 0.007
        # Score should be crushed.
        self.assertLess(decision_bad.metadata['disagreement_penalty'], 0.1, "High disagreement should crush score.")

    def test_cvar_gate(self):
        gate = CVaRGate(limit=0.05)
        # Series with huge loss
        crashes = pd.Series([0.01]*95 + [-0.50]*5) # 5% tail are -0.50
        # CVaR should be ~0.50 (loss)
        val = gate.check_portfolio(crashes)
        self.assertFalse(val, "Should reject portfolio with 50% CVaR vs 5% limit")

if __name__ == '__main__':
    unittest.main()
