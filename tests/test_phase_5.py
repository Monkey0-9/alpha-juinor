
import unittest
import sys
import os
import numpy as np
from unittest.mock import MagicMock

# Add project root
sys.path.insert(0, os.getcwd())

from execution.impact_gate import ImpactGate, ImpactDecision
from execution.regime_aware_executor import RegimeAwareExecutor

class TestPhase5(unittest.TestCase):

    def setUp(self):
        # Mock Regime Controller
        self.mock_regime = MagicMock()
        self.mock_regime.get_current_state.return_value = MagicMock(
            regime=MagicMock(value="NORMAL")
        )

    def test_impact_gate_approve_small(self):
        """Test small order approval."""
        gate = ImpactGate(regime_controller=self.mock_regime)

        # Small order: 100 shares, ADV 1M -> .01% participation
        res = gate.check_impact(
            symbol="AAPL", side="BUY", quantity=100, price=150.0,
            volatility=0.02, adv=1_000_000
        )

        self.assertEqual(res.decision, ImpactDecision.APPROVE)
        self.assertEqual(res.approved_qty, 100)

    def test_impact_gate_reject_huge(self):
        """Test huge order rejection/reduction."""
        gate = ImpactGate(regime_controller=self.mock_regime)

        # Huge order: 50% of ADV -> Should trigger ADV limit
        res = gate.check_impact(
            symbol="AAPL", side="BUY", quantity=500_000, price=150.0,
            volatility=0.02, adv=1_000_000
        )

        # Should be REDUCED or REJECTED depending on strictness
        # Default BASE_MAX_ADV_PCT is 0.10 (10%)
        # So it should reduce to approx 100k
        self.assertIn(res.decision, [ImpactDecision.REDUCE, ImpactDecision.SLICE])

        if res.decision == ImpactDecision.REDUCE:
            self.assertLess(res.approved_qty, 500_000)
            self.assertLessEqual(res.approved_qty, 1_000_000 * 0.15) # Buffer
        elif res.decision == ImpactDecision.SLICE:
            # If sliced, we verify it created slices
            self.assertGreater(res.slice_count, 1)


    def test_executor_regime_slices(self):
        """Test executor slicing varies by regime."""
        executor = RegimeAwareExecutor()

        # NORMAL regime
        executor.set_regime("NORMAL")
        plan_normal = executor.execute_order("AAPL", 1000, "BUY")
        self.assertEqual(plan_normal["profile"]["urgency"].value, "passive")

        # VOLATILE regime
        executor.set_regime("VOLATILE")
        plan_vol = executor.execute_order("AAPL", 1000, "BUY")
        self.assertEqual(plan_vol["profile"]["urgency"].value, "patient")
        # Patient should might have more slices or longer horizon
        self.assertGreaterEqual(plan_vol["profile"]["time_horizon_minutes"], plan_normal["profile"]["time_horizon_minutes"])

    def test_executor_crisis_block(self):
        """Test executor blocks in crisis."""
        executor = RegimeAwareExecutor()
        executor.set_regime("CRISIS")

        plan = executor.execute_order("AAPL", 1000, "BUY")
        self.assertEqual(plan["status"], "SKIP_CRISIS_MODE")
        self.assertEqual(len(plan["slices"]), 0)

if __name__ == "__main__":
    unittest.main()
