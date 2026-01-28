"""
tests/integration/test_phase2_integration.py

Verifies "Start to End" integration of Phase 2 components:
1. CapitalStressTester in ExecutionGatekeeper
2. DecisionExplainer in decide_execution
"""
import unittest
from governance.execution_decision import decide_execution
from execution.gates import ExecutionGatekeeper

class TestPhase2Integration(unittest.TestCase):
    def test_explainer_integration(self):
        """Test that decision layer returns explanation text"""
        config = {
            "execution": {
                "min_notional_usd": 100,
                "min_conviction": 0.1
            }
        }

        result = decide_execution(
            cycle_id="TEST",
            symbol="AAPL",
            target_weight=0.1,
            current_weight=0.0,
            nav_usd=100000,
            price=150.0,
            conviction=0.8,
            data_quality=1.0,
            risk_scaled_weight=0.1,
            skipping_history={},
            market_open=True,
            config=config
        )

        self.assertEqual(result['decision'], 'EXECUTE')
        self.assertIn('explanation', result)
        self.assertIn('Decision: EXECUTE AAPL', result['explanation'])
        self.assertIn('Confidence:', result['explanation'])

    def test_stress_tester_integration(self):
        """Test that ExecutionGatekeeper uses CapitalStressTester"""
        from unittest.mock import patch

        gate = ExecutionGatekeeper(adv_limit_pct=0.1, max_impact_bps=20)

        # Test impact calculation logic using internal stress tester
        # Small order -> OK
        with patch.object(gate, 'is_market_open', return_value=True):
            is_ok, reason, qty = gate.validate_execution(
                symbol="MSFT",
                qty=100, # Small
                side="buy",
                price=300,
                adv_30d=100_000_000, # Large ADV ($100M)
                volatility=0.015
            )
            self.assertTrue(is_ok, f"Small order failed: {reason}")

            # Massive order -> Fail on Impact or ADV
            # 20% of ADV -> Fail ADV
            is_ok, reason, qty = gate.validate_execution(
                symbol="MSFT",
                qty=200000, # 20% of 1M
                side="buy",
                price=300,
                adv_30d=1000000,
                volatility=0.015
            )
            self.assertFalse(is_ok)
            self.assertEqual(reason, "ADV_LIMIT_EXCEEDED")

if __name__ == "__main__":
    unittest.main()
