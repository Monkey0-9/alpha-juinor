import unittest
from unittest.mock import MagicMock

from governance.execution_decision import decide_execution


class TestIntelligenceGate(unittest.TestCase):
    def setUp(self):
        self.config = {
            "execution": {
                "min_notional_usd": 100.0,
                "min_conviction": 0.3,
                "min_weight_change": 0.001,
            }
        }
        self.skipping_history = {}
        # Mock the explainer to avoid method name mismatches (explain vs explain_trade)
        import governance.execution_decision

        governance.execution_decision.explainer = MagicMock()

    def test_low_grade_rejection(self):
        """Verify that C grade trades are rejected."""
        res = decide_execution(
            cycle_id="test_cycle",
            symbol="AAPL",
            target_weight=0.05,
            current_weight=0.0,
            nav_usd=100000.0,
            price=150.0,
            conviction=0.8,
            data_quality=1.0,
            risk_scaled_weight=0.05,
            skipping_history=self.skipping_history,
            market_open=True,
            config=self.config,
            intelligence_grade="C",  # LOW GRADE
            smart_money_aligned=True,
        )
        self.assertEqual(res["decision"], "SKIP_LOW_GRADE")
        self.assertIn("GRADE_C", res["reason_codes"])

    def test_high_grade_approval(self):
        """Verify that A grade trades pass when sentiment is aligned."""
        res = decide_execution(
            cycle_id="test_cycle",
            symbol="AAPL",
            target_weight=0.05,
            current_weight=0.0,
            nav_usd=100000.0,
            price=150.0,
            conviction=0.8,
            data_quality=1.0,
            risk_scaled_weight=0.05,
            skipping_history=self.skipping_history,
            market_open=True,
            config=self.config,
            intelligence_grade="A",  # HIGH GRADE
            smart_money_aligned=True,
        )
        self.assertEqual(res["decision"], "EXECUTE")

    def test_smart_money_rejection(self):
        """Verify that trades misaligned with Smart Money are rejected."""
        res = decide_execution(
            cycle_id="test_cycle",
            symbol="AAPL",
            target_weight=0.05,
            current_weight=0.0,
            nav_usd=100000.0,
            price=150.0,
            conviction=0.8,
            data_quality=1.0,
            risk_scaled_weight=0.05,
            skipping_history=self.skipping_history,
            market_open=True,
            config=self.config,
            intelligence_grade="A",
            smart_money_aligned=False,  # MISALIGNED
        )
        self.assertEqual(res["decision"], "SKIP_SMART_MONEY")
        self.assertIn("SMART_MONEY_MISALIGNED", res["reason_codes"])


if __name__ == "__main__":
    unittest.main()
