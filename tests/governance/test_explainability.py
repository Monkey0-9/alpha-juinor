"""
tests/governance/test_explainability.py

Tests for Operator Trust & Explainability.
"""
import unittest
from governance.explainer import DecisionExplainer, DecisionExplanation

class TestExplainability(unittest.TestCase):
    def setUp(self):
        self.explainer = DecisionExplainer()

    def test_explain_trade_approved(self):
        """Test explanation for approved trade"""
        explanation = self.explainer.explain_trade(
            symbol="AAPL",
            action="BUY",
            signal_strength=2.5,
            signal_components={"Momentum": 1.5, "MeanReversion": 1.0},
            position_size=100,
            adv=10000,
            risk_metrics={
                "portfolio_beta": (0.85, 1.0, "<"),
                "sector_exposure": (0.15, 0.20, "<")
            },
            governance_approved=True,
            confidence_interval=(0.75, 0.95)
        )

        self.assertEqual(explanation.action, "BUY")
        self.assertEqual(explanation.governance_status, "APPROVED")
        self.assertTrue(explanation.risk_checks["governance"])
        self.assertGreater(len(explanation.reasoning), 3)

    def test_explain_trade_blocked(self):
        """Test explanation for blocked trade"""
        explanation = self.explainer.explain_trade(
            symbol="TSLA",
            action="BUY",
            signal_strength=1.0,
            signal_components={"Momentum": 0.5},
            position_size=1000,
            adv=5000,  # 20% of ADV - too high
            risk_metrics={
                "sector_exposure": (0.25, 0.20, "<")  # Exceeds limit
            },
            governance_approved=False,
            confidence_interval=(0.3, 0.6)
        )

        self.assertEqual(explanation.governance_status, "BLOCKED")
        self.assertFalse(explanation.risk_checks["governance"])
        self.assertFalse(explanation.risk_checks["liquidity"])

    def test_explain_skip(self):
        """Test 'Why NOT trade' explanation"""
        explanation = self.explainer.explain_skip(
            symbol="GME",
            skip_reasons=[
                ("governance", "Symbol in QUARANTINE (data quality < 0.6)"),
                ("risk", "Tech sector at 25% (max 20%)"),
                ("confidence", "Alpha signal 1.1σ (threshold 1.5σ)")
            ]
        )

        self.assertEqual(explanation.action, "SKIP")
        self.assertEqual(explanation.governance_status, "BLOCKED")
        self.assertGreater(len(explanation.reasoning), 3)

    def test_confidence_bands_strong_signal(self):
        """Test confidence bands for strong signal"""
        low, high = self.explainer.compute_confidence_bands(
            signal_strength=3.0,
            historical_ic=0.08,
            regime_uncertainty=0.05
        )

        # Strong signal → high confidence, tight bands
        self.assertGreater(low, 0.5)
        self.assertGreater(high, 0.8)
        self.assertLess(high - low, 0.3)

    def test_confidence_bands_weak_signal(self):
        """Test confidence bands for weak signal"""
        low, high = self.explainer.compute_confidence_bands(
            signal_strength=0.5,
            historical_ic=0.02,
            regime_uncertainty=0.2
        )

        # Weak signal → low confidence, wide bands
        self.assertLess(low, 0.3)
        self.assertLess(high, 0.7)

    def test_format_explanation(self):
        """Test explanation formatting"""
        explanation = DecisionExplanation(
            symbol="AAPL",
            action="BUY",
            reasoning=["✓ Alpha Signal: 2.5σ", "✓ Risk Check: beta 0.85 < 1.0"],
            confidence_low=0.75,
            confidence_high=0.95,
            risk_checks={"beta": True},
            governance_status="APPROVED"
        )

        formatted = self.explainer.format_explanation(explanation)

        self.assertIn("BUY AAPL", formatted)
        self.assertIn("Confidence:", formatted)
        self.assertIn("APPROVED", formatted)

if __name__ == "__main__":
    unittest.main()
