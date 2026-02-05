import unittest
from decimal import Decimal
from unittest.mock import MagicMock

from intelligence.autonomous_brain import AutonomousTradingBrain


class TestSymmetricIntelligence(unittest.TestCase):
    def setUp(self):
        self.brain = AutonomousTradingBrain()
        # Mock modules
        self.brain._regime = MagicMock()
        self.brain._smart_money = MagicMock()

    def test_regime_veto_relaxation_for_sells(self):
        """Test that SELL signals are penalized but NOT killed by regime veto."""
        # Setup: Regime avoids "Momentum"
        self.brain._regime.analyze.return_value.avoid_strategies = ["Momentum"]
        self.brain._regime.analyze.return_value.primary_regime.value = "BULL_TREND"

        # Scenario 1: Momentum BUY (Should be HARD VETOED)
        signals = [{
            "symbol": "AAPL",
            "strategy": "Momentum",
            "action": "BUY",
            "confidence": 0.8,
            "entry": Decimal("150"),
            "stop": Decimal("140"),
            "target": Decimal("180")
        }]

        sm_map = {} # No smart money info
        regime_info = {
            "primary": "BULL_TREND",
            "avoid": ["Momentum"]
        }

        ranked = self.brain._rank_opportunities(signals, sm_map, regime_info)
        self.assertEqual(ranked[0]["score"], 0.0, "Momentum BUY should be vetoed (score 0)")
        self.assertEqual(ranked[0]["grade"], "F")

        # Scenario 2: Momentum SELL (Should be PENALIZED but ALIVE)
        signals_sell = [{
            "symbol": "TSLA",
            "strategy": "Momentum",
            "action": "SELL",
            "confidence": 0.8,
            "entry": Decimal("200"),
            "stop": Decimal("210"),
            "target": Decimal("180")
        }]

        ranked_sell = self.brain._rank_opportunities(signals_sell, sm_map, regime_info)
        score = ranked_sell[0]["score"]

        # Original 0.8 -> Penalized by 0.5 = 0.4.
        # Plus RR bonus? RR is 2.0 (20/10). No bonus (<2.5).
        # So expected score around 0.4.
        self.assertGreater(score, 0.0, "Momentum SELL should NOT be vetoed")
        self.assertLess(score, 0.8, "Momentum SELL should be penalized")
        self.assertIn("Regime Avoids (Penalized but Kept for Exit)", ranked_sell[0]["reasoning"])

if __name__ == '__main__':
    unittest.main()
