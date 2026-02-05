import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from analytics.regime_analyzer import MarketRegime, RegimeAnalysis
from intelligence.autonomous_brain import AutonomousTradingBrain
from intelligence.smart_money_detector import SmartMoneyDetector, SmartMoneySignal


class TestVastIntelligence(unittest.TestCase):

    def setUp(self):
        self.brain = AutonomousTradingBrain()
        self.detector = SmartMoneyDetector()

    def test_hard_regime_veto(self):
        """Test that BUY strategies avoided by the regime are hard-vetoed (F grade)."""
        signals = [
            {
                "symbol": "AAPL",
                "strategy": "MeanReversion_RSI",  # Assume this is avoided in TRENDING
                "confidence": 0.8,
                "action": "BUY",  # BUY is vetoed, SELL is penalized (Symmetric Intel)
            }
        ]
        regime_info = {
            "primary": "TRENDING_UP",
            "avoid": ["MeanReversion"],  # Strategies checking this substring
            "recommended": ["Momentum"],
        }
        smart_money = {}

        results = self.brain._rank_opportunities(signals, smart_money, regime_info)

        self.assertEqual(results[0]["grade"], "F")
        self.assertEqual(results[0]["score"], 0.0)
        self.assertIn("HARD VETO", results[0]["reasoning"])

    def test_smart_money_alignment_requirement(self):
        """Test that 'A' grade requires Smart Money alignment."""
        signals = [
            {
                "symbol": "TSLA",
                "strategy": "Breakout",
                "confidence": 0.95,  # Very high base confidence
                "action": "BUY",
                "entry": 100,
                "target": 120,
                "stop": 90,  # RR = 2.0
            }
        ]

        # Case 1: No Smart Money -> Capped at B+ (0.75)
        regime_info = {"avoid": [], "recommended": []}
        smart_money = {}  # Empty

        results = self.brain._rank_opportunities(signals, smart_money, regime_info)
        self.assertLessEqual(results[0]["score"], 0.75)
        self.assertIn("Capped", results[0]["reasoning"])

        # Case 2: Aligned Smart Money -> Can exceed 0.75
        smart_money = {
            "TSLA": {
                "direction": "BULLISH",
                "activity": "ACCUMULATION",
                "confidence": 0.8,
            }
        }
        # Reset signal score for fairness in test (since list is mutable)
        signals[0]["score"] = 0.95  # Reset

        results = self.brain._rank_opportunities(signals, smart_money, regime_info)
        # 0.95 base + 0.15 SM + ... > 0.75
        self.assertGreater(results[0]["score"], 0.8)
        self.assertTrue(results[0]["smart_money_aligned"])

    def test_smart_money_strict_volume(self):
        """Test that SmartMoneyDetector ignores weak volume spikes."""
        # Setup: CREATE A VALID SIGNAL (Accumulation) so we get a result
        # Price: 100, 101, 100, 101...
        prices = [100 if i % 2 == 0 else 101 for i in range(100)]

        # Volume: High on UP days (odd indices), Low on DOWN days (even indices)
        # Up days (100 -> 101): indices 1, 3, 5...
        # Down days (101 -> 100): indices 2, 4, 6...
        base_vol = [1000 if i % 2 == 0 else 1500 for i in range(100)]

        # Case 1: Weak Spike (2.2 sigma) on the LAST day
        # Last index 99 is UP (100->101). Normal vol 1500.
        # Mean ~1250. Std ~250.
        # Target 2.2 sigma: (X - 1250) / 250 = 2.2 => X = 1800
        # Let's just manually set it relative to mean/std inside detector logic
        # Actually easier to just force values or trust the math.
        # Let's construct a synthetic scenario where we CONTROL mean/std

        vol_arr = np.array(base_vol)
        vol_mean = np.mean(vol_arr[-50:])
        vol_std = np.std(vol_arr[-50:])

        # Make the last one ~2.2 sigma
        target_weak = vol_mean + 2.2 * vol_std
        vol_arr[-1] = target_weak

        prices_s = pd.Series(prices)
        volumes_s = pd.Series(vol_arr)

        signal = self.detector.analyze("TEST", prices_s, volumes_s)

        # It SHOULD satisfy Accumulation (because of the pattern)
        self.assertIsNotNone(signal, "Should detect Accumulation")

        # But SHOULD NOT have Volume Spike reasoning (because < 2.5)
        volume_reasons = [r for r in signal.reasoning if "Volume Spike" in r]
        self.assertEqual(
            len(volume_reasons), 0, "Should not detect weak volume spike (2.2 sigma)"
        )

        # Case 2: Strong Spike (3.0 sigma)
        target_strong = vol_mean + 3.0 * vol_std
        vol_arr[-1] = target_strong
        volumes_s = pd.Series(vol_arr)

        signal = self.detector.analyze("TEST", prices_s, volumes_s)
        self.assertIsNotNone(signal)

        volume_reasons = [r for r in signal.reasoning if "Volume Spike" in r]
        self.assertTrue(
            len(volume_reasons) > 0, "Should detect strong volume spike (3.0 sigma)"
        )


if __name__ == "__main__":
    unittest.main()
