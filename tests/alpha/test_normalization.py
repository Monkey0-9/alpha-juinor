"""
tests/alpha/test_normalization.py

Tests for Mandatory Alpha Normalization.
"""
import unittest
import numpy as np
import pandas as pd
from alpha_families.normalization import AlphaNormalizer

class TestAlphaNormalization(unittest.TestCase):
    def setUp(self):
        self.normalizer = AlphaNormalizer()

    def test_z_score_calculation(self):
        """Test basic Z-score logic"""
        # Create history > 10 items
        data = [1.0, 2.0, 3.0, 4.0, 5.0] * 3 # 15 items
        history = pd.Series(data)
        # Mean=3, Std=1.46 (approx)
        val = 6.0

        z, conf = self.normalizer.normalize_signal(val, history)

        # (6-3)/1.46 ~ 2.05
        # Allow wider range due to simple stats
        self.assertTrue(1.5 < z < 2.5, f"Z-score {z} out of range")
        self.assertTrue(0.0 <= conf <= 1.0)

    def test_clipping_logic(self):
        """Test outlier clipping"""
        history = pd.Series([0.0] * 10 + [0.1])
        val = 100.0 # Extreme outlier

        z, conf = self.normalizer.normalize_signal(val, history)

        self.assertEqual(z, 5.0) # Clipped to max_z

    def test_distribution_construction(self):
        """Test distribution payload creation"""
        dist = self.normalizer.construct_distribution(
            z_score=2.0,
            confidence=0.8,
            volatility=0.2
        )

        self.assertIn('mu', dist)
        self.assertIn('sigma', dist)
        self.assertIn('confidence', dist)
        self.assertTrue(dist['mu'] > 0) # Positive signal
        self.assertTrue(dist['p_loss'] < 0.5) # Profitable exp

    def test_repair_logic(self):
        """Test fixing broken distributions"""
        broken = {
            'mu': np.nan,
            'sigma': -1.0,
            'confidence': 999.0
        }

        fixed = self.normalizer.repair_distribution(broken)

        self.assertEqual(fixed['mu'], 0.0)
        self.assertEqual(fixed['sigma'], 0.1) # Default
        self.assertEqual(fixed['confidence'], 1.0) # Clipped

if __name__ == "__main__":
    unittest.main()
