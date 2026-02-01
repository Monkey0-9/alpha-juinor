
import unittest
import sys
import os
import pandas as pd
from datetime import datetime

# Add project root
sys.path.insert(0, os.getcwd())

from contracts.alpha_model import AlphaOutput, AlphaDecision
from ml.feature_store import get_feature_store

class TestPhase2(unittest.TestCase):

    def test_alpha_contract_valid(self):
        """Test valid alpha output creation."""
        alpha = AlphaOutput(
            mu=0.01,
            sigma=0.02,
            cvar_95=-0.03,
            confidence=0.8,
            provider="test_model",
            model_version="v1",
            input_schema_hash="abc",
            decision=AlphaDecision.BUY
        )
        self.assertEqual(alpha.decision, "BUY")
        self.assertEqual(alpha.mu, 0.01)

    def test_alpha_contract_invalid_range(self):
        """Test invalid ranges raise validation errors."""
        with self.assertRaises(ValueError):
            AlphaOutput(
                mu=0.9, # Too high
                sigma=0.02,
                cvar_95=-0.03,
                confidence=0.8,
                provider="test",
                model_version="v1",
                input_schema_hash="abc"
            )

    def test_alpha_contract_consistency(self):
        """Test cvar must be negative."""
        with self.assertRaises(ValueError):
             AlphaOutput(
                mu=0.01,
                sigma=0.02,
                cvar_95=0.01, # Positive CVaR invalid
                confidence=0.8,
                provider="test",
                model_version="v1",
                input_schema_hash="abc"
            )

    def test_feature_store_logic(self):
        """Test on-the-fly feature computation."""
        store = get_feature_store()

        # Mock DataFrame
        dates = pd.date_range("2024-01-01", periods=30)
        data = {
            "open": [100 + i for i in range(30)],
            "high": [105 + i for i in range(30)],
            "low": [95 + i for i in range(30)],
            "close": [102 + i for i in range(30)],
            "volume": [1000 for _ in range(30)],
            "date": dates
        }
        df = pd.DataFrame(data)

        # Test private computation method directly to avoid DB dependency in unit test
        features = store._compute_standard_features(df.copy())

        self.assertIn("returns_1d", features.columns)
        self.assertIn("volatility_20d", features.columns)

        # Check calculation
        # Returns for linear growth roughly constant
        latest_ret = features.iloc[-1]['returns_1d']
        self.assertTrue(0 < latest_ret < 0.05)

if __name__ == "__main__":
    unittest.main()
