"""
tests/test_phase_6.py

Unit tests for Phase 6: ML Governance (SafePredictor, Governor)
"""

import unittest
import sys
import os
import numpy as np
import pandas as pd
from unittest.mock import MagicMock

sys.path.insert(0, os.getcwd())

from ml.safe_predict import SafePredictor
from ml.governor import ModelGovernor
from contracts.alpha_model import AlphaOutput


class TestPhase6(unittest.TestCase):

    def setUp(self):
        self.mock_model = MagicMock()
        self.mock_model.predict.return_value = np.array([0.05])

    def test_safe_predict_imputation(self):
        """Test that missing features are imputed and prediction succeeds."""
        predictor = SafePredictor(model_id="test_model")

        # Expected ["A", "B"], provided only ["A"]
        features = pd.DataFrame({"A": [1.0]})
        expected = ["A", "B"]

        output = predictor.predict(self.mock_model, features, expected)

        self.assertIsInstance(output, AlphaOutput)
        self.assertEqual(output.mu, 0.05)  # Use 'mu' not 'signal'
        self.assertEqual(output.confidence, 0.5)  # Reduced due to imputation

    def test_governor_small_sample(self):
        """Test PSI calculation with small samples."""
        gov = ModelGovernor()

        expected = np.array([1, 2, 3])
        actual = np.array([1, 2, 3])

        psi = gov.calculate_psi(expected, actual)
        self.assertEqual(psi, 0.0)

    def test_safe_predict_failure(self):
        """Test graceful failure on model crash."""
        crash_model = MagicMock()
        crash_model.predict.side_effect = Exception("Boom")

        predictor = SafePredictor(model_id="crash_test")
        features = pd.DataFrame({"A": [1.0]})

        output = predictor.predict(crash_model, features, ["A"])

        # Should return neutral output (mu=0.0, confidence=0.0)
        self.assertEqual(output.mu, 0.0)
        self.assertEqual(output.confidence, 0.0)


if __name__ == "__main__":
    unittest.main()
