"""
tests/ml/test_model_governor.py
"""
import unittest
import numpy as np
from ml.governor import ModelGovernor

class TestModelGovernor(unittest.TestCase):
    def test_psi_no_drift(self):
        gov = ModelGovernor()
        # Same distribution
        base = np.random.normal(0, 1, 1000)
        curr = np.random.normal(0, 1, 1000)

        res = gov.check_drift("test_m", base, curr)
        self.assertFalse(res['drift_detected'], f"PSI {res['psi']} should be low")
        self.assertTrue(res['psi'] < 0.2)

    def test_psi_drift(self):
        gov = ModelGovernor()
        # Different distribution
        base = np.random.normal(0, 1, 1000)
        curr = np.random.normal(2, 1, 1000) # Shifted mean

        res = gov.check_drift("test_m", base, curr)
        self.assertTrue(res['drift_detected'])
        self.assertTrue(res['psi'] > 0.2)

if __name__ == "__main__":
    unittest.main()
