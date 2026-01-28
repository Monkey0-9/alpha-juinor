"""
tests/risk/test_pnl_decomposition.py

Tests for Economic Truth Engine (OLS).
"""
import unittest
import numpy as np
import pandas as pd
from risk.pnl_decomposition import PnLDecomposer

class TestPnLDecomposition(unittest.TestCase):
    def setUp(self):
        self.decomposer = PnLDecomposer(risk_free_rate=0.0)

    def test_perfect_beta(self):
        # r_strategy = 1.5 * r_benchmark + 0
        dates = pd.date_range("2024-01-01", periods=100)
        np.random.seed(42)
        bench_ret = pd.Series(np.random.normal(0.001, 0.01, 100), index=dates)
        strat_ret = 1.5 * bench_ret

        res = self.decomposer.decompose("TEST", strat_ret, bench_ret)

        self.assertTrue(res.valid)
        self.assertAlmostEqual(res.beta, 1.5, places=5)
        self.assertAlmostEqual(res.alpha_bps, 0.0, places=1)
        self.assertAlmostEqual(res.r_squared, 1.0, places=5)
        self.assertAlmostEqual(res.correlation, 1.0, places=5)

    def test_pure_alpha(self):
        # r_strategy = 0 * r_benchmark + 0.0005 daily (approx 12.6% annual)
        dates = pd.date_range("2024-01-01", periods=100)
        np.random.seed(42)
        bench_ret = pd.Series(np.random.normal(0.001, 0.01, 100), index=dates)
        strat_ret = pd.Series(np.zeros(100) + 0.0005, index=dates)
        # Note: Constant return has 0 covariance with benchmark noise typically

        res = self.decomposer.decompose("TEST", strat_ret, bench_ret)

        # Beta should be near 0
        self.assertAlmostEqual(res.beta, 0.0, delta=0.1)
        # Alpha should be annual of 0.0005 -> ~12.6% = 1260 bps
        expected_bps = 0.0005 * 252 * 10000
        self.assertAlmostEqual(res.alpha_bps, expected_bps, delta=10) # Float variance

    def test_aligned_dates(self):
        # Unaligned indices
        dates1 = pd.date_range("2024-01-01", periods=50)
        dates2 = pd.date_range("2024-01-10", periods=50) # Shifted start

        s1 = pd.Series(np.random.rand(50), index=dates1)
        s2 = pd.Series(np.random.rand(50), index=dates2)

        res = self.decomposer.decompose("TEST", s1, s2)
        # Only ~40 days overlap
        self.assertTrue(res.valid) # 40 > 30 min

    def test_insufficient_data(self):
        dates = pd.date_range("2024-01-01", periods=10)
        s1 = pd.Series(np.random.rand(10), index=dates)

        res = self.decomposer.decompose("TEST", s1, s1)
        self.assertFalse(res.valid)

if __name__ == "__main__":
    unittest.main()
