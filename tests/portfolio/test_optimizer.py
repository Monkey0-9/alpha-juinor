"""
tests/portfolio/test_optimizer.py
"""
import unittest
import numpy as np
from portfolio.optimizer import PortfolioOptimizer, Constraint, CVXPY_AVAILABLE

class TestOptimizer(unittest.TestCase):
    def setUp(self):
        self.opt = PortfolioOptimizer()
        self.tickers = ["A", "B", "C"]
        self.mu = np.array([0.1, 0.2, 0.15])
        self.cov = np.array([
            [0.1, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [0.0, 0.0, 0.1]
        ])
        self.current = np.array([0.3, 0.3, 0.4])
        self.liquidity_costs = np.array([0.01, 0.01, 0.01])

    def test_optimization_basics(self):
        if not CVXPY_AVAILABLE:
            print("Skipping optimizer test: CVXPY not installed.")
            return

        res = self.opt.optimize(
            self.tickers, self.mu, self.cov, self.current,
            constraints=[], liquidity_costs=self.liquidity_costs, cvar_limit=0.05
        )

        self.assertEqual(res.status, "SUCCESS")
        weights = list(res.weights.values())
        self.assertAlmostEqual(sum(weights), 1.0, places=4)
        print("Optimization weights:", res.weights)

if __name__ == "__main__":
    unittest.main()
