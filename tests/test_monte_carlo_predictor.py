"""
Tests for Monte Carlo Price Predictor with Markov Chain.
"""

import unittest

import numpy as np
import pandas as pd

from intelligence.monte_carlo_predictor import (
    MarketRegime,
    MarkovChainRegimeDetector,
    MonteCarloPricePredictor,
    PricePrediction,
    get_mc_predictor,
)


class TestMonteCarloPricePredictor(unittest.TestCase):
    """Test cases for MonteCarloPricePredictor."""

    def setUp(self):
        """Create predictor and sample price data."""
        self.predictor = MonteCarloPricePredictor(n_simulations=1000, seed=42)

        # Generate sample price series (random walk)
        np.random.seed(42)
        returns = np.random.normal(0.0005, 0.02, 100)
        prices = 100 * np.cumprod(1 + returns)
        self.prices = pd.Series(prices)

    def test_gbm_simulation_shape(self):
        """Test that GBM produces correct output shape."""
        paths = self.predictor.simulate_gbm_paths(
            s0=100.0, mu=0.10, sigma=0.20, n_days=20
        )

        self.assertEqual(paths.shape, (1000, 20))

    def test_gbm_simulation_positive_prices(self):
        """Test that GBM always produces positive prices."""
        paths = self.predictor.simulate_gbm_paths(
            s0=100.0, mu=0.10, sigma=0.40, n_days=50
        )

        self.assertTrue(np.all(paths > 0))

    def test_mean_reversion_simulation_converges(self):
        """Test that OU process paths stay near long-term mean."""
        paths = self.predictor.simulate_mean_reversion_paths(
            s0=80.0,  # Start below mean
            mu=0.05,
            sigma=0.15,
            theta=5.0,  # Fast mean reversion
            long_term_mean=100.0,
            n_days=60,
        )

        # Final prices should be closer to mean than starting price
        final_mean = np.mean(paths[:, -1])
        self.assertGreater(final_mean, 80.0)
        self.assertLess(abs(final_mean - 100.0), abs(80.0 - 100.0))

    def test_predict_returns_valid_structure(self):
        """Test that predict returns PricePrediction with all fields."""
        prediction = self.predictor.predict("TEST", self.prices)

        self.assertIsInstance(prediction, PricePrediction)
        self.assertEqual(prediction.symbol, "TEST")
        self.assertGreater(prediction.current_price, 0)
        self.assertGreater(prediction.pred_1d, 0)
        self.assertGreater(prediction.pred_5d, 0)
        self.assertGreater(prediction.pred_20d, 0)

    def test_predict_confidence_intervals(self):
        """Test that confidence intervals are ordered correctly."""
        prediction = self.predictor.predict("TEST", self.prices)

        # 5th < 50th < 95th percentile
        self.assertLess(prediction.range_5d[0], prediction.range_5d[1])
        self.assertLess(prediction.range_5d[1], prediction.range_5d[2])

    def test_predict_probability_bounds(self):
        """Test that probabilities are between 0 and 1."""
        prediction = self.predictor.predict("TEST", self.prices)

        self.assertGreaterEqual(prediction.prob_up_1d, 0.0)
        self.assertLessEqual(prediction.prob_up_1d, 1.0)
        self.assertGreaterEqual(prediction.prob_up_5d, 0.0)
        self.assertLessEqual(prediction.prob_up_5d, 1.0)

    def test_fair_value_range(self):
        """Test fair value range calculation."""
        fv_range = self.predictor.get_fair_value_range(self.prices, horizon_days=5)

        self.assertIn("p5", fv_range)
        self.assertIn("p25", fv_range)
        self.assertIn("p50", fv_range)
        self.assertIn("p75", fv_range)
        self.assertIn("p95", fv_range)

        # Percentiles should be ordered
        self.assertLess(fv_range["p5"], fv_range["p25"])
        self.assertLess(fv_range["p25"], fv_range["p50"])
        self.assertLess(fv_range["p50"], fv_range["p75"])
        self.assertLess(fv_range["p75"], fv_range["p95"])

    def test_singleton_accessor(self):
        """Test that get_mc_predictor returns singleton."""
        p1 = get_mc_predictor()
        p2 = get_mc_predictor()

        self.assertIs(p1, p2)

    def test_insufficient_data_raises(self):
        """Test that insufficient data raises ValueError."""
        short_prices = pd.Series([100, 101, 102])

        with self.assertRaises(ValueError):
            self.predictor.predict("TEST", short_prices)


if __name__ == "__main__":
    unittest.main()
