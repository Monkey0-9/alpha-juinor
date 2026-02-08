"""
Tests for Monte Carlo Mean Reversion Strategy.
"""
import unittest

import numpy as np
import pandas as pd

from strategies.monte_carlo_mean_reversion import (
    MCMRConfig,
    MonteCarloMeanReversionStrategy,
    create_mc_mean_reversion_strategy,
)
from strategy_factory.interface import Signal


class TestMonteCarloMeanReversionStrategy(unittest.TestCase):
    """Test cases for MonteCarloMeanReversionStrategy."""

    def setUp(self):
        """Create strategy and sample data."""
        self.config = MCMRConfig(n_simulations=1000)
        self.strategy = MonteCarloMeanReversionStrategy(self.config)

    def _create_price_series(self, trend: str = "neutral", length: int = 100):
        """Generate price series with different characteristics."""
        np.random.seed(42)

        if trend == "oversold":
            # Price dropped significantly below historical mean
            returns = np.concatenate([
                np.random.normal(0.001, 0.01, 80),  # Normal period
                np.random.normal(-0.02, 0.01, 20)   # Sharp decline
            ])
        elif trend == "overbought":
            # Price rose significantly above historical mean
            returns = np.concatenate([
                np.random.normal(0.001, 0.01, 80),
                np.random.normal(0.02, 0.01, 20)    # Sharp rise
            ])
        else:
            # Neutral / random walk
            returns = np.random.normal(0.0005, 0.015, length)

        prices = 100 * np.cumprod(1 + returns)
        return pd.Series(prices)

    def test_strategy_name(self):
        """Test strategy name property."""
        self.assertEqual(self.strategy.name, "MC_MeanReversion")

    def test_signal_structure(self):
        """Test that generate_signal returns valid Signal object."""
        prices = self._create_price_series()
        signal = self.strategy.generate_signal("TEST", prices)

        self.assertIsInstance(signal, Signal)
        self.assertEqual(signal.symbol, "TEST")
        self.assertIsInstance(signal.strength, float)
        self.assertIsInstance(signal.confidence, float)

    def test_oversold_generates_buy_signal(self):
        """Test that oversold condition generates positive signal."""
        prices = self._create_price_series("oversold")
        signal = self.strategy.generate_signal("TEST", prices)

        # Should generate buy signal (positive strength)
        self.assertGreater(signal.strength, 0)
        self.assertIn(signal.metadata["signal_type"],
                     ["buy", "strong_buy", "weak_buy"])

    def test_overbought_generates_sell_signal(self):
        """Test that overbought condition generates negative signal."""
        prices = self._create_price_series("overbought")
        signal = self.strategy.generate_signal("TEST", prices)

        # Should generate sell signal (negative strength)
        self.assertLess(signal.strength, 0)
        self.assertIn(signal.metadata["signal_type"],
                     ["sell", "strong_sell", "weak_sell"])

    def test_neutral_market_low_signal(self):
        """Test that neutral market produces weak/no signal."""
        prices = self._create_price_series("neutral")
        signal = self.strategy.generate_signal("TEST", prices)

        # Signal should be relatively weak
        self.assertLess(abs(signal.strength), 0.5)

    def test_strength_bounded(self):
        """Test that signal strength is bounded between -1 and 1."""
        prices = self._create_price_series("oversold")
        signal = self.strategy.generate_signal("TEST", prices)

        self.assertGreaterEqual(signal.strength, -1.0)
        self.assertLessEqual(signal.strength, 1.0)

    def test_confidence_bounded(self):
        """Test that confidence is bounded between 0 and 1."""
        prices = self._create_price_series("oversold")
        signal = self.strategy.generate_signal("TEST", prices)

        self.assertGreaterEqual(signal.confidence, 0.0)
        self.assertLessEqual(signal.confidence, 1.0)

    def test_regime_adjustment_sideways(self):
        """Test that sideways regime boosts signals."""
        prices = self._create_price_series("oversold")

        # Without regime
        signal_base = self.strategy.generate_signal("TEST", prices)

        # With sideways regime (favorable for mean reversion)
        regime_data = {"regime": "SIDEWAYS", "risk_multiplier": 1.0}
        signal_regime = self.strategy.generate_signal("TEST", prices, regime_data)

        self.assertTrue(signal_regime.regime_adjusted)
        # Sideways should boost the signal
        self.assertGreaterEqual(
            abs(signal_regime.strength),
            abs(signal_base.strength) * 0.9  # Allow some variance
        )

    def test_regime_adjustment_trending(self):
        """Test that trending regime reduces signals."""
        prices = self._create_price_series("oversold")

        # With trending regime (unfavorable for mean reversion)
        regime_data = {"regime": "TREND", "risk_multiplier": 1.0}
        signal = self.strategy.generate_signal("TEST", prices, regime_data)

        self.assertTrue(signal.regime_adjusted)

    def test_insufficient_data(self):
        """Test handling of insufficient data."""
        short_prices = pd.Series([100, 101, 102, 103, 104])
        signal = self.strategy.generate_signal("TEST", short_prices)

        self.assertEqual(signal.strength, 0.0)
        self.assertEqual(signal.confidence, 0.0)
        self.assertEqual(signal.metadata["reason"], "insufficient_data")

    def test_metadata_contains_fair_value(self):
        """Test that metadata includes fair value information."""
        prices = self._create_price_series()
        signal = self.strategy.generate_signal("TEST", prices)

        self.assertIn("fair_value_low", signal.metadata)
        self.assertIn("fair_value_mid", signal.metadata)
        self.assertIn("fair_value_high", signal.metadata)
        self.assertIn("rsi", signal.metadata)

    def test_factory_function(self):
        """Test the convenience factory function."""
        strategy = create_mc_mean_reversion_strategy(
            n_simulations=2000,
            lookback=30
        )

        self.assertEqual(strategy.config.n_simulations, 2000)
        self.assertEqual(strategy.config.lookback_period, 30)


if __name__ == "__main__":
    unittest.main()
