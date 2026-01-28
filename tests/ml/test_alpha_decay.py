"""
tests/ml/test_alpha_decay.py

Tests for Alpha Decay & Strategy Death Detection.
"""
import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from ml.alpha_decay import AlphaDecayMonitor, DecayMetrics

class TestAlphaDecayMonitor(unittest.TestCase):
    def setUp(self):
        self.monitor = AlphaDecayMonitor(
            decay_threshold=0.3,
            critical_threshold=0.5,
            min_ic_baseline=0.02
        )

    def test_rolling_ic_perfect_correlation(self):
        """Test IC calculation with perfect correlation"""
        dates = pd.date_range("2024-01-01", periods=100)
        np.random.seed(42)

        # Perfect positive correlation
        signals = pd.Series(np.random.randn(100), index=dates)
        forward_returns = signals * 0.01  # Perfect correlation

        ic = self.monitor.compute_rolling_ic(signals, forward_returns, window=30)

        # Should be close to 1.0 (Spearman correlation)
        self.assertGreater(ic.iloc[-1], 0.95)

    def test_rolling_ic_zero_correlation(self):
        """Test IC with uncorrelated signals"""
        dates = pd.date_range("2024-01-01", periods=100)
        np.random.seed(42)

        signals = pd.Series(np.random.randn(100), index=dates)
        forward_returns = pd.Series(np.random.randn(100), index=dates)

        ic = self.monitor.compute_rolling_ic(signals, forward_returns, window=30)

        # Should be close to 0
        self.assertLess(abs(ic.iloc[-1]), 0.3)

    def test_decay_detection_healthy(self):
        """Test decay detection for healthy strategy"""
        dates = pd.date_range("2024-01-01", periods=252)
        historical_ic = pd.Series(np.random.normal(0.05, 0.01, 252), index=dates)
        current_ic = 0.05

        is_decayed, decay_score, status = self.monitor.detect_decay(current_ic, historical_ic)

        self.assertFalse(is_decayed)
        self.assertEqual(status, "HEALTHY")
        self.assertLess(decay_score, 0.3)

    def test_decay_detection_degraded(self):
        """Test decay detection for degraded strategy"""
        dates = pd.date_range("2024-01-01", periods=252)
        historical_ic = pd.Series(np.random.normal(0.05, 0.01, 252), index=dates)
        current_ic = 0.025  # 50% drop from baseline

        is_decayed, decay_score, status = self.monitor.detect_decay(current_ic, historical_ic)

        self.assertTrue(is_decayed)
        self.assertIn(status, ["DEGRADED", "CRITICAL"])
        self.assertGreater(decay_score, 0.3)

    def test_decay_detection_critical(self):
        """Test decay detection for critical strategy"""
        dates = pd.date_range("2024-01-01", periods=252)
        historical_ic = pd.Series(np.random.normal(0.05, 0.01, 252), index=dates)
        current_ic = -0.01  # Negative IC

        is_decayed, decay_score, status = self.monitor.detect_decay(current_ic, historical_ic)

        self.assertTrue(is_decayed)
        self.assertEqual(status, "CRITICAL")

    def test_capacity_estimation(self):
        """Test capacity estimation logic"""
        capacity = self.monitor.estimate_capacity(
            strategy_sharpe=1.5,
            avg_position_size=100000,
            avg_daily_volume=1000000,
            participation_rate=0.05
        )

        # Should be positive and reasonable
        self.assertGreater(capacity, 0)
        self.assertLess(capacity, 1000000)  # Less than full ADV

    def test_recommend_retirement_continue(self):
        """Test retirement recommendation for healthy strategy"""
        metrics = DecayMetrics(
            strategy_id="TEST",
            date="2024-01-01",
            rolling_ic_30d=0.05,
            rolling_ic_60d=0.05,
            rolling_ic_90d=0.05,
            decay_score=0.1,
            capacity_utilization=0.5,
            status="HEALTHY",
            recommendation=""
        )

        rec = self.monitor.recommend_retirement(metrics)
        self.assertEqual(rec, "CONTINUE")

    def test_recommend_retirement_quarantine(self):
        """Test retirement recommendation for degraded strategy"""
        metrics = DecayMetrics(
            strategy_id="TEST",
            date="2024-01-01",
            rolling_ic_30d=0.02,
            rolling_ic_60d=0.02,
            rolling_ic_90d=0.02,
            decay_score=0.4,
            capacity_utilization=0.5,
            status="DEGRADED",
            recommendation=""
        )

        rec = self.monitor.recommend_retirement(metrics)
        self.assertEqual(rec, "QUARANTINE")

    def test_recommend_retirement_retire(self):
        """Test retirement recommendation for critical strategy"""
        metrics = DecayMetrics(
            strategy_id="TEST",
            date="2024-01-01",
            rolling_ic_30d=-0.01,
            rolling_ic_60d=-0.01,
            rolling_ic_90d=-0.01,
            decay_score=0.8,
            capacity_utilization=0.5,
            status="CRITICAL",
            recommendation=""
        )

        rec = self.monitor.recommend_retirement(metrics)
        self.assertEqual(rec, "RETIRE")

    def test_analyze_strategy_full_pipeline(self):
        """Test full analysis pipeline"""
        dates = pd.date_range("2024-01-01", periods=200)
        np.random.seed(42)

        # Create correlated signals and returns
        signals = pd.Series(np.random.randn(200), index=dates)
        forward_returns = signals * 0.005 + np.random.randn(200) * 0.002

        metrics = self.monitor.analyze_strategy(
            strategy_id="TEST_STRATEGY",
            signals=signals,
            forward_returns=forward_returns,
            current_aum=100000,
            avg_daily_volume=1000000,
            strategy_sharpe=1.2
        )

        # Verify all fields populated
        self.assertIsNotNone(metrics.rolling_ic_30d)
        self.assertIsNotNone(metrics.rolling_ic_60d)
        self.assertIsNotNone(metrics.rolling_ic_90d)
        self.assertIsNotNone(metrics.decay_score)
        self.assertIsNotNone(metrics.status)
        self.assertIsNotNone(metrics.recommendation)

if __name__ == "__main__":
    unittest.main()
