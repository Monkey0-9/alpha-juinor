"""
Phase 10 Verification - Top 1% Hedge Fund Components.

Tests all new modules:
1. Factor Zoo
2. Factor Orthogonalization
3. Transaction Cost Model
4. Microstructure Analysis
5. Dynamic Leverage
6. Alpha Decay Detection
7. Alternative Data
8. P&L Attribution
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.getcwd())


class TestPhase10Components(unittest.TestCase):
    def setUp(self):
        # Mock price data
        np.random.seed(42)
        dates = pd.date_range("2022-01-01", periods=300)
        self.prices = pd.Series(
            np.cumsum(np.random.randn(300) * 0.01) + 100,
            index=dates
        )
        self.volumes = pd.Series(
            np.random.randint(100000, 1000000, 300),
            index=dates
        )

    def test_factor_zoo(self):
        """Test Factor Zoo calculations."""
        print("\n[TEST] Factor Zoo...")
        from factors.factor_zoo import get_factor_zoo

        zoo = get_factor_zoo()

        # Test individual factors
        mom_12_1 = zoo.calc_momentum_12_1(self.prices)
        self.assertIsNotNone(mom_12_1)
        print(f" -> Momentum 12-1: {mom_12_1:.4f}")

        vol_20 = zoo.calc_realized_vol_20(self.prices)
        self.assertGreater(vol_20, 0)
        print(f" -> Realized Vol 20: {vol_20:.4f}")

        rsi = zoo.calc_rsi_14(self.prices)
        self.assertGreaterEqual(rsi, 0)
        self.assertLessEqual(rsi, 100)
        print(f" -> RSI 14: {rsi:.2f}")

        # Test all factors
        factors = zoo.calculate_all_factors(
            self.prices, volumes=self.volumes
        )
        self.assertGreater(len(factors), 5)
        print(f" -> Total factors calculated: {len(factors)}")

        print(" -> Factor Zoo: OK")

    def test_factor_orthogonalization(self):
        """Test factor orthogonalization."""
        print("\n[TEST] Factor Orthogonalization...")
        from factors.factor_orthogonalization import FactorOrthogonalizer

        # Create correlated factors
        factor_data = pd.DataFrame({
            "factor_a": np.random.randn(100),
            "factor_b": np.random.randn(100),
        })
        factor_data["factor_c"] = factor_data["factor_a"] * 0.8 + np.random.randn(100) * 0.2

        ortho = FactorOrthogonalizer(min_correlation=0.5)
        ortho.fit(factor_data)

        ortho_factors = ortho.transform(factor_data)
        self.assertEqual(len(ortho_factors.columns), 3)
        print(f" -> Orthogonalized {len(ortho_factors.columns)} factors")

        # Check orthogonality (correlation should be ~0)
        corr = ortho_factors.corr()
        off_diag = corr.values[~np.eye(3, dtype=bool)]
        max_corr = np.abs(off_diag).max()
        print(f" -> Max off-diagonal correlation: {max_corr:.4f}")
        self.assertLess(max_corr, 0.1)

        print(" -> Factor Orthogonalization: OK")

    def test_transaction_cost_model(self):
        """Test transaction cost estimation."""
        print("\n[TEST] Transaction Cost Model...")
        from execution.transaction_cost_model import get_transaction_cost_model

        tcm = get_transaction_cost_model()

        estimate = tcm.estimate_cost(
            symbol="AAPL",
            shares=1000,
            price=150.0,
            direction="BUY",
            adv=5_000_000,
            volatility=0.02,
            urgency="MEDIUM"
        )

        self.assertGreater(estimate.total_cost, 0)
        self.assertGreater(estimate.cost_bps, 0)
        print(f" -> Total cost: ${estimate.total_cost:.2f}")
        print(f" -> Cost in bps: {estimate.cost_bps:.2f}")
        print(f" -> Recommended algo: {estimate.recommended_algo}")

        print(" -> Transaction Cost Model: OK")

    def test_microstructure(self):
        """Test microstructure analysis."""
        print("\n[TEST] Microstructure Analysis...")
        from execution.microstructure import get_microstructure_analyzer

        analyzer = get_microstructure_analyzer()

        # Create mock trades
        trades = pd.DataFrame({
            "price": self.prices.values[-100:],
            "volume": self.volumes.values[-100:]
        })

        signal = analyzer.analyze("AAPL", trades)

        self.assertIsNotNone(signal.vpin)
        self.assertGreaterEqual(signal.vpin, 0)
        self.assertLessEqual(signal.vpin, 1)
        print(f" -> VPIN: {signal.vpin:.4f}")
        print(f" -> Flow Direction: {signal.flow_direction.value}")

        print(" -> Microstructure Analysis: OK")

    def test_dynamic_leverage(self):
        """Test dynamic leverage engine."""
        print("\n[TEST] Dynamic Leverage...")
        from risk.dynamic_leverage import get_leverage_engine

        engine = get_leverage_engine()

        # Bull quiet - should be high leverage
        decision = engine.compute_leverage(
            regime="BULL_QUIET",
            vix=15,
            current_drawdown=0.02
        )
        self.assertGreater(decision.target_leverage, 1.0)
        print(f" -> BULL_QUIET, VIX=15: {decision.target_leverage:.2f}x")

        # Crisis - should be minimal leverage
        decision = engine.compute_leverage(
            regime="CRISIS",
            vix=45,
            current_drawdown=0.20
        )
        self.assertLess(decision.target_leverage, 0.3)
        print(f" -> CRISIS, VIX=45: {decision.target_leverage:.2f}x")

        print(" -> Dynamic Leverage: OK")

    def test_alpha_decay(self):
        """Test alpha decay detection."""
        print("\n[TEST] Alpha Decay Detection...")
        from monitoring.alpha_decay import get_alpha_decay_detector

        detector = get_alpha_decay_detector()

        # Simulate strategy returns
        for i in range(100):
            # First 50 days: good performance
            if i < 50:
                ret = 0.001 + np.random.randn() * 0.01
            # Last 50 days: decaying performance
            else:
                ret = -0.0005 + np.random.randn() * 0.012
            detector.record_return("TestStrategy", ret)

        # Check for decay
        alert = detector.check_decay("TestStrategy")

        report = detector.get_health_report()
        print(f" -> Strategy status: {report.get('TestStrategy', {}).get('status')}")

        adjustments = detector.get_weight_adjustments()
        print(f" -> Weight adjustment: {adjustments.get('TestStrategy', 1.0):.2f}")

        print(" -> Alpha Decay Detection: OK")

    def test_alternative_data(self):
        """Test alternative data engine."""
        print("\n[TEST] Alternative Data...")
        from alternative_data import get_alternative_data_engine

        engine = get_alternative_data_engine()

        # Simulate data
        engine.simulate_data("AAPL")

        signal = engine.get_composite_signal("AAPL")

        self.assertGreaterEqual(signal.composite_score, -1)
        self.assertLessEqual(signal.composite_score, 1)
        print(f" -> Composite Score: {signal.composite_score:.4f}")
        print(f" -> Confidence: {signal.confidence:.4f}")
        print(f" -> Reasons: {signal.reasons}")

        print(" -> Alternative Data: OK")

    def test_pnl_attribution(self):
        """Test P&L attribution."""
        print("\n[TEST] P&L Attribution...")
        from analytics.pnl_attribution import get_pnl_attributor

        attributor = get_pnl_attributor()

        attribution = attributor.attribute_daily(
            portfolio_return=0.015,
            position_returns={"AAPL": 0.02, "MSFT": 0.01},
            position_weights={"AAPL": 0.5, "MSFT": 0.5},
            factor_exposures={
                "AAPL": {"market": 1.1, "value": 0.2, "momentum": 0.5},
                "MSFT": {"market": 0.9, "value": 0.3, "momentum": 0.3}
            },
            factor_returns={"market": 0.01, "value": 0.002, "momentum": 0.003}
        )

        print(f" -> Total Return: {attribution.total_return:.4f}")
        print(f" -> Alpha: {attribution.alpha:.4f}")
        print(f" -> Market Beta: {attribution.market_beta:.4f}")

        summary = attributor.get_summary()
        print(f" -> Days tracked: {summary['days_tracked']}")

        print(" -> P&L Attribution: OK")


if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 10: Top 1% Hedge Fund Verification")
    print("=" * 60)
    unittest.main(verbosity=2)
