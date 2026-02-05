"""
Phase 9 Verification Script.

Tests all Elite Trading System components:
1. Rate Limiter
2. Multi-Timeframe Engine
3. Return Maximizer
4. Auto Pilot
5. Error Handler
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.getcwd())


class TestPhase9Integration(unittest.TestCase):
    def setUp(self):
        # Mock market data
        dates = pd.date_range("2022-01-01", periods=300)
        self.daily_data = pd.DataFrame({
            "open": np.random.normal(100, 5, 300),
            "high": np.random.normal(105, 5, 300),
            "low": np.random.normal(95, 5, 300),
            "close": np.cumsum(np.random.randn(300) * 0.5) + 100,
            "volume": np.random.randint(100000, 1000000, 300)
        }, index=dates)

    def test_rate_limiter(self):
        """Test Token Bucket rate limiter."""
        print("\n[TEST] Rate Limiter...")
        from data.rate_limiter import get_rate_limiter

        limiter = get_rate_limiter()

        # Should acquire successfully
        self.assertTrue(limiter.can_request("alpaca"))
        self.assertTrue(limiter.acquire("alpaca", timeout=1.0))

        status = limiter.get_status("alpaca")
        print(f" -> Alpaca status: {status}")
        self.assertIn("daily_used", status)
        print(" -> Rate Limiter: OK")

    def test_multi_timeframe_engine(self):
        """Test Multi-Timeframe Strategy Engine."""
        print("\n[TEST] Multi-Timeframe Engine...")
        from strategies.multi_timeframe_engine import (
            get_mtf_engine, Timeframe
        )

        engine = get_mtf_engine()

        # Generate signals
        swing_sig = engine.generate_swing_signal("AAPL", self.daily_data)
        if swing_sig:
            print(f" -> Swing Signal: {swing_sig.direction}, str={swing_sig.strength:.2f}")

        momentum_sig = engine.generate_momentum_signal("AAPL", self.daily_data)
        if momentum_sig:
            print(f" -> Momentum Signal: {momentum_sig.direction}, str={momentum_sig.strength:.2f}")

        # Run full analysis
        result = engine.run_full_analysis(
            "AAPL", daily_data=self.daily_data
        )
        if result:
            print(f" -> Aggregated: {result.action}, confidence={result.final_confidence:.2f}")

        print(" -> Multi-Timeframe Engine: OK")

    def test_return_maximizer(self):
        """Test Return Maximizer."""
        print("\n[TEST] Return Maximizer...")
        from strategies.return_maximizer import (
            get_return_maximizer, OpportunityScore
        )

        maximizer = get_return_maximizer()

        # Create opportunities
        opps = [
            maximizer.score_opportunity("AAPL", 0.15, 0.20, 0.60, 0.8),
            maximizer.score_opportunity("MSFT", 0.12, 0.18, 0.55, 0.7),
            maximizer.score_opportunity("GOOGL", 0.10, 0.15, 0.52, 0.6),
        ]

        for opp in opps:
            print(f" -> {opp.symbol}: Kelly={opp.kelly_fraction:.2%}, Rec={opp.recommended_weight:.2%}")

        # Optimize
        weights = maximizer.optimize_portfolio(opps, "NEUTRAL")
        print(f" -> Optimized Weights: {weights}")

        # Check target
        expected = {o.symbol: o.expected_return for o in opps}
        check = maximizer.check_target_achievable(weights, expected)
        print(f" -> Target Check: {check['recommendation']}")

        print(" -> Return Maximizer: OK")

    def test_auto_pilot(self):
        """Test Auto Pilot automation."""
        print("\n[TEST] Auto Pilot...")
        from orchestration.auto_pilot import get_autopilot, AutoAction

        pilot = get_autopilot()

        # Test decision
        decision = pilot.decide_action(
            symbol="AAPL",
            signal_direction=1,
            signal_strength=0.8,
            signal_confidence=0.75,
            current_price=150.0,
            pm_score=0.7,
            risk_passed=True,
            portfolio_value=100000
        )

        print(f" -> Decision: {decision.action.value}, qty={decision.quantity}")
        self.assertEqual(decision.action, AutoAction.BUY)
        self.assertGreater(decision.quantity, 0)

        # Test low confidence (should not buy)
        decision_low = pilot.decide_action(
            symbol="MSFT",
            signal_direction=1,
            signal_strength=0.3,
            signal_confidence=0.4,  # Below threshold
            current_price=300.0,
            pm_score=0.3,
            risk_passed=True
        )
        self.assertFalse(decision_low.approved)
        print(f" -> Low Confidence: {decision_low.reason}")

        print(" -> Auto Pilot: OK")

    def test_error_handler(self):
        """Test Global Error Handler."""
        print("\n[TEST] Error Handler...")
        from governance.error_handler import (
            get_error_handler, ErrorSeverity, error_boundary
        )

        handler = get_error_handler()

        # Test error handling
        try:
            raise ValueError("Test error")
        except Exception as e:
            recovered = handler.handle_error(
                "data_provider", e, ErrorSeverity.LOW
            )
            print(f" -> Error handled, recovered={recovered}")

        # Test circuit breaker
        self.assertTrue(handler.can_proceed("data_provider"))

        # Test health report
        report = handler.get_health_report()
        print(f" -> System paused: {report['system_paused']}")
        self.assertFalse(report["system_paused"])

        # Test error boundary decorator
        @error_boundary("test_subsystem", ErrorSeverity.LOW, default_return=42)
        def failing_function():
            raise RuntimeError("Intentional failure")

        result = failing_function()
        self.assertEqual(result, 42)
        print(" -> Error Boundary returned default on failure")

        print(" -> Error Handler: OK")


if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 9: Elite Trading System Verification")
    print("=" * 60)
    unittest.main(verbosity=2)
