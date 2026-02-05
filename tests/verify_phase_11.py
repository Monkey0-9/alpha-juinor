"""
Phase 11 Verification - True Top 1% Hedge Fund Components.

Tests all new Renaissance/Two Sigma/Citadel-style modules.
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.getcwd())


class TestPhase11Components(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        dates = pd.date_range("2022-01-01", periods=300)
        self.returns = pd.Series(
            np.random.randn(300) * 0.01,
            index=dates
        )
        self.prices = (1 + self.returns).cumprod() * 100

    def test_hmm_predictor(self):
        """Test Hidden Markov Model predictor."""
        print("\n[TEST] HMM Predictor...")
        from ml.hmm_predictor import get_hmm_predictor, MarketState

        hmm = get_hmm_predictor()
        hmm.fit(self.returns)

        prediction = hmm.predict(self.returns)

        self.assertIsInstance(prediction.current_state, MarketState)
        self.assertGreater(prediction.confidence, 0)
        print(f" -> State: {prediction.current_state.name}")
        print(f" -> Confidence: {prediction.confidence:.2f}")
        print(" -> HMM Predictor: OK")

    def test_kernel_features(self):
        """Test kernel feature transformation."""
        print("\n[TEST] Kernel Features...")
        from ml.kernel_features import KernelFeatureTransformer, KernelPCA

        # Test transformer
        X = np.random.randn(100, 5)
        transformer = KernelFeatureTransformer(kernel_type="rbf")
        X_transformed = transformer.fit_transform(X)

        self.assertEqual(X_transformed.shape[0], 100)
        print(f" -> Transformed shape: {X_transformed.shape}")

        # Test Kernel PCA
        kpca = KernelPCA(n_components=3)
        X_reduced = kpca.fit_transform(X)
        self.assertEqual(X_reduced.shape[1], 3)
        print(f" -> KPCA shape: {X_reduced.shape}")
        print(" -> Kernel Features: OK")

    def test_rl_agent(self):
        """Test RL trading agent."""
        print("\n[TEST] RL Trading Agent...")
        from ml.rl_trading_agent import get_rl_agent

        agent = get_rl_agent()
        state = np.random.randn(20)

        action = agent.select_action(state, training=True)
        self.assertIn(action, [0, 1, 2])

        signal, conf = agent.get_trading_signal(state, 0.0)
        self.assertIn(signal, ["BUY", "SELL", "HOLD"])
        print(f" -> Signal: {signal}, Confidence: {conf:.2f}")
        print(" -> RL Trading Agent: OK")

    def test_kelly_optimizer(self):
        """Test Kelly criterion optimizer."""
        print("\n[TEST] Kelly Optimizer...")
        from portfolio.kelly_optimizer import get_kelly_optimizer

        kelly = get_kelly_optimizer()

        expected_returns = {"AAPL": 0.001, "MSFT": 0.0008, "GOOG": 0.0012}
        cov = pd.DataFrame(
            np.eye(3) * 0.0004,
            index=["AAPL", "MSFT", "GOOG"],
            columns=["AAPL", "MSFT", "GOOG"]
        )

        allocation = kelly.optimize(expected_returns, cov)

        self.assertEqual(len(allocation.weights), 3)
        print(f" -> Weights: {allocation.weights}")
        print(f" -> Expected Growth: {allocation.expected_growth:.4f}")
        print(" -> Kelly Optimizer: OK")

    def test_market_neutral(self):
        """Test market neutral constructor."""
        print("\n[TEST] Market Neutral...")
        from portfolio.market_neutral import get_market_neutral_constructor

        constructor = get_market_neutral_constructor()

        alpha_scores = {"AAPL": 0.5, "MSFT": 0.3, "GOOG": -0.4, "AMZN": -0.3}
        betas = {"AAPL": 1.1, "MSFT": 0.9, "GOOG": 1.2, "AMZN": 1.0}

        portfolio = constructor.construct(alpha_scores, betas)

        self.assertGreater(len(portfolio.long_weights), 0)
        self.assertGreater(len(portfolio.short_weights), 0)
        print(f" -> Gross Exposure: {portfolio.gross_exposure:.2f}")
        print(f" -> Net Beta: {portfolio.portfolio_beta:.4f}")
        print(" -> Market Neutral: OK")

    def test_risk_parity(self):
        """Test risk parity optimizer."""
        print("\n[TEST] Risk Parity...")
        from portfolio.risk_parity import get_risk_parity_optimizer

        optimizer = get_risk_parity_optimizer()

        cov = pd.DataFrame(
            [[0.04, 0.01, 0.005],
             [0.01, 0.02, 0.003],
             [0.005, 0.003, 0.01]],
            index=["AAPL", "MSFT", "GOOG"],
            columns=["AAPL", "MSFT", "GOOG"]
        )

        allocation = optimizer.optimize(cov)

        print(f" -> Weights: {allocation.weights}")
        print(f" -> Risk Contributions: {allocation.risk_contributions}")
        print(f" -> Diversification Ratio: {allocation.diversification_ratio:.2f}")
        print(" -> Risk Parity: OK")

    def test_volatility_arb(self):
        """Test volatility arbitrage."""
        print("\n[TEST] Volatility Arb...")
        from strategies.volatility_arb import get_volatility_arb

        vol_arb = get_volatility_arb()

        signal = vol_arb.generate_signal("AAPL", self.prices, implied_vol=0.25)

        self.assertIsNotNone(signal.realized_vol)
        print(f" -> RV: {signal.realized_vol:.2%}, IV: {signal.implied_vol:.2%}")
        print(f" -> Signal: {signal.signal}")
        print(" -> Volatility Arb: OK")

    def test_event_driven(self):
        """Test event-driven alpha."""
        print("\n[TEST] Event-Driven Alpha...")
        from strategies.event_driven import get_event_driven_alpha

        engine = get_event_driven_alpha()

        signal = engine.earnings_surprise_signal(
            symbol="AAPL",
            expected_eps=1.50,
            actual_eps=1.65
        )

        self.assertEqual(signal.direction, 1)  # Positive surprise
        print(f" -> Direction: {signal.direction}, Confidence: {signal.confidence:.2f}")
        print(" -> Event-Driven Alpha: OK")

    def test_deep_ensemble(self):
        """Test deep learning ensemble."""
        print("\n[TEST] Deep Ensemble...")
        from ml.deep_ensemble import get_deep_ensemble

        ensemble = get_deep_ensemble()

        features = np.random.randn(20, 10)
        prediction = ensemble.predict(features, "AAPL")

        self.assertGreaterEqual(prediction.prediction, -1)
        self.assertLessEqual(prediction.prediction, 1)
        print(f" -> Prediction: {prediction.prediction:.4f}")
        print(f" -> LSTM: {prediction.lstm_pred:.4f}")
        print(f" -> Transformer: {prediction.transformer_pred:.4f}")
        print(f" -> CNN: {prediction.cnn_pred:.4f}")
        print(" -> Deep Ensemble: OK")

    def test_streaming_engine(self):
        """Test streaming engine."""
        print("\n[TEST] Streaming Engine...")
        from engine.streaming_engine import get_streaming_engine, MarketTick
        import time

        engine = get_streaming_engine()

        # Simulate ticks
        for i in range(50):
            tick = MarketTick(
                symbol="AAPL",
                price=150 + np.random.randn() * 0.5,
                volume=np.random.randint(100, 1000),
                timestamp=time.time()
            )
            engine.on_tick(tick)

        stats = engine.get_stats()
        print(f" -> Ticks processed: {stats['ticks_processed']}")
        print(f" -> Signals generated: {stats['signals_generated']}")
        print(" -> Streaming Engine: OK")

    def test_ultra_fast_router(self):
        """Test ultra-fast router."""
        print("\n[TEST] Ultra-Fast Router...")
        from execution.ultra_fast_router import get_ultra_fast_router, OrderType
        import time

        router = get_ultra_fast_router()

        # Submit order
        order = router.submit_order(
            symbol="AAPL",
            side="BUY",
            quantity=100,
            order_type=OrderType.MARKET
        )

        time.sleep(0.1)  # Wait for execution

        report = router.get_execution_report(order.order_id)
        stats = router.get_latency_stats()

        if stats:
            print(f" -> Mean latency: {stats.get('mean_ms', 0):.2f}ms")
            print(f" -> P99 latency: {stats.get('p99_ms', 0):.2f}ms")
        print(" -> Ultra-Fast Router: OK")


if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 11: True Top 1% Hedge Fund Verification")
    print("=" * 60)
    unittest.main(verbosity=2)
