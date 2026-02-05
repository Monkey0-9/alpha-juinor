"""
Model Orchestrator - Single Interface to ALL ML Models.

This is the CRITICAL missing piece - connects all existing models
to actual trading decisions.

Features:
- Parallel execution of all models
- Confidence-weighted ensemble
- Model health monitoring
- Graceful fallbacks
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

logger = logging.getLogger(__name__)


@dataclass
class ModelPrediction:
    """Single model prediction."""
    model_name: str
    signal: float  # -1 to 1
    confidence: float  # 0 to 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    success: bool = True
    error: Optional[str] = None


@dataclass
class EnsemblePrediction:
    """Combined prediction from all models."""
    symbol: str
    final_signal: float
    final_confidence: float
    direction: str  # "LONG", "SHORT", "NEUTRAL"
    model_predictions: Dict[str, ModelPrediction] = field(default_factory=dict)
    weights_used: Dict[str, float] = field(default_factory=dict)
    total_latency_ms: float = 0.0


class ModelOrchestrator:
    """
    Orchestrates ALL ML models for trading decisions.

    Integrates:
    - HMM Predictor (regime detection)
    - Deep Ensemble (LSTM + Transformer + CNN)
    - RL Trading Agent (position recommendations)
    - LLM Signal Generator (market context)
    - NLP Sentiment (news analysis)
    - Genetic Optimizer (parameter evolution)
    """

    def __init__(self, max_workers: int = 8):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        # Model weights (adaptive)
        self.model_weights = {
            "hmm": 0.20,
            "deep_ensemble": 0.25,
            "rl_agent": 0.15,
            "llm_signals": 0.15,
            "nlp_sentiment": 0.10,
            "momentum": 0.10,
            "technical": 0.05
        }

        # Model health tracking
        self.model_health = {name: 1.0 for name in self.model_weights}
        self.model_errors = {name: 0 for name in self.model_weights}

        # Lazy-loaded models
        self._hmm = None
        self._deep_ensemble = None
        self._rl_agent = None
        self._llm_generator = None
        self._nlp_analyzer = None

    @property
    def hmm(self):
        """Lazy load HMM predictor."""
        if self._hmm is None:
            try:
                from ml.hmm_predictor import get_hmm_predictor
                self._hmm = get_hmm_predictor()
            except Exception as e:
                logger.warning(f"HMM not available: {e}")
        return self._hmm

    @property
    def deep_ensemble(self):
        """Lazy load deep ensemble."""
        if self._deep_ensemble is None:
            try:
                from ml.deep_ensemble import get_deep_ensemble
                self._deep_ensemble = get_deep_ensemble()
            except Exception as e:
                logger.warning(f"Deep ensemble not available: {e}")
        return self._deep_ensemble

    @property
    def rl_agent(self):
        """Lazy load RL agent."""
        if self._rl_agent is None:
            try:
                from ml.rl_trading_agent import get_rl_agent
                self._rl_agent = get_rl_agent()
            except Exception as e:
                logger.warning(f"RL agent not available: {e}")
        return self._rl_agent

    @property
    def llm_generator(self):
        """Lazy load LLM generator."""
        if self._llm_generator is None:
            try:
                from ml.llm_signals import get_llm_generator
                self._llm_generator = get_llm_generator()
            except Exception as e:
                logger.warning(f"LLM generator not available: {e}")
        return self._llm_generator

    @property
    def nlp_analyzer(self):
        """Lazy load NLP analyzer."""
        if self._nlp_analyzer is None:
            try:
                from ml.nlp_sentiment import get_nlp_analyzer
                self._nlp_analyzer = get_nlp_analyzer()
            except Exception as e:
                logger.warning(f"NLP analyzer not available: {e}")
        return self._nlp_analyzer

    def _run_hmm(
        self,
        symbol: str,
        returns: pd.Series
    ) -> ModelPrediction:
        """Run HMM regime prediction."""
        start = time.time()
        try:
            if self.hmm is None or len(returns) < 50:
                return ModelPrediction("hmm", 0.0, 0.0, success=False)

            # Fit and predict
            self.hmm.fit(returns)
            result = self.hmm.predict(returns)

            # Convert regime to signal
            state = result.current_state.name
            if "BULL" in state:
                signal = 0.8 if "HIGH" in state else 0.5
            elif "BEAR" in state:
                signal = -0.8 if "HIGH" in state else -0.5
            else:
                signal = 0.0

            return ModelPrediction(
                model_name="hmm",
                signal=signal,
                confidence=result.confidence,
                metadata={"state": state, "probs": result.state_probabilities},
                latency_ms=(time.time() - start) * 1000
            )
        except Exception as e:
            self.model_errors["hmm"] += 1
            return ModelPrediction("hmm", 0.0, 0.0, success=False, error=str(e))

    def _run_deep_ensemble(
        self,
        symbol: str,
        features: np.ndarray
    ) -> ModelPrediction:
        """Run deep ensemble prediction."""
        start = time.time()
        try:
            if self.deep_ensemble is None:
                return ModelPrediction("deep_ensemble", 0.0, 0.0, success=False)

            result = self.deep_ensemble.predict(features, symbol)

            return ModelPrediction(
                model_name="deep_ensemble",
                signal=float(result.prediction),
                confidence=float(result.confidence),
                metadata={"model_weights": result.model_weights},
                latency_ms=(time.time() - start) * 1000
            )
        except Exception as e:
            self.model_errors["deep_ensemble"] += 1
            return ModelPrediction("deep_ensemble", 0.0, 0.0, success=False, error=str(e))

    def _run_rl_agent(
        self,
        symbol: str,
        state: np.ndarray
    ) -> ModelPrediction:
        """Run RL agent prediction."""
        start = time.time()
        try:
            if self.rl_agent is None:
                return ModelPrediction("rl_agent", 0.0, 0.0, success=False)

            action = self.rl_agent.select_action(state, epsilon=0.0)

            # Action: 0=HOLD, 1=BUY, 2=SELL
            signal_map = {0: 0.0, 1: 1.0, 2: -1.0}
            signal = signal_map.get(action, 0.0)

            return ModelPrediction(
                model_name="rl_agent",
                signal=signal,
                confidence=0.7,  # RL doesn't give confidence
                metadata={"action": action},
                latency_ms=(time.time() - start) * 1000
            )
        except Exception as e:
            self.model_errors["rl_agent"] += 1
            return ModelPrediction("rl_agent", 0.0, 0.0, success=False, error=str(e))

    def _run_llm_signals(
        self,
        symbol: str,
        context: Dict
    ) -> ModelPrediction:
        """Run LLM signal generation."""
        start = time.time()
        try:
            if self.llm_generator is None:
                return ModelPrediction("llm_signals", 0.0, 0.0, success=False)

            from ml.llm_signals import MarketContext

            market_ctx = MarketContext(
                symbol=symbol,
                current_price=context.get("price", 100),
                price_change_1d=context.get("return_1d", 0),
                price_change_5d=context.get("return_5d", 0),
                volume_ratio=context.get("volume_ratio", 1.0),
                rsi=context.get("rsi", 50),
                sentiment=context.get("sentiment", 0),
                sector_performance=context.get("sector_perf", 0),
                market_regime=context.get("regime", "NEUTRAL")
            )

            result = self.llm_generator.generate_signal(market_ctx)

            signal_map = {"BUY": 1.0, "SELL": -1.0, "HOLD": 0.0}
            signal = signal_map.get(result.signal, 0.0)

            return ModelPrediction(
                model_name="llm_signals",
                signal=signal,
                confidence=result.confidence,
                metadata={"reasoning": result.reasoning[:100]},
                latency_ms=(time.time() - start) * 1000
            )
        except Exception as e:
            self.model_errors["llm_signals"] += 1
            return ModelPrediction("llm_signals", 0.0, 0.0, success=False, error=str(e))

    def _run_nlp_sentiment(
        self,
        symbol: str,
        news_text: Optional[str] = None
    ) -> ModelPrediction:
        """Run NLP sentiment analysis."""
        start = time.time()
        try:
            if self.nlp_analyzer is None or not news_text:
                return ModelPrediction("nlp_sentiment", 0.0, 0.0, success=False)

            result = self.nlp_analyzer.analyze_text(news_text, source="news")

            return ModelPrediction(
                model_name="nlp_sentiment",
                signal=result.sentiment_score,
                confidence=result.confidence,
                metadata={"keywords": result.keywords},
                latency_ms=(time.time() - start) * 1000
            )
        except Exception as e:
            self.model_errors["nlp_sentiment"] += 1
            return ModelPrediction("nlp_sentiment", 0.0, 0.0, success=False, error=str(e))

    def _run_momentum(
        self,
        returns: pd.Series
    ) -> ModelPrediction:
        """Calculate momentum signal."""
        start = time.time()
        try:
            if len(returns) < 20:
                return ModelPrediction("momentum", 0.0, 0.5, success=True)

            mom_5 = returns.iloc[-5:].sum()
            mom_20 = returns.iloc[-20:].sum()
            mom_60 = returns.sum() if len(returns) >= 60 else returns.sum()

            signal = 0.5 * mom_5 + 0.3 * mom_20 + 0.2 * mom_60
            signal = float(np.clip(signal * 10, -1, 1))

            return ModelPrediction(
                model_name="momentum",
                signal=signal,
                confidence=0.6,
                metadata={"mom_5": mom_5, "mom_20": mom_20},
                latency_ms=(time.time() - start) * 1000
            )
        except Exception as e:
            return ModelPrediction("momentum", 0.0, 0.0, success=False, error=str(e))

    def _run_technical(
        self,
        prices: pd.Series
    ) -> ModelPrediction:
        """Calculate technical signal."""
        start = time.time()
        try:
            if len(prices) < 20:
                return ModelPrediction("technical", 0.0, 0.5, success=True)

            ma_10 = prices.rolling(10).mean().iloc[-1]
            ma_20 = prices.rolling(20).mean().iloc[-1]
            current = prices.iloc[-1]

            score = 0.0
            if current > ma_10:
                score += 0.3
            if current > ma_20:
                score += 0.3
            if ma_10 > ma_20:
                score += 0.2

            # RSI
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain.iloc[-1] / (loss.iloc[-1] + 1e-10)
            rsi = 100 - (100 / (1 + rs))

            if rsi < 30:
                score += 0.2
            elif rsi > 70:
                score -= 0.2

            return ModelPrediction(
                model_name="technical",
                signal=float(np.clip(score, -1, 1)),
                confidence=0.5,
                metadata={"rsi": rsi, "ma_signal": ma_10 > ma_20},
                latency_ms=(time.time() - start) * 1000
            )
        except Exception as e:
            return ModelPrediction("technical", 0.0, 0.0, success=False, error=str(e))

    def predict(
        self,
        symbol: str,
        prices: pd.Series,
        features: Optional[np.ndarray] = None,
        news_text: Optional[str] = None,
        context: Optional[Dict] = None
    ) -> EnsemblePrediction:
        """
        Generate ensemble prediction using ALL models.

        This is the main entry point that combines all model outputs.
        """
        start_time = time.time()
        predictions = {}

        returns = prices.pct_change().dropna()
        if features is None:
            # Generate features from prices
            features = np.column_stack([
                returns.values[-20:] if len(returns) >= 20 else np.zeros(20),
                np.ones(20) * (prices.iloc[-1] / prices.mean() if len(prices) > 0 else 1)
            ])

        # Build context if not provided
        if context is None:
            context = {
                "price": prices.iloc[-1] if len(prices) > 0 else 100,
                "return_1d": returns.iloc[-1] if len(returns) > 0 else 0,
                "return_5d": returns.iloc[-5:].sum() if len(returns) >= 5 else 0,
                "volume_ratio": 1.0,
                "rsi": 50,
                "sentiment": 0,
                "regime": "NEUTRAL"
            }

        # Run all models in parallel
        futures = {}

        futures["hmm"] = self.executor.submit(self._run_hmm, symbol, returns)
        futures["deep_ensemble"] = self.executor.submit(
            self._run_deep_ensemble, symbol, features
        )
        futures["rl_agent"] = self.executor.submit(
            self._run_rl_agent, symbol, features.flatten()[:10]
        )
        futures["llm_signals"] = self.executor.submit(
            self._run_llm_signals, symbol, context
        )
        futures["nlp_sentiment"] = self.executor.submit(
            self._run_nlp_sentiment, symbol, news_text
        )
        futures["momentum"] = self.executor.submit(self._run_momentum, returns)
        futures["technical"] = self.executor.submit(self._run_technical, prices)

        # Collect results
        for name, future in futures.items():
            try:
                predictions[name] = future.result(timeout=5)
            except Exception as e:
                predictions[name] = ModelPrediction(
                    name, 0.0, 0.0, success=False, error=str(e)
                )

        # Calculate weighted ensemble signal
        total_weight = 0.0
        weighted_signal = 0.0
        weights_used = {}

        for name, pred in predictions.items():
            if pred.success and pred.confidence > 0:
                # Adjust weight by model health
                effective_weight = (
                    self.model_weights.get(name, 0.1) *
                    self.model_health.get(name, 1.0) *
                    pred.confidence
                )
                weighted_signal += pred.signal * effective_weight
                total_weight += effective_weight
                weights_used[name] = effective_weight

        if total_weight > 0:
            final_signal = weighted_signal / total_weight
        else:
            final_signal = 0.0

        # Determine direction
        if final_signal > 0.2:
            direction = "LONG"
        elif final_signal < -0.2:
            direction = "SHORT"
        else:
            direction = "NEUTRAL"

        # Calculate confidence
        successful_models = sum(1 for p in predictions.values() if p.success)
        final_confidence = (successful_models / len(predictions)) * min(abs(final_signal) + 0.5, 1.0)

        return EnsemblePrediction(
            symbol=symbol,
            final_signal=float(final_signal),
            final_confidence=float(final_confidence),
            direction=direction,
            model_predictions=predictions,
            weights_used=weights_used,
            total_latency_ms=(time.time() - start_time) * 1000
        )

    def batch_predict(
        self,
        market_data: Dict[str, pd.Series],
        news_data: Optional[Dict[str, str]] = None
    ) -> Dict[str, EnsemblePrediction]:
        """Batch prediction for multiple symbols."""
        results = {}

        for symbol, prices in market_data.items():
            news = news_data.get(symbol) if news_data else None
            results[symbol] = self.predict(symbol, prices, news_text=news)

        return results

    def get_model_health(self) -> Dict[str, float]:
        """Get health status of all models."""
        return self.model_health.copy()

    def update_model_health(self, name: str, success: bool):
        """Update model health based on prediction success."""
        if name in self.model_health:
            if success:
                self.model_health[name] = min(1.0, self.model_health[name] + 0.01)
            else:
                self.model_health[name] = max(0.1, self.model_health[name] - 0.1)


# Global singleton
_orchestrator: Optional[ModelOrchestrator] = None


def get_model_orchestrator() -> ModelOrchestrator:
    """Get or create global model orchestrator."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = ModelOrchestrator()
    return _orchestrator
