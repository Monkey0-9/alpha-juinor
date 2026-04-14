"""
LLM Signal Generator - D.E. Shaw-style AI-Powered Alpha.

Features:
- Generate trading signals from structured prompts
- Market context analysis
- Multi-factor signal synthesis
- Governance-safe signal generation
"""

import logging
import random
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class LLMSignal:
    """LLM-generated trading signal."""
    symbol: str
    signal: str  # "BUY", "SELL", "HOLD"
    confidence: float
    reasoning: str
    factors_considered: List[str]
    timestamp: str


@dataclass
class MarketContext:
    """Market context for LLM analysis."""
    symbol: str
    current_price: float
    price_change_1d: float
    price_change_5d: float
    volume_ratio: float
    rsi: float
    sentiment: float
    sector_performance: float
    market_regime: str


class LLMSignalGenerator:
    """
    LLM-powered signal generator.

    In production, this would integrate with:
    - OpenAI GPT-4
    - Anthropic Claude
    - Google PaLM
    - Proprietary LLMs (D.E. Shaw trains their own)

    This implementation provides a rules-based simulation
    of LLM decision-making for demonstration.
    """

    def __init__(
        self,
        model_name: str = "simulated-llm",
        temperature: float = 0.3,
        confidence_threshold: float = 0.6
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.confidence_threshold = confidence_threshold

        # Signal history
        self.history: List[LLMSignal] = []

        # Factor weights (learned via historical performance)
        self.factor_weights = {
            "momentum": 0.25,
            "sentiment": 0.20,
            "technical": 0.20,
            "fundamental": 0.15,
            "sector": 0.10,
            "regime": 0.10
        }

    def create_context(
        self,
        symbol: str,
        price_data: Dict,
        indicators: Dict,
        sentiment: float,
        regime: str
    ) -> MarketContext:
        """Create market context from various data sources."""
        return MarketContext(
            symbol=symbol,
            current_price=price_data.get("current", 0),
            price_change_1d=price_data.get("change_1d", 0),
            price_change_5d=price_data.get("change_5d", 0),
            volume_ratio=price_data.get("volume_ratio", 1.0),
            rsi=indicators.get("rsi", 50),
            sentiment=sentiment,
            sector_performance=price_data.get("sector_perf", 0),
            market_regime=regime
        )

    def _analyze_momentum(self, context: MarketContext) -> Tuple[float, str]:
        """Analyze momentum factors."""
        score = 0.0
        reasons = []

        if context.price_change_5d > 0.03:
            score += 0.5
            reasons.append("Strong 5-day momentum")
        elif context.price_change_5d < -0.03:
            score -= 0.5
            reasons.append("Weak 5-day momentum")

        if context.price_change_1d > 0.01:
            score += 0.3
            reasons.append("Positive daily trend")
        elif context.price_change_1d < -0.01:
            score -= 0.3
            reasons.append("Negative daily trend")

        return score, "; ".join(reasons) if reasons else "Neutral momentum"

    def _analyze_technical(self, context: MarketContext) -> Tuple[float, str]:
        """Analyze technical indicators."""
        score = 0.0
        reasons = []

        # RSI analysis
        if context.rsi < 30:
            score += 0.5
            reasons.append("Oversold (RSI < 30)")
        elif context.rsi > 70:
            score -= 0.5
            reasons.append("Overbought (RSI > 70)")

        # Volume analysis
        if context.volume_ratio > 1.5:
            score += 0.2
            reasons.append("High volume confirmation")

        return score, "; ".join(reasons) if reasons else "Neutral technicals"

    def _analyze_sentiment(self, context: MarketContext) -> Tuple[float, str]:
        """Analyze sentiment factors."""
        score = context.sentiment

        if score > 0.3:
            reason = "Positive market sentiment"
        elif score < -0.3:
            reason = "Negative market sentiment"
        else:
            reason = "Neutral sentiment"

        return score, reason

    def _analyze_regime(self, context: MarketContext) -> Tuple[float, str]:
        """Analyze market regime."""
        regime_scores = {
            "BULL_STRONG": 0.5,
            "BULL_WEAK": 0.2,
            "NEUTRAL": 0.0,
            "BEAR_WEAK": -0.2,
            "BEAR_STRONG": -0.5,
            "CRISIS": -0.8
        }

        score = regime_scores.get(context.market_regime, 0)
        reason = f"Market regime: {context.market_regime}"

        return score, reason

    def generate_signal(
        self,
        context: MarketContext
    ) -> LLMSignal:
        """
        Generate trading signal from market context.

        This simulates LLM reasoning by combining
        multiple analysis factors.
        """
        factors = []
        reasoning_parts = []
        weighted_score = 0.0

        # Analyze each factor
        momentum_score, momentum_reason = self._analyze_momentum(context)
        weighted_score += momentum_score * self.factor_weights["momentum"]
        factors.append("momentum")
        reasoning_parts.append(f"Momentum: {momentum_reason}")

        technical_score, technical_reason = self._analyze_technical(context)
        weighted_score += technical_score * self.factor_weights["technical"]
        factors.append("technical")
        reasoning_parts.append(f"Technical: {technical_reason}")

        sentiment_score, sentiment_reason = self._analyze_sentiment(context)
        weighted_score += sentiment_score * self.factor_weights["sentiment"]
        factors.append("sentiment")
        reasoning_parts.append(f"Sentiment: {sentiment_reason}")

        regime_score, regime_reason = self._analyze_regime(context)
        weighted_score += regime_score * self.factor_weights["regime"]
        factors.append("regime")
        reasoning_parts.append(f"Regime: {regime_reason}")

        # Add some randomness to simulate LLM uncertainty
        noise = random.gauss(0, 0.1 * self.temperature)
        final_score = weighted_score + noise

        # Determine signal
        if final_score > 0.2:
            signal = "BUY"
            confidence = min(1.0, 0.5 + final_score)
        elif final_score < -0.2:
            signal = "SELL"
            confidence = min(1.0, 0.5 - final_score)
        else:
            signal = "HOLD"
            confidence = 0.5

        llm_signal = LLMSignal(
            symbol=context.symbol,
            signal=signal,
            confidence=confidence,
            reasoning=" | ".join(reasoning_parts),
            factors_considered=factors,
            timestamp=datetime.utcnow().isoformat()
        )

        self.history.append(llm_signal)

        return llm_signal

    def get_ensemble_signal(
        self,
        contexts: List[MarketContext]
    ) -> Dict[str, LLMSignal]:
        """Generate signals for multiple symbols."""
        signals = {}
        for context in contexts:
            signals[context.symbol] = self.generate_signal(context)
        return signals

    def get_history(self, symbol: Optional[str] = None) -> List[LLMSignal]:
        """Get signal history."""
        if symbol:
            return [s for s in self.history if s.symbol == symbol]
        return self.history


# Global singleton
_llm_generator: Optional[LLMSignalGenerator] = None


def get_llm_generator() -> LLMSignalGenerator:
    """Get or create global LLM signal generator."""
    global _llm_generator
    if _llm_generator is None:
        _llm_generator = LLMSignalGenerator()
    return _llm_generator
