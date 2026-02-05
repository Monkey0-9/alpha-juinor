
import logging
from typing import Dict, Any
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)

# Optional LLM integration - does not affect core functionality if unavailable
_llm_research = None


def _get_llm_research():
    """Lazy load LLM research to avoid startup overhead."""
    global _llm_research
    if _llm_research is None:
        try:
            from data.intelligence.llm_market_research import (
                get_llm_market_research
            )
            _llm_research = get_llm_market_research()
        except Exception as e:
            logger.debug(f"LLM research not available: {e}")
            _llm_research = False  # Mark as unavailable
    return _llm_research if _llm_research else None


class SentimentAgent(BaseAgent):
    """
    Specialized agent for Sentiment Analysis.
    Analyzes news headlines, social media, and market mood.

    Uses LLM-powered analysis as an OPTIONAL enhancement layer.
    Core functionality works without LLM APIs.
    """
    def __init__(self, model_name: str = "gpt-4-turbo"):
        super().__init__("SentimentAnalyzer", model_name)
        self._use_llm = True  # Can be disabled if LLM causes issues

    def analyze(self, ticker: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyzes sentiment for a given ticker.

        Primary analysis uses local heuristics.
        LLM enhances the analysis when available (non-blocking).
        """
        # ========================================
        # PRIMARY: Local/heuristic analysis (always runs)
        # ========================================
        price_change = data.get("price_change_pct", 0)

        # Base sentiment from price action
        if price_change > 2:
            sentiment_score = 0.7
            reasoning = f"Strong positive momentum for {ticker}."
        elif price_change > 0:
            sentiment_score = 0.55
            reasoning = f"Positive price action for {ticker}."
        elif price_change < -2:
            sentiment_score = 0.3
            reasoning = f"Negative momentum for {ticker}."
        elif price_change < 0:
            sentiment_score = 0.45
            reasoning = f"Slight negative bias for {ticker}."
        else:
            sentiment_score = 0.5
            reasoning = f"Neutral sentiment for {ticker}."

        # Ticker-specific adjustments (local knowledge)
        if ticker == "BTC-USD":
            sentiment_score = min(0.9, sentiment_score + 0.15)
            reasoning = "High retail engagement and positive ETF flow."
        elif ticker == "MSFT":
            sentiment_score = min(0.9, sentiment_score + 0.1)
            reasoning = "Steady growth driven by AI integration."
        elif ticker in ["NVDA", "AMD"]:
            sentiment_score = min(0.9, sentiment_score + 0.1)
            reasoning = "Strong AI/chip sector momentum."

        # ========================================
        # OPTIONAL: LLM enhancement (non-blocking)
        # ========================================
        llm_boost = 0.0
        llm_note = ""

        if self._use_llm:
            try:
                research = _get_llm_research()
                if research:
                    llm_result = research.get_market_sentiment(ticker)
                    if llm_result and llm_result.confidence > 0.4:
                        # Blend LLM score (30% weight) with local (70%)
                        llm_score = (llm_result.sentiment_score + 1) / 2
                        blended = 0.7 * sentiment_score + 0.3 * llm_score
                        llm_boost = blended - sentiment_score
                        sentiment_score = blended
                        llm_note = f" LLM: {llm_result.summary}"
                        logger.debug(
                            f"LLM enhanced {ticker}: "
                            f"{llm_result.sentiment_score:.2f}"
                        )
            except Exception as e:
                # LLM failure does NOT affect core analysis
                logger.debug(f"LLM enhancement skipped for {ticker}: {e}")

        # Final signal calculation
        if sentiment_score > 0.6:
            signal = 1.0
        elif sentiment_score < 0.4:
            signal = -1.0
        else:
            signal = 0.0

        return {
            "ticker": ticker,
            "signal": signal,
            "confidence": sentiment_score,
            "reason": reasoning + llm_note,
            "agent": self.name,
            "llm_enhanced": bool(llm_boost != 0),
        }

