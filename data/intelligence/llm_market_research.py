"""
LLM-Powered Market Research Service.

Provides real-time market sentiment analysis and fundamental research
using Google AI (Gemini) and DeepSeek LLM APIs.

Features:
- Ticker-specific sentiment analysis
- Fundamental company research
- News headline interpretation
- Broad market outlook analysis
- Response caching for efficiency
"""

import logging
import json
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from dataclasses import dataclass

from data.providers.llm_provider import get_llm_provider, LLMProvider

logger = logging.getLogger(__name__)


@dataclass
class SentimentResult:
    """Market sentiment analysis result."""
    ticker: str
    sentiment_score: float  # -1 (bearish) to 1 (bullish)
    confidence: float  # 0 to 1
    summary: str
    key_factors: List[str]
    timestamp: str
    provider: str


@dataclass
class FundamentalResearch:
    """Fundamental research result."""
    ticker: str
    outlook: str  # "bullish", "bearish", "neutral"
    target_direction: str  # "up", "down", "sideways"
    key_strengths: List[str]
    key_risks: List[str]
    sector_position: str
    summary: str
    timestamp: str


class LLMMarketResearch:
    """
    LLM-powered market research and sentiment analysis.

    Usage:
        research = LLMMarketResearch()
        sentiment = research.get_market_sentiment("AAPL")
        fundamentals = research.get_fundamental_research("MSFT")
    """

    def __init__(self, llm_provider: Optional[LLMProvider] = None):
        """Initialize with LLM provider."""
        self.llm = llm_provider or get_llm_provider()
        self._sentiment_cache: Dict[str, SentimentResult] = {}
        self._cache_ttl = timedelta(minutes=30)

    def get_market_sentiment(self, ticker: str) -> SentimentResult:
        """
        Get real-time market sentiment for a ticker.

        Uses LLM to analyze current market conditions and sentiment.

        Args:
            ticker: Stock ticker symbol (e.g., "AAPL")

        Returns:
            SentimentResult with score, confidence, and analysis
        """
        # Check cache
        cache_key = f"sentiment_{ticker}"
        if cache_key in self._sentiment_cache:
            cached = self._sentiment_cache[cache_key]
            cached_time = datetime.fromisoformat(cached.timestamp)
            if datetime.utcnow() - cached_time < self._cache_ttl:
                return cached

        prompt = f"""Analyze the current market sentiment for {ticker}.

Consider:
1. Recent price action and momentum
2. Market news and events
3. Sector performance
4. Institutional activity
5. Technical indicators

Respond with JSON:
{{
    "sentiment_score": <float -1 to 1, where -1=very bearish, 0=neutral, 1=very bullish>,
    "confidence": <float 0 to 1>,
    "summary": "<one sentence summary>",
    "key_factors": ["<factor1>", "<factor2>", "<factor3>"]
}}"""

        result = self.llm.generate_json(prompt)

        if result:
            sentiment = SentimentResult(
                ticker=ticker,
                sentiment_score=float(result.get("sentiment_score", 0)),
                confidence=float(result.get("confidence", 0.5)),
                summary=result.get("summary", "Unable to determine sentiment"),
                key_factors=result.get("key_factors", []),
                timestamp=datetime.utcnow().isoformat(),
                provider=self.llm.available_providers[0] if self.llm.providers else "none"
            )
        else:
            # Fallback to neutral if LLM fails
            sentiment = SentimentResult(
                ticker=ticker,
                sentiment_score=0.0,
                confidence=0.3,
                summary="LLM analysis unavailable - defaulting to neutral",
                key_factors=["LLM unavailable"],
                timestamp=datetime.utcnow().isoformat(),
                provider="fallback"
            )

        # Cache result
        self._sentiment_cache[cache_key] = sentiment
        logger.info(f"LLM sentiment for {ticker}: {sentiment.sentiment_score:.2f} ({sentiment.summary})")

        return sentiment

    def get_fundamental_research(self, ticker: str) -> FundamentalResearch:
        """
        Get fundamental research and analysis for a ticker.

        Analyzes company fundamentals, competitive position, and outlook.

        Args:
            ticker: Stock ticker symbol

        Returns:
            FundamentalResearch with outlook and key factors
        """
        prompt = f"""Provide fundamental analysis for {ticker}.

Analyze:
1. Business model and competitive advantages
2. Financial health and growth prospects
3. Industry position and trends
4. Key risks and opportunities
5. Valuation considerations

Respond with JSON:
{{
    "outlook": "<bullish|bearish|neutral>",
    "target_direction": "<up|down|sideways>",
    "key_strengths": ["<strength1>", "<strength2>"],
    "key_risks": ["<risk1>", "<risk2>"],
    "sector_position": "<leader|challenger|follower>",
    "summary": "<two sentence summary>"
}}"""

        result = self.llm.generate_json(prompt)

        if result:
            return FundamentalResearch(
                ticker=ticker,
                outlook=result.get("outlook", "neutral"),
                target_direction=result.get("target_direction", "sideways"),
                key_strengths=result.get("key_strengths", []),
                key_risks=result.get("key_risks", []),
                sector_position=result.get("sector_position", "unknown"),
                summary=result.get("summary", "Analysis unavailable"),
                timestamp=datetime.utcnow().isoformat()
            )
        else:
            return FundamentalResearch(
                ticker=ticker,
                outlook="neutral",
                target_direction="sideways",
                key_strengths=[],
                key_risks=["LLM analysis unavailable"],
                sector_position="unknown",
                summary="Unable to perform fundamental analysis",
                timestamp=datetime.utcnow().isoformat()
            )

    def analyze_news_headlines(
        self,
        ticker: str,
        headlines: List[str]
    ) -> Dict[str, Any]:
        """
        Analyze news headlines for market impact.

        Args:
            ticker: Stock ticker symbol
            headlines: List of recent news headlines

        Returns:
            Dict with sentiment score, market impact, and analysis
        """
        if not headlines:
            return {
                "sentiment_score": 0.0,
                "market_impact": "none",
                "summary": "No headlines to analyze",
                "bullish_headlines": [],
                "bearish_headlines": []
            }

        headlines_text = "\n".join(f"- {h}" for h in headlines[:10])

        prompt = f"""Analyze these news headlines for {ticker}:

{headlines_text}

Determine the overall sentiment and market impact.

Respond with JSON:
{{
    "sentiment_score": <float -1 to 1>,
    "market_impact": "<high|medium|low|none>",
    "summary": "<one sentence summary>",
    "bullish_headlines": [<indices of bullish headlines>],
    "bearish_headlines": [<indices of bearish headlines>]
}}"""

        result = self.llm.generate_json(prompt)

        if result:
            return {
                "sentiment_score": float(result.get("sentiment_score", 0)),
                "market_impact": result.get("market_impact", "low"),
                "summary": result.get("summary", ""),
                "bullish_headlines": result.get("bullish_headlines", []),
                "bearish_headlines": result.get("bearish_headlines", [])
            }
        else:
            return {
                "sentiment_score": 0.0,
                "market_impact": "unknown",
                "summary": "Unable to analyze headlines",
                "bullish_headlines": [],
                "bearish_headlines": []
            }

    def get_market_outlook(self) -> Dict[str, Any]:
        """
        Get broad market outlook and conditions.

        Returns:
            Dict with market sentiment, key levels, and outlook
        """
        prompt = """Analyze the current broad market conditions.

Consider:
1. Major indices (SPY, QQQ, DIA)
2. Market breadth and internals
3. Volatility (VIX)
4. Sector rotation
5. Economic indicators

Respond with JSON:
{
    "overall_sentiment": "<bullish|bearish|neutral>",
    "sentiment_score": <float -1 to 1>,
    "volatility_regime": "<low|normal|high|extreme>",
    "leading_sectors": ["<sector1>", "<sector2>"],
    "lagging_sectors": ["<sector1>", "<sector2>"],
    "key_levels": {"SPY_support": <float>, "SPY_resistance": <float>},
    "outlook": "<one sentence market outlook>",
    "risk_factors": ["<risk1>", "<risk2>"]
}"""

        result = self.llm.generate_json(prompt)

        if result:
            return result
        else:
            return {
                "overall_sentiment": "neutral",
                "sentiment_score": 0.0,
                "volatility_regime": "normal",
                "leading_sectors": [],
                "lagging_sectors": [],
                "key_levels": {},
                "outlook": "Market outlook unavailable",
                "risk_factors": ["LLM analysis unavailable"]
            }

    def get_sentiment_score(self, ticker: str) -> float:
        """
        Quick sentiment score for a ticker (-1 to 1).

        Convenience method for integration with existing systems.
        """
        result = self.get_market_sentiment(ticker)
        return result.sentiment_score

    def clear_cache(self):
        """Clear the sentiment cache."""
        self._sentiment_cache.clear()


# ---------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------
_research: Optional[LLMMarketResearch] = None


def get_llm_market_research() -> LLMMarketResearch:
    """Get or create the global LLM market research instance."""
    global _research
    if _research is None:
        _research = LLMMarketResearch()
    return _research


# ---------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    logging.basicConfig(level=logging.INFO)

    research = get_llm_market_research()

    print("Testing market sentiment...")
    sentiment = research.get_market_sentiment("AAPL")
    print(f"AAPL Sentiment: {sentiment.sentiment_score:.2f}")
    print(f"Summary: {sentiment.summary}")
    print(f"Factors: {sentiment.key_factors}")

    print("\nTesting market outlook...")
    outlook = research.get_market_outlook()
    print(f"Market: {outlook.get('overall_sentiment')} ({outlook.get('sentiment_score', 0):.2f})")
    print(f"Outlook: {outlook.get('outlook')}")
