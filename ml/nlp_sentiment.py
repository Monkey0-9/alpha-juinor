"""
NLP Sentiment Analyzer - Point72/D.E. Shaw-style Text Analysis.

Features:
- News headline sentiment
- Social media sentiment
- Earnings call transcript analysis
- SEC filing text extraction
"""

import logging
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import time

logger = logging.getLogger(__name__)


@dataclass
class SentimentResult:
    """Sentiment analysis result."""
    text: str
    sentiment_score: float  # -1 to 1
    confidence: float
    keywords: List[str]
    source: str


@dataclass
class AggregatedSentiment:
    """Aggregated sentiment for a symbol."""
    symbol: str
    overall_sentiment: float
    news_sentiment: float
    social_sentiment: float
    filing_sentiment: float
    sample_count: int
    signal: str  # "BULLISH", "BEARISH", "NEUTRAL"


class NLPSentimentAnalyzer:
    """
    Financial NLP sentiment analysis.

    Techniques:
    - Keyword-based scoring
    - Negation handling
    - Financial-specific lexicon
    - Context-aware weighting
    """

    def __init__(self):
        # Financial-specific sentiment lexicons
        self.positive_words = {
            "beat", "exceed", "outperform", "upgrade", "bullish", "surge",
            "profit", "growth", "strong", "record", "positive", "gain",
            "rally", "breakout", "momentum", "buy", "accumulate", "upside",
            "improve", "accelerate", "expand", "increase", "optimistic",
            "confident", "robust", "solid", "healthy", "favorable"
        }

        self.negative_words = {
            "miss", "disappoint", "downgrade", "bearish", "plunge", "loss",
            "weak", "decline", "sell", "crash", "risk", "concern", "warning",
            "lawsuit", "investigation", "fraud", "scandal", "bankruptcy",
            "default", "cut", "reduce", "slowdown", "recession", "volatile",
            "uncertain", "difficult", "challenge", "pressure", "headwind"
        }

        self.negation_words = {
            "not", "no", "never", "neither", "nobody", "nothing",
            "nowhere", "hardly", "barely", "doesn't", "isn't", "wasn't",
            "shouldn't", "wouldn't", "couldn't", "won't", "can't", "don't"
        }

        self.intensifiers = {
            "very": 1.5, "extremely": 2.0, "significantly": 1.5,
            "substantially": 1.5, "massive": 2.0, "huge": 1.8,
            "slightly": 0.5, "somewhat": 0.7, "marginally": 0.5
        }

        # Cache for sentiment results
        self.cache: Dict[str, List[SentimentResult]] = defaultdict(list)

    def preprocess(self, text: str) -> List[str]:
        """Preprocess text for analysis."""
        # Lowercase and tokenize
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()
        return tokens

    def analyze_text(
        self,
        text: str,
        source: str = "unknown"
    ) -> SentimentResult:
        """Analyze sentiment of a single text."""
        tokens = self.preprocess(text)

        if not tokens:
            return SentimentResult(
                text=text[:100],
                sentiment_score=0.0,
                confidence=0.0,
                keywords=[],
                source=source
            )

        positive_count = 0
        negative_count = 0
        keywords = []

        # Track negation window
        negation_active = False
        negation_window = 3
        since_negation = 0

        # Current intensifier
        current_intensifier = 1.0

        for i, token in enumerate(tokens):
            # Check for negation
            if token in self.negation_words:
                negation_active = True
                since_negation = 0
                continue

            # Update negation window
            if negation_active:
                since_negation += 1
                if since_negation > negation_window:
                    negation_active = False

            # Check for intensifier
            if token in self.intensifiers:
                current_intensifier = self.intensifiers[token]
                continue

            # Check for positive word
            if token in self.positive_words:
                score = current_intensifier
                if negation_active:
                    negative_count += score
                    keywords.append(f"not-{token}")
                else:
                    positive_count += score
                    keywords.append(token)

            # Check for negative word
            if token in self.negative_words:
                score = current_intensifier
                if negation_active:
                    positive_count += score
                    keywords.append(f"not-{token}")
                else:
                    negative_count += score
                    keywords.append(token)

            # Reset intensifier
            current_intensifier = 1.0

        # Calculate sentiment score
        total = positive_count + negative_count
        if total > 0:
            sentiment_score = (positive_count - negative_count) / total
        else:
            sentiment_score = 0.0

        # Confidence based on keyword density
        confidence = min(1.0, total / (len(tokens) * 0.1))

        return SentimentResult(
            text=text[:100],
            sentiment_score=sentiment_score,
            confidence=confidence,
            keywords=keywords[:10],
            source=source
        )

    def analyze_headlines(
        self,
        symbol: str,
        headlines: List[str]
    ) -> float:
        """Analyze multiple news headlines."""
        if not headlines:
            return 0.0

        sentiments = []
        for headline in headlines:
            result = self.analyze_text(headline, source="news")
            sentiments.append(result.sentiment_score * result.confidence)
            self.cache[symbol].append(result)

        return sum(sentiments) / len(sentiments)

    def analyze_social(
        self,
        symbol: str,
        posts: List[str]
    ) -> float:
        """Analyze social media posts."""
        if not posts:
            return 0.0

        sentiments = []
        for post in posts:
            result = self.analyze_text(post, source="social")
            sentiments.append(result.sentiment_score * result.confidence)
            self.cache[symbol].append(result)

        return sum(sentiments) / len(sentiments)

    def get_aggregated_sentiment(self, symbol: str) -> AggregatedSentiment:
        """Get aggregated sentiment for a symbol."""
        results = self.cache.get(symbol, [])

        if not results:
            return AggregatedSentiment(
                symbol=symbol,
                overall_sentiment=0.0,
                news_sentiment=0.0,
                social_sentiment=0.0,
                filing_sentiment=0.0,
                sample_count=0,
                signal="NEUTRAL"
            )

        # Aggregate by source
        news = [r for r in results if r.source == "news"]
        social = [r for r in results if r.source == "social"]
        filings = [r for r in results if r.source == "filing"]

        def avg_sentiment(items: List[SentimentResult]) -> float:
            if not items:
                return 0.0
            return sum(r.sentiment_score * r.confidence for r in items) / len(items)

        news_sent = avg_sentiment(news)
        social_sent = avg_sentiment(social)
        filing_sent = avg_sentiment(filings)

        # Weighted overall (news more reliable than social)
        overall = (news_sent * 0.5 + social_sent * 0.3 + filing_sent * 0.2)

        # Determine signal
        if overall > 0.2:
            signal = "BULLISH"
        elif overall < -0.2:
            signal = "BEARISH"
        else:
            signal = "NEUTRAL"

        return AggregatedSentiment(
            symbol=symbol,
            overall_sentiment=overall,
            news_sentiment=news_sent,
            social_sentiment=social_sent,
            filing_sentiment=filing_sent,
            sample_count=len(results),
            signal=signal
        )

    def clear_cache(self, symbol: Optional[str] = None):
        """Clear sentiment cache."""
        if symbol:
            self.cache[symbol] = []
        else:
            self.cache.clear()


# Global singleton
_nlp_analyzer: Optional[NLPSentimentAnalyzer] = None


def get_nlp_analyzer() -> NLPSentimentAnalyzer:
    """Get or create global NLP analyzer."""
    global _nlp_analyzer
    if _nlp_analyzer is None:
        _nlp_analyzer = NLPSentimentAnalyzer()
    return _nlp_analyzer
