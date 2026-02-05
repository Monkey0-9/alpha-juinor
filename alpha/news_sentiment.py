#!/usr/bin/env python3
"""
ALTERNATIVE DATA: NEWS SENTIMENT ALPHA
=======================================

S-Class Initiative 1: Extract alpha from news sentiment.
Uses VADER (built into NLTK) for sentiment analysis.
Falls back to simple keyword matching if NLTK unavailable.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Optional
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NEWS_SENTIMENT")

# Try to import NLTK VADER, fallback to simple method
try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    logger.warning("NLTK not available. Using simple keyword sentiment.")


@dataclass
class SentimentSignal:
    """Sentiment-based trading signal."""
    symbol: str
    sentiment_score: float  # -1 to +1
    headline_count: int
    signal_strength: float  # -1 to +1 (trading signal)
    confidence: float


class NewsSentimentAlpha:
    """
    Generate alpha signals from news sentiment.

    Strategy: Contrarian sentiment
    - Extreme positive sentiment (> 0.5) -> Potential overbought
    - Extreme negative sentiment (< -0.5) -> Potential oversold
    """

    def __init__(self):
        if VADER_AVAILABLE:
            self.analyzer = SentimentIntensityAnalyzer()
        else:
            self.analyzer = None

        # Keyword-based fallback
        self.positive_words = {
            'surge', 'soar', 'rally', 'gain', 'profit', 'beat', 'strong',
            'growth', 'up', 'high', 'bull', 'buy', 'upgrade', 'outperform'
        }
        self.negative_words = {
            'fall', 'crash', 'drop', 'loss', 'miss', 'weak', 'decline',
            'down', 'low', 'bear', 'sell', 'downgrade', 'underperform', 'fear'
        }

        # Thresholds
        self.extreme_threshold = 0.5

    def analyze_headline(self, headline: str) -> float:
        """Analyze sentiment of a single headline."""
        if self.analyzer:
            scores = self.analyzer.polarity_scores(headline)
            return scores['compound']
        else:
            # Simple keyword matching
            headline_lower = headline.lower()
            pos_count = sum(1 for w in self.positive_words if w in headline_lower)
            neg_count = sum(1 for w in self.negative_words if w in headline_lower)

            total = pos_count + neg_count
            if total == 0:
                return 0.0
            return (pos_count - neg_count) / total

    def generate_signal(
        self,
        symbol: str,
        headlines: List[str]
    ) -> SentimentSignal:
        """
        Generate trading signal from headlines.

        Contrarian logic:
        - Very positive news = Market may be overbought -> Slight bearish signal
        - Very negative news = Market may be oversold -> Slight bullish signal
        - Neutral = No signal
        """
        if not headlines:
            return SentimentSignal(
                symbol=symbol,
                sentiment_score=0.0,
                headline_count=0,
                signal_strength=0.0,
                confidence=0.0
            )

        # Analyze all headlines
        scores = [self.analyze_headline(h) for h in headlines]
        avg_sentiment = np.mean(scores)

        # Contrarian signal generation
        if avg_sentiment > self.extreme_threshold:
            # Extremely positive -> potential reversal, slight bearish
            signal_strength = -0.3 * (avg_sentiment - self.extreme_threshold)
        elif avg_sentiment < -self.extreme_threshold:
            # Extremely negative -> potential reversal, slight bullish
            signal_strength = -0.3 * (avg_sentiment + self.extreme_threshold)
        else:
            # Neutral zone -> no signal
            signal_strength = 0.0

        # Confidence based on headline count
        confidence = min(len(headlines) / 10, 1.0)

        return SentimentSignal(
            symbol=symbol,
            sentiment_score=float(avg_sentiment),
            headline_count=len(headlines),
            signal_strength=float(signal_strength),
            confidence=float(confidence)
        )


def fetch_sample_headlines(symbol: str) -> List[str]:
    """
    Fetch sample headlines for testing.
    In production, this would use RSS feeds or news APIs.
    """
    # Simulated headlines for testing
    sample_headlines = {
        "SPY": [
            "Stock market rallies to new highs amid strong earnings",
            "Investors optimistic as Fed signals rate pause",
            "Tech stocks lead market gains for third straight day",
            "Economic data shows continued growth momentum",
        ],
        "QQQ": [
            "Tech giants beat earnings expectations",
            "AI boom drives Nasdaq to record levels",
            "Semiconductor stocks surge on demand outlook",
        ],
        "GLD": [
            "Gold prices steady amid dollar weakness",
            "Investors seek safe haven as uncertainty rises",
        ],
        "TLT": [
            "Bond yields fall as inflation cools",
            "Treasury market rallies on Fed pivot hopes",
        ]
    }
    return sample_headlines.get(symbol, [])


def demo():
    """Demonstrate sentiment alpha generation."""
    print("=" * 60)
    print("     NEWS SENTIMENT ALPHA (S-Class Initiative 1)")
    print("=" * 60)

    alpha = NewsSentimentAlpha()
    print(f"VADER Available: {VADER_AVAILABLE}")

    symbols = ["SPY", "QQQ", "GLD", "TLT"]

    for symbol in symbols:
        headlines = fetch_sample_headlines(symbol)
        signal = alpha.generate_signal(symbol, headlines)

        print(f"\n{symbol}:")
        print(f"  Headlines: {signal.headline_count}")
        print(f"  Sentiment: {signal.sentiment_score:+.2f}")
        print(f"  Signal: {signal.signal_strength:+.2f}")
        print(f"  Confidence: {signal.confidence:.1%}")


if __name__ == "__main__":
    demo()
