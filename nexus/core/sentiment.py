import logging
import asyncio
import feedparser
import re
import numpy as np
from typing import Dict, Any
import datetime

logger = logging.getLogger(__name__)

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except (ImportError, OSError, RuntimeError):
    # transformers may fail to import due to environment or protobuf/tensorflow
    # compatibility issues. Fall back to lighter sentiment options.
    TRANSFORMERS_AVAILABLE = False


class SentimentEngine:
    BULLISH_TERMS = {
        "surge", "soar", "jump", "climb", "gain", "rally", "upbeat", "beat", "beats",
        "growth", "buy", "upgrade", "outperform", "record", "profit", "bullish",
        "strong", "higher", "breakout", "momentum", "upside", "positive", "expansion",
        "opportunity", "innovation", "leadership", "dominant", "accelerate",
    }
    BEARISH_TERMS = {
        "plunge", "tumble", "fall", "drop", "decline", "slump", "miss", "misses",
        "loss", "sell", "downgrade", "underperform", "warning", "bearish", "weak",
        "lower", "crash", "lawsuit", "investigation", "recession", "inflation",
        "slowdown", "downturn", "debt", "default", "bankruptcy", "volatile",
    }
    INTENSIFIERS = {"very", "extremely", "highly", "significantly", "remarkably", "substantially"}
    NEGATORS = {"not", "no", "never", "neither", "nor", "cannot", "don't", "won't"}

    def __init__(self, use_finbert: bool = False):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._CACHE_TTL = 3600
        self.use_finbert = use_finbert
        self._finbert = None
        self._vader = None
        if VADER_AVAILABLE:
            try:
                self._vader = SentimentIntensityAnalyzer()
            except (ValueError, OSError, RuntimeError):
                pass

    def _score_headline_advanced(self, title: str) -> float:
        words = re.findall(r'\b\w+\b', title.lower())
        word_set = set(words)
        bull_matches = word_set.intersection(self.BULLISH_TERMS)
        bear_matches = word_set.intersection(self.BEARISH_TERMS)
        score = 0.0
        for w in words:
            if w in self.INTENSIFIERS:
                score += 0.15 if not any(n in word_set for n in self.NEGATORS) else -0.10
            if w in self.NEGATORS:
                score *= -0.5
        for term in bull_matches:
            score += 0.4
        for term in bear_matches:
            score -= 0.4
        if self._vader:
            try:
                vs = self._vader.polarity_scores(title)
                score = score * 0.6 + vs['compound'] * 0.4
            except (ValueError, TypeError, AttributeError):
                pass
        if TEXTBLOB_AVAILABLE:
            try:
                blob = TextBlob(title)
                score = score * 0.7 + blob.sentiment.polarity * 0.3
            except (ValueError, TypeError, AttributeError):
                pass
        return float(np.clip(score, -1.0, 1.0))

    async def get_sentiment(self, symbol: str) -> float:
        now = datetime.datetime.now().timestamp()
        if symbol in self._cache:
            entry = self._cache[symbol]
            if now - entry["timestamp"] < self._CACHE_TTL:
                return float(entry["score"])
        try:
            score = await asyncio.to_thread(self._fetch_and_score, symbol)
            self._cache[symbol] = {"timestamp": now, "score": score}
            return score
        except (TimeoutError, ConnectionError, OSError, ValueError) as e:
            logger.debug("Sentiment fetch failed for %s: %s", symbol, e)
            return 0.0

    def _fetch_and_score(self, symbol: str) -> float:
        url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"
        try:
            feed = feedparser.parse(url)
        except (TimeoutError, ConnectionError, OSError, ValueError, KeyError):
            return 0.0
        if not feed.entries:
            return 0.0
        scores = []
        for entry in feed.entries[:10]:
            title = entry.title
            score = self._score_headline_advanced(title)
            scores.append(score)
        if not scores:
            return 0.0
        weights = np.linspace(1.0, 0.5, len(scores))
        weighted = np.average(scores, weights=weights)
        return float(np.clip(weighted, -1.0, 1.0))