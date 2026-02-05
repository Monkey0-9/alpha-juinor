"""
Background LLM Sentiment Fetcher.

Runs LLM sentiment analysis asynchronously in the background
so it NEVER blocks the main market data pipeline.

Key design:
- Main data pipeline (Yahoo, Alpaca, etc.) is unaffected
- LLM sentiment is fetched in background thread
- Results are cached and available when ready
- If LLM is slow/unavailable, system continues with local analysis
"""

import logging
import threading
import time
from typing import Dict, Optional, Set
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class BackgroundSentimentFetcher:
    """
    Fetches LLM sentiment in background without blocking.

    Usage:
        fetcher = BackgroundSentimentFetcher()
        fetcher.request_sentiment("AAPL")  # Non-blocking

        # Later, check if ready
        result = fetcher.get_sentiment("AAPL")  # Returns cached or None
    """

    def __init__(self, max_workers: int = 2):
        """
        Initialize background fetcher.

        Args:
            max_workers: Max concurrent LLM requests (keep low to avoid rate limits)
        """
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="llm_sentiment"
        )
        self._cache: Dict[str, dict] = {}
        self._pending: Set[str] = set()
        self._cache_ttl = timedelta(minutes=30)
        self._research = None
        self._lock = threading.Lock()

    def _get_research(self):
        """Lazy load LLM research module."""
        if self._research is None:
            try:
                from data.intelligence.llm_market_research import (
                    get_llm_market_research
                )
                self._research = get_llm_market_research()
            except Exception as e:
                logger.debug(f"LLM research not available: {e}")
        return self._research

    def _fetch_sentiment_async(self, ticker: str):
        """Background task to fetch sentiment."""
        try:
            research = self._get_research()
            if not research:
                return

            result = research.get_market_sentiment(ticker)

            with self._lock:
                self._cache[ticker] = {
                    "score": result.sentiment_score,
                    "confidence": result.confidence,
                    "summary": result.summary,
                    "factors": result.key_factors,
                    "timestamp": datetime.utcnow().isoformat(),
                    "provider": result.provider
                }
                self._pending.discard(ticker)

            logger.debug(
                f"Background LLM sentiment for {ticker}: "
                f"{result.sentiment_score:.2f}"
            )

        except Exception as e:
            logger.debug(f"Background LLM fetch failed for {ticker}: {e}")
            with self._lock:
                self._pending.discard(ticker)

    def request_sentiment(self, ticker: str) -> bool:
        """
        Request sentiment fetch in background (non-blocking).

        Args:
            ticker: Stock ticker symbol

        Returns:
            True if request was queued, False if already pending/cached
        """
        with self._lock:
            # Already have fresh cache?
            if ticker in self._cache:
                cached = self._cache[ticker]
                ts = datetime.fromisoformat(cached["timestamp"])
                if datetime.utcnow() - ts < self._cache_ttl:
                    return False  # Already cached

            # Already pending?
            if ticker in self._pending:
                return False

            self._pending.add(ticker)

        # Submit to background thread pool
        self._executor.submit(self._fetch_sentiment_async, ticker)
        return True

    def get_sentiment(self, ticker: str) -> Optional[dict]:
        """
        Get cached sentiment if available (non-blocking).

        Args:
            ticker: Stock ticker symbol

        Returns:
            Cached sentiment dict or None if not ready
        """
        with self._lock:
            if ticker not in self._cache:
                return None

            cached = self._cache[ticker]
            ts = datetime.fromisoformat(cached["timestamp"])

            if datetime.utcnow() - ts > self._cache_ttl:
                return None  # Expired

            return cached

    def get_score(self, ticker: str, default: float = 0.0) -> float:
        """
        Get sentiment score if available, else default.

        This is the fastest method - just returns a float.
        """
        result = self.get_sentiment(ticker)
        return result["score"] if result else default

    def prefetch_batch(self, tickers: list):
        """
        Prefetch sentiment for multiple tickers in background.

        Call this early in your pipeline so results are ready later.
        """
        for ticker in tickers[:10]:  # Limit to avoid rate limits
            self.request_sentiment(ticker)

    def is_pending(self, ticker: str) -> bool:
        """Check if a sentiment fetch is in progress."""
        with self._lock:
            return ticker in self._pending

    def shutdown(self):
        """Shutdown the background executor."""
        self._executor.shutdown(wait=False)


# ---------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------
_fetcher: Optional[BackgroundSentimentFetcher] = None


def get_background_sentiment_fetcher() -> BackgroundSentimentFetcher:
    """Get or create the global background fetcher."""
    global _fetcher
    if _fetcher is None:
        _fetcher = BackgroundSentimentFetcher()
    return _fetcher


# ---------------------------------------------------------------------
# Convenience functions for integration
# ---------------------------------------------------------------------

def prefetch_llm_sentiment(tickers: list):
    """
    Prefetch LLM sentiment for tickers in background.

    Call this early in your trading cycle so results are
    ready when you need them.
    """
    try:
        fetcher = get_background_sentiment_fetcher()
        fetcher.prefetch_batch(tickers)
    except Exception:
        pass  # Never block on LLM issues


def get_llm_sentiment_score(ticker: str, default: float = 0.0) -> float:
    """
    Get LLM sentiment score if available (non-blocking).

    Returns cached score or default immediately.
    """
    try:
        fetcher = get_background_sentiment_fetcher()
        return fetcher.get_score(ticker, default)
    except Exception:
        return default
