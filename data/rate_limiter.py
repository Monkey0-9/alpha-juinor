"""
Smart Rate Limiter for Data Providers.

Uses Token Bucket algorithm with per-provider limits.
Prevents hitting API rate limits by intelligently throttling requests.
"""

import time
import threading
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class ProviderLimit:
    """Rate limit configuration for a data provider."""
    requests_per_minute: int = 60
    requests_per_day: int = 10000
    burst_size: int = 10  # Max requests in quick succession


@dataclass
class TokenBucket:
    """Token Bucket rate limiter implementation."""
    capacity: float
    fill_rate: float  # tokens per second
    tokens: float = field(default=0.0)
    last_update: float = field(default_factory=time.time)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self):
        self.tokens = self.capacity

    def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens. Returns True if successful.
        """
        with self.lock:
            now = time.time()
            elapsed = now - self.last_update
            self.tokens = min(
                self.capacity,
                self.tokens + elapsed * self.fill_rate
            )
            self.last_update = now

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

    def wait_for_token(self, tokens: int = 1, timeout: float = 60.0) -> bool:
        """
        Wait until tokens are available or timeout.
        """
        start = time.time()
        while time.time() - start < timeout:
            if self.consume(tokens):
                return True
            time.sleep(0.1)
        return False

    def tokens_available(self) -> int:
        """Return current available tokens (approximate)."""
        with self.lock:
            now = time.time()
            elapsed = now - self.last_update
            return int(min(
                self.capacity,
                self.tokens + elapsed * self.fill_rate
            ))


class SmartRateLimiter:
    """
    Intelligent rate limiter with per-provider tracking.

    Features:
    - Token bucket per provider (minute-level)
    - Daily quota tracking
    - Automatic backoff on approaching limits
    - Provider-specific configurations
    """

    # Default limits per provider
    DEFAULT_LIMITS: Dict[str, ProviderLimit] = {
        "alpaca": ProviderLimit(200, 10000, 20),
        "yahoo": ProviderLimit(100, 2000, 10),
        "polygon": ProviderLimit(5, 500, 2),  # Free tier
        "alpha_vantage": ProviderLimit(5, 500, 2),
        "finnhub": ProviderLimit(60, 1000, 10),
        "twelvedata": ProviderLimit(8, 800, 3),
        "binance": ProviderLimit(1200, 100000, 100),
        "coingecko": ProviderLimit(10, 10000, 5),
        "fred": ProviderLimit(120, 10000, 20),
        "stooq": ProviderLimit(30, 1000, 5),
        "default": ProviderLimit(30, 5000, 5),
    }

    def __init__(self):
        self.buckets: Dict[str, TokenBucket] = {}
        self.daily_counts: Dict[str, int] = defaultdict(int)
        self.daily_reset: float = time.time()
        self.lock = threading.Lock()
        self._init_buckets()

    def _init_buckets(self):
        """Initialize token buckets for all known providers."""
        for provider, limits in self.DEFAULT_LIMITS.items():
            fill_rate = limits.requests_per_minute / 60.0
            self.buckets[provider] = TokenBucket(
                capacity=limits.burst_size,
                fill_rate=fill_rate
            )

    def _get_bucket(self, provider: str) -> TokenBucket:
        """Get or create bucket for provider."""
        provider = provider.lower()
        if provider not in self.buckets:
            limits = self.DEFAULT_LIMITS.get(
                provider, self.DEFAULT_LIMITS["default"]
            )
            fill_rate = limits.requests_per_minute / 60.0
            self.buckets[provider] = TokenBucket(
                capacity=limits.burst_size,
                fill_rate=fill_rate
            )
        return self.buckets[provider]

    def _check_daily_reset(self):
        """Reset daily counters if 24 hours passed."""
        with self.lock:
            if time.time() - self.daily_reset > 86400:
                self.daily_counts.clear()
                self.daily_reset = time.time()
                logger.info("RateLimiter: Daily counters reset")

    def can_request(self, provider: str) -> bool:
        """
        Check if a request can be made without waiting.
        """
        provider = provider.lower()
        self._check_daily_reset()

        limits = self.DEFAULT_LIMITS.get(
            provider, self.DEFAULT_LIMITS["default"]
        )

        # Check daily limit
        if self.daily_counts[provider] >= limits.requests_per_day:
            logger.warning(f"RateLimiter: {provider} daily limit reached")
            return False

        # Check minute-level bucket
        bucket = self._get_bucket(provider)
        return bucket.tokens_available() > 0

    def acquire(
        self,
        provider: str,
        timeout: float = 30.0,
        tokens: int = 1
    ) -> bool:
        """
        Acquire permission to make a request. Blocks until available.

        Returns True if acquired, False if timeout or daily limit hit.
        """
        provider = provider.lower()
        self._check_daily_reset()

        limits = self.DEFAULT_LIMITS.get(
            provider, self.DEFAULT_LIMITS["default"]
        )

        # Check daily limit first
        with self.lock:
            if self.daily_counts[provider] >= limits.requests_per_day:
                logger.error(
                    f"RateLimiter: {provider} daily limit "
                    f"({limits.requests_per_day}) exhausted"
                )
                return False

        # Wait for token
        bucket = self._get_bucket(provider)
        if bucket.wait_for_token(tokens, timeout):
            with self.lock:
                self.daily_counts[provider] += tokens
            return True

        logger.warning(
            f"RateLimiter: Timeout waiting for {provider} token"
        )
        return False

    def record_request(self, provider: str, count: int = 1):
        """Record a request was made (for external tracking)."""
        provider = provider.lower()
        with self.lock:
            self.daily_counts[provider] += count

    def get_status(self, provider: str) -> Dict:
        """Get current rate limit status for a provider."""
        provider = provider.lower()
        limits = self.DEFAULT_LIMITS.get(
            provider, self.DEFAULT_LIMITS["default"]
        )
        bucket = self._get_bucket(provider)

        return {
            "provider": provider,
            "tokens_available": bucket.tokens_available(),
            "daily_used": self.daily_counts.get(provider, 0),
            "daily_limit": limits.requests_per_day,
            "requests_per_minute": limits.requests_per_minute,
        }

    def get_all_status(self) -> Dict[str, Dict]:
        """Get status of all providers."""
        return {p: self.get_status(p) for p in self.DEFAULT_LIMITS.keys()}


# Global singleton instance
_rate_limiter: Optional[SmartRateLimiter] = None


def get_rate_limiter() -> SmartRateLimiter:
    """Get or create the global rate limiter instance."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = SmartRateLimiter()
    return _rate_limiter
