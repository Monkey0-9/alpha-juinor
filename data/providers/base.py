from abc import ABC, abstractmethod
import pandas as pd
import logging
from typing import List, Optional

from data.smart_cache import get_data_cache

logger = logging.getLogger(__name__)


class DataProvider(ABC):
    """
    Abstract interface for fetching market data.
    Enables swapping distinct data sources without changing downstream code.

    Uses SmartDataCache to prevent redundant API calls and rate limit hits.
    HIGH PERFORMANCE: Fetch once, cache forever (until TTL expires).
    """

    # Provider capability declarations
    supports_ohlcv = False
    supports_latest_quote = False
    supports_sentiment = False
    supports_news = False

    # Authentication and availability status
    _authenticated = False
    disabled = False

    # Provider name (override in subclass)
    provider_name = "default"

    def fetch_ohlcv_cached(
        self,
        ticker: str,
        start_date: str,
        end_date: str = None
    ) -> pd.DataFrame:
        """
        Fetch OHLCV with smart caching.
        Only calls API if cache miss. HIGH PERFORMANCE - no rate limit hits.
        """
        cache = get_data_cache()

        # Check cache first - if cached, return immediately (no API call)
        cached = cache.get(ticker, "daily", start_date)
        if cached is not None and not cached.empty:
            logger.debug(f"{self.provider_name}: Cache HIT for {ticker}")
            return cached

        # Check if we should even try fetching
        if not cache.should_fetch(ticker, "daily"):
            logger.debug(f"{self.provider_name}: Skip fetch {ticker} (already tried today)")
            return pd.DataFrame()

        # Cache miss - fetch from API (happens ONCE per day per symbol)
        logger.info(f"{self.provider_name}: Fetching {ticker} from API (once/day)")
        try:
            data = self.fetch_ohlcv(ticker, start_date, end_date)
            if data is not None and not data.empty:
                cache.set(ticker, data, "daily", start_date)
            cache.record_fetch(ticker, "daily")  # Mark as fetched
            return data if data is not None else pd.DataFrame()
        except Exception as e:
            logger.error(f"{self.provider_name}: Fetch failed {ticker}: {e}")
            cache.record_fetch(ticker, "daily")  # Prevent retry spam
            return pd.DataFrame()

    @abstractmethod
    def fetch_ohlcv(
        self, ticker: str, start_date: str, end_date: str = None
    ) -> pd.DataFrame:
        """Fetch OHLCV data (Synchronous). Override in subclass."""
        pass

    async def fetch_ohlcv_async(
        self, ticker: str, start_date: str, end_date: str = None
    ) -> pd.DataFrame:
        """Fetch OHLCV with caching (Asynchronous)."""
        import asyncio
        return await asyncio.to_thread(
            self.fetch_ohlcv_cached, ticker, start_date, end_date
        )

    @abstractmethod
    def get_panel(
        self, tickers: List[str], start_date: str, end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Fetch multiple tickers into a MultiIndex panel."""
        pass

    def get_panel_cached(
        self, tickers: List[str], start_date: str, end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch panel with per-symbol caching.
        HIGH PERFORMANCE: Each symbol fetched max once per day.
        """
        frames = {}
        for ticker in tickers:
            data = self.fetch_ohlcv_cached(ticker, start_date, end_date)
            if not data.empty:
                frames[ticker] = data

        if not frames:
            return pd.DataFrame()

        # Combine into panel
        return pd.concat(frames, axis=1)

    async def get_panel_async(
        self, tickers: List[str], start_date: str, end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Fetch multiple tickers with caching (Asynchronous)."""
        import asyncio
        return await asyncio.to_thread(
            self.get_panel_cached, tickers, start_date, end_date
        )

    def is_available(self) -> bool:
        """Check if provider is authenticated and not disabled."""
        return self._authenticated and not self.disabled

    def get_latest_quote(self, ticker: str) -> Optional[float]:
        """Synchronous latest price quote."""
        if not self.supports_latest_quote:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support latest quotes"
            )
        return None

    async def get_latest_quote_async(self, ticker: str) -> Optional[float]:
        """Asynchronous latest price quote."""
        if not self.supports_latest_quote:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support latest quotes"
            )
        return None

    async def start_streaming(self, tickers: List[str]):
        """Optional: Start WebSocket streaming for real-time updates."""
        pass

    async def stop_streaming(self):
        """Optional: Stop WebSocket streaming."""
        pass
