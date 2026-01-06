from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Optional

class DataProvider(ABC):
    """
    Abstract interface for fetching market data.
    Enables swapping distinct data sources (Yahoo, CCXT, Alpaca) without changing downstream code.
    """

    # Provider capability declarations - must be overridden by subclasses
    supports_ohlcv = False
    supports_latest_quote = False
    supports_sentiment = False
    supports_news = False

    # Authentication and availability status
    _authenticated = False
    disabled = False

    @abstractmethod
    def fetch_ohlcv(self, ticker: str, start_date: str, end_date: str = None) -> pd.DataFrame:
        """Fetch OHLCV data (Synchronous)."""
        pass

    async def fetch_ohlcv_async(self, ticker: str, start_date: str, end_date: str = None) -> pd.DataFrame:
        """Fetch OHLCV data (Asynchronous). Default: wraps sync in thread."""
        import asyncio
        return await asyncio.to_thread(self.fetch_ohlcv, ticker, start_date, end_date)

    @abstractmethod
    def get_panel(self, tickers: List[str], start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
        """Fetch and combine multiple tickers into a MultiIndex panel (Synchronous)."""
        pass

    async def get_panel_async(self, tickers: List[str], start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
        """Fetch and combine multiple tickers into a MultiIndex panel (Asynchronous). Default: wraps sync in thread."""
        import asyncio
        return await asyncio.to_thread(self.get_panel, tickers, start_date, end_date)

    def is_available(self) -> bool:
        """Check if provider is authenticated and not disabled."""
        return self._authenticated and not self.disabled

    def get_latest_quote(self, ticker: str) -> Optional[float]:
        """Synchronous latest price quote."""
        if not self.supports_latest_quote:
            raise NotImplementedError(f"{self.__class__.__name__} does not support latest quotes")
        return None

    async def get_latest_quote_async(self, ticker: str) -> Optional[float]:
        """Asynchronous latest price quote."""
        if not self.supports_latest_quote:
            raise NotImplementedError(f"{self.__class__.__name__} does not support latest quotes")
        return None

    async def start_streaming(self, tickers: List[str]):
        """Optional: Start WebSocket streaming for real-time updates."""
        pass

    async def stop_streaming(self):
        """Optional: Stop WebSocket streaming."""
        pass
