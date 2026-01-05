from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Optional

class DataProvider(ABC):
    """
    Abstract interface for fetching market data.
    Enables swapping distinct data sources (Yahoo, CCXT, Alpaca) without changing downstream code.
    """
    
    @abstractmethod
    def fetch_ohlcv(self, ticker: str, start_date: str, end_date: str = None) -> pd.DataFrame:
        """
        Fetch OHLCV data for a given ticker.
        Returns DataFrame with columns: Open, High, Low, Close, Volume.
        index should be DatetimeIndex.
        """
        pass

    @abstractmethod
    def get_panel(self, tickers: List[str], start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
        """Fetch and combine multiple tickers into a MultiIndex panel."""
        pass
