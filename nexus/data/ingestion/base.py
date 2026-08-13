from abc import ABC, abstractmethod
from typing import List
import pandas as pd
from datetime import datetime


class DataSource(ABC):
    """
    Abstract base class for all market data ingestion sources.
    """

    @abstractmethod
    def fetch_historical_ohlcv(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data for given symbols.

        Args:
            symbols: List of ticker symbols.
            start_date: Start datetime.
            end_date: End datetime.
            interval: Data interval (e.g., '1d', '1m', '1h').

        Returns:
            A pandas DataFrame with canonical columns:
            ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']
            The index should be a simple RangeIndex.
        """
        pass

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Name of the data source (e.g., 'yahoo', 'alpaca')."""
        pass
