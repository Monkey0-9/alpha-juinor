from abc import ABC, abstractmethod
from datetime import datetime
from typing import List
from ...models.market import MarketBar

class DataProvider(ABC):
    """
    Abstract base class for all market data providers.
    """
    @abstractmethod
    def get_name(self) -> str:
        """Returns the unique name of the provider."""
        pass

    @abstractmethod
    async def get_historical_data(
        self, 
        symbol: str, 
        start: datetime, 
        end: datetime, 
        interval: str = "1d"
    ) -> List[MarketBar]:
        """Fetches historical market data for a given symbol and range."""
        pass
