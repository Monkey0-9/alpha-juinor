import pandas as pd
import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

class DataQualityResult(BaseModel):
    symbol: str
    score: float
    is_valid: bool
    errors: List[str]

class ProviderAdapter:
    """
    Interface for data providers.
    """
    async def fetch_price_history(self, symbols: List[str], start: datetime.date, end: datetime.date) -> Dict[str, pd.DataFrame]:
        raise NotImplementedError

    async def fetch_corporate_actions(self, symbol: str) -> List[Dict[str, Any]]:
        raise NotImplementedError
