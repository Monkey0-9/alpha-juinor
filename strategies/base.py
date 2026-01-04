# strategies/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List
import pandas as pd
import numpy as np

class BaseStrategy(ABC):
    """
    Formal Strategy interface as per institutional requirements.
    Wraps Alpha generation and Risk logic into a single unit.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = config.get('name', self.__class__.__name__)
        self.tickers = config.get('tickers', [])
        
    @abstractmethod
    def generate_signals(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals (e.g. conviction 0..1) from market data.
        market_data: MultiIndex DataFrame (ticker, field)
        Returns a DataFrame indexed by timestamp with columns as tickers.
        """
        pass
    
    @abstractmethod
    def calculate_risk(self, signals: pd.DataFrame, market_data: pd.DataFrame) -> pd.DataFrame:
        """
        Adjust signals based on strategy-specific risk (e.g. stop losses).
        """
        return signals # Default: No adjustment
    
    def on_order_fill(self, trade_data: Dict[str, Any]):
        """Callback for execution feedback."""
        pass
