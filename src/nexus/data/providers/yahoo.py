import yfinance as yf
import pandas as pd
from datetime import datetime
from typing import List
from .base import DataProvider
from ...models.market import MarketBar
from ..validator import DataValidator

class YahooDataProvider(DataProvider):
    """
    Yahoo Finance implementation of the DataProvider.
    Uses yfinance to fetch historical data and converts it to validated MarketBars.
    """
    def get_name(self) -> str:
        return "yahoo"

    async def get_historical_data(
        self, 
        symbol: str, 
        start: datetime, 
        end: datetime, 
        interval: str = "1d"
    ) -> List[MarketBar]:
        ticker = yf.Ticker(symbol)
        
        # yfinance is synchronous, but we wrap it in a thread/process in an elite system
        # For now, we use history() directly
        df = ticker.history(start=start, end=end, interval=interval)
        
        if df.empty:
            return []
            
        df = DataValidator.validate_ohlc(df, ticker=symbol)
        
        bars = []
        for timestamp, row in df.iterrows():
            bars.append(MarketBar(
                symbol=symbol,
                timestamp=timestamp.to_pydatetime(),
                open=float(row['Open']),
                high=float(row['High']),
                low=float(row['Low']),
                close=float(row['Close']),
                volume=int(row['Volume'])
            ))
            
        return bars
