
import os
import requests
import pandas as pd
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class AlphaVantageDataProvider:
    """
    Alpha Vantage Data Provider.
    Good for Forex, Macro, and backup Equity data.
    Requires API Key.
    """
    
    BASE_URL = "https://www.alphavantage.co/query"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("ALPHA_VANTAGE_API_KEY")
        if not self.api_key:
            logger.warning("Alpha Vantage API Key missing. Some backup data may be unavailable.")

    def fetch_ohlcv(self, symbol: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        if not self.api_key: return pd.DataFrame()
        
        try:
            params = {
                "function": "TIME_SERIES_DAILY",
                "symbol": symbol,
                "apikey": self.api_key,
                "outputsize": "full",
                "datatype": "json"
            }
            resp = requests.get(self.BASE_URL, params=params, timeout=10)
            data = resp.json()
            
            if "Time Series (Daily)" not in data:
                return pd.DataFrame()
                
            ts_data = data["Time Series (Daily)"]
            df = pd.DataFrame.from_dict(ts_data, orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            # Rename columns: "1. open" -> "Open"
            df = df.rename(columns={
                "1. open": "Open",
                "2. high": "High",
                "3. low": "Low",
                "4. close": "Close",
                "5. volume": "Volume"
            })
            
            # Convert to float
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                df[col] = df[col].astype(float)
                
            if start_date:
                df = df[df.index >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df.index <= pd.to_datetime(end_date)]
                
            return df[["Open", "High", "Low", "Close", "Volume"]]
            
        except Exception as e:
            logger.error(f"Alpha Vantage Fetch Failed {symbol}: {e}")
            return pd.DataFrame()
