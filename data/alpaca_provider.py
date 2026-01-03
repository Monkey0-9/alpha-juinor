# data/alpaca_provider.py
import os
import pandas as pd
import requests
import time
from datetime import datetime, timedelta
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class AlpacaDataProvider:
    """
    Alpaca Markets data provider (FREE for paper trading).
    Supports historical bars and real-time quotes.
    
    Setup:
    1. Sign up at alpaca.markets
    2. Get Paper Trading API keys
    3. Set environment variables:
       - ALPACA_API_KEY
       - ALPACA_SECRET_KEY
       - ALPACA_BASE_URL (default: https://paper-api.alpaca.markets)
    """
    
    def __init__(self, paper=True):
        self.api_key = os.getenv("ALPACA_API_KEY")
        self.secret_key = os.getenv("ALPACA_SECRET_KEY")
        
        if paper:
            self.base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
            self.data_url = "https://data.alpaca.markets"
        else:
            self.base_url = "https://api.alpaca.markets"
            self.data_url = "https://data.alpaca.markets"
        
        self.headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key
        }
        
        if not self.api_key or not self.secret_key:
            logger.warning("Alpaca API keys not found. Set ALPACA_API_KEY and ALPACA_SECRET_KEY env vars.")
    
    def _request_with_retry(self, url: str, params: Optional[Dict] = None, retries: int = 3) -> requests.Response:
        """Retry wrapper for Alpaca Data API requests."""
        for i in range(retries):
            try:
                response = requests.get(url, headers=self.headers, params=params, timeout=10)
                if response.status_code == 429: # Rate limit
                    time.sleep(2 ** i)
                    continue
                response.raise_for_status()
                return response
            except Exception as e:
                if i == retries - 1:
                    raise
                time.sleep(1)
        return None

    def fetch_ohlcv(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch historical OHLCV bars from Alpaca.
        Free tier: Unlimited for paper trading.
        """
        if not self.api_key:
            logger.error("Cannot fetch data: Alpaca API keys not configured")
            return pd.DataFrame()
        
        # Ensure dates are in Alpaca friendly format (YYYY-MM-DD or RFC3339)
        # Alpaca expects dates like 2021-01-01
        
        try:
            url = f"{self.data_url}/v2/stocks/{ticker}/bars"
            
            params = {
                "start": start_date,
                "end": end_date,
                "timeframe": "1Day",
                "limit": 10000,
                "adjustment": "all"
            }
            
            response = self._request_with_retry(url, params=params)
            
            data = response.json()
            
            if "bars" not in data or not data["bars"]:
                logger.warning(f"No data returned for {ticker}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            bars = data["bars"]
            df = pd.DataFrame(bars)
            
            # Rename columns to match our standard
            df = df.rename(columns={
                "t": "Date",
                "o": "Open",
                "h": "High",
                "l": "Low",
                "c": "Close",
                "v": "Volume"
            })
            
            # Parse dates
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date")
            
            # Select only OHLCV
            df = df[["Open", "High", "Low", "Close", "Volume"]]
            
            logger.info(f"Fetched {len(df)} bars for {ticker} from Alpaca")
            return df
            
        except Exception as e:
            logger.error(f"Alpaca fetch failed for {ticker}: {e}")
            return pd.DataFrame()
    
    def get_latest_quote(self, ticker: str) -> Optional[float]:
        """Get latest trade price (real-time)."""
        try:
            url = f"{self.data_url}/v2/stocks/{ticker}/trades/latest"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            data = response.json()
            if "trade" in data and "p" in data["trade"]:
                return float(data["trade"]["p"])
            return None
        except Exception as e:
            logger.error(f"Failed to get latest quote for {ticker}: {e}")
            return None
