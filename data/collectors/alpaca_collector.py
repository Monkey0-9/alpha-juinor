# data/alpaca_provider.py
import os
import pandas as pd
import requests
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional
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
    
    def __init__(self, api_key: Optional[str] = None, secret_key: Optional[str] = None, base_url: Optional[str] = None, paper=True):
        self.api_key = api_key or os.getenv("ALPACA_API_KEY")
        self.secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY")
        
        if base_url:
            self.base_url = base_url
            self.data_url = "https://data.alpaca.markets"
        elif paper:
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

    def _is_crypto(self, ticker: str) -> bool:
        """Helper to identify if ticker is crypto (BTC-USD, ETH/USD etc)."""
        crypto_keywords = ["-USD", "/USD", "BTC", "ETH", "LTC", "DOGE"]
        return any(k in ticker.upper() for k in crypto_keywords)
    
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
            # ROUTING: Stocks use /v2/stocks, Crypto uses /v1beta3/crypto/us
            if self._is_crypto(ticker):
                # Standardize crypto format (Alpaca Crypto v1beta3 uses BTC/USD)
                clean_ticker = ticker.upper().replace("-", "/")
                url = f"{self.data_url}/v1beta3/crypto/us/bars"
                params = {
                    "symbols": clean_ticker,
                    "start": start_date,
                    "end": end_date,
                    "timeframe": "1Day"
                }
            else:
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
            
            # Map response data based on asset type
            if self._is_crypto(ticker):
                clean_ticker = ticker.upper().replace("-", "/")
                if "bars" not in data or not data["bars"] or clean_ticker not in data["bars"]:
                    logger.warning(f"No crypto data returned for {ticker}")
                    return pd.DataFrame()
                bars = data["bars"][clean_ticker]
            else:
                if "bars" not in data or not data["bars"]:
                    logger.warning(f"No stock data returned for {ticker}")
                    return pd.DataFrame()
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
            
            # Parse dates and enforce UTC
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date")
            
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            else:
                df.index = df.index.tz_convert("UTC")
            
            # Select only OHLCV
            df = df[["Open", "High", "Low", "Close", "Volume"]]
            
            logger.info(f"Fetched {len(df)} bars for {ticker} from Alpaca")
            return df
            
        except Exception as e:
            logger.error(f"Alpaca fetch failed for {ticker}: {e}")
            return pd.DataFrame()
    
    def get_all_assets(self, asset_class: str = "us_equity") -> List[Dict]:
        """Fetch all tradable assets from Alpaca."""
        url = f"{self.base_url}/v2/assets"
        params = {"asset_class": asset_class, "status": "active"}
        response = self._request_with_retry(url, params=params)
        data = response.json()
        return [
            {
                "symbol": a["symbol"],
                "exchange": a["exchange"],
                "tradable": a["tradable"],
                "status": a["status"]
            } for a in data if a["tradable"]
        ]

    def enrich_universe_data(self, tickers: List[str]) -> pd.DataFrame:
        """
        Enrich universe with ADV, Price, and Market Cap.
        (Uses snapshots and fundamental estimates).
        """
        # Batch Fetch Snapshots for Last Price and Vol
        chunk_size = 200
        all_stats = []
        
        for i in range(0, len(tickers), chunk_size):
            chunk = tickers[i : i + chunk_size]
            url = f"{self.data_url}/v2/stocks/snapshots"
            params = {"symbols": ",".join(chunk)}
            res = self._request_with_retry(url, params=params)
            data = res.json()
            
            for tk, snap in data.items():
                close = snap.get("latestTrade", {}).get("p", 0.0)
                vol = snap.get("dailyBar", {}).get("v", 0)
                
                all_stats.append({
                    "symbol": tk,
                    "last_price": close,
                    "avg_volume": vol, # Daily proxy
                    "avg_dollar_volume_30d": close * vol, # Simplify for scaling
                    "market_cap": close * 1e7, # PLACEHOLDER: Real mcap would need fundamentals API
                    "status": "tradable",
                    "exchange": "NYSE", # Default
                    "listed_only": True
                })
        
        return pd.DataFrame(all_stats)

    def get_latest_quote(self, ticker: str) -> Optional[float]:
        """Get latest trade price (real-time)."""
        try:
            if self._is_crypto(ticker):
                clean_ticker = ticker.upper().replace("-", "/")
                url = f"{self.data_url}/v1beta3/crypto/us/latest/trades"
                params = {"symbols": clean_ticker}
                response = self._request_with_retry(url, params=params)
                data = response.json()
                if "trades" in data and clean_ticker in data["trades"]:
                    return float(data["trades"][clean_ticker]["p"])
            else:
                url = f"{self.data_url}/v2/stocks/{ticker}/trades/latest"
                response = self._request_with_retry(url)
                data = response.json()
                if "trade" in data and "p" in data["trade"]:
                    return float(data["trade"]["p"])
            return None
        except Exception as e:
            logger.error(f"Failed to get latest quote for {ticker}: {e}")
            return None

    def get_panel(self, tickers: List[str], start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Build a MultiIndex price panel for backtesting.
        """
        data = {}
        for ticker in tickers:
            df = self.fetch_ohlcv(ticker, start_date, end_date)
            if not df.empty:
                for col in df.columns:
                    data[(ticker, col)] = df[col]
        
        if not data:
            return pd.DataFrame()
            
        panel = pd.DataFrame(data)
        panel.columns = pd.MultiIndex.from_tuples(panel.columns)
        return panel
