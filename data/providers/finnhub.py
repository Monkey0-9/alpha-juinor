from .base import DataProvider
import pandas as pd
import requests
from datetime import datetime
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

class FinnhubDataProvider(DataProvider):
    """
    Finnhub.io Data Provider.
    Requires API key. Free tier has 60 calls/minute.
    """

    BASE_URL = "https://finnhub.io/api/v1"

    def __init__(self, enable_cache: bool = True, cache_ttl_hours: int = 24):
        self.enable_cache = enable_cache
        from data.cache.market_cache import get_cache
        self.cache = get_cache(ttl_hours=cache_ttl_hours) if enable_cache else None

        from configs.config_manager import get_config
        config = get_config()
        self.api_key = config.get('data_providers', {}).get('finnhub_api_key')
        if not self.api_key:
            # Fallback to env var
            import os
            self.api_key = os.getenv("FINNHUB_API_KEY")
        
        self._authenticated = bool(self.api_key)

    def fetch_ohlcv(self, ticker: str, start_date: str, end_date: str = None) -> pd.DataFrame:
        """Finnhub stock candles: /stock/candle"""
        if not self.api_key: return pd.DataFrame()
        
        if end_date is None:
            end_date = datetime.today().strftime('%Y-%m-%d')

        if self.enable_cache:
            cached_data = self.cache.get(ticker, start_date, end_date)
            if cached_data is not None:
                return cached_data

        try:
            start_ts = int(pd.to_datetime(start_date).timestamp())
            end_ts = int(pd.to_datetime(end_date).timestamp())

            url = f"{self.BASE_URL}/stock/candle"
            params = {
                'symbol': ticker.upper(),
                'resolution': 'D',
                'from': start_ts,
                'to': end_ts,
                'token': self.api_key
            }

            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code != 200:
                logger.warning(f"Finnhub API Error: {resp.text}")
                return pd.DataFrame()

            data = resp.json()
            if data.get('s') != 'ok':
                return pd.DataFrame()

            df = pd.DataFrame({
                'Open': data['o'],
                'High': data['h'],
                'Low': data['l'],
                'Close': data['c'],
                'Volume': data['v'],
                'timestamp': pd.to_datetime(data['t'], unit='s')
            })
            df.set_index('timestamp', inplace=True)
            
            from utils.timezone import normalize_index_utc
            df = normalize_index_utc(df)

            if self.enable_cache and not df.empty:
                self.cache.set(ticker, start_date, end_date, df)

            return df
        except Exception as e:
            logger.error(f"Finnhub fetch failed for {ticker}: {e}")
            return pd.DataFrame()

    def get_latest_quote(self, ticker: str) -> Optional[float]:
        """Finnhub quote: /quote"""
        if not self.api_key: return None
        try:
            url = f"{self.BASE_URL}/quote"
            params = {'symbol': ticker.upper(), 'token': self.api_key}
            resp = requests.get(url, params=params, timeout=5)
            if resp.status_code == 200:
                return float(resp.json().get('c', 0))
            return None
        except Exception:
            return None

    def get_panel(self, tickers: List[str], start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
        data = {}
        for ticker in tickers:
            df = self.fetch_ohlcv(ticker, start_date, end_date)
            if not df.empty:
                for col in df.columns:
                    data[(ticker, col)] = df[col]
        if not data: return pd.DataFrame()
        panel = pd.DataFrame(data)
        panel.columns = pd.MultiIndex.from_tuples(panel.columns)
        return panel
