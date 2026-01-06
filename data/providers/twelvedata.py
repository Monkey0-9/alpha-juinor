from .base import DataProvider
import pandas as pd
import requests
from datetime import datetime
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

class TwelveDataDataProvider(DataProvider):
    """
    TwelveData Data Provider.
    Requires API key. Free tier has 8 calls/minute.
    """

    BASE_URL = "https://api.twelvedata.com"

    def __init__(self, enable_cache: bool = True, cache_ttl_hours: int = 24):
        self.enable_cache = enable_cache
        from data.cache.market_cache import get_cache
        self.cache = get_cache(ttl_hours=cache_ttl_hours) if enable_cache else None

        from configs.config_manager import get_config
        config = get_config()
        self.api_key = config.get('data_providers', {}).get('twelvedata_api_key')
        if not self.api_key:
            import os
            self.api_key = os.getenv("TWELVEDATA_API_KEY")
        
        self._authenticated = bool(self.api_key)

    def fetch_ohlcv(self, ticker: str, start_date: str, end_date: str = None) -> pd.DataFrame:
        if not self.api_key: return pd.DataFrame()
        
        if end_date is None:
            end_date = datetime.today().strftime('%Y-%m-%d')

        if self.enable_cache:
            cached_data = self.cache.get(ticker, start_date, end_date)
            if cached_data is not None:
                return cached_data

        try:
            url = f"{self.BASE_URL}/time_series"
            params = {
                'symbol': ticker.upper(),
                'interval': '1day',
                'start_date': start_date,
                'end_date': end_date,
                'apikey': self.api_key
            }

            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code != 200:
                logger.warning(f"TwelveData API Error: {resp.text}")
                return pd.DataFrame()

            data = resp.json()
            if data.get('status') != 'ok' or 'values' not in data:
                return pd.DataFrame()

            df = pd.DataFrame(data['values'])
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
            df.sort_index(inplace=True)

            # Rename columns
            df = df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })

            # Select only OHLCV
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            from utils.timezone import normalize_index_utc
            df = normalize_index_utc(df)

            if self.enable_cache and not df.empty:
                self.cache.set(ticker, start_date, end_date, df)

            return df
        except Exception as e:
            logger.error(f"TwelveData fetch failed for {ticker}: {e}")
            return pd.DataFrame()

    def get_latest_quote(self, ticker: str) -> Optional[float]:
        if not self.api_key: return None
        try:
            url = f"{self.BASE_URL}/price"
            params = {'symbol': ticker.upper(), 'apikey': self.api_key}
            resp = requests.get(url, params=params, timeout=5)
            if resp.status_code == 200:
                return float(resp.json().get('price', 0))
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
