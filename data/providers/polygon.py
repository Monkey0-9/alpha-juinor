from .base import DataProvider
import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

class PolygonDataProvider(DataProvider):
    """
    Polygon.io Data Provider for EOD validation.
    Requires API key (free tier: 5 calls/minute, 5k/month).
    """

    BASE_URL = "https://api.polygon.io"

    def __init__(self, enable_cache: bool = True, cache_ttl_hours: int = 24):
        self.enable_cache = enable_cache
        from data.cache.market_cache import get_cache
        self.cache = get_cache(ttl_hours=cache_ttl_hours) if enable_cache else None

        # Get API key from config
        from configs.config_manager import get_config
        config = get_config()
        self.api_key = config.get('data_providers', {}).get('polygon_api_key')
        if not self.api_key:
            raise ValueError("Polygon API key not found in config. Add 'data_providers.polygon_api_key' to golden_config.yaml")

    def fetch_ohlcv(self, ticker: str, start_date: str, end_date: str = None) -> pd.DataFrame:
        if end_date is None:
            end_date = datetime.today().strftime('%Y-%m-%d')

        # Try cache first
        if self.enable_cache:
            cached_data = self.cache.get(ticker, start_date, end_date)
            if cached_data is not None:
                print(f"   [Cache Hit] {ticker} from cache")
                return cached_data

        print(f"   [Download] Fetching {ticker} from Polygon.io...")

        # Polygon aggregates API: /v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from}/{to}
        # For EOD: multiplier=1, timespan=day
        start_ts = int(pd.to_datetime(start_date).timestamp() * 1000)
        end_ts = int(pd.to_datetime(end_date).timestamp() * 1000)

        url = f"{self.BASE_URL}/v2/aggs/ticker/{ticker.upper()}/range/1/day/{start_ts}/{end_ts}"
        params = {
            'apiKey': self.api_key,
            'limit': 50000  # Max per request
        }

        try:
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code != 200:
                logger.warning(f"Polygon API Error: {resp.text}")
                return pd.DataFrame()

            data = resp.json()
            if not data.get('results'):
                return pd.DataFrame()

            # Parse results: each bar has t (timestamp ms), o, h, l, c, v, vw, n
            results = data['results']
            df = pd.DataFrame(results)
            df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
            df.set_index('timestamp', inplace=True)

            # Rename columns to standard OHLCV
            df = df.rename(columns={
                'o': 'Open',
                'h': 'High',
                'l': 'Low',
                'c': 'Close',
                'v': 'Volume'
            })

            # Select only OHLCV columns
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

            # Ensure numeric types
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Use centralized utility
            from utils.timezone import normalize_index_utc
            df = normalize_index_utc(df)

            # INSTITUTIONAL: Proactive Validation
            from data.validator import DataValidator
            df = DataValidator.validate_ohlc(df, ticker=ticker)

            # Cache the result
            if self.enable_cache and not df.empty:
                self.cache.set(ticker, start_date, end_date, df)

            return df

        except Exception as e:
            logger.error(f"Polygon fetch failed for {ticker}: {e}")
            return pd.DataFrame()

    def get_panel(self, tickers: List[str], start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
        """Fetch and combine multiple tickers into a MultiIndex panel."""
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
