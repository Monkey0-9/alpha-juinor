"""
AlphaVantage Data Provider for Daily and Intraday Data.

Supports:
- Daily adjusted OHLCV
- Intraday 1-minute bars
- Full provenance tracking
"""

import pandas as pd
import requests
import time
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple
from .base import DataProvider

logger = logging.getLogger(__name__)

# API Configuration
ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"


class AlphaVantageProvider(DataProvider):
    """
    AlphaVantage API Provider.
    Free tier: 25 requests/day, 5 requests/minute
    Premium tier: Higher limits

    Supports: Daily, Intraday (1min, 5min, 15min, 30min, 60min)
    """

    supports_ohlcv = True
    supports_latest_quote = True

    def __init__(self, api_key: str = None, enable_cache: bool = True):
        self.api_key = api_key or self._load_api_key()
        self.enable_cache = enable_cache
        self.cache = {} if enable_cache else None
        self._authenticated = bool(self.api_key)

        # Rate limiting
        self._min_interval = 12.0  # seconds (5 requests/minute)
        self._last_request = 0

    def _load_api_key(self) -> str:
        """Load API key from config or environment"""
        import os
        from configs.config_manager import get_config

        # Try config first
        config = get_config()
        api_key = config.get('data_providers', {}).get('alpha_vantage_api_key')
        if api_key:
            return api_key

        # Try environment
        return os.getenv('ALPHA_VANTAGE_API_KEY', '')

    def _rate_limit(self):
        """Enforce API rate limits"""
        now = time.time()
        elapsed = now - self._last_request
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request = time.time()

    def fetch_ohlcv(self, ticker: str, start_date: str,
                    end_date: str = None) -> pd.DataFrame:
        """
        Fetch daily adjusted OHLCV data from AlphaVantage.
        Returns DataFrame with columns: Open, High, Low, Close, Adjusted_Close, Volume
        """
        if end_date is None:
            end_date = datetime.today().strftime('%Y-%m-%d')

        if not self._authenticated:
            logger.warning("AlphaVantage API key not configured")
            return pd.DataFrame()

        # Check cache
        cache_key = f"{ticker}_{start_date}_{end_date}"
        if self.enable_cache and cache_key in self.cache:
            logger.debug(f"Cache hit for {ticker}")
            return self.cache[cache_key]

        self._rate_limit()

        params = {
            'function': 'TIME_SERIES_DAILY_ADJUSTED',
            'symbol': ticker,
            'outputsize': 'full',  # Get full history
            'apikey': self.api_key
        }

        try:
            response = requests.get(ALPHA_VANTAGE_BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            # Parse time series
            time_series_key = 'Time Series (Daily)'
            if time_series_key not in data:
                error_msg = data.get('Note') or data.get('Information') or 'Unknown error'
                logger.warning(f"AlphaVantage API error for {ticker}: {error_msg}")
                return pd.DataFrame()

            time_series = data[time_series_key]

            # Convert to DataFrame
            records = []
            for date_str, values in time_series.items():
                date = datetime.strptime(date_str, '%Y-%m-%d')
                if date >= datetime.strptime(start_date, '%Y-%m-%d') and date <= datetime.strptime(end_date, '%Y-%m-%d'):
                    records.append({
                        'Date': date,
                        'Open': float(values['1. open']),
                        'High': float(values['2. high']),
                        'Low': float(values['3. low']),
                        'Close': float(values['4. close']),
                        'Adjusted_Close': float(values['5. adjusted close']),
                        'Volume': int(values['6. volume'])
                    })

            if not records:
                return pd.DataFrame()

            df = pd.DataFrame(records)
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            df = df.sort_index()

            # Cache result
            if self.enable_cache:
                self.cache[cache_key] = df

            return df

        except Exception as e:
            logger.error(f"AlphaVantage fetch failed for {ticker}: {e}")
            return pd.DataFrame()

    def fetch_intraday(self, ticker: str, start_date: str,
                       end_date: str = None, interval: str = '1min') -> pd.DataFrame:
        """
        Fetch intraday data from AlphaVantage.
        interval: '1min', '5min', '15min', '30min', '60min'
        """
        if end_date is None:
            end_date = datetime.today().strftime('%Y-%m-%d')

        if not self._authenticated:
            logger.warning("AlphaVantage API key not configured")
            return pd.DataFrame()

        self._rate_limit()

        params = {
            'function': f'TIME_SERIES_INTRADAY',
            'symbol': ticker,
            'interval': interval,
            'outputsize': 'full',
            'apikey': self.api_key
        }

        try:
            response = requests.get(ALPHA_VANTAGE_BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            # Parse time series (format depends on interval)
            time_series_key = f'Time Series ({interval})'
            if time_series_key not in data:
                return pd.DataFrame()

            time_series = data[time_series_key]

            records = []
            for timestamp_str, values in time_series.items():
                dt = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                if dt.date() >= datetime.strptime(start_date, '%Y-%m-%d').date() and \
                   dt.date() <= datetime.strptime(end_date, '%Y-%m-%d').date():
                    records.append({
                        'Date': dt.date(),
                        'Time': dt.time(),
                        'Datetime': dt,
                        'Open': float(values['1. open']),
                        'High': float(values['2. high']),
                        'Low': float(values['3. low']),
                        'Close': float(values['4. close']),
                        'Volume': int(values['5. volume'])
                    })

            if not records:
                return pd.DataFrame()

            df = pd.DataFrame(records)
            df = df.sort_values('Datetime')

            return df

        except Exception as e:
            logger.error(f"AlphaVantage intraday fetch failed for {ticker}: {e}")
            return pd.DataFrame()

    async def fetch_ohlcv_async(self, ticker: str, start_date: str,
                                end_date: str = None) -> pd.DataFrame:
        """Async wrapper for OHLCV fetch"""
        import asyncio
        return await asyncio.to_thread(self.fetch_ohlcv, ticker, start_date, end_date)

    async def fetch_intraday_async(self, ticker: str, start_date: str,
                                   end_date: str = None, interval: str = '1min') -> pd.DataFrame:
        """Async wrapper for intraday fetch"""
        import asyncio
        return await asyncio.to_thread(
            self.fetch_intraday, ticker, start_date, end_date, interval
        )

    def get_panel(self, tickers: List[str], start_date: str,
                  end_date: str = None) -> pd.DataFrame:
        """Fetch and combine multiple tickers"""
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

    def get_latest_quote(self, ticker: str) -> Optional[float]:
        """Get latest price quote using Global Quote endpoint"""
        if not self._authenticated:
            return None

        self._rate_limit()

        params = {
            'function': 'GLOBAL_QUOTE',
            'symbol': ticker,
            'apikey': self.api_key
        }

        try:
            response = requests.get(ALPHA_VANTAGE_BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            quote = data.get('Global Quote', {})
            if quote:
                return float(quote.get('05. price', 0))
            return None

        except Exception as e:
            logger.error(f"AlphaVantage quote failed for {ticker}: {e}")
            return None


class AlphaVantageBatchProvider:
    """
    Batch provider for AlphaVantage with optimal query construction.
    AlphaVantage allows batching by using the same API call for all data.
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key or self._load_api_key()
        self._authenticated = bool(self.api_key)

    def _load_api_key(self) -> str:
        import os
        from configs.config_manager import get_config
        config = get_config()
        return config.get('data_providers', {}).get('alpha_vantage_api_key') or os.getenv('ALPHA_VANTAGE_API_KEY', '')

    def fetch_batch_daily(self, tickers: List[str],
                          start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """
        Fetch daily data for multiple tickers.
        AlphaVantage processes one ticker per call, so we iterate.
        """
        results = {}
        provider = AlphaVantageProvider(self.api_key)

        for ticker in tickers:
            df = provider.fetch_ohlcv(ticker, start_date, end_date)
            if not df.empty:
                results[ticker] = df

        return results

