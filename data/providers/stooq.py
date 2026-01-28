"""
Stooq Data Provider - Free Historical Data.

Stooq provides free historical data without API keys.
Supports: Daily OHLCV for stocks, ETFs, indices, and crypto.
"""

import pandas as pd
import requests
import logging
import io
from datetime import datetime, timedelta
from typing import List, Optional
from .base import DataProvider

logger = logging.getLogger(__name__)

# Stooq API Configuration
STOOQ_BASE_URL = "https://stooq.com/q/d/l/"


class StooqProvider(DataProvider):
    """
    Stooq.com Data Provider.

    Advantages:
    - Completely free, no API key required
    - Good historical coverage

    Limitations:
    - Daily data only (no intraday)
    - May have gaps for delisted symbols
    - Rate limited (be respectful)

    Data format: CSV with columns:
    Date,Open,High,Low,Close,Volume
    """

    supports_ohlcv = True
    supports_latest_quote = False

    def __init__(self, enable_cache: bool = True):
        self.enable_cache = enable_cache
        self.cache = {} if enable_cache else None

    def _build_url(self, ticker: str, start_date: str, end_date: str) -> str:
        """Build Stooq CSV download URL"""
        # Stooq uses format: https://stooq.com/q/d/l/?s={ticker}&d1={start}&d2={end}
        # Dates are in YYYYMMDD format
        start_str = start_date.replace('-', '')
        end_str = end_date.replace('-', '')

        return f"{STOOQ_BASE_URL}?s={ticker}&d1={start_str}&d2={end_str}"

    def fetch_ohlcv(self, ticker: str, start_date: str,
                    end_date: str = None) -> pd.DataFrame:
        """
        Fetch daily OHLCV data from Stooq.

        Args:
            ticker: Symbol (e.g., 'aapl.us' for US stocks, 'btc.usd' for crypto)
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format (defaults to today)

        Returns:
            DataFrame with Date, Open, High, Low, Close, Volume columns
        """
        if end_date is None:
            end_date = datetime.today().strftime('%Y-%m-%d')

        # Check cache
        cache_key = f"{ticker}_{start_date}_{end_date}"
        if self.enable_cache and cache_key in self.cache:
            logger.debug(f"Cache hit for {ticker}")
            return self.cache[cache_key]

        # Adjust ticker format for Stooq
        stooq_ticker = self._normalize_ticker(ticker)
        url = self._build_url(stooq_ticker, start_date, end_date)

        try:
            logger.debug(f"Fetching {ticker} from Stooq: {url}")

            response = requests.get(url, timeout=30)
            response.raise_for_status()

            # Parse CSV
            df = pd.read_csv(io.StringIO(response.text))

            if df.empty:
                logger.warning(f"No data returned from Stooq for {ticker}")
                return pd.DataFrame()

            # Ensure required columns exist
            required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in required_cols):
                logger.warning(f"Stooq response missing columns for {ticker}: {df.columns.tolist()}")
                return pd.DataFrame()

            # Parse dates
            df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
            df.set_index('Date', inplace=True)
            df = df.sort_index()

            # Ensure numeric types
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Remove rows with NaN
            df = df.dropna(subset=['Close', 'Volume'])

            # Cache result
            if self.enable_cache and not df.empty:
                self.cache[cache_key] = df

            return df

        except requests.exceptions.RequestException as e:
            logger.error(f"Stooq request failed for {ticker}: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Stooq parsing failed for {ticker}: {e}")
            return pd.DataFrame()

    def _normalize_ticker(self, ticker: str) -> str:
        """
        Normalize ticker symbol for Stooq format.

        Stooq format:
        - US stocks: 'aapl.us' (lowercase, .us suffix)
        - UK stocks: 'bp.l' (lowercase, .l suffix)
        - Crypto: 'btc.usd' (lowercase)
        """
        # Remove common suffixes
        normalized = ticker.upper()

        # Handle common cases
        if '-USD' in normalized or '-USDT' in normalized or '-USDC' in normalized:
            # Crypto pair
            base = normalized.replace('-USD', '').replace('-USDT', '').replace('-USDC', '')
            return f"{base.lower()}.usd"

        if '=' in normalized:
            # Forex pair
            return normalized.replace('=', '').lower()

        if '.' in normalized:
            # Already has suffix, lowercase it
            return normalized.lower()

        # US stock by default
        return f"{normalized.lower()}.us"

    async def fetch_ohlcv_async(self, ticker: str, start_date: str,
                                end_date: str = None) -> pd.DataFrame:
        """Async wrapper for OHLCV fetch"""
        import asyncio
        return await asyncio.to_thread(self.fetch_ohlcv, ticker, start_date, end_date)

    def get_panel(self, tickers: List[str], start_date: str,
                  end_date: str = None) -> pd.DataFrame:
        """Fetch and combine multiple tickers into a panel"""
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

    def fetch_fx_pair(self, pair: str, start_date: str,
                      end_date: str = None) -> pd.DataFrame:
        """
        Fetch forex pair data.
        pair: e.g., 'EURUSD', 'GBPUSD', 'USDJPY'
        """
        # Stooq uses lowercase format: e.g., eur.usd
        normalized = pair.replace('=', '').lower()
        return self.fetch_ohlcv(normalized, start_date, end_date)

    def fetch_crypto(self, symbol: str, quote: str = 'USD',
                     start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Fetch cryptocurrency data.
        symbol: e.g., 'BTC', 'ETH', 'SOL'
        quote: Quote currency (default: USD)
        """
        # Stooq format: btc.usd
        ticker = f"{symbol.lower()}.{quote.lower()}"
        return self.fetch_ohlcv(ticker, start_date, end_date)

    def get_index_components(self, index_symbol: str) -> List[str]:
        """
        Get components of a major index (approximate).
        Note: Stooq doesn't provide index constituents, this is a static mapping.
        """
        sp500_sample = [
            'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA', 'JPM',
            'V', 'JNJ', 'WMT', 'PG', 'MA', 'UNH', 'HD', 'DIS', 'BAC', 'ADBE',
            'CRM', 'PYPL', 'PFE', 'KO', 'TMO', 'COST', 'ABT', 'DHR', 'ABBV',
            'MRK', 'LLY', 'AVGO', 'PEP', 'CSCO', 'MCD', 'WFC', 'T', 'NKE',
            'VZ', 'D', 'NEE', 'XOM', 'UNP', 'ORCL', 'LIN', 'AMGN', 'HON'
        ]

        if index_symbol in ['SPY', 'SP500', '^GSPC']:
            return sp500_sample
        elif index_symbol in ['QQQ', '^IXIC']:
            return sp500_sample  # Simplified
        elif index_symbol in ['DIA', '^DJI']:
            return sp500_sample[:30]  # DJIA subset

        return sp500_sample


class StooqIndexProvider:
    """
    Provider for major index data from Stooq.
    Indices are typically available directly by symbol.
    """

    def fetch_index(self, index_symbol: str, start_date: str,
                    end_date: str = None) -> pd.DataFrame:
        """
        Fetch index data.
        index_symbol: e.g., 'SPX' (S&P 500), 'NDX' (Nasdaq 100), 'DJI' (Dow Jones)
        """
        provider = StooqProvider()

        # Stooq index symbols
        index_map = {
            'SPX': '^spx',      # S&P 500
            'NDX': '^ndx',      # Nasdaq 100
            'DJI': '^dji',      # Dow Jones
            'COMP': '^comp',    # Nasdaq Composite
            'RUT': '^rut',      # Russell 2000
            'DAX': '^dax',      # German DAX
            'FTSE': '^ftse',    # UK FTSE 100
            'NIKKEI': '^nkx',   # Japan Nikkei 225
        }

        stooq_symbol = index_map.get(index_symbol, f"^{index_symbol.lower()}")
        return provider.fetch_ohlcv(stooq_symbol, start_date, end_date)


# Alias for backward compatibility
StooqDataProvider = StooqProvider

