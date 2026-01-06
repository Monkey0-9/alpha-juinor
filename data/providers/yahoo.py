from .base import DataProvider
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import asyncio
from typing import List, Optional, Dict
import numpy as np
from utils.timezone import normalize_index_utc
from data.cache.market_cache import get_cache

class YahooDataProvider(DataProvider):
    # Yahoo capabilities - free provider, always available
    supports_ohlcv = True
    supports_latest_quote = True

    def __init__(self, enable_cache: bool = True, cache_ttl_hours: int = 24):
        self.enable_cache = enable_cache
        self.cache = get_cache(ttl_hours=cache_ttl_hours) if enable_cache else None
        # Yahoo is free, so always authenticated
        self._authenticated = True
    
    async def fetch_ohlcv_async(self, ticker: str, start_date: str, end_date: str = None) -> pd.DataFrame:
        """Async wrapper for blocking yfinance call."""
        return await asyncio.to_thread(self.fetch_ohlcv, ticker, start_date, end_date)

    def fetch_ohlcv(self, ticker: str, start_date: str, end_date: str = None) -> pd.DataFrame:
        if end_date is None:
            end_date = datetime.today().strftime('%Y-%m-%d')
        
        # Try cache first
        if self.enable_cache:
            cached_data = self.cache.get(ticker, start_date, end_date)
            if cached_data is not None:
                print(f"   [Cache Hit] {ticker} from cache")
                return cached_data
        
        print(f"   [Download] Fetching {ticker} from Yahoo Finance...")
            
        # yfinance download
        # yfinance download
        # Suppress yfinance error output for cleaner logs
        import sys
        import os
        import logging
        from contextlib import contextmanager
        
        # Disable yfinance logging to avoid 404 errors cluttering stdout
        logging.getLogger('yfinance').setLevel(logging.CRITICAL)
        logging.getLogger('yfinance').propagate = False

        @contextmanager
        def suppress_stderr():
            with open(os.devnull, "w") as devnull:
                old_stderr = sys.stderr
                sys.stderr = devnull
                try:
                    yield
                finally:
                    sys.stderr = old_stderr

        try:
            with suppress_stderr():
                df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
        except Exception:
            # Catch yfinance internal errors (e.g. delisted)
            return pd.DataFrame()
        
        if df.empty:
            print(f"   [Warning] No data found for {ticker}")
            return pd.DataFrame()
            
        # Standardize columns
        # yfinance might return MultiIndex columns (Price, Ticker) even for one ticker
        if isinstance(df.columns, pd.MultiIndex):
            try:
                # If the second level has the ticker or blank, drop it
                df.columns = df.columns.droplevel(1)
            except Exception:
                pass

        # Ensure standard naming (in case of subtle case differences or extra cols)
        # We only want the standard OHLCV
        # Using .copy() to avoid SettingWithCopy warnings downstream
        try:
            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        except KeyError:
            # Fallback if case is different (e.g. 'Date' index is implicit)
            # Find columns case-insensitively if needed, but usually auto_adjust gives Proper Case
            pass
            
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

    async def get_panel_async(self, tickers: List[str], start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
        """Parallel fetch for Yahoo tickers."""
        tasks = [self.fetch_ohlcv_async(ticker, start_date, end_date) for ticker in tickers]
        dfs = await asyncio.gather(*tasks)
        
        data = {}
        for ticker, df in zip(tickers, dfs):
            if not df.empty:
                for col in df.columns:
                    data[(ticker, col)] = df[col]
        
        if not data: return pd.DataFrame()
        panel = pd.DataFrame(data)
        panel.columns = pd.MultiIndex.from_tuples(panel.columns)
        return panel

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

    async def get_latest_quote_async(self, ticker: str) -> Optional[float]:
        """Async wrapper for latest quote."""
        return await asyncio.to_thread(self.get_latest_quote, ticker)

    def get_latest_quote(self, ticker: str) -> Optional[float]:
        """Fetch the latest price for a ticker using Yahoo Finance."""
        try:
            # We fetch a tiny slice of data to get the latest price
            # Using period='1d' and interval='1m' for real-time-ish price
            data = yf.download(ticker, period="1d", interval="1m", progress=False, auto_adjust=True)
            if not data.empty:
                return float(data["Close"].iloc[-1].iloc[0]) if isinstance(data["Close"].iloc[-1], pd.Series) else float(data["Close"].iloc[-1])
            return None
        except Exception:
            return None
