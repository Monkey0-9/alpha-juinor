from .base import DataProvider
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import asyncio
from typing import List, Optional, Dict
import numpy as np
from mini_quant_fund.utils.timezone import normalize_index_utc
from mini_quant_fund.data.cache.market_cache import get_cache

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
            
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
        except Exception as e:
            print(f"   [Error] yfinance failed for {ticker}: {e}")
            return pd.DataFrame()
        
        if df.empty:
            print(f"   [Warning] No data found for {ticker}")
            return pd.DataFrame()
            
        # Robust MultiIndex flattening
        if isinstance(df.columns, pd.MultiIndex):
            try:
                # If Level 0 is the ticker, we want the price columns from Level 1
                # If Level 1 is the ticker, we want Level 0
                if ticker in df.columns.levels[0]:
                    df.columns = df.columns.droplevel(0)
                else:
                    df.columns = df.columns.droplevel(1)
            except Exception:
                # Fallback: flatten to strings
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

        # Ensure standard naming
        try:
            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        except KeyError:
            # Try lowercase or other common variants if needed, but usually auto_adjust is standard
            pass
            
        # Use centralized utility
        from mini_quant_fund.utils.timezone import normalize_index_utc
        df = normalize_index_utc(df)
        
        # INSTITUTIONAL: Proactive Validation
        from mini_quant_fund.data.validator import DataValidator
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
            data = yf.download(ticker, period="1d", interval="1m", progress=False, auto_adjust=True)
            if not data.empty:
                # Handle possible MultiIndex in latest quote too
                if isinstance(data.columns, pd.MultiIndex):
                    if ticker in data.columns.levels[0]:
                        val = data["Close"].iloc[-1]
                    else:
                        val = data["Close"].iloc[-1] # Usually works if dropped
                else:
                    val = data["Close"].iloc[-1]
                
                if isinstance(val, pd.Series):
                    return float(val.iloc[0])
                return float(val)
            return None
        except Exception:
            return None
