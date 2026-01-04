from abc import ABC, abstractmethod
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Optional, Dict
import numpy as np

class DataProvider(ABC):
    """
    Abstract interface for fetching market data.
    Enables swapping distinct data sources (Yahoo, Bloomberg, Polygon) without changing downstream code.
    """
    
    @abstractmethod
    def fetch_ohlcv(self, ticker: str, start_date: str, end_date: str = None) -> pd.DataFrame:
        """
        Fetch OHLCV data for a given ticker.
        Returns DataFrame with columns: Open, High, Low, Close, Volume.
        index should be DatetimeIndex.
        """
        pass

class YahooDataProvider(DataProvider):
    def fetch_ohlcv(self, ticker: str, start_date: str, end_date: str = None) -> pd.DataFrame:
        print(f"   [Download] Fetching {ticker} from Yahoo Finance...")
        
        if end_date is None:
            end_date = datetime.today().strftime('%Y-%m-%d')
            
        # yfinance download
        df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
        
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
            
        df.index = pd.to_datetime(df.index)
        
        return df

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
