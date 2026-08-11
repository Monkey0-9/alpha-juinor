import pandas as pd
import yfinance as yf
from typing import List
from datetime import datetime
import logging
from .base import DataSource

logger = logging.getLogger(__name__)

class YahooFinanceSource(DataSource):
    """
    Yahoo Finance data ingestion source using yfinance.
    """

    @property
    def source_name(self) -> str:
        return "yahoo"

    def fetch_historical_ohlcv(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d",
    ) -> pd.DataFrame:
        if not symbols:
            return pd.DataFrame()
        
        # yfinance expects date strings or datetime objects.
        # It handles multiple symbols efficiently if passed as a space-separated string.
        tickers_str = " ".join(symbols)
        logger.info(f"Fetching data from Yahoo Finance for {len(symbols)} symbols...")
        
        try:
            # group_by='ticker' makes the columns MultiIndex if multiple tickers
            df = yf.download(
                tickers_str,
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                interval=interval,
                group_by="ticker",
                auto_adjust=False, # We want raw data for our validation to handle adjustments
                threads=True,
            )
        except Exception as e:
            logger.error(f"Failed to fetch data from Yahoo Finance: {e}")
            raise
            
        if df.empty:
            return pd.DataFrame()
            
        # Process and unpivot the MultiIndex columns if multiple symbols
        processed_dfs = []
        if len(symbols) > 1:
            for symbol in symbols:
                if symbol not in df.columns.levels[0]:
                    continue
                sym_df = df[symbol].copy()
                sym_df['symbol'] = symbol
                processed_dfs.append(sym_df)
            if not processed_dfs:
                return pd.DataFrame()
            combined_df = pd.concat(processed_dfs)
        else:
            combined_df = df.copy()
            combined_df['symbol'] = symbols[0]
            
        # Format the dataframe to canonical schema
        combined_df = combined_df.reset_index()
        # Rename standard columns
        rename_map = {
            'Date': 'timestamp',
            'Datetime': 'timestamp',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'Adj Close': 'adj_close'
        }
        combined_df = combined_df.rename(columns=rename_map)
        
        # Ensure column names are lowercase
        combined_df.columns = [str(c).lower() for c in combined_df.columns]
        
        # Convert timestamp to UTC standard
        if combined_df['timestamp'].dt.tz is None:
            combined_df['timestamp'] = combined_df['timestamp'].dt.tz_localize('UTC')
        else:
            combined_df['timestamp'] = combined_df['timestamp'].dt.tz_convert('UTC')
            
        # Reorder columns
        canonical_cols = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'adj_close']
        available_cols = [c for c in canonical_cols if c in combined_df.columns]
        combined_df = combined_df[available_cols]
        
        return combined_df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
