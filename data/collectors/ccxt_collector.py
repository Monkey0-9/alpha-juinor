# data/collectors/ccxt_collector.py
import asyncio
import ccxt.async_support as ccxt_async
from typing import List, Optional, Dict
import pandas as pd
from datetime import datetime
from .base import DataProvider

class CCXTDataProvider(DataProvider):
    """
    Asynchronous data provider for crypto exchanges using CCXT.
    Default: Binance.
    """
    
    def __init__(self, exchange_id: str = 'binance', config: Dict = None):
        if config is None:
            config = {'enableRateLimit': True}
        
        # Instantiate the exchange
        exchange_class = getattr(ccxt_async, exchange_id)
        self.exchange = exchange_class(config)
        self.timeframe = '1d' # Default to daily bars

    async def _fetch_ohlcv_async(self, symbol: str, start_date: str, limit: int = 1000) -> pd.DataFrame:
        """Internal async method to fetch data."""
        since = self.exchange.parse8601(f"{start_date}T00:00:00Z")
        
        try:
            ohlcv = await self.exchange.fetch_ohlcv(symbol, self.timeframe, since=since, limit=limit)
            if not ohlcv:
                return pd.DataFrame()
                
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            print(f"   [Error] CCXT failed to fetch {symbol}: {e}")
            return pd.DataFrame()

    def fetch_ohlcv(self, ticker: str, start_date: str, end_date: str = None) -> pd.DataFrame:
        """Synchronous wrapper for institutional engine compatibility."""
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # This is tricky if called from an async context, 
            # but for our current engine.run(sync), this works.
            # In a production 24/7 async loop, we'd use await directly.
            import nest_asyncio
            nest_asyncio.apply()
            
        return loop.run_until_complete(self._fetch_ohlcv_async(ticker, start_date))

    def get_panel(self, tickers: List[str], start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
        """Fetch multiple crypto pairs concurrently."""
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
            
        async def fetch_all():
            tasks = [self._fetch_ohlcv_async(tk, start_date) for tk in tickers]
            results = await asyncio.gather(*tasks)
            data = {}
            for ticker, df in zip(tickers, results):
                if not df.empty:
                    for col in df.columns:
                        data[(ticker, col)] = df[col]
            return data

        data = loop.run_until_complete(fetch_all())
        
        if not data:
            return pd.DataFrame()
            
        panel = pd.DataFrame(data)
        panel.columns = pd.MultiIndex.from_tuples(panel.columns)
        return panel

    async def close(self):
        await self.exchange.close()
