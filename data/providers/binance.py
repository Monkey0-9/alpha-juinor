
import requests
import pandas as pd
import logging
import time
from datetime import datetime
from typing import Optional, List, Dict, Set
import asyncio
import aiohttp
import websockets
import json
from data.providers.base import DataProvider
from utils.timezone import normalize_index_utc

logger = logging.getLogger(__name__)

class BinanceDataProvider(DataProvider):
    """
    Binance Public API Collector (Spot).
    No API Key required for public data.
    """
    supports_ohlcv = True
    supports_latest_quote = True
    _authenticated = True

    BASE_URL = "https://api.binance.com/api/v3"

    def __init__(self):
        # Binance is free, so always authenticated
        self._authenticated = True
        self._session: Optional[aiohttp.ClientSession] = None
        self._price_cache: Dict[str, float] = {}
        self._ws_task: Optional[asyncio.Task] = None
        self._streaming_tickers: Set[str] = set()

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=100, keepalive_timeout=60))
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def get_latest_quote_async(self, symbol: str) -> Optional[float]:
        # Fast-Path: WebSocket Cache
        if symbol in self._price_cache:
            return self._price_cache[symbol]
            
        try:
            formatted_sym = symbol.replace("-", "").replace("/", "").upper()
            if formatted_sym.endswith("USD"): formatted_sym += "T"
            
            url = f"{self.BASE_URL}/ticker/price?symbol={formatted_sym}"
            session = await self._get_session()
            async with session.get(url, timeout=5) as resp:
                data = await resp.json()
                price = float(data['price'])
                self._price_cache[symbol] = price # Update cache for next time
                return price
        except Exception as e:
            logger.warning(f"Binance Async Price Fetch Failed {symbol}: {e}")
            return self._price_cache.get(symbol, 0.0)

    async def start_streaming(self, tickers: List[str]):
        """Starts a persistent WebSocket connection for the given tickers."""
        crypto_tickers = [tk for tk in tickers if "-USD" in tk]
        if not crypto_tickers: return
        
        self._streaming_tickers.update(crypto_tickers)
        if self._ws_task and not self._ws_task.done():
            # Already running, but maybe we need to resubscribe for new tickers?
            # For Binance simplicity, we'll just restart if the ticker set grew significantly
            return

        self._ws_task = asyncio.create_task(self._ws_manager())
        logger.info(f"Binance WebSocket Manager started for {len(self._streaming_tickers)} assets.")

    async def _ws_manager(self):
        """Persistent WebSocket loop with auto-reconnect."""
        while True:
            try:
                # Binance stream names: btcusdt@markPrice or btcusdt@ticker
                # Using @ticker for every 1000ms update, or @aggTrade for real-time
                streams = [f"{tk.replace('-USD', '').lower()}usdt@ticker" for tk in self._streaming_tickers]
                stream_url = f"wss://stream.binance.com:9443/ws/{'/'.join(streams)}"
                
                async with websockets.connect(stream_url) as ws:
                    logger.info("Binance WebSocket Connected.")
                    async for msg in ws:
                        data = json.loads(msg)
                        if 's' in data and 'c' in data:
                            # Normalize symbol back to BTC-USD
                            raw_sym = data['s'].replace('USDT', '')
                            # Add dash back if it's missing (simplified)
                            norm_sym = f"{raw_sym}-USD"
                            self._price_cache[norm_sym] = float(data['c'])
            except Exception as e:
                logger.warning(f"Binance WebSocket Error: {e}. Reconnecting in 5s...")
                await asyncio.sleep(5)

    def get_latest_quote(self, symbol: str) -> Optional[float]:
        # Keep sync for compatibility, but recommend async
        try:
            formatted_sym = symbol.replace("-", "").replace("/", "").upper()
            if formatted_sym.endswith("USD"): formatted_sym += "T"
            url = f"{self.BASE_URL}/ticker/price?symbol={formatted_sym}"
            import requests
            resp = requests.get(url, timeout=5)
            data = resp.json()
            return float(data['price'])
        except Exception as e:
            logger.warning(f"Binance Price Fetch Failed {symbol}: {e}")
            return 0.0

    async def fetch_ohlcv_async(self, symbol: str, start_date: str, end_date: Optional[str] = None, interval: str = "1d") -> pd.DataFrame:
        formatted_sym = symbol.replace("-", "").upper()
        if formatted_sym.endswith("USD"): formatted_sym += "T"
        
        start_ts = int(pd.to_datetime(start_date).timestamp() * 1000)
        end_ts = int(pd.to_datetime(end_date).timestamp() * 1000) if end_date else int(time.time() * 1000)
        
        all_candles = []
        current_start = start_ts
        session = await self._get_session()

        while True:
            params = {"symbol": formatted_sym, "interval": interval, "startTime": current_start, "endTime": end_ts, "limit": 1000}
            try:
                async with session.get(f"{self.BASE_URL}/klines", params=params, timeout=10) as resp:
                    if resp.status_code != 200: break
                    data = await resp.json()
                    if not data: break
                    all_candles.extend(data)
                    current_start = data[-1][0] + 1
                    if current_start >= end_ts: break
            except Exception:
                break
        
        if not all_candles: return pd.DataFrame()
        df = pd.DataFrame(all_candles, columns=["OpenTime", "Open", "High", "Low", "Close", "Volume", "CloseTime", "QuoteAssetVolume", "Trades", "TakerBuyBase", "TakerBuyQuote", "Ignore"])
        df["date"] = pd.to_datetime(df["OpenTime"], unit="ms")
        for col in ["Open", "High", "Low", "Close", "Volume"]: df[col] = df[col].astype(float)
        return df.set_index("date")[["Open", "High", "Low", "Close", "Volume"]]

    def fetch_ohlcv(self, symbol: str, start_date: str, end_date: Optional[str] = None, interval: str = "1d") -> pd.DataFrame:
        # Wrapper around async for compatibility if needed, or keep sync implementation
        # For hot-path we should only use async.
        # Repeating implementation for now to avoid complexity in mixed-mode.
        formatted_sym = symbol.replace("-", "").upper()
        if formatted_sym.endswith("USD"): formatted_sym += "T"
        start_ts = int(pd.to_datetime(start_date).timestamp() * 1000)
        end_ts = int(pd.to_datetime(end_date).timestamp() * 1000) if end_date else int(time.time() * 1000)
        all_candles = []
        current_start = start_ts
        import requests
        while True:
            params = {"symbol": formatted_sym, "interval": interval, "startTime": current_start, "endTime": end_ts, "limit": 1000}
            try:
                resp = requests.get(f"{self.BASE_URL}/klines", params=params, timeout=10)
                if resp.status_code != 200: break
                data = resp.json()
                if not data: break
                all_candles.extend(data)
                current_start = data[-1][0] + 1
                if current_start >= end_ts: break
            except Exception: break
        if not all_candles: return pd.DataFrame()
        df = pd.DataFrame(all_candles, columns=["OpenTime", "Open", "High", "Low", "Close", "Volume", "CloseTime", "QuoteAssetVolume", "Trades", "TakerBuyBase", "TakerBuyQuote", "Ignore"])
        df["date"] = pd.to_datetime(df["OpenTime"], unit="ms")
        for col in ["Open", "High", "Low", "Close", "Volume"]: df[col] = df[col].astype(float)
        return df.set_index("date")[["Open", "High", "Low", "Close", "Volume"]]

    async def get_panel_async(self, tickers: List[str], start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
        """Fetch and combine multiple tickers in parallel."""
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
        """Synchronous panel fetch."""
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
