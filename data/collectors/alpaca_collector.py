# data/alpaca_provider.py
import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import requests

from config.secrets_manager import secrets
from data.providers.base import DataProvider
from data.router.entitlement_router import router
from utils.retry import retry

logger = logging.getLogger(__name__)


class AlpacaDataProvider(DataProvider):
    """
    Alpaca Markets data provider (FREE for paper trading).
    """

    supports_ohlcv = True
    supports_latest_quote = True

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        base_url: Optional[str] = None,
        paper=True,
    ):
        self.api_key = api_key or secrets.get_secret("ALPACA_API_KEY")
        self.secret_key = secret_key or secrets.get_secret("ALPACA_SECRET_KEY")

        if base_url:
            self.base_url = base_url
        elif paper:
            self.base_url = os.getenv(
                "ALPACA_BASE_URL", "https://paper-api.alpaca.markets"
            )
        else:
            self.base_url = "https://api.alpaca.markets"
        self.data_url = "https://data.alpaca.markets"

        self.headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
        }

        if not self.api_key or not self.secret_key:
            msg = (
                "Alpaca API keys not found. "
                "Set ALPACA_API_KEY and ALPACA_SECRET_KEY env vars."
            )
            logger.warning(msg)
            self._authenticated = False
        else:
            self._authenticated = True

    def _is_crypto(self, ticker: str) -> bool:
        """Helper to identify if ticker is crypto (BTC-USD, ETH/USD etc)."""
        crypto_keywords = ["-USD", "/USD", "BTC", "ETH", "LTC", "DOGE"]
        return any(k in ticker.upper() for k in crypto_keywords)

    @retry(exceptions=(requests.exceptions.RequestException,), tries=3, delay=1, backoff=2)
    def _request_with_retry(
        self, url: str, params: Optional[Dict] = None, **kwargs
    ) -> requests.Response:
        """Retry wrapper for Alpaca Data API requests.
        Institutional: No retry on 403/400.
        """
        response = requests.get(
            url, headers=self.headers, params=params, timeout=10
        )

        # Institutional Rule: Do NOT retry on Permission
        # or Bad Request errors
        if response.status_code in [403, 400]:
            err_msg = (
                f"[ALPACA] Institutional Violation: "
                f"{response.status_code} {response.text}"
            )
            logger.error(err_msg)
            # Raise exception immediately to skip retry
            response.raise_for_status()

        if response.status_code == 429:  # Rate limit
            # This will trigger retry due to RequestException or manual raise
            response.raise_for_status()

        response.raise_for_status()
        return response

    def fetch_ohlcv(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch historical OHLCV bars from Alpaca.

        Free tier: Unlimited for paper trading.
        """
        # --- SECTION A: ROUTER ENFORCEMENT ---
        try:
            start_dt = pd.to_datetime(start_date)
            days = (datetime.now() - start_dt).days
        except ValueError:
            days = 365

        selection = router.select_provider(ticker, days)
        if selection["provider"] != "alpaca":
            provider = selection["provider"]
            logger.info(
                f"[DATA_ROUTER] symbol={ticker} required_days={days} "
                f"selected_provider={provider} reason=Alpaca_not_selected"
            )
            return pd.DataFrame()  # Stop
        # ----------------------------------------

        if not self.api_key:
            logger.error("Cannot fetch data: Alpaca API keys not configured")
            return pd.DataFrame()

        # Ensure dates are in Alpaca friendly format (YYYY-MM-DD or RFC3339)
        try:
            # ROUTING: Stocks use /v2/stocks, Crypto uses /v1beta3/crypto/us
            if self._is_crypto(ticker):
                # Standardize crypto format (Alpaca uses BTC/USD)
                clean_ticker = ticker.upper().replace("-", "/")
                url = f"{self.data_url}/v1beta3/crypto/us/bars"
                params = {
                    "symbols": clean_ticker,
                    "start": start_date,
                    "end": end_date,
                    "timeframe": "1Day",
                }
            else:
                url = f"{self.data_url}/v2/stocks/{ticker}/bars"
                params = {
                    "start": start_date,
                    "end": end_date,
                    "timeframe": "1Day",
                    "limit": 10000,
                    "adjustment": "all",
                }
            # Make Request
            try:
                response = self._request_with_retry(url, params=params)
                if response is None:
                    return pd.DataFrame()

                data = response.json()
            except requests.exceptions.HTTPError as http_err:
                if http_err.response.status_code in [403, 400]:
                    code = http_err.response.status_code
                    router.block_provider("alpaca", ticker, f"HTTP_{code}")
                    return pd.DataFrame()
                raise http_err

            if self._is_crypto(ticker):
                clean_ticker = ticker.upper().replace("-", "/")
                if (
                    "bars" not in data
                    or not data["bars"]
                    or clean_ticker not in data["bars"]
                ):
                    logger.warning(f"No crypto data returned for {ticker}")
                    return pd.DataFrame()
                bars = data["bars"][clean_ticker]
            else:
                if "bars" not in data or not data["bars"]:
                    logger.warning(f"No stock data returned for {ticker}")
                    return pd.DataFrame()
                bars = data["bars"]
            df = pd.DataFrame(bars)

            # Rename columns to match our standard
            df = df.rename(
                columns={
                    "t": "Date",
                    "o": "Open",
                    "h": "High",
                    "l": "Low",
                    "c": "Close",
                    "v": "Volume",
                }
            )

            # Parse dates and enforce UTC
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date")

            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            else:
                df.index = df.index.tz_convert("UTC")

            # Select only OHLCV
            df = df[["Open", "High", "Low", "Close", "Volume"]]

            logger.info(f"Fetched {len(df)} bars for {ticker} from Alpaca")
            return df

            logger.info(f"Fetched {len(df)} bars for {ticker} from Alpaca")
            return df

        except requests.exceptions.HTTPError as http_err:
            if http_err.response.status_code in [403, 400]:
                code = http_err.response.status_code
                logger.error(f"[ALPACA] ENTITLEMENT ERROR: {code} for {ticker}")
                # Mark unusable to prevent retry storms
                try:
                    from data.governance.provider_router import (
                        mark_provider_unavailable,
                    )

                    mark_provider_unavailable("alpaca", ticker)
                except ImportError:
                    pass
            else:
                logger.error(f"Alpaca fetch failed for {ticker}: {http_err}")
            return pd.DataFrame()

        except Exception as err:
            logger.error(f"Alpaca error for {ticker}: {err}")
            return pd.DataFrame()

    async def fetch_ohlcv_async(
        self, ticker: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Async wrapper for Alpaca fetch."""
        import asyncio

        return await asyncio.to_thread(self.fetch_ohlcv, ticker, start_date, end_date)

    def get_all_assets(self, asset_class: str = "us_equity") -> List[Dict]:
        """Fetch all tradable assets from Alpaca."""
        url = f"{self.base_url}/v2/assets"
        params = {"asset_class": asset_class, "status": "active"}
        response = self._request_with_retry(url, params=params)
        data = response.json()
        return [
            {
                "symbol": a["symbol"],
                "exchange": a["exchange"],
                "tradable": a["tradable"],
                "status": a["status"],
            }
            for a in data
            if a["tradable"]
        ]

    def enrich_universe_data(self, tickers: List[str]) -> pd.DataFrame:
        """
        Enrich universe with ADV, Price, and Market Cap.
        (Uses snapshots and fundamental estimates).
        """
        # Batch Fetch Snapshots for Last Price and Vol
        chunk_size = 200
        all_stats = []

        for i in range(0, len(tickers), chunk_size):
            chunk = tickers[i : i + chunk_size]
            url = f"{self.data_url}/v2/stocks/snapshots"
            params = {"symbols": ",".join(chunk)}
            res = self._request_with_retry(url, params=params)
            data = res.json()

            for tk, snap in data.items():
                close = snap.get("latestTrade", {}).get("p", 0.0)
                vol = snap.get("dailyBar", {}).get("v", 0)

                all_stats.append(
                    {
                        "symbol": tk,
                        "last_price": close,
                        "avg_volume": vol,
                        "avg_dollar_volume_30d": close * vol,
                        "market_cap": close * 1e7,
                        "status": "tradable",
                        "exchange": "NYSE",
                        "listed_only": True,
                    }
                )

        return pd.DataFrame(all_stats)

    def get_latest_quote(self, ticker: str) -> Optional[float]:
        """Get latest trade price (real-time)."""
        try:
            if self._is_crypto(ticker):
                clean_ticker = ticker.upper().replace("-", "/")
                url = f"{self.data_url}/v1beta3/crypto/us/latest/trades"
                params = {"symbols": clean_ticker}
                response = self._request_with_retry(url, params=params)
                data = response.json()
                if "trades" in data and clean_ticker in data["trades"]:
                    return float(data["trades"][clean_ticker]["p"])
            else:
                url = f"{self.data_url}/v2/stocks/{ticker}/trades/latest"
                response = self._request_with_retry(url)
                data = response.json()
                if "trade" in data and "p" in data["trade"]:
                    return float(data["trade"]["p"])
            return None
        except Exception as e:
            logger.error(f"Failed to get latest quote for {ticker}: {e}")
            return None

    def get_panel(
        self,
        tickers: List[str],
        start_date: str,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Build a MultiIndex price panel for backtesting.
        """
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

    async def get_panel_async(
        self,
        tickers: List[str],
        start_date: str,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Async wrapper for Alpaca panel fetch."""
        import asyncio

        return await asyncio.to_thread(self.get_panel, tickers, start_date, end_date)

    async def get_latest_quote_async(self, ticker: str) -> Optional[float]:
        """Async wrapper for Alpaca latest quote."""
        import asyncio

        return await asyncio.to_thread(self.get_latest_quote, ticker)

    async def get_latest_prices_async(self, tickers: List[str]) -> Dict[str, float]:
        """Fetch multiple latest prices in parallel."""
        import asyncio

        tasks = [self.get_latest_quote_async(tk) for tk in tickers]
        results = await asyncio.gather(*tasks)
        return {tk: pr for tk, pr in zip(tickers, results) if pr is not None}
