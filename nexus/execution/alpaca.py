import os
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import aiohttp
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from dotenv import load_dotenv
from nexus.research.simulator import TradeSimulator

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

ALPACA_API_BASE_URL = "https://api.alpaca.markets"
ALPACA_PAPER_BASE_URL = "https://paper-api.alpaca.markets"
ALPACA_DATA_BASE_URL = "https://data.alpaca.markets"

@dataclass
class AlpacaCredentials:
    api_key: str
    api_secret: str
    paper_trading: bool = True

    def get_headers(self) -> Dict[str, str]:
        return {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
            "Content-Type": "application/json"
        }

class AlpacaClient:
    """Alpaca execution client with live and simulated paper trading fallback."""

    def __init__(self, credentials: Optional[AlpacaCredentials] = None):
        self.session: Optional[aiohttp.ClientSession] = None
        self.simulator = TradeSimulator()
        self.simulated = False

        if credentials is None:
            api_key = os.getenv("ALPACA_API_KEY", "")
            api_secret = os.getenv("ALPACA_API_SECRET", "")
            paper_trading = os.getenv("ALPACA_PAPER_TRADING", "true").lower() == "true"
            if api_key and api_secret:
                # Strip potential whitespace or quotes from copy-paste errors
                api_key = api_key.strip().strip('"').strip("'")
                api_secret = api_secret.strip().strip('"').strip("'")
                self.credentials = AlpacaCredentials(api_key, api_secret, paper_trading)
                self.enabled = True
            else:
                logger.warning("Alpaca credentials missing. Using simulated paper trading mode.")
                self.credentials = None
                self.enabled = True
                self.simulated = True
        else:
            self.credentials = credentials
            self.enabled = True

        self.base_url = ALPACA_PAPER_BASE_URL if self.credentials and self.credentials.paper_trading else ALPACA_API_BASE_URL
        self.data_url = ALPACA_DATA_BASE_URL

    async def _get_session(self) -> aiohttp.ClientSession:
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session

    async def close(self) -> None:
        if self.session and not self.session.closed:
            await self.session.close()

    async def get_account(self) -> Dict[str, Any]:
        if self.simulated:
            account = self.simulator.get_account(self._current_prices())
            # If we fell back due to missing credentials, report it
            err = "Invalid API Keys in .env" if self.credentials else "No keys provided"
            return {**account, "enabled": True, "simulated": True, "error": err}

        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/v2/account", headers=self.credentials.get_headers()) as response:
                data = await response.json()
                if response.status in {200, 201}:
                    return {
                        "enabled": True,
                        "simulated": False,
                        "status": data.get("status"),
                        "account_id": data.get("id"),
                        "buying_power": float(data.get("buying_power", 0)),
                        "cash": float(data.get("cash", 0)),
                        "portfolio_value": float(data.get("portfolio_value", 0)),
                        "equity": float(data.get("equity", 0)),
                        "last_equity": float(data.get("last_equity", 0)),
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                if response.status == 401:
                    logger.warning("Alpaca API Keys invalid (401). Falling back to full simulation mode.")
                    self.simulated = True
                    return await self.get_account()
                return {"enabled": True, "simulated": False, "status": "ERROR", "error": f"HTTP {response.status}"}
        except Exception as e:
            logger.warning(f"Alpaca account request failed: {e}")
            return {"enabled": True, "simulated": False, "error": str(e)}

    async def get_positions(self) -> List[Dict[str, Any]]:
        if self.simulated:
            return [
                {
                    "symbol": pos["symbol"],
                    "qty": float(pos["qty"]),
                    "avg_price": float(pos["avg_price"]),
                    "market_value": float(pos["qty"] * self._current_prices().get(pos["symbol"], 0.0)),
                    "unrealized_pl": 0.0,
                    "unrealized_plpc": 0.0,
                    "current_price": float(self._current_prices().get(pos["symbol"], 0.0)),
                    "side": "long" if pos["qty"] > 0 else "short"
                }
                for pos in self.simulator.get_positions()
            ]

        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/v2/positions", headers=self.credentials.get_headers()) as response:
                if response.status in {200, 201}:
                    positions = await response.json()
                    return [{
                        "symbol": p.get("symbol"),
                        "qty": float(p.get("qty", 0)),
                        "avg_price": float(p.get("avg_entry_price", 0)),
                        "market_value": float(p.get("market_value", 0)),
                        "unrealized_pl": float(p.get("unrealized_pl", 0)),
                        "unrealized_plpc": float(p.get("unrealized_plpc", 0)),
                        "current_price": float(p.get("current_price", 0)),
                        "side": p.get("side")
                    } for p in positions]
                return []
        except Exception as e:
            logger.warning(f"Failed to fetch Alpaca positions: {e}")
            return []

    async def submit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        asset_class: str = "equity",
        order_type: str = "market",
        time_in_force: str = "day",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        strategy: Optional[str] = None,
        extended_hours: bool = False
    ) -> Dict[str, Any]:
        symbol = symbol.upper()
        if self.simulated:
            bars = await self.get_bars(symbol, timeframe="1Min", limit=1)
            current_price = float(bars[-1]["close"]) if bars else 100.0
            if order_type == "limit" and limit_price is not None:
                result = self.simulator.execute_limit_order(symbol, qty, limit_price, side)
            else:
                result = self.simulator.execute_market_order(symbol, qty, current_price, side)
            return {
                **result,
                "success": True,
                "simulated": True,
                "asset_class": asset_class,
                "strategy": strategy or "default"
            }

        if asset_class != "equity":
            return {
                "success": False,
                "error": f"Unsupported asset class '{asset_class}' for live Alpaca execution. Use paper mode or equity instruments.",
                "asset_class": asset_class
            }

        try:
            session = await self._get_session()
            order_data = {
                "symbol": symbol,
                "qty": str(qty),
                "side": side,
                "type": order_type,
                "time_in_force": time_in_force,
                "extended_hours": extended_hours
            }
            if limit_price is not None:
                order_data["limit_price"] = str(round(limit_price, 4))
            if stop_price is not None:
                order_data["stop_price"] = str(round(stop_price, 4))
            if strategy:
                order_data["client_order_id"] = strategy[:32]

            async with session.post(
                f"{self.base_url}/v2/orders",
                headers=self.credentials.get_headers(),
                json=order_data
            ) as response:
                data = await response.json()
                if response.status in {200, 201}:
                    return {"success": True, "order_id": data.get("id"), "status": data.get("status"), "asset_class": asset_class}
                logger.error(f"Alpaca order rejected: {response.status} {data}")
                return {"success": False, "error": data.get("message", data), "asset_class": asset_class}
        except Exception as e:
            logger.error(f"Order submit failed: {e}")
            return {"success": False, "error": str(e), "asset_class": asset_class}

    async def get_orders(self, status: str = "all", limit: int = 50) -> List[Dict[str, Any]]:
        if self.simulated:
            return self.simulator.order_history[-limit:]

        try:
            session = await self._get_session()
            params = {"status": status, "limit": limit}
            async with session.get(f"{self.base_url}/v2/orders", headers=self.credentials.get_headers(), params=params) as response:
                if response.status in {200, 201}:
                    return await response.json()
                return []
        except Exception as e:
            logger.warning(f"Failed to fetch Alpaca orders: {e}")
            return []

    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        if self.simulated:
            return {"success": True, "order_id": order_id, "simulated": True}

        try:
            session = await self._get_session()
            async with session.delete(f"{self.base_url}/v2/orders/{order_id}", headers=self.credentials.get_headers()) as response:
                if response.status in {200, 201}:
                    return {"success": True, "order_id": order_id}
                data = await response.json()
                return {"success": False, "error": data.get("message", "Unable to cancel order")}
        except Exception as e:
            logger.warning(f"Order cancel failed: {e}")
            return {"success": False, "error": str(e)}

    async def get_assets(self, asset_class: str = "us_equity", status: str = "active", tradable: bool = True, page_size: int = 200) -> List[Dict[str, Any]]:
        """Fetch the full Alpaca asset universe for the requested asset class."""
        if self.simulated:
            return self._generate_simulated_universe()

        assets: List[Dict[str, Any]] = []
        page_token: Optional[str] = None

        try:
            session = await self._get_session()
            while True:
                params = {
                    "asset_class": asset_class,
                    "status": status,
                    "tradable": str(tradable).lower(),
                    "limit": page_size,
                }
                if page_token:
                    params["page_token"] = page_token

                async with session.get(
                    f"{self.base_url}/v2/assets",
                    headers=self.credentials.get_headers(),
                    params=params
                ) as response:
                    if response.status not in {200, 201}:
                        logger.warning(f"Failed to fetch assets: HTTP {response.status}")
                        break

                    result = await response.json()
                    if isinstance(result, dict):
                        page_assets = result.get("assets", [])
                        page_token = result.get("next_page_token") or response.headers.get("x-next-page-token")
                    else:
                        page_assets = result
                        page_token = response.headers.get("x-next-page-token")

                    if not page_assets:
                        break

                    assets.extend(page_assets)
                    if not page_token or len(page_assets) < page_size:
                        break

            return assets
        except Exception as e:
            logger.warning(f"Failed to fetch Alpaca assets: {e}")
            return []

    async def get_clock(self) -> Dict[str, Any]:
        if self.simulated:
            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "is_open": False,
                "next_open": None,
                "next_close": None,
                "session": "SIMULATED"
            }

        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/v2/clock", headers=self.credentials.get_headers()) as response:
                if response.status in {200, 201}:
                    return await response.json()
                return {"is_open": False}
        except Exception as e:
            logger.warning(f"Failed to fetch market clock: {e}")
            return {"is_open": False}

    async def get_bars(
        self,
        symbol: str,
        timeframe: str = "1Min",
        limit: int = 100,
        start: str = None,
        feed: str = "iex"
    ) -> List[Dict[str, Any]]:
        symbol = symbol.upper()
        if self.simulated:
            return self._generate_synthetic_bars(symbol, timeframe, limit)

        try:
            session = await self._get_session()
            for feed_candidate in [feed, "sip"]:
                params = {"timeframe": timeframe, "limit": limit, "feed": feed_candidate}
                if start:
                    params["start"] = start
                async with session.get(
                    f"{self.data_url}/v2/stocks/{symbol}/bars",
                    headers=self.credentials.get_headers(),
                    params=params
                ) as response:
                    if response.status in {200, 201}:
                        data = await response.json()
                        bars = data.get("bars") or []
                        if bars:
                            return bars
                        break
                    if response.status in {403, 422}:
                        logger.warning(
                            f"Alpaca bars request for {symbol} failed with feed={feed_candidate}: {response.status}. Trying alternate feed."
                        )
                        continue
                    if response.status == 401:
                        logger.warning("Alpaca API Keys invalid (401). Falling back to full simulation mode.")
                        self.simulated = True
                        return await self.get_bars(symbol, timeframe, limit, start, feed)
                    logger.warning(
                        f"Alpaca bars request for {symbol} returned HTTP {response.status}: {await response.text()}"
                    )
                    break
            
            # Final fallback: retry once after a short sleep if it's a transient network error
            return []
        except aiohttp.ClientConnectorError as e:
            logger.error(f"DNS or Connection Error for {symbol}: {e}. Retrying with alternate DNS logic if possible.")
            # This is where we'd implement alternate resolver if needed, for now just returning empty to allow engine to use fallback
            return []
        except Exception as e:
            logger.warning(f"Failed to fetch bars for {symbol}: {e}")
            return []

    async def close_position(self, symbol: str) -> Dict[str, Any]:
        symbol = symbol.upper()
        if self.simulated:
            bars = await self.get_bars(symbol, timeframe="1Min", limit=1)
            current_price = float(bars[-1]["close"]) if bars else 100.0
            result = self.simulator.close_position(symbol, current_price)
            return {**result, "success": True, "simulated": True}

        try:
            session = await self._get_session()
            async with session.delete(
                f"{self.base_url}/v2/positions/{symbol}",
                headers=self.credentials.get_headers()
            ) as response:
                if response.status in {200, 201}:
                    return {"success": True, "symbol": symbol}
                return {"success": False, "error": "Failed to close"}
        except Exception as e:
            logger.warning(f"Failed to close position for {symbol}: {e}")
            return {"success": False, "error": str(e)}

    def _generate_synthetic_bars(self, symbol: str, timeframe: str, limit: int) -> List[Dict[str, Any]]:
        now = datetime.now(timezone.utc)
        prices = np.cumprod(1 + np.random.normal(0, 0.0008, limit)) * 100
        bars = []
        for idx in range(limit):
            ts = now - pd.Timedelta(minutes=(limit - idx))
            bars.append({
                "t": ts.isoformat() + "Z",
                "o": float(prices[idx] * (1 - np.random.random() * 0.001)),
                "h": float(prices[idx] * (1 + np.random.random() * 0.001)),
                "l": float(prices[idx] * (1 - np.random.random() * 0.001)),
                "c": float(prices[idx]),
                "v": int(1000 + np.random.randint(0, 1000)),
                "close": float(prices[idx])
            })
        return bars

    def _current_prices(self) -> Dict[str, float]:
        return {pos["symbol"]: 100.0 for pos in self.simulator.get_positions()}

    def _generate_simulated_universe(self) -> List[Dict[str, Any]]:
        return [
            {"symbol": s, "name": f"{s} Corp", "exchange": "NASDAQ", "tradable": True, "status": "active"}
            for s in ["AAPL", "MSFT", "GOOG", "AMZN", "NVDA", "SPY", "QQQ"]
        ]

_client: Optional[AlpacaClient] = None

def get_client() -> AlpacaClient:
    global _client
    if _client is None:
        _client = AlpacaClient()
    return _client
