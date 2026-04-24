"""
HUGEFUNDS - Alpaca Paper Trading Integration
Real Trading Execution Module - Beyond Simulation
"""

import os
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import aiohttp
import asyncio

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    if os.path.exists(env_path):
        load_dotenv(env_path)
        logging.getLogger('HugeFunds.Alpaca').debug(f"Loaded .env from {env_path}")
except ImportError:
    pass  # python-dotenv not installed, rely on system env vars

logger = logging.getLogger('HugeFunds.Alpaca')

# Alpaca API Configuration
ALPACA_PAPER_BASE_URL = "https://paper-api.alpaca.markets"
ALPACA_DATA_BASE_URL = "https://data.alpaca.markets"
ALPACA_MARKETS_URL = "https://api.alpaca.markets"

@dataclass
class AlpacaCredentials:
    """Alpaca API credentials"""
    api_key: str
    api_secret: str
    paper_trading: bool = True
    
    def get_headers(self) -> Dict[str, str]:
        """Get API request headers"""
        return {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
            "Content-Type": "application/json"
        }

class AlpacaClient:
    """
    Alpaca API Client for Paper Trading
    Enables real order execution, position management, and account monitoring
    """
    
    def __init__(self, credentials: Optional[AlpacaCredentials] = None):
        """
        Initialize Alpaca client
        
        Args:
            credentials: Alpaca API credentials (optional, will try env vars)
        """
        if credentials is None:
            # Try to load from environment variables
            api_key = os.getenv("ALPACA_API_KEY", "")
            api_secret = os.getenv("ALPACA_API_SECRET", "")
            paper_trading = os.getenv("ALPACA_PAPER_TRADING", "true").lower() == "true"
            
            if not api_key or not api_secret:
                logger.warning("Alpaca credentials not found. Trading functionality disabled.")
                self.credentials = None
                self.enabled = False
            else:
                self.credentials = AlpacaCredentials(api_key, api_secret, paper_trading)
                self.enabled = True
                logger.info(f"Alpaca client initialized (Paper Trading: {paper_trading})")
        else:
            self.credentials = credentials
            self.enabled = True
            logger.info(f"Alpaca client initialized (Paper Trading: {credentials.paper_trading})")
        
        self.base_url = ALPACA_PAPER_BASE_URL
        self.data_url = ALPACA_DATA_BASE_URL
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def close(self):
        """Close the client session"""
        if self.session and not self.session.closed:
            await self.session.close()
    
    async def get_account(self) -> Dict[str, Any]:
        """
        Get account information
        
        Returns:
            Account details including buying power, equity, etc.
        """
        if not self.enabled:
            return {"error": "Alpaca not configured", "enabled": False}
        
        try:
            session = await self._get_session()
            headers = self.credentials.get_headers()
            
            async with session.get(
                f"{self.base_url}/v2/account",
                headers=headers
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"Account info retrieved: {data.get('status', 'unknown')}")
                    return {
                        "enabled": True,
                        "status": "connected",
                        "account_id": data.get("id"),
                        "account_number": data.get("account_number"),
                        "status": data.get("status"),
                        "currency": data.get("currency"),
                        "buying_power": float(data.get("buying_power", 0)),
                        "cash": float(data.get("cash", 0)),
                        "portfolio_value": float(data.get("portfolio_value", 0)),
                        "equity": float(data.get("equity", 0)),
                        "initial_margin": float(data.get("initial_margin", 0)),
                        "maintenance_margin": float(data.get("maintenance_margin", 0)),
                        "daytrade_count": data.get("daytrade_count", 0),
                        "last_equity": float(data.get("last_equity", 0)),
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to get account: {response.status} - {error_text}")
                    return {
                        "enabled": True,
                        "status": "error",
                        "error": f"HTTP {response.status}: {error_text}",
                        "timestamp": datetime.now().isoformat()
                    }
        except Exception as e:
            logger.error(f"Exception getting account: {e}")
            return {
                "enabled": True,
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get current positions
        
        Returns:
            List of current open positions
        """
        if not self.enabled:
            return []
        
        try:
            session = await self._get_session()
            headers = self.credentials.get_headers()
            
            async with session.get(
                f"{self.base_url}/v2/positions",
                headers=headers
            ) as response:
                if response.status == 200:
                    positions = await response.json()
                    logger.info(f"Retrieved {len(positions)} positions")
                    
                    formatted_positions = []
                    for pos in positions:
                        formatted_positions.append({
                            "symbol": pos.get("symbol"),
                            "exchange": pos.get("exchange"),
                            "asset_class": pos.get("asset_class"),
                            "qty": float(pos.get("qty", 0)),
                            "avg_entry_price": float(pos.get("avg_entry_price", 0)),
                            "side": pos.get("side"),
                            "market_value": float(pos.get("market_value", 0)),
                            "cost_basis": float(pos.get("cost_basis", 0)),
                            "unrealized_pl": float(pos.get("unrealized_pl", 0)),
                            "unrealized_plpc": float(pos.get("unrealized_plpc", 0)),
                            "current_price": float(pos.get("current_price", 0)),
                            "lastday_price": float(pos.get("lastday_price", 0)),
                            "change_today": float(pos.get("change_today", 0)),
                            "timestamp": datetime.now().isoformat()
                        })
                    
                    return formatted_positions
                elif response.status == 404:
                    # No positions
                    return []
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to get positions: {response.status} - {error_text}")
                    return []
        except Exception as e:
            logger.error(f"Exception getting positions: {e}")
            return []
    
    async def submit_order(
        self,
        symbol: str,
        qty: float,
        side: str,  # "buy" or "sell"
        order_type: str = "market",  # "market", "limit", "stop", "stop_limit"
        time_in_force: str = "day",  # "day", "gtc", "opg", "cls", "ioc", "fok"
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        client_order_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Submit an order to Alpaca
        
        Args:
            symbol: Stock symbol (e.g., "AAPL")
            qty: Quantity to trade
            side: "buy" or "sell"
            order_type: Order type
            time_in_force: Time in force
            limit_price: Limit price (for limit orders)
            stop_price: Stop price (for stop orders)
            client_order_id: Custom order ID
            
        Returns:
            Order details including order ID and status
        """
        if not self.enabled:
            return {"error": "Alpaca not configured", "enabled": False}
        
        # Validate inputs
        if side not in ["buy", "sell"]:
            return {"error": f"Invalid side: {side}. Must be 'buy' or 'sell'"}
        
        if qty <= 0:
            return {"error": f"Invalid quantity: {qty}. Must be positive"}
        
        try:
            session = await self._get_session()
            headers = self.credentials.get_headers()
            
            # Build order data
            order_data = {
                "symbol": symbol.upper(),
                "qty": str(qty),
                "side": side,
                "type": order_type,
                "time_in_force": time_in_force
            }
            
            if limit_price:
                order_data["limit_price"] = str(limit_price)
            if stop_price:
                order_data["stop_price"] = str(stop_price)
            if client_order_id:
                order_data["client_order_id"] = client_order_id
            
            logger.info(f"Submitting {side} order: {qty} shares of {symbol}")
            
            async with session.post(
                f"{self.base_url}/v2/orders",
                headers=headers,
                json=order_data
            ) as response:
                data = await response.json()
                
                if response.status == 200:
                    logger.info(f"Order submitted successfully: {data.get('id')}")
                    return {
                        "success": True,
                        "order_id": data.get("id"),
                        "client_order_id": data.get("client_order_id"),
                        "symbol": data.get("symbol"),
                        "qty": float(data.get("qty", 0)),
                        "side": data.get("side"),
                        "type": data.get("type"),
                        "status": data.get("status"),
                        "created_at": data.get("created_at"),
                        "filled_qty": float(data.get("filled_qty", 0)),
                        "filled_avg_price": float(data.get("filled_avg_price", 0)) if data.get("filled_avg_price") else None,
                        "message": f"Order {data.get('status')}"
                    }
                else:
                    logger.error(f"Order failed: {data}")
                    return {
                        "success": False,
                        "error": data.get("message", "Unknown error"),
                        "code": data.get("code"),
                        "symbol": symbol,
                        "qty": qty,
                        "side": side
                    }
        except Exception as e:
            logger.error(f"Exception submitting order: {e}")
            return {
                "success": False,
                "error": str(e),
                "symbol": symbol,
                "qty": qty,
                "side": side
            }
    
    async def get_orders(
        self,
        status: Optional[str] = None,
        limit: int = 50,
        after: Optional[str] = None,
        until: Optional[str] = None,
        direction: str = "desc"
    ) -> List[Dict[str, Any]]:
        """
        Get orders
        
        Args:
            status: Filter by status (open, closed, all)
            limit: Maximum number of orders to return
            after: Filter orders after this timestamp
            until: Filter orders until this timestamp
            direction: Sort direction (asc, desc)
            
        Returns:
            List of orders
        """
        if not self.enabled:
            return []
        
        try:
            session = await self._get_session()
            headers = self.credentials.get_headers()
            
            params = {
                "limit": limit,
                "direction": direction
            }
            if status:
                params["status"] = status
            if after:
                params["after"] = after
            if until:
                params["until"] = until
            
            async with session.get(
                f"{self.base_url}/v2/orders",
                headers=headers,
                params=params
            ) as response:
                if response.status == 200:
                    orders = await response.json()
                    logger.info(f"Retrieved {len(orders)} orders")
                    
                    formatted_orders = []
                    for order in orders:
                        formatted_orders.append({
                            "order_id": order.get("id"),
                            "client_order_id": order.get("client_order_id"),
                            "symbol": order.get("symbol"),
                            "qty": float(order.get("qty", 0)),
                            "filled_qty": float(order.get("filled_qty", 0)),
                            "side": order.get("side"),
                            "type": order.get("type"),
                            "status": order.get("status"),
                            "created_at": order.get("created_at"),
                            "filled_avg_price": float(order.get("filled_avg_price", 0)) if order.get("filled_avg_price") else None,
                            "limit_price": float(order.get("limit_price", 0)) if order.get("limit_price") else None,
                            "stop_price": float(order.get("stop_price", 0)) if order.get("stop_price") else None
                        })
                    
                    return formatted_orders
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to get orders: {response.status} - {error_text}")
                    return []
        except Exception as e:
            logger.error(f"Exception getting orders: {e}")
            return []
    
    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Cancel an open order
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            Cancellation result
        """
        if not self.enabled:
            return {"error": "Alpaca not configured", "enabled": False}
        
        try:
            session = await self._get_session()
            headers = self.credentials.get_headers()
            
            async with session.delete(
                f"{self.base_url}/v2/orders/{order_id}",
                headers=headers
            ) as response:
                if response.status == 200:
                    logger.info(f"Order cancelled: {order_id}")
                    return {
                        "success": True,
                        "order_id": order_id,
                        "message": "Order cancelled successfully"
                    }
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to cancel order: {response.status} - {error_text}")
                    return {
                        "success": False,
                        "order_id": order_id,
                        "error": f"HTTP {response.status}: {error_text}"
                    }
        except Exception as e:
            logger.error(f"Exception cancelling order: {e}")
            return {
                "success": False,
                "order_id": order_id,
                "error": str(e)
            }
    
    async def close_position(self, symbol: str) -> Dict[str, Any]:
        """
        Close a position (liquidate)
        
        Args:
            symbol: Symbol of position to close
            
        Returns:
            Close result
        """
        if not self.enabled:
            return {"error": "Alpaca not configured", "enabled": False}
        
        try:
            session = await self._get_session()
            headers = self.credentials.get_headers()
            
            async with session.delete(
                f"{self.base_url}/v2/positions/{symbol}",
                headers=headers
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"Position closed: {symbol}")
                    return {
                        "success": True,
                        "symbol": symbol,
                        "qty": float(data.get("qty", 0)),
                        "side": data.get("side"),
                        "status": data.get("status"),
                        "message": f"Position in {symbol} closed"
                    }
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to close position: {response.status} - {error_text}")
                    return {
                        "success": False,
                        "symbol": symbol,
                        "error": f"HTTP {response.status}: {error_text}"
                    }
        except Exception as e:
            logger.error(f"Exception closing position: {e}")
            return {
                "success": False,
                "symbol": symbol,
                "error": str(e)
            }
    
    async def close_all_positions(self) -> Dict[str, Any]:
        """
        Close all positions (EMERGENCY KILL SWITCH)
        
        Returns:
            Close all result
        """
        if not self.enabled:
            return {"error": "Alpaca not configured", "enabled": False}
        
        try:
            session = await self._get_session()
            headers = self.credentials.get_headers()
            
            # Get current positions first
            positions = await self.get_positions()
            
            # Close all positions
            async with session.delete(
                f"{self.base_url}/v2/positions",
                headers=headers
            ) as response:
                if response.status == 200:
                    logger.info("ALL POSITIONS CLOSED - EMERGENCY KILL SWITCH ACTIVATED")
                    return {
                        "success": True,
                        "message": "EMERGENCY: All positions liquidated",
                        "positions_closed": len(positions),
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to close all positions: {response.status} - {error_text}")
                    return {
                        "success": False,
                        "error": f"HTTP {response.status}: {error_text}"
                    }
        except Exception as e:
            logger.error(f"Exception closing all positions: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_clock(self) -> Dict[str, Any]:
        """
        Get market clock (open/close status)
        
        Returns:
            Market clock information
        """
        if not self.enabled:
            return {"error": "Alpaca not configured", "enabled": False}
        
        try:
            session = await self._get_session()
            headers = self.credentials.get_headers()
            
            async with session.get(
                f"{self.base_url}/v2/clock",
                headers=headers
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "enabled": True,
                        "timestamp": data.get("timestamp"),
                        "is_open": data.get("is_open"),
                        "next_open": data.get("next_open"),
                        "next_close": data.get("next_close"),
                        "message": "Market is OPEN" if data.get("is_open") else "Market is CLOSED"
                    }
                else:
                    error_text = await response.text()
                    return {
                        "enabled": True,
                        "error": f"HTTP {response.status}: {error_text}"
                    }
        except Exception as e:
            return {
                "enabled": True,
                "error": str(e)
            }
    
    async def get_news(self, symbol: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get news for a specific symbol
        
        Args:
            symbol: Stock symbol
            limit: Number of news items to return
            
        Returns:
            List of news items
        """
        if not self.enabled:
            return []
        
        try:
            session = await self._get_session()
            headers = self.credentials.get_headers()
            
            async with session.get(
                f"{self.data_url}/v1beta1/news?symbols={symbol}&limit={limit}",
                headers=headers
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("news", [])
                else:
                    logger.warning(f"Failed to get news for {symbol}: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Error getting news: {e}")
            return []
    
    async def get_calendar(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get market calendar
        
        Args:
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            
        Returns:
            List of trading days
        """
        if not self.enabled:
            return []
        
        try:
            session = await self._get_session()
            headers = self.credentials.get_headers()
            
            params = {}
            if start:
                params["start"] = start
            if end:
                params["end"] = end
            
            async with session.get(
                f"{self.base_url}/v2/calendar",
                headers=headers,
                params=params
            ) as response:
                if response.status == 200:
                    calendar = await response.json()
                    return calendar
                else:
                    return []
        except Exception as e:
            logger.error(f"Exception getting calendar: {e}")
            return []

# Global Alpaca client instance
alpaca_client: Optional[AlpacaClient] = None

def get_alpaca_client() -> AlpacaClient:
    """Get or create Alpaca client singleton"""
    global alpaca_client
    if alpaca_client is None:
        alpaca_client = AlpacaClient()
    return alpaca_client

async def initialize_alpaca():
    """Initialize Alpaca client on startup"""
    client = get_alpaca_client()
    if client.enabled:
        account = await client.get_account()
        # Check if account info was retrieved successfully
        if account and account.get("status") == "ACTIVE":
            logger.info("[OK] Alpaca paper trading CONNECTED and READY")
            return True
        else:
            logger.warning(f"[WARN] Alpaca connection issue: {account.get('error', 'Unknown')}")
            return False
    else:
        logger.info("[INFO] Alpaca not configured. Set ALPACA_API_KEY and ALPACA_API_SECRET env vars")
        return False

async def close_alpaca():
    """Close Alpaca client on shutdown"""
    global alpaca_client
    if alpaca_client:
        await alpaca_client.close()
        logger.info("Alpaca client closed")
