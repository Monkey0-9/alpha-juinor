# brokers/ccxt_broker.py
import ccxt.async_support as ccxt_async
import asyncio
import logging
from typing import Dict, List, Optional, Any
from backtest.execution import Order, OrderType, OrderStatus, Trade

logger = logging.getLogger(__name__)

class CCXTExecutionHandler:
    """
    Live/Paper execution handler for crypto using CCXT.
    Converts internal Order objects to exchange calls.
    """
    
    def __init__(self, exchange_id: str, api_key: str = "", secret: str = "", paper: bool = True):
        exchange_class = getattr(ccxt_async, exchange_id)
        self.exchange = exchange_class({
            'apiKey': api_key,
            'secret': secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
        if paper and hasattr(self.exchange, 'set_sandbox_mode'):
            self.exchange.set_sandbox_mode(True)
        self.paper = paper

    async def submit_order(self, order: Order) -> Optional[str]:
        """Submit order to exchange. Returns external order ID."""
        symbol = order.ticker
        side = 'buy' if order.quantity > 0 else 'sell'
        amount = abs(order.quantity)
        
        try:
            if order.order_type == OrderType.MARKET:
                res = await self.exchange.create_order(symbol, 'market', side, amount)
            else:
                res = await self.exchange.create_order(symbol, 'limit', side, amount, order.limit_price)
            
            logger.info(f"CCXT: Order submitted {res['id']}")
            return res['id']
        except Exception as e:
            logger.error(f"CCXT Submission Error: {e}")
            return None

    async def get_order_status(self, external_id: str, symbol: str) -> Dict[str, Any]:
        """Fetch status from exchange."""
        try:
            return await self.exchange.fetch_order(external_id, symbol)
        except Exception as e:
            logger.error(f"CCXT Status Error: {e}")
            return {}

    async def close(self):
        await self.exchange.close()

    def sync_submit(self, order: Order) -> Optional[str]:
        """Synchronous wrapper."""
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
        return loop.run_until_complete(self.submit_order(order))
