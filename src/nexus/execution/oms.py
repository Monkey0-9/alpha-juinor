import logging
from typing import Dict, Optional, List, Callable
from ..models.trade import Order, Trade, OrderStatus
from .broker import BrokerAdapter
from ..core.context import engine_context

class OMS:
    """
    Order Management System.
    Manages the lifecycle of orders, tracks state transitions, 
    and facilitates reconciliation with broker fills.
    """
    def __init__(self, broker: BrokerAdapter):
        self.broker = broker
        self.active_orders: Dict[str, Order] = {}
        self.trades: List[Trade] = []
        self.logger = engine_context.get_logger("oms")
        
        # Register fill callback with broker
        self.broker.set_fill_callback(self._on_fill)

    async def submit_order(self, order: Order) -> bool:
        """Route order to broker and update internal tracking."""
        self.logger.info(f"Submitting order: {order.order_id} {order.side} {order.quantity} {order.symbol}")
        
        success = await self.broker.submit_order(order)
        if success:
            order.status = OrderStatus.SUBMITTED
            self.active_orders[order.order_id] = order
            return True
        else:
            order.status = OrderStatus.REJECTED
            self.logger.error(f"Order submission failed: {order.order_id}")
            return False

    async def cancel_order(self, order_id: str) -> bool:
        """Attempt to cancel an active order."""
        if order_id not in self.active_orders:
            self.logger.warn(f"Cancellation failed: Order {order_id} not found/active")
            return False
            
        success = await self.broker.cancel_order(order_id)
        if success:
            self.active_orders[order_id].status = OrderStatus.CANCELLED
            self.active_orders.pop(order_id)
            self.logger.info(f"Order cancelled: {order_id}")
            return True
        return False

    def _on_fill(self, trade: Trade):
        """Callback handled when broker notifies of a fill."""
        order_id = trade.order_id
        if order_id not in self.active_orders:
            self.logger.error(f"Received fill for non-tracked order: {order_id}")
            return
            
        order = self.active_orders[order_id]
        order.filled_quantity += trade.quantity
        order.average_price = ((order.average_price * (order.filled_quantity - trade.quantity)) + 
                                (trade.price * trade.quantity)) / order.filled_quantity
        
        self.trades.append(trade)
        
        if order.filled_quantity >= order.quantity - 1e-6: # Tolerance for float
            order.status = OrderStatus.FILLED
            self.active_orders.pop(order_id)
            self.logger.info(f"Order fully filled: {order_id}")
        else:
            order.status = OrderStatus.PARTIAL
            self.logger.info(f"Order partially filled: {order_id} ({order.filled_quantity}/{order.quantity})")

    def get_order(self, order_id: str) -> Optional[Order]:
        return self.active_orders.get(order_id)

    def get_active_orders(self) -> List[Order]:
        return list(self.active_orders.values())
