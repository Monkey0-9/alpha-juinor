from abc import ABC, abstractmethod
from typing import List, Optional, Callable
from ..models.trade import Order, Trade

class BrokerAdapter(ABC):
    """
    Abstract interface for all broker connectivity.
    Ensures research and execution share the same order/trade definitions.
    """
    @abstractmethod
    async def submit_order(self, order: Order) -> bool:
        """Submits an order to the broker."""
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancels an existing order."""
        pass

    @abstractmethod
    def set_fill_callback(self, callback: Callable[[Trade], None]):
        """Sets the callback function to be called when a trade/fill occurs."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Name of the broker adapter."""
        pass
