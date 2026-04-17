import numpy as np
from numba import njit
from abc import ABC, abstractmethod
from ..models.trade import Order, Trade, OrderSide

class CostModel(ABC):
    """Base class for transaction cost and slippage models."""
    @abstractmethod
    def calculate_cost(self, order: Order, price: float, volume: float) -> float:
        """Returns the total cost (slippage + commission) for an execution."""
        pass

class InstitutionalCostModel(CostModel):
    """
    Standard institutional cost model.
    - Commission: Fixed bps of value
    - Slippage: Square root impact model (Impact ~ sigma * sqrt(Size/Volume))
    """
    def __init__(self, commission_bps: float = 1.0, slippage_coeff: float = 0.1):
        self.commission_bps = commission_bps / 10000.0
        self.slippage_coeff = slippage_coeff

    def calculate_cost(self, order: Order, price: float, volume: float) -> float:
        return _compute_institutional_cost(order.quantity, price, volume, self.commission_bps, self.slippage_coeff)

    def get_execution_price(self, order: Order, price: float, volume: float) -> float:
        """Returns the price adjusted for slippage."""
        cost = self.calculate_cost(order, price, volume)
        slippage_per_unit = cost / order.quantity
        
        if order.side in [OrderSide.BUY, OrderSide.COVER]:
            return price + slippage_per_unit
        else:
            return price - slippage_per_unit

@njit
def _compute_institutional_cost(quantity: float, price: float, volume: float, commission_bps: float, slippage_coeff: float) -> float:
    value = quantity * price
    # 1. Commission
    commission = value * commission_bps
    # 2. Slippage (Market Impact)
    if volume > 0:
        participation_rate = quantity / volume
        slippage_bps = slippage_coeff * np.sqrt(participation_rate)
        slippage = value * slippage_bps
    else:
        slippage = value * 0.01 
    return commission + slippage
