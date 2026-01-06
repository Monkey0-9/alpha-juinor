
import logging
from typing import List, Dict
from backtest.execution import Order

logger = logging.getLogger(__name__)

class MockBroker:
    """
    Mock Broker that implements the interface expected by LiveEngine.
    Maintains a simulated local account and position state.
    """
    def __init__(self, initial_capital: float = 100000.0):
        self.equity = float(initial_capital)
        self.positions = {} # {ticker: quantity}
        logger.info(f"MockBroker initialized with ${self.equity:,.2f}")

    def get_account(self) -> Dict:
        return {
            "status": "ACTIVE",
            "equity": self.equity,
            "cash": self.equity # Simplified
        }

    def get_positions(self) -> Dict[str, float]:
        return self.positions

    def submit_orders(self, orders: List[Order]) -> List[Dict]:
        results = []
        for order in orders:
            # Simplified mock execution: assume immediate fill at current price
            # In real mock, we'd need market data to get prices, 
            # but here we just log and update state if needed.
            # Actually, LiveEngine calculates qty based on current_prices, 
            # so we just accept it here.
            
            logger.info(f"MockBroker: Executing {order.order_type.value} {order.ticker} qty={order.quantity}")
            
            # Update local state
            ticker = order.ticker
            current_qty = self.positions.get(ticker, 0.0)
            self.positions[ticker] = current_qty + order.quantity
            
            results.append({"status": "filled", "order_id": order.id})
        return results
    def cancel_all_orders(self) -> bool:
        """Cancel all simulated open orders (Mock implementation)."""
        logger.info("MockBroker: Cancelled all open orders.")
        return True
