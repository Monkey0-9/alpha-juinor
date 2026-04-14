
import logging
import math
from typing import List, Dict, Optional
from mini_quant_fund.backtest.execution import Order
from mini_quant_fund.execution.market_impact import MarketImpactModel

logger = logging.getLogger(__name__)

class MockBroker:
    """
    Institutional Mock Broker.
    Narrowing the 'Real vs Paper' gap by applying Almgren-Chriss market impact.
    """
    def __init__(self, initial_capital: float = 100000.0):
        self.equity = float(initial_capital)
        self.positions = {} # {ticker: quantity}
        self.impact_model = MarketImpactModel()
        logger.info(f"Institutional MockBroker initialized with ${self.equity:,.2f} and MarketImpact logic.")

    def submit_order(self, symbol: str, qty: float, side: str, order_type: str = "market", time_in_force: str = "day", limit_price: float = None, stop_price: float = None) -> Dict:
        """
        Submits order with REALISTIC SLIPPAGE injection.
        """
        # Realistic Price Simulation (Mocking current mid)
        # In production, this would fetch from self.market_data_feed
        current_mid = limit_price or 150.0 
        
        # 1. Calculate Market Impact (Slippage)
        impact = self.impact_model.estimate_impact(
            symbol=symbol,
            side=side.upper(),
            quantity=qty,
            order_type=order_type.upper(),
            price=current_mid,
            participation_rate=0.05 # Conservative participation
        )
        
        # 2. Adjust execution price based on impact
        slippage = impact.market_impact
        execution_price = current_mid + slippage if side.lower() == 'buy' else current_mid - slippage
        
        logger.info(f"[SLIPPAGE] Applied ${slippage:.4f} impact to {symbol} {side}. Executed @ ${execution_price:.2f}")

        # 3. Update local state
        current_qty = self.positions.get(symbol, 0.0)
        signed_qty = qty if side.lower() == 'buy' else -qty
        
        # Deduct trade cost from equity
        trade_cost = qty * execution_price
        if side.lower() == 'buy':
            self.equity -= trade_cost
        else:
            self.equity += trade_cost
            
        self.positions[symbol] = current_qty + signed_qty

        return {
            "success": True,
            "order": {
                "id": "inst_mock_id", 
                "symbol": symbol, 
                "qty": qty, 
                "side": side,
                "execution_price": execution_price,
                "impact_bps": impact.impact_bps
            },
            "error": None,
            "mapped_symbol": symbol
        }

    def get_account(self) -> Dict:
        """Get account details."""
        return {
            "equity": self.equity,
            "buying_power": self.equity, # Simplified
            "cash": self.equity,
            "status": "ACTIVE",
            "currency": "USD"
        }

    def get_positions(self) -> Dict[str, float]:
        """Get current positions as {ticker: quantity}."""
        # Filter out zero positions
        return {k: v for k, v in self.positions.items() if v != 0}
