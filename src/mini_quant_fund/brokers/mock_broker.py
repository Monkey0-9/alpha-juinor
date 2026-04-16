
import logging
import math
from typing import List, Dict, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    from mini_quant_fund.execution.market_impact import MarketImpactModel
    HAS_MARKET_IMPACT = True
except ImportError:
    logger.warning("MarketImpactModel not found. Falling back to constant slippage.")
    HAS_MARKET_IMPACT = False

class MockBroker:
    """
    Institutional Mock Broker for high-fidelity backtesting and simulation.
    Applies market impact models to simulate realistic execution costs.
    """
    
    def __init__(self, initial_capital: float = 1000000.0, commission_bps: float = 1.0):
        self.equity = float(initial_capital)
        self.cash = float(initial_capital)
        self.positions = {} # {symbol: quantity}
        self.commission_bps = commission_bps
        
        if HAS_MARKET_IMPACT:
            self.impact_model = MarketImpactModel()
        else:
            self.impact_model = None
            
        logger.info(f"MockBroker initialized: Capital=${self.equity:,.2f}, Commission={commission_bps} bps")

    def submit_order(self, 
                     symbol: str, 
                     qty: float, 
                     side: str, 
                     order_type: str = "MARKET", 
                     limit_price: Optional[float] = None) -> Dict[str, Any]:
        """
        Executes a mock order with slippage and commission.
        """
        side = side.upper()
        # Mock current price (in a real system this would come from a data feed)
        current_mid = limit_price or 100.0
        
        # 1. Calculate Slippage (Market Impact)
        slippage_bps = 0.0
        if self.impact_model:
            impact = self.impact_model.estimate_impact(
                symbol=symbol,
                side=side,
                quantity=qty,
                order_type=order_type,
                price=current_mid
            )
            execution_price = current_mid + impact.market_impact if side == "BUY" else current_mid - impact.market_impact
            slippage_bps = impact.impact_bps
        else:
            # Simple constant slippage fallback: 5 bps
            execution_price = current_mid * (1 + 0.0005 if side == "BUY" else 1 - 0.0005)
            slippage_bps = 5.0
            
        # 2. Apply Commission
        commission = (qty * execution_price) * (self.commission_bps / 10000)
        
        # 3. Update State
        total_cost = (qty * execution_price) + commission if side == "BUY" else -(qty * execution_price) + commission
        
        if side == "BUY" and total_cost > self.cash:
            return {"success": False, "error": "Insufficient funds"}
            
        self.cash -= total_cost
        
        current_qty = self.positions.get(symbol, 0.0)
        new_qty = current_qty + (qty if side == "BUY" else -qty)
        self.positions[symbol] = new_qty
        
        # Clean up zero positions
        if abs(self.positions[symbol]) < 1e-8:
            del self.positions[symbol]
            
        logger.info(f"Executed {side} {qty} {symbol} @ {execution_price:.4f} (Impact: {slippage_bps:.2f} bps)")
        
        return {
            "success": True,
            "order_id": f"mock_{int(datetime.now().timestamp())}",
            "execution_price": execution_price,
            "commission": commission,
            "slippage_bps": slippage_bps
        }

    def get_account(self) -> Dict[str, float]:
        """Returns account summary."""
        return {
            "cash": self.cash,
            "equity": self.equity, # Should calculate based on position values
            "positions_count": len(self.positions)
        }

    def get_positions(self) -> Dict[str, float]:
        """Returns current positions."""
        return self.positions.copy()
