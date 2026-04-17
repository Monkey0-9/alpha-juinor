from typing import Dict, List, Optional
from datetime import datetime
from ..models.trade import Position, Trade, OrderSide, PortfolioState
from ..models.market import MarketBar
from ..core.context import engine_context

class PortfolioTracker:
    """
    Handles cash and position accounting for backtesting and simulation.
    Ensures realistic PnL calculations including commissions.
    """
    def __init__(self, initial_cash: float = 100000.0):
        self.cash = initial_cash
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_history: List[Dict[str, Any]] = []
        self.logger = engine_context.get_logger("portfolio_tracker")

    def update_with_trade(self, trade: Trade):
        """Updates portfolio state based on an execution fill."""
        symbol = trade.symbol
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
            
        pos = self.positions[symbol]
        
        # Calculate signed quantity change
        qty_change = trade.quantity if trade.side in [OrderSide.BUY, OrderSide.COVER] else -trade.quantity
        
        # Update cash (including commission)
        total_cost = (trade.quantity * trade.price) + trade.commission
        if trade.side in [OrderSide.BUY, OrderSide.COVER]:
            self.cash -= total_cost
        else:
            self.cash += (trade.quantity * trade.price) - trade.commission

        # Update position
        old_qty = pos.quantity
        new_qty = old_qty + qty_change
        
        if old_qty == 0:
            pos.average_entry_price = trade.price
        elif (old_qty > 0 and qty_change > 0) or (old_qty < 0 and qty_change < 0):
            # Adding to position: update average price
            total_basis = (abs(old_qty) * pos.average_entry_price) + (trade.quantity * trade.price)
            pos.average_entry_price = total_basis / abs(new_qty)
        else:
            # Reducing position: realize PnL
            reduced_qty = min(abs(old_qty), abs(qty_change))
            pnl = reduced_qty * (trade.price - pos.average_entry_price)
            if old_qty < 0: pnl = -pnl # Short position
            pos.realized_pnl += pnl
            
        pos.quantity = new_qty
        self.trades.append(trade)
        
    def mark_to_market(self, timestamp: datetime, prices: Dict[str, float]):
        """Updates unrealized PnL and records total equity."""
        total_equity = self.cash
        for symbol, pos in self.positions.items():
            if symbol in prices:
                price = prices[symbol]
                pos.last_price = price
                pos.unrealized_pnl = pos.quantity * (price - pos.average_entry_price)
                total_equity += (pos.quantity * price)
        
        self.equity_history.append({
            "timestamp": timestamp,
            "equity": total_equity,
            "cash": self.cash
        })

    def get_state(self) -> PortfolioState:
        """Returns the current portfolio state."""
        equity = self.cash + sum(p.quantity * p.last_price for p in self.positions.values())
        return PortfolioState(
            timestamp=datetime.utcnow(),
            cash=self.cash,
            equity=equity,
            positions=self.positions.copy()
        )
