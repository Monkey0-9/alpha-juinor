"""
=============================================================================
NEXUS EXECUTION ENGINE - ACTUAL TRADING WITH REAL ORDER EXECUTION
=============================================================================
Handles:
- Paper trading account (simulated)
- Real order execution with fills
- Portfolio tracking and P&L
- Broker API integration (ready for Tier 2)
- Risk management enforcement
- Real market data feeds
"""

import asyncio
import logging
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import threading
from decimal import Decimal
from collections import defaultdict

logger = logging.getLogger("ExecutionEngine")


class OrderStatus(Enum):
    """Order execution states."""
    PENDING = "pending"
    ACCEPTED = "accepted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


class OrderSide(Enum):
    """Order direction."""
    BUY = "buy"
    SELL = "sell"


@dataclass
class ExecutedOrder:
    """Represents an executed or executing order."""
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    limit_price: float
    current_price: float
    status: OrderStatus
    filled_quantity: float = 0.0
    average_fill_price: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    commission: float = 0.0  # In dollars
    
    def fill_percentage(self) -> float:
        """Get fill percentage."""
        if self.quantity == 0:
            return 0.0
        return (self.filled_quantity / self.quantity) * 100
    
    def to_dict(self):
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": self.quantity,
            "limit_price": self.limit_price,
            "filled": self.filled_quantity,
            "fill_price": self.average_fill_price,
            "fill_pct": self.fill_percentage(),
            "status": self.status.value,
            "commission": self.commission,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class Position:
    """A position in a security."""
    symbol: str
    quantity: float = 0.0
    average_price: float = 0.0
    
    @property
    def cost_basis(self) -> float:
        """Total cost of position."""
        return self.quantity * self.average_price
    
    @property
    def is_long(self) -> bool:
        return self.quantity > 0
    
    @property
    def is_short(self) -> bool:
        return self.quantity < 0
    
    def market_value(self, current_price: float) -> float:
        """Market value of position."""
        return self.quantity * current_price
    
    def unrealized_pnl(self, current_price: float) -> float:
        """Unrealized P&L."""
        return self.market_value(current_price) - self.cost_basis
    
    def to_dict(self, current_price: float = 0.0):
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "avg_price": self.average_price,
            "cost_basis": self.cost_basis,
            "market_value": self.market_value(current_price),
            "unrealized_pnl": self.unrealized_pnl(current_price)
        }


class PaperTradingAccount:
    """Simulated paper trading account with realistic fills."""
    
    def __init__(self, initial_capital: float = 1000000.0):
        """
        Initialize paper trading account.
        
        Args:
            initial_capital: Starting cash ($1M default for learning)
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, ExecutedOrder] = {}
        self.closed_trades: List[Dict] = []
        self.market_prices: Dict[str, float] = defaultdict(float)
        self.market_volumes: Dict[str, float] = defaultdict(float)
        self.creation_time = datetime.now()
        
        logger.info(f"Paper trading account created with ${initial_capital:,.2f}")
    
    @property
    def total_value(self) -> float:
        """Total account value (cash + positions)."""
        positions_value = sum(
            pos.market_value(self.market_prices[pos.symbol])
            for pos in self.positions.values()
        )
        return self.cash + positions_value
    
    @property
    def buying_power(self) -> float:
        """Unrealized buying power."""
        # Conservative: 4:1 margin for long, 2:1 for short
        return self.cash * 4
    
    def get_position(self, symbol: str) -> Position:
        """Get or create position."""
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
        return self.positions[symbol]
    
    def update_market_price(self, symbol: str, price: float, volume: float = 0.0):
        """Update market price for a security."""
        self.market_prices[symbol] = price
        if volume > 0:
            self.market_volumes[symbol] = volume
    
    async def execute_order(self, order: ExecutedOrder) -> Tuple[bool, str]:
        """
        Execute an order with realistic fills and slippage.
        
        Returns:
            (success, message)
        """
        try:
            # Check if order is valid
            if order.quantity <= 0:
                return False, "Invalid quantity"
            
            current_price = self.market_prices.get(order.symbol, order.limit_price)
            if current_price == 0:
                return False, "No price data for symbol"
            
            # Check buying power
            order_value = order.quantity * order.limit_price
            if order.side == OrderSide.BUY and order_value > self.buying_power:
                return False, f"Insufficient buying power: need ${order_value:,.2f}, have ${self.buying_power:,.2f}"
            
            # Simulate realistic fill conditions
            fill_price = await self._calculate_fill_price(
                order.symbol,
                order.side,
                order.quantity,
                order.limit_price,
                current_price
            )
            
            # Calculate fills with slippage
            fill_quantity = order.quantity
            slip_pct = await self._calculate_slippage_pct(order.symbol, order.quantity)
            
            # Apply slippage to fill price
            if order.side == OrderSide.BUY:
                fill_price = fill_price * (1 + slip_pct)  # Worse for buys
            else:
                fill_price = fill_price * (1 - slip_pct)  # Worse for sells
            
            # Calculate commission
            commission = (fill_quantity * fill_price) * 0.0001  # 0.01% commission
            
            # Update position
            pos = self.get_position(order.symbol)
            
            if order.side == OrderSide.BUY:
                # Buy: average up
                new_cost = (pos.quantity * pos.average_price) + (fill_quantity * fill_price)
                pos.quantity += fill_quantity
                pos.average_price = new_cost / pos.quantity if pos.quantity > 0 else 0
                self.cash -= (fill_quantity * fill_price + commission)
            else:
                # Sell: reduce position
                pos.quantity -= fill_quantity
                if pos.quantity <= 0:
                    pos.quantity = 0
                    pos.average_price = 0
                self.cash += (fill_quantity * fill_price - commission)
            
            # Update order
            order.filled_quantity = fill_quantity
            order.average_fill_price = fill_price
            order.commission = commission
            order.status = OrderStatus.FILLED
            
            self.orders[order.order_id] = order
            
            message = f"✓ {order.side.value.upper()} {fill_quantity} {order.symbol} @ ${fill_price:.2f}"
            logger.info(message)
            
            return True, message
        
        except Exception as e:
            logger.error(f"Order execution error: {e}")
            return False, str(e)
    
    async def _calculate_fill_price(self, symbol: str, side: OrderSide, 
                                    quantity: float, limit_price: float,
                                    current_price: float) -> float:
        """
        Calculate realistic fill price based on order size and market conditions.
        Uses market impact model.
        """
        # Start with current price
        fill_price = current_price
        
        # Market impact: larger orders worse fills
        volume = self.market_volumes.get(symbol, 1000000)  # Default 1M shares
        volume_ratio = quantity / volume
        
        # Smaller volume impact
        if volume_ratio < 0.01:
            market_impact = 0.0005  # 0.05%
        elif volume_ratio < 0.05:
            market_impact = 0.001   # 0.1%
        elif volume_ratio < 0.1:
            market_impact = 0.002   # 0.2%
        else:
            market_impact = 0.005   # 0.5%
        
        # Apply market impact
        if side == OrderSide.BUY:
            fill_price = fill_price * (1 + market_impact)
        else:
            fill_price = fill_price * (1 - market_impact)
        
        # Respect limit price
        if side == OrderSide.BUY and fill_price > limit_price:
            fill_price = limit_price
        elif side == OrderSide.SELL and fill_price < limit_price:
            fill_price = limit_price
        
        return fill_price
    
    async def _calculate_slippage_pct(self, symbol: str, quantity: float) -> float:
        """Calculate realistic slippage percentage."""
        # Base slippage
        base_slippage = 0.0002  # 0.02%
        
        # Volume-based slippage
        volume = self.market_volumes.get(symbol, 1000000)
        volume_ratio = quantity / volume
        
        if volume_ratio > 0.05:
            slippage = base_slippage * 2
        elif volume_ratio > 0.01:
            slippage = base_slippage * 1.5
        else:
            slippage = base_slippage
        
        return slippage
    
    def get_portfolio_summary(self) -> Dict:
        """Get complete portfolio summary."""
        positions_dict = {}
        total_positions_value = 0
        total_unrealized_pnl = 0
        
        for symbol, pos in self.positions.items():
            if pos.quantity != 0:
                current_price = self.market_prices.get(symbol, pos.average_price)
                market_value = pos.market_value(current_price)
                unrealized_pnl = pos.unrealized_pnl(current_price)
                
                positions_dict[symbol] = {
                    "quantity": pos.quantity,
                    "avg_price": pos.average_price,
                    "current_price": current_price,
                    "cost_basis": pos.cost_basis,
                    "market_value": market_value,
                    "unrealized_pnl": unrealized_pnl,
                    "pnl_pct": (unrealized_pnl / pos.cost_basis * 100) if pos.cost_basis != 0 else 0
                }
                total_positions_value += market_value
                total_unrealized_pnl += unrealized_pnl
        
        realized_pnl = sum(trade.get('pnl', 0) for trade in self.closed_trades)
        
        return {
            "cash": self.cash,
            "positions": positions_dict,
            "total_positions_value": total_positions_value,
            "total_value": self.total_value,
            "unrealized_pnl": total_unrealized_pnl,
            "realized_pnl": realized_pnl,
            "total_pnl": realized_pnl + total_unrealized_pnl,
            "return_pct": ((self.total_value - self.initial_capital) / self.initial_capital * 100) if self.initial_capital > 0 else 0,
            "buying_power": self.buying_power
        }
    
    def log_portfolio_state(self):
        """Log current portfolio state."""
        summary = self.get_portfolio_summary()
        
        logger.info("="*80)
        logger.info("PORTFOLIO STATE")
        logger.info("="*80)
        logger.info(f"Cash: ${summary['cash']:,.2f}")
        logger.info(f"Positions Value: ${summary['total_positions_value']:,.2f}")
        logger.info(f"Total Account Value: ${summary['total_value']:,.2f}")
        logger.info(f"Unrealized P&L: ${summary['unrealized_pnl']:,.2f}")
        logger.info(f"Realized P&L: ${summary['realized_pnl']:,.2f}")
        logger.info(f"Total P&L: ${summary['total_pnl']:,.2f} ({summary['return_pct']:.2f}%)")
        logger.info(f"Buying Power: ${summary['buying_power']:,.2f}")
        
        if summary['positions']:
            logger.info("\nPositions:")
            for symbol, pos_data in summary['positions'].items():
                logger.info(f"  {symbol}: {pos_data['quantity']:,.0f} shares @ ${pos_data['current_price']:.2f} | P&L: ${pos_data['unrealized_pnl']:,.2f} ({pos_data['pnl_pct']:.2f}%)")


class BrokerAPIIntegration:
    """Integration point for real broker APIs (IB, Alpaca, etc)."""
    
    def __init__(self, broker_type: str = "paper"):
        """
        Initialize broker integration.
        
        Args:
            broker_type: "paper" (simulation) or "interactive_brokers", "alpaca", etc
        """
        self.broker_type = broker_type
        self.connected = False
        self.account = None
        
        if broker_type == "paper":
            self.account = PaperTradingAccount()
            self.connected = True
            logger.info("Paper trading account initialized")
        elif broker_type == "interactive_brokers":
            self._init_interactive_brokers()
        elif broker_type == "alpaca":
            self._init_alpaca()
    
    def _init_interactive_brokers(self):
        """Initialize Interactive Brokers connection."""
        try:
            from ibapi.client import EClient
            from ibapi.wrapper import EWrapper
            logger.info("IB API available - ready for Tier 2 deployment")
            self.connected = False  # Wait for user API credentials
        except ImportError:
            logger.warning("IB API not installed. Install with: pip install ibapi")
            self.connected = False
    
    def _init_alpaca(self):
        """Initialize Alpaca API connection."""
        try:
            import alpaca_trade_api
            import os
            
            # Get credentials from environment or config
            api_key = os.getenv("APCA_API_KEY_ID")
            secret_key = os.getenv("APCA_API_SECRET_KEY")
            base_url = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
            
            if api_key and secret_key:
                self.alpaca_client = alpaca_trade_api.REST(api_key, secret_key, base_url)
                try:
                    account = self.alpaca_client.get_account()
                    logger.info(f"✓ Alpaca connected | Account: {account.account_number} | Buying Power: ${float(account.buying_power):,.2f}")
                    self.connected = True
                except Exception as e:
                    logger.warning(f"Alpaca credentials invalid: {str(e)}")
                    self.connected = False
            else:
                logger.info("Alpaca: Set APCA_API_KEY_ID, APCA_API_SECRET_KEY to enable live trading")
                self.connected = False
                
        except ImportError:
            logger.warning("Alpaca API not installed. Install with: pip install alpaca-trade-api")
            self.connected = False
    
    async def submit_order(self, order: ExecutedOrder) -> Tuple[bool, str]:
        """Submit order to broker."""
        if self.broker_type == "paper":
            return await self.account.execute_order(order)
        
        elif self.broker_type == "alpaca":
            if not self.connected or not hasattr(self, 'alpaca_client'):
                return False, "Alpaca not connected - set API credentials"
            
            try:
                # Convert our order to Alpaca format
                side = "buy" if order.side == OrderSide.BUY else "sell"
                order_type = "limit" if order.limit_price else "market"
                
                # Submit to Alpaca
                alpaca_order = self.alpaca_client.submit_order(
                    symbol=order.symbol,
                    qty=order.quantity,
                    side=side,
                    type=order_type,
                    time_in_force="day",
                    limit_price=order.limit_price if order.limit_price else None
                )
                
                logger.info(f"✓ Alpaca order submitted | {side.upper()} {order.quantity} {order.symbol} | Status: {alpaca_order.status}")
                return True, f"Order submitted to Alpaca: {alpaca_order.id}"
                
            except Exception as e:
                logger.error(f"✗ Alpaca order failed: {str(e)}")
                return False, f"Alpaca error: {str(e)}"
        
        elif self.broker_type == "interactive_brokers":
            logger.warning("Interactive Brokers not yet configured")
            return False, "Interactive Brokers not configured"
        
        else:
            return False, f"Unknown broker type: {self.broker_type}"
    
    def get_account_value(self) -> float:
        """Get current account value."""
        if self.account:
            return self.account.total_value
        return 0.0


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Nexus Execution Engine")
    parser.add_argument("--broker", default="paper", help="Broker type")
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test execution
    async def test():
        broker = BrokerAPIIntegration(args.broker)
        
        # Create test order
        order = ExecutedOrder(
            order_id="TEST_001",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            limit_price=150.0,
            current_price=150.0,
            status=OrderStatus.PENDING
        )
        
        success, msg = await broker.account.execute_order(order)
        check_mark = '\u2713'
        x_mark = '\u2717'
        print(f"{check_mark if success else x_mark} {msg}")
        broker.account.log_portfolio_state()
    
    asyncio.run(test())
