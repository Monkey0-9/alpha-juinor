"""
Auto Trader - Autonomous Full Market Trading.

Features:
- Daily market scanning
- Automatic trade execution
- Position sizing and risk management
- Portfolio rebalancing
"""

import logging
import os
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TradeOrder:
    """Trade order to execute."""
    symbol: str
    side: str  # "buy" or "sell"
    quantity: int
    order_type: str = "market"
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "day"


@dataclass
class PortfolioPosition:
    """Current portfolio position."""
    symbol: str
    quantity: int
    avg_cost: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float


@dataclass
class TradingSession:
    """Daily trading session result."""
    date: str
    stocks_scanned: int
    signals_generated: int
    orders_placed: int
    orders_filled: int
    total_value_traded: float
    positions_opened: int
    positions_closed: int


class AutoTrader:
    """
    Autonomous trading system for the FULL MARKET.

    Daily workflow:
    1. Scan entire market
    2. Generate signals
    3. Select best opportunities
    4. Size positions
    5. Execute trades
    6. Monitor and rebalance
    """

    def __init__(
        self,
        max_positions: int = 50,
        position_size_pct: float = 0.02,  # 2% per position
        max_sector_exposure: float = 0.25,
        daily_turnover_limit: float = 0.20,
        use_paper: bool = True
    ):
        self.max_positions = max_positions
        self.position_size_pct = position_size_pct
        self.max_sector_exposure = max_sector_exposure
        self.daily_turnover_limit = daily_turnover_limit
        self.use_paper = use_paper

        self.api = None
        self.positions: Dict[str, PortfolioPosition] = {}
        self.pending_orders: List[TradeOrder] = []
        self.session_history: List[TradingSession] = []

    def initialize(self):
        """Initialize Alpaca API connection."""
        try:
            import alpaca_trade_api as tradeapi

            api_key = os.getenv("ALPACA_API_KEY")
            api_secret = os.getenv("ALPACA_SECRET_KEY")

            if self.use_paper:
                base_url = "https://paper-api.alpaca.markets"
            else:
                base_url = "https://api.alpaca.markets"

            self.api = tradeapi.REST(api_key, api_secret, base_url)

            # Verify connection
            account = self.api.get_account()
            logger.info(
                f"Connected to Alpaca: ${float(account.equity):,.2f} equity, "
                f"${float(account.buying_power):,.2f} buying power"
            )

            return True

        except Exception as e:
            logger.error(f"Failed to initialize Alpaca: {e}")
            return False

    def get_account_info(self) -> Dict:
        """Get current account information."""
        if not self.api:
            return {}

        try:
            account = self.api.get_account()
            return {
                "equity": float(account.equity),
                "buying_power": float(account.buying_power),
                "cash": float(account.cash),
                "portfolio_value": float(account.portfolio_value),
                "day_trade_count": int(account.daytrade_count),
                "status": account.status
            }
        except Exception as e:
            logger.error(f"Failed to get account: {e}")
            return {}

    def get_positions(self) -> Dict[str, PortfolioPosition]:
        """Get current positions."""
        if not self.api:
            return {}

        try:
            positions = self.api.list_positions()

            self.positions = {}
            for pos in positions:
                self.positions[pos.symbol] = PortfolioPosition(
                    symbol=pos.symbol,
                    quantity=int(pos.qty),
                    avg_cost=float(pos.avg_entry_price),
                    current_price=float(pos.current_price),
                    market_value=float(pos.market_value),
                    unrealized_pnl=float(pos.unrealized_pl),
                    unrealized_pnl_pct=float(pos.unrealized_plpc)
                )

            return self.positions

        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return {}

    def calculate_position_size(
        self,
        symbol: str,
        signal_strength: float,
        price: float,
        account_value: float
    ) -> int:
        """Calculate position size in shares."""
        # Base position size
        base_value = account_value * self.position_size_pct

        # Adjust by signal strength
        adjusted_value = base_value * abs(signal_strength)

        # Convert to shares
        shares = int(adjusted_value / price)

        # Minimum 1 share
        return max(1, shares)

    def generate_orders(
        self,
        signals: List,
        account_value: float,
        current_prices: Dict[str, float]
    ) -> List[TradeOrder]:
        """Generate trade orders from signals."""
        orders = []

        # Get current positions
        current_positions = set(self.positions.keys())

        # Track new position count
        new_positions = 0
        available_slots = self.max_positions - len(current_positions)

        for signal in signals:
            symbol = signal.symbol
            price = current_prices.get(symbol, 0)

            if price <= 0:
                continue

            # Handle LONG signals
            if signal.signal_type == "LONG":
                if symbol not in current_positions:
                    if new_positions >= available_slots:
                        continue

                    qty = self.calculate_position_size(
                        symbol, signal.alpha_score, price, account_value
                    )

                    orders.append(TradeOrder(
                        symbol=symbol,
                        side="buy",
                        quantity=qty
                    ))
                    new_positions += 1

            # Handle SHORT signals (if enabled)
            elif signal.signal_type == "SHORT":
                # Close existing long
                if symbol in current_positions:
                    pos = self.positions[symbol]
                    if pos.quantity > 0:
                        orders.append(TradeOrder(
                            symbol=symbol,
                            side="sell",
                            quantity=pos.quantity
                        ))

        return orders

    def execute_order(self, order: TradeOrder) -> bool:
        """Execute a single order."""
        if not self.api:
            logger.warning("API not initialized")
            return False

        try:
            self.api.submit_order(
                symbol=order.symbol,
                qty=order.quantity,
                side=order.side,
                type=order.order_type,
                time_in_force=order.time_in_force,
                limit_price=order.limit_price,
                stop_price=order.stop_price
            )

            logger.info(
                f"Order submitted: {order.side.upper()} {order.quantity} {order.symbol}"
            )
            return True

        except Exception as e:
            logger.error(f"Order failed for {order.symbol}: {e}")
            return False

    def execute_orders(self, orders: List[TradeOrder]) -> Tuple[int, int]:
        """Execute all orders."""
        success = 0
        failed = 0

        for order in orders:
            if self.execute_order(order):
                success += 1
            else:
                failed += 1

            # Rate limiting
            time.sleep(0.1)

        return success, failed

    def close_losing_positions(self, max_loss_pct: float = -0.05) -> int:
        """Close positions exceeding max loss."""
        self.get_positions()
        closed = 0

        for symbol, pos in self.positions.items():
            if pos.unrealized_pnl_pct < max_loss_pct:
                order = TradeOrder(
                    symbol=symbol,
                    side="sell",
                    quantity=abs(pos.quantity)
                )
                if self.execute_order(order):
                    closed += 1
                    logger.info(f"Closed losing position: {symbol} ({pos.unrealized_pnl_pct:.1%})")

        return closed

    def run_daily_session(
        self,
        market_signals: List,
        current_prices: Dict[str, float]
    ) -> TradingSession:
        """Run a complete daily trading session."""
        session_date = datetime.now().strftime("%Y-%m-%d")

        # Get account info
        account = self.get_account_info()
        account_value = account.get("equity", 0)

        if account_value <= 0:
            logger.error("Invalid account value")
            return TradingSession(
                date=session_date,
                stocks_scanned=0,
                signals_generated=0,
                orders_placed=0,
                orders_filled=0,
                total_value_traded=0,
                positions_opened=0,
                positions_closed=0
            )

        # Generate orders from signals
        orders = self.generate_orders(
            market_signals,
            account_value,
            current_prices
        )

        # Execute orders
        filled, failed = self.execute_orders(orders)

        # Calculate total value traded
        total_traded = sum(
            current_prices.get(o.symbol, 0) * o.quantity
            for o in orders[:filled]
        )

        # Manage existing positions
        closed = self.close_losing_positions()

        session = TradingSession(
            date=session_date,
            stocks_scanned=len(market_signals),
            signals_generated=len([s for s in market_signals if s.signal_type != "NEUTRAL"]),
            orders_placed=len(orders),
            orders_filled=filled,
            total_value_traded=total_traded,
            positions_opened=filled,
            positions_closed=closed
        )

        self.session_history.append(session)

        logger.info(
            f"Session complete: {session.orders_filled}/{session.orders_placed} orders filled, "
            f"${session.total_value_traded:,.0f} traded"
        )

        return session


# Global singleton
_trader: Optional[AutoTrader] = None


def get_auto_trader() -> AutoTrader:
    """Get or create global auto trader."""
    global _trader
    if _trader is None:
        _trader = AutoTrader()
    return _trader
