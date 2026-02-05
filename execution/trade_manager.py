"""
Live Trade Manager - Complete Trade Lifecycle
================================================

Manages the complete lifecycle of trades:
1. Entry Management
2. Position Tracking
3. Exit Management (Stop Loss, Take Profit)
4. Trailing Stops
5. Partial Exits
6. P&L Tracking

Never miss an exit. Protect profits.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, getcontext
from enum import Enum
from typing import Any, Dict, List, Optional
import threading

logger = logging.getLogger(__name__)

getcontext().prec = 50


class TradeStatus(Enum):
    """Trade status."""
    PENDING = "PENDING"
    OPEN = "OPEN"
    PARTIAL_CLOSE = "PARTIAL_CLOSE"
    CLOSED = "CLOSED"
    CANCELLED = "CANCELLED"


class ExitReason(Enum):
    """Reason for trade exit."""
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT_1 = "TAKE_PROFIT_1"
    TAKE_PROFIT_2 = "TAKE_PROFIT_2"
    TAKE_PROFIT_3 = "TAKE_PROFIT_3"
    TRAILING_STOP = "TRAILING_STOP"
    MANUAL = "MANUAL"
    TIME_STOP = "TIME_STOP"
    SIGNAL_EXIT = "SIGNAL_EXIT"
    SMART_CUT = "SMART_CUT"         # Cut based on future prediction loss
    PROFIT_EXT = "PROFIT_EXT"       # Extended based on prediction growth


@dataclass
class LiveTrade:
    """A live trade being managed."""
    trade_id: str
    symbol: str
    side: str  # LONG, SHORT

    # Entry
    entry_price: Decimal
    entry_time: datetime
    quantity: int

    # Current
    current_price: Decimal
    current_quantity: int

    # Exits
    stop_loss: Decimal
    take_profit_1: Decimal
    take_profit_2: Decimal
    take_profit_3: Decimal

    # Trailing
    trailing_stop_pct: Optional[Decimal]
    trailing_stop_price: Optional[Decimal]
    highest_price: Decimal  # For trailing
    lowest_price: Decimal   # For short trailing

    # Status
    status: TradeStatus

    # P&L
    realized_pnl: Decimal = Decimal("0")
    unrealized_pnl: Decimal = Decimal("0")

    # Exits taken
    exits: List[Dict] = field(default_factory=list)

    # Strategy info
    strategy: str = ""

    def update_price(self, new_price: Decimal):
        """Update current price and P&L."""
        self.current_price = new_price

        # Update high/low for trailing
        if new_price > self.highest_price:
            self.highest_price = new_price
        if new_price < self.lowest_price:
            self.lowest_price = new_price

        # Update trailing stop
        if self.trailing_stop_pct:
            if self.side == "LONG":
                new_trailing = self.highest_price * (1 - self.trailing_stop_pct)
                if self.trailing_stop_price is None or new_trailing > self.trailing_stop_price:
                    self.trailing_stop_price = new_trailing
            else:
                new_trailing = self.lowest_price * (1 + self.trailing_stop_pct)
                if self.trailing_stop_price is None or new_trailing < self.trailing_stop_price:
                    self.trailing_stop_price = new_trailing

        # Update unrealized P&L
        if self.side == "LONG":
            self.unrealized_pnl = (new_price - self.entry_price) * self.current_quantity
        else:
            self.unrealized_pnl = (self.entry_price - new_price) * self.current_quantity


@dataclass
class ExitOrder:
    """An exit order to place."""
    trade_id: str
    symbol: str
    side: str  # SELL for long exit, BUY for short exit
    quantity: int
    price: Optional[Decimal]
    order_type: str  # MARKET, LIMIT
    reason: ExitReason


class TradeManager:
    """
    Manages all live trades.

    Tracks positions, manages exits, protects profits.
    """

    def __init__(self):
        """Initialize the trade manager."""
        self.trades: Dict[str, LiveTrade] = {}
        self.closed_trades: List[LiveTrade] = []
        self._lock = threading.Lock()
        self._trade_counter = 0

        # Stats
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = Decimal("0")

        logger.info("[TRADE] Trade Manager initialized - MANAGING POSITIONS")

    def open_trade(
        self,
        symbol: str,
        side: str,
        entry_price: Decimal,
        quantity: int,
        stop_loss: Decimal,
        take_profit_1: Decimal,
        take_profit_2: Optional[Decimal] = None,
        take_profit_3: Optional[Decimal] = None,
        trailing_stop_pct: Optional[float] = None,
        strategy: str = ""
    ) -> str:
        """Open a new trade."""
        with self._lock:
            self._trade_counter += 1
            trade_id = f"T{self._trade_counter:06d}"

            # Default TPs
            if take_profit_2 is None:
                if side == "LONG":
                    take_profit_2 = take_profit_1 * Decimal("1.5")
                else:
                    take_profit_2 = take_profit_1 * Decimal("0.5")

            if take_profit_3 is None:
                if side == "LONG":
                    take_profit_3 = take_profit_1 * Decimal("2")
                else:
                    take_profit_3 = take_profit_1 * Decimal("0.3")

            trade = LiveTrade(
                trade_id=trade_id,
                symbol=symbol,
                side=side,
                entry_price=entry_price,
                entry_time=datetime.utcnow(),
                quantity=quantity,
                current_price=entry_price,
                current_quantity=quantity,
                stop_loss=stop_loss,
                take_profit_1=take_profit_1,
                take_profit_2=take_profit_2,
                take_profit_3=take_profit_3,
                trailing_stop_pct=Decimal(str(trailing_stop_pct)) if trailing_stop_pct else None,
                trailing_stop_price=None,
                highest_price=entry_price,
                lowest_price=entry_price,
                status=TradeStatus.OPEN,
                strategy=strategy
            )

            self.trades[trade_id] = trade
            self.total_trades += 1

            logger.info(
                f"[TRADE] Opened {trade_id}: {side} {quantity} {symbol} @ {entry_price} | "
                f"SL: {stop_loss} | TP: {take_profit_1}"
            )

            return trade_id

    def update_trade_parameters(
        self,
        trade_id: str,
        stop_loss: Optional[Decimal] = None,
        take_profit_1: Optional[Decimal] = None
    ):
        """Dynamically update trade parameters (Elite Management)."""
        with self._lock:
            if trade_id in self.trades:
                trade = self.trades[trade_id]
                if stop_loss:
                    trade.stop_loss = stop_loss
                    logger.info(f"[TRADE] {trade_id} SL updated to {stop_loss}")
                if take_profit_1:
                    trade.take_profit_1 = take_profit_1
                    logger.info(f"[TRADE] {trade_id} TP updated to {take_profit_1}")

    def update_prices(self, prices: Dict[str, Decimal]):
        """Update all trade prices."""
        with self._lock:
            for trade_id, trade in self.trades.items():
                if trade.status != TradeStatus.OPEN:
                    continue

                if trade.symbol in prices:
                    trade.update_price(prices[trade.symbol])

    def check_exits(self) -> List[ExitOrder]:
        """Check all trades for exit conditions."""
        exit_orders = []

        with self._lock:
            for trade_id, trade in list(self.trades.items()):
                if trade.status != TradeStatus.OPEN:
                    continue

                orders = self._check_trade_exits(trade)
                exit_orders.extend(orders)

        return exit_orders

    def _check_trade_exits(self, trade: LiveTrade) -> List[ExitOrder]:
        """Check a single trade for exits."""
        orders = []

        price = trade.current_price

        if trade.side == "LONG":
            # Stop loss
            if price <= trade.stop_loss:
                orders.append(self._create_exit_order(
                    trade, trade.current_quantity, ExitReason.STOP_LOSS
                ))

            # Trailing stop
            elif trade.trailing_stop_price and price <= trade.trailing_stop_price:
                orders.append(self._create_exit_order(
                    trade, trade.current_quantity, ExitReason.TRAILING_STOP
                ))

            # Take profits
            elif price >= trade.take_profit_3:
                orders.append(self._create_exit_order(
                    trade, trade.current_quantity, ExitReason.TAKE_PROFIT_3
                ))
            elif price >= trade.take_profit_2:
                # 50% partial close
                qty = max(1, trade.current_quantity // 2)
                orders.append(self._create_exit_order(
                    trade, qty, ExitReason.TAKE_PROFIT_2
                ))
            elif price >= trade.take_profit_1:
                # 30% partial close
                qty = max(1, trade.current_quantity * 30 // 100)
                orders.append(self._create_exit_order(
                    trade, qty, ExitReason.TAKE_PROFIT_1
                ))

        else:  # SHORT
            # Stop loss
            if price >= trade.stop_loss:
                orders.append(self._create_exit_order(
                    trade, trade.current_quantity, ExitReason.STOP_LOSS
                ))

            # Trailing stop
            elif trade.trailing_stop_price and price >= trade.trailing_stop_price:
                orders.append(self._create_exit_order(
                    trade, trade.current_quantity, ExitReason.TRAILING_STOP
                ))

            # Take profits (for short, TPs are below entry)
            elif price <= trade.take_profit_3:
                orders.append(self._create_exit_order(
                    trade, trade.current_quantity, ExitReason.TAKE_PROFIT_3
                ))
            elif price <= trade.take_profit_2:
                qty = max(1, trade.current_quantity // 2)
                orders.append(self._create_exit_order(
                    trade, qty, ExitReason.TAKE_PROFIT_2
                ))
            elif price <= trade.take_profit_1:
                qty = max(1, trade.current_quantity * 30 // 100)
                orders.append(self._create_exit_order(
                    trade, qty, ExitReason.TAKE_PROFIT_1
                ))

        return orders

    def _create_exit_order(
        self,
        trade: LiveTrade,
        quantity: int,
        reason: ExitReason
    ) -> ExitOrder:
        """Create an exit order."""
        return ExitOrder(
            trade_id=trade.trade_id,
            symbol=trade.symbol,
            side="SELL" if trade.side == "LONG" else "BUY",
            quantity=quantity,
            price=None,  # Market order
            order_type="MARKET",
            reason=reason
        )

    def process_exit(
        self,
        trade_id: str,
        quantity: int,
        exit_price: Decimal,
        reason: ExitReason
    ):
        """Process a completed exit."""
        with self._lock:
            if trade_id not in self.trades:
                return

            trade = self.trades[trade_id]

            # Calculate P&L for this exit
            if trade.side == "LONG":
                pnl = (exit_price - trade.entry_price) * quantity
            else:
                pnl = (trade.entry_price - exit_price) * quantity

            trade.realized_pnl += pnl
            trade.current_quantity -= quantity

            trade.exits.append({
                "timestamp": datetime.utcnow(),
                "quantity": quantity,
                "price": exit_price,
                "reason": reason.value,
                "pnl": pnl
            })

            # Full close
            if trade.current_quantity <= 0:
                trade.status = TradeStatus.CLOSED
                self.closed_trades.append(trade)
                del self.trades[trade_id]

                self.total_pnl += trade.realized_pnl

                if trade.realized_pnl > 0:
                    self.winning_trades += 1
                else:
                    self.losing_trades += 1

                logger.info(
                    f"[TRADE] Closed {trade_id}: {reason.value} | "
                    f"P&L: ${trade.realized_pnl:.2f}"
                )
            else:
                trade.status = TradeStatus.PARTIAL_CLOSE

                logger.info(
                    f"[TRADE] Partial exit {trade_id}: {quantity} @ {exit_price} | "
                    f"Remaining: {trade.current_quantity}"
                )

    def close_trade(
        self,
        trade_id: str,
        exit_price: Decimal,
        reason: ExitReason = ExitReason.MANUAL
    ):
        """Manually close a trade."""
        with self._lock:
            if trade_id not in self.trades:
                return

            trade = self.trades[trade_id]
            self.process_exit(trade_id, trade.current_quantity, exit_price, reason)

    def get_open_trades(self) -> List[LiveTrade]:
        """Get all open trades."""
        with self._lock:
            return list(self.trades.values())

    def get_trade(self, trade_id: str) -> Optional[LiveTrade]:
        """Get a specific trade."""
        with self._lock:
            return self.trades.get(trade_id)

    def get_position(self, symbol: str) -> Dict[str, Any]:
        """Get aggregate position for a symbol."""
        with self._lock:
            long_qty = 0
            short_qty = 0
            total_cost = Decimal("0")

            for trade in self.trades.values():
                if trade.symbol == symbol and trade.status == TradeStatus.OPEN:
                    if trade.side == "LONG":
                        long_qty += trade.current_quantity
                        total_cost += trade.entry_price * trade.current_quantity
                    else:
                        short_qty += trade.current_quantity
                        total_cost += trade.entry_price * trade.current_quantity

            net_qty = long_qty - short_qty
            avg_price = total_cost / (long_qty + short_qty) if (long_qty + short_qty) > 0 else Decimal("0")

            return {
                "symbol": symbol,
                "net_quantity": net_qty,
                "long_quantity": long_qty,
                "short_quantity": short_qty,
                "average_price": avg_price
            }

    def get_stats(self) -> Dict[str, Any]:
        """Get trading statistics."""
        with self._lock:
            win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0

            return {
                "total_trades": self.total_trades,
                "open_trades": len(self.trades),
                "closed_trades": len(self.closed_trades),
                "winning_trades": self.winning_trades,
                "losing_trades": self.losing_trades,
                "win_rate": win_rate,
                "total_pnl": float(self.total_pnl),
                "average_pnl": float(self.total_pnl / len(self.closed_trades)) if self.closed_trades else 0
            }


# Singleton
_manager: Optional[TradeManager] = None


def get_trade_manager() -> TradeManager:
    """Get or create the Trade Manager."""
    global _manager
    if _manager is None:
        _manager = TradeManager()
    return _manager
