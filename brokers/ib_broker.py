"""
Interactive Brokers Futures/Forex Adapter
=========================================

Institutional-grade broker integration for:
- Futures trading (CME, ICE, Eurex)
- Forex spot and forwards
- Options on futures
- Global commodities

Uses IB TWS API.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class AssetClass(Enum):
    """Asset class types."""

    EQUITY = "STK"
    FUTURES = "FUT"
    FOREX = "CASH"
    OPTIONS = "OPT"
    COMMODITY = "CMDTY"


@dataclass
class IBContract:
    """Interactive Brokers contract specification."""

    symbol: str
    asset_class: AssetClass
    exchange: str
    currency: str = "USD"
    expiry: Optional[str] = None  # For futures/options
    strike: Optional[float] = None  # For options
    right: Optional[str] = None  # "C" or "P" for options


class IBBrokerAdapter:
    """
    Interactive Brokers adapter for multi-asset trading.

    Features:
    - Futures across global exchanges
    - Forex spot and forwards
    - Commodities
    - Real-time market data
    - Order management
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7497,  # TWS paper trading port
        client_id: int = 1,
    ):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.connected = False
        self.positions: Dict[str, float] = {}
        self.orders: Dict[int, Dict] = {}

        # Simulated mode if IB not available
        self.simulation_mode = True
        logger.info("IBBrokerAdapter initialized in simulation mode")

    def connect(self) -> bool:
        """
        Connect to IB TWS/Gateway.

        Returns:
            True if connected successfully
        """
        try:
            # In real implementation, would use ibapi
            # from ib api.client import EClient
            # from ibapi.wrapper import EWrapper
            # self.app = TWSClient(...).connect()

            # For now, simulate connection
            logger.info(
                f"Simulating connection to IB at {self.host}:{self.port}"
            )
            self.connected = True
            return True

        except Exception as e:
            logger.error(f"Failed to connect to IB: {e}")
            return False

    def create_futures_contract(
        self,
        symbol: str,
        exchange: str,
        expiry: str,
    ) -> IBContract:
        """
        Create futures contract specification.

        Args:
            symbol: Futures symbol (e.g., "ES", "CL", "GC")
            exchange: Exchange code (e.g., "CME", "NYMEX", "COMEX")
            expiry: Expiry date YYYYMM format

        Returns:
            IBContract specification
        """
        return IBContract(
            symbol=symbol,
            asset_class=AssetClass.FUTURES,
            exchange=exchange,
            currency="USD",
            expiry=expiry,
        )

    def create_forex_contract(
        self, base_currency: str, quote_currency: str
    ) -> IBContract:
        """
        Create forex contract specification.

        Args:
            base_currency: Base currency (e.g., "EUR")
            quote_currency: Quote currency (e.g., "USD")

        Returns:
            IBContract specification
        """
        return IBContract(
            symbol=base_currency,
            asset_class=AssetClass.FOREX,
            exchange="IDEALPRO",  # IB's FX exchange
            currency=quote_currency,
        )

    def place_order(
        self,
        contract: IBContract,
        action: str,
        quantity: int,
        order_type: str = "MKT",
        limit_price: Optional[float] = None,
    ) -> int:
        """
        Place order.

        Args:
            contract: Contract specification
            action: "BUY" or "SELL"
            quantity: Order size
            order_type: "MKT", "LMT", "STP", etc.
            limit_price: Limit price for limit orders

        Returns:
            Order ID
        """
        order_id = len(self.orders) + 1000

        order = {
            "contract": contract,
            "action": action,
            "quantity": quantity,
            "order_type": order_type,
            "limit_price": limit_price,
            "status": "SUBMITTED",
            "filled_qty": 0,
            "timestamp": datetime.now(),
        }

        self.orders[order_id] = order

        if self.simulation_mode:
            # Simulate immediate fill for market orders
            if order_type == "MKT":
                order["status"] = "FILLED"
                order["filled_qty"] = quantity
                self._update_position(contract.symbol, quantity if action == "BUY" else -quantity)

        logger.info(
            f"Order {order_id}: {action} {quantity} {contract.symbol} @ {order_type}"
        )

        return order_id

    def get_order_status(self, order_id: int) -> Optional[Dict]:
        """Get order status."""
        return self.orders.get(order_id)

    def get_positions(self) -> Dict[str, float]:
        """Get current positions."""
        return self.positions.copy()

    def get_market_data(self, contract: IBContract) -> Dict[str, float]:
        """
        Get real-time market data.

        Args:
            contract: Contract specification

        Returns:
            Dictionary with bid, ask, last, volume
        """
        if self.simulation_mode:
            # Simulated market data
            import random

            base_price = {
                "ES": 4500,  # S&P 500 futures
                "NQ": 15000,  # Nasdaq futures
                "CL": 75,  # Crude oil
                "GC": 2000,  # Gold
                "EUR": 1.10,  # EUR/USD
            }.get(contract.symbol, 100)

            spread = base_price * 0.0001  # 1bp spread

            return {
                "bid": base_price - spread / 2,
                "ask": base_price + spread / 2,
                "last": base_price,
                "volume": random.randint(10000, 100000),
                "timestamp": datetime.now(),
            }

        # Real implementation would request market data via IB API
        return {}

    def _update_position(self, symbol: str, quantity_change: float):
        """Update position tracking."""
        current = self.positions.get(symbol, 0)
        self.positions[symbol] = current + quantity_change

        if abs(self.positions[symbol]) < 1e-6:
            del self.positions[symbol]

    def close_all_positions(self):
        """Close all open positions."""
        for symbol, qty in list(self.positions.items()):
            action = "SELL" if qty > 0 else "BUY"
            # Create contract (simplified)
            contract = IBContract(
                symbol=symbol,
                asset_class=AssetClass.FUTURES,
                exchange="CME",
            )
            self.place_order(contract, action, abs(qty))

        logger.info("Closed all positions")

    def disconnect(self):
        """Disconnect from IB."""
        self.connected = False
        logger.info("Disconnected from IB")


class FuturesRollCalendar:
    """
    Futures roll calendar management.

    Handles contract expiry and rolling to next contract.
    """

    def __init__(self):
        self.roll_dates: Dict[str, List[str]] = {}

    def get_active_contract(self, symbol: str, as_of_date: datetime) -> str:
        """
        Get active futures contract for a symbol.

        Args:
            symbol: Futures symbol
            as_of_date: Date to check

        Returns:
            Expiry code (e.g., "202603" for March 2026)
        """
        # Simplified: use next quarter month
        month = as_of_date.month
        year = as_of_date.year

        # Roll to next quarter month
        quarter_months = [3, 6, 9, 12]
        next_quarter = min([m for m in quarter_months if m >= month], default=3)

        if next_quarter < month:
            year += 1

        expiry = f"{year}{next_quarter:02d}"
        return expiry

    def should_roll(
        self, symbol: str, current_expiry: str, as_of_date: datetime
    ) -> bool:
        """
        Check if position should be rolled to next contract.

        Args:
            symbol: Futures symbol
            current_expiry: Current contract expiry
            as_of_date: Current date

        Returns:
            True if should roll
        """
        # Simple rule: roll 5 days before expiry
        expiry_year = int(current_expiry[:4])
        expiry_month = int(current_expiry[4:6])

        expiry_date = datetime(expiry_year, expiry_month, 15)  # Mid-month
        days_to_expiry = (expiry_date - as_of_date).days

        return days_to_expiry <= 5
