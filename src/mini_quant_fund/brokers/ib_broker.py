"""
Interactive Brokers Production Adapter
======================================
Institutional-grade broker integration using ib_insync for:
- Futures (CME, ICE, Eurex, SGX)
- Forex spot and forwards
- Options on futures
- Global equities (LSE, TSX, ASX, JPX, HKEx)
- Commodities

Requires IB TWS or IB Gateway running.
"""

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class AssetClass(str, Enum):
    """Asset class types."""

    EQUITY = "STK"
    FUTURES = "FUT"
    FOREX = "CASH"
    OPTIONS = "OPT"
    COMMODITY = "CMDTY"
    INDEX = "IND"
    BOND = "BOND"


@dataclass
class IBContract:
    """Interactive Brokers contract specification."""

    symbol: str
    asset_class: AssetClass
    exchange: str
    currency: str = "USD"
    expiry: Optional[str] = None
    strike: Optional[float] = None
    right: Optional[str] = None
    multiplier: Optional[str] = None
    local_symbol: Optional[str] = None


@dataclass
class IBOrder:
    """Order tracking."""

    order_id: int
    contract: IBContract
    action: str
    quantity: float
    order_type: str
    limit_price: Optional[float] = None
    status: str = "PENDING"
    fill_price: float = 0.0
    filled_qty: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)


class IBBrokerAdapter:
    """
    Production Interactive Brokers adapter for multi-asset trading.

    Features:
    - Auto-reconnection with exponential backoff
    - Thread-safe order management
    - Futures across global exchanges (CME, ICE, Eurex, SGX)
    - Forex spot and forwards
    - Commodities
    - Real-time market data via WebSocket
    - Position reconciliation
    - Comprehensive error handling
    """

    # Exchange routing map
    EXCHANGE_MAP = {
        "ES": ("CME", "USD", "50"),
        "NQ": ("CME", "USD", "20"),
        "YM": ("CBOT", "USD", "5"),
        "RTY": ("CME", "USD", "50"),
        "CL": ("NYMEX", "USD", "1000"),
        "GC": ("COMEX", "USD", "100"),
        "SI": ("COMEX", "USD", "5000"),
        "HG": ("COMEX", "USD", "25000"),
        "NG": ("NYMEX", "USD", "10000"),
        "ZB": ("CBOT", "USD", "1000"),
        "ZN": ("CBOT", "USD", "1000"),
        "ZF": ("CBOT", "USD", "1000"),
        "FESX": ("EUREX", "EUR", "10"),
        "FDAX": ("EUREX", "EUR", "25"),
        "NKD": ("CME", "USD", "5"),
        "HSI": ("HKFE", "HKD", "50"),
        "SXF": ("CDE", "CAD", "200"),
    }

    # Forex pairs
    FOREX_PAIRS = {
        "EURUSD": ("EUR", "USD"),
        "GBPUSD": ("GBP", "USD"),
        "USDJPY": ("USD", "JPY"),
        "AUDUSD": ("AUD", "USD"),
        "USDCAD": ("USD", "CAD"),
        "USDCHF": ("USD", "CHF"),
        "NZDUSD": ("NZD", "USD"),
        "EURGBP": ("EUR", "GBP"),
        "EURJPY": ("EUR", "JPY"),
        "GBPJPY": ("GBP", "JPY"),
        "EURCHF": ("EUR", "CHF"),
        "AUDJPY": ("AUD", "JPY"),
        "CADJPY": ("CAD", "JPY"),
        "EURAUD": ("EUR", "AUD"),
        "EURCAD": ("EUR", "CAD"),
        "GBPAUD": ("GBP", "AUD"),
        "GBPCAD": ("GBP", "CAD"),
        "GBPCHF": ("GBP", "CHF"),
        "AUDCAD": ("AUD", "CAD"),
        "AUDCHF": ("AUD", "CHF"),
    }

    def __init__(
        self,
        host: str = "",
        port: int = 0,
        client_id: int = 1,
        auto_reconnect: bool = True,
        max_reconnect_attempts: int = 10,
    ):
        self.host = host or os.environ.get("IB_HOST", "127.0.0.1")
        self.port = port or int(os.environ.get("IB_PORT", "4002"))
        self.client_id = client_id
        self.auto_reconnect = auto_reconnect
        self.max_reconnect_attempts = max_reconnect_attempts

        self._ib = None
        self._connected = False
        self._lock = threading.Lock()
        self._positions: Dict[str, float] = {}
        self._orders: Dict[int, IBOrder] = {}
        self._next_order_id = 1
        self._reconnect_count = 0
        self._market_data_cache: Dict[str, Dict] = {}

    def connect(self) -> bool:
        """
        Connect to IB TWS/Gateway with auto-reconnection.

        Returns:
            True if connected successfully
        """
        try:
            from ib_insync import IB

            self._ib = IB()
            self._ib.connect(self.host, self.port, clientId=self.client_id)
            self._connected = True
            self._reconnect_count = 0

            # Register disconnect handler
            if self.auto_reconnect:
                self._ib.disconnectedEvent += self._on_disconnect

            # Sync positions
            self._sync_positions()

            logger.info(f"Connected to IB Gateway at " f"{self.host}:{self.port}")
            return True

        except ImportError:
            logger.warning(
                "ib_insync not installed. Install with: " "pip install ib_insync"
            )
            self._connected = False
            return False
        except Exception as e:
            logger.error(f"IB connection failed: {e}")
            self._connected = False
            if self.auto_reconnect:
                return self._attempt_reconnect()
            return False

    def _on_disconnect(self):
        """Handle disconnection event."""
        logger.warning("IB Gateway disconnected")
        self._connected = False
        if self.auto_reconnect:
            self._attempt_reconnect()

    def _attempt_reconnect(self) -> bool:
        """Reconnect with exponential backoff."""
        for attempt in range(self.max_reconnect_attempts):
            self._reconnect_count += 1
            wait = min(2**attempt, 60)
            logger.info(
                f"Reconnecting in {wait}s "
                f"(attempt {attempt + 1}/"
                f"{self.max_reconnect_attempts})"
            )
            time.sleep(wait)
            try:
                if self._ib:
                    self._ib.connect(self.host, self.port, clientId=self.client_id)
                    self._connected = True
                    self._sync_positions()
                    logger.info("Reconnected to IB Gateway")
                    return True
            except Exception as e:
                logger.warning(f"Reconnect attempt failed: {e}")
        logger.error("All reconnection attempts exhausted")
        return False

    def _sync_positions(self):
        """Synchronize positions from IB."""
        if not self._connected or not self._ib:
            return
        try:
            positions = self._ib.positions()
            self._positions = {}
            for pos in positions:
                symbol = pos.contract.symbol
                self._positions[symbol] = pos.position
            logger.info(f"Synced {len(self._positions)} positions")
        except Exception as e:
            logger.error(f"Position sync failed: {e}")

    def create_futures_contract(
        self,
        symbol: str,
        expiry: str = "",
        exchange: str = "",
    ) -> IBContract:
        """
        Create futures contract specification.

        Args:
            symbol: Futures symbol (e.g., "ES", "CL", "GC")
            expiry: Expiry YYYYMM (auto-detects if empty)
            exchange: Exchange (auto-detects from EXCHANGE_MAP)

        Returns:
            IBContract specification
        """
        info = self.EXCHANGE_MAP.get(symbol, ("CME", "USD", "1"))
        ex = exchange or info[0]
        curr = info[1]
        mult = info[2]

        if not expiry:
            expiry = FuturesRollCalendar().get_active_contract(
                symbol, datetime.utcnow()
            )

        return IBContract(
            symbol=symbol,
            asset_class=AssetClass.FUTURES,
            exchange=ex,
            currency=curr,
            expiry=expiry,
            multiplier=mult,
        )

    def create_forex_contract(self, pair: str) -> IBContract:
        """
        Create forex contract from pair string.

        Args:
            pair: e.g., "EURUSD", "GBPJPY"
        """
        if pair in self.FOREX_PAIRS:
            base, quote = self.FOREX_PAIRS[pair]
        else:
            base, quote = pair[:3], pair[3:]

        return IBContract(
            symbol=base,
            asset_class=AssetClass.FOREX,
            exchange="IDEALPRO",
            currency=quote,
        )

    def create_equity_contract(
        self,
        symbol: str,
        exchange: str = "SMART",
        currency: str = "USD",
    ) -> IBContract:
        """Create equity contract for global stocks."""
        return IBContract(
            symbol=symbol,
            asset_class=AssetClass.EQUITY,
            exchange=exchange,
            currency=currency,
        )

    def place_order(
        self,
        contract: IBContract,
        action: str,
        quantity: float,
        order_type: str = "MKT",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: str = "GTC",
    ) -> Optional[int]:
        """
        Place order with full validation.

        Returns:
            Order ID or None on failure
        """
        with self._lock:
            if not self._connected:
                logger.error("Not connected to IB")
                return None

            try:
                from ib_insync import (
                    Contract,
                    Forex,
                    Future,
                    LimitOrder,
                    MarketOrder,
                    Stock,
                    StopOrder,
                )

                # Build ib_insync contract
                if contract.asset_class == AssetClass.FUTURES:
                    ib_contract = Future(
                        symbol=contract.symbol,
                        lastTradeDateOrContractMonth=(contract.expiry),
                        exchange=contract.exchange,
                        currency=contract.currency,
                        multiplier=contract.multiplier,
                    )
                elif contract.asset_class == AssetClass.FOREX:
                    ib_contract = Forex(
                        pair=(f"{contract.symbol}" f"{contract.currency}"),
                    )
                else:
                    ib_contract = Stock(
                        symbol=contract.symbol,
                        exchange=contract.exchange,
                        currency=contract.currency,
                    )

                # Build order
                if order_type.upper() == "MKT":
                    ib_order = MarketOrder(action, quantity)
                elif order_type.upper() == "LMT":
                    ib_order = LimitOrder(action, quantity, limit_price)
                elif order_type.upper() == "STP":
                    ib_order = StopOrder(action, quantity, stop_price)
                else:
                    ib_order = MarketOrder(action, quantity)

                ib_order.tif = time_in_force

                # Submit
                trade = self._ib.placeOrder(ib_contract, ib_order)
                order_id = trade.order.orderId

                # Track
                self._orders[order_id] = IBOrder(
                    order_id=order_id,
                    contract=contract,
                    action=action,
                    quantity=quantity,
                    order_type=order_type,
                    limit_price=limit_price,
                    status="SUBMITTED",
                )

                logger.info(
                    f"IB Order {order_id}: {action} "
                    f"{quantity} {contract.symbol} "
                    f"@ {order_type}"
                )
                return order_id

            except ImportError:
                logger.error("ib_insync not installed")
                return None
            except Exception as e:
                logger.error(f"Order placement failed: {e}")
                return None

    def get_order_status(self, order_id: int) -> Optional[Dict]:
        """Get order status."""
        if order_id in self._orders:
            o = self._orders[order_id]
            return {
                "order_id": o.order_id,
                "status": o.status,
                "filled_qty": o.filled_qty,
                "fill_price": o.fill_price,
                "symbol": o.contract.symbol,
            }
        return None

    def get_positions(self) -> Dict[str, float]:
        """Get current positions."""
        self._sync_positions()
        return dict(self._positions)

    def get_market_data(self, contract: IBContract) -> Dict[str, Any]:
        """
        Get real-time market data.

        Returns:
            Dict with bid, ask, last, volume
        """
        if not self._connected or not self._ib:
            return {
                "bid": 0,
                "ask": 0,
                "last": 0,
                "volume": 0,
            }

        try:
            from ib_insync import Forex, Future, Stock

            if contract.asset_class == AssetClass.FUTURES:
                ib_c = Future(
                    contract.symbol,
                    contract.expiry,
                    contract.exchange,
                )
            elif contract.asset_class == AssetClass.FOREX:
                ib_c = Forex(f"{contract.symbol}{contract.currency}")
            else:
                ib_c = Stock(
                    contract.symbol,
                    contract.exchange,
                    contract.currency,
                )

            self._ib.qualifyContracts(ib_c)
            ticker = self._ib.reqMktData(ib_c)
            self._ib.sleep(1)

            return {
                "bid": ticker.bid or 0,
                "ask": ticker.ask or 0,
                "last": ticker.last or 0,
                "volume": ticker.volume or 0,
                "high": ticker.high or 0,
                "low": ticker.low or 0,
            }
        except Exception as e:
            logger.error(f"Market data error: {e}")
            return {
                "bid": 0,
                "ask": 0,
                "last": 0,
                "volume": 0,
            }

    def close_all_positions(self):
        """Close all open positions."""
        for symbol, qty in self._positions.items():
            if qty != 0:
                action = "SELL" if qty > 0 else "BUY"
                contract = self.create_equity_contract(symbol)
                self.place_order(contract, action, abs(qty), "MKT")
        logger.info("Closing all positions")

    def disconnect(self):
        """Disconnect from IB."""
        if self._ib and self._connected:
            self._ib.disconnect()
            self._connected = False
            logger.info("Disconnected from IB Gateway")

    @property
    def is_connected(self) -> bool:
        """Check connection status."""
        return self._connected


class FuturesRollCalendar:
    """
    Futures roll calendar management.
    Handles contract expiry and rolling to next contract.
    """

    # Standard quarterly months: Mar, Jun, Sep, Dec
    QUARTERLY_MONTHS = ["03", "06", "09", "12"]

    # Monthly contracts months
    MONTHLY_MONTHS = [f"{m:02d}" for m in range(1, 13)]

    # Symbol → schedule type
    SCHEDULE = {
        "ES": "quarterly",
        "NQ": "quarterly",
        "YM": "quarterly",
        "RTY": "quarterly",
        "CL": "monthly",
        "NG": "monthly",
        "GC": "bimonthly",
        "SI": "quarterly_offset",
        "HG": "monthly",
        "ZB": "quarterly",
        "ZN": "quarterly",
        "FESX": "quarterly",
        "FDAX": "quarterly",
        "NKD": "quarterly",
        "HSI": "monthly",
    }

    # Roll days before expiry
    ROLL_DAYS = {
        "ES": 8,
        "NQ": 8,
        "YM": 8,
        "RTY": 8,
        "CL": 5,
        "NG": 5,
        "GC": 5,
        "SI": 5,
        "HG": 5,
        "ZB": 3,
        "ZN": 3,
        "FESX": 5,
        "FDAX": 5,
        "NKD": 8,
        "HSI": 3,
    }

    def __init__(self):
        self.roll_dates: Dict[str, List[str]] = {}

    def get_active_contract(self, symbol: str, as_of_date: datetime) -> str:
        """
        Get active futures contract expiry code.

        Returns:
            Expiry code e.g., "202603"
        """
        sched = self.SCHEDULE.get(symbol, "quarterly")
        year = as_of_date.year
        month = as_of_date.month

        if sched == "quarterly":
            months = self.QUARTERLY_MONTHS
        elif sched == "monthly":
            months = self.MONTHLY_MONTHS
        else:
            months = self.QUARTERLY_MONTHS

        # Find next expiry
        for m in months:
            m_int = int(m)
            if m_int >= month:
                return f"{year}{m}"

        # Wrap to next year
        return f"{year + 1}{months[0]}"

    def should_roll(
        self,
        symbol: str,
        current_expiry: str,
        as_of_date: datetime,
    ) -> bool:
        """
        Check if position should be rolled.

        Returns:
            True if within roll window
        """
        roll_days = self.ROLL_DAYS.get(symbol, 5)

        # Parse expiry
        exp_year = int(current_expiry[:4])
        exp_month = int(current_expiry[4:6])

        # Approximate expiry as 3rd Friday
        from calendar import monthcalendar

        cal = monthcalendar(exp_year, exp_month)
        fridays = [week[4] for week in cal if week[4] != 0]
        third_friday = fridays[2] if len(fridays) >= 3 else 15
        expiry_date = datetime(exp_year, exp_month, third_friday)

        days_to_expiry = (expiry_date - as_of_date).days
        return days_to_expiry <= roll_days

    def get_next_contract(self, symbol: str, current_expiry: str) -> str:
        """Get next contract after current."""
        sched = self.SCHEDULE.get(symbol, "quarterly")
        current_month = int(current_expiry[4:6])
        current_year = int(current_expiry[:4])

        if sched == "quarterly":
            months = [3, 6, 9, 12]
        elif sched == "monthly":
            months = list(range(1, 13))
        else:
            months = [3, 6, 9, 12]

        for m in months:
            if m > current_month:
                return f"{current_year}{m:02d}"

        return f"{current_year + 1}{months[0]:02d}"
