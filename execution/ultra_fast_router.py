"""
Ultra-Fast Execution - Citadel-style Low-Latency Trading.

Features:
- Parallel order submission
- Exchange-specific routing
- Latency optimization
- Smart order types
"""

import logging
import time
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
import threading

logger = logging.getLogger(__name__)


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    IOC = "IOC"  # Immediate or Cancel
    FOK = "FOK"  # Fill or Kill
    TWAP = "TWAP"
    VWAP = "VWAP"


class ExecutionVenue(Enum):
    PRIMARY = "PRIMARY"  # Main exchange
    DARK_POOL = "DARK_POOL"
    SOR = "SOR"  # Smart Order Routing


@dataclass
class FastOrder:
    """High-performance order structure."""
    order_id: str
    symbol: str
    side: str  # BUY, SELL
    quantity: float
    order_type: OrderType
    limit_price: Optional[float] = None
    venue: ExecutionVenue = ExecutionVenue.PRIMARY

    # Timing
    created_at: float = field(default_factory=time.time)
    submitted_at: Optional[float] = None
    filled_at: Optional[float] = None

    # Fill info
    fill_price: Optional[float] = None
    fill_quantity: float = 0.0
    status: str = "PENDING"


@dataclass
class ExecutionReport:
    """Execution report with latency metrics."""
    order_id: str
    symbol: str
    side: str
    requested_quantity: float
    filled_quantity: float
    average_price: float

    # Latency breakdown
    submission_latency_ms: float
    total_latency_ms: float

    # Quality metrics
    slippage_bps: float
    market_impact_bps: float


class UltraFastRouter:
    """
    Ultra-low latency order router.

    Features:
    - Parallel submission to multiple venues
    - Smart order routing
    - Latency monitoring
    - Execution quality tracking
    """

    def __init__(
        self,
        max_workers: int = 4,
        default_timeout: float = 1.0
    ):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.default_timeout = default_timeout

        # Order tracking
        self.orders: Dict[str, FastOrder] = {}
        self.order_counter = 0
        self.lock = threading.Lock()

        # Latency stats
        self.latency_history: List[float] = []

        # Venue routing rules
        self.venue_rules = {
            "large_order_threshold": 10000,  # Shares
            "dark_pool_symbols": set()
        }

    def generate_order_id(self) -> str:
        """Generate unique order ID."""
        with self.lock:
            self.order_counter += 1
            return f"ORD-{int(time.time())}-{self.order_counter:06d}"

    def select_venue(
        self,
        symbol: str,
        quantity: float,
        order_type: OrderType
    ) -> ExecutionVenue:
        """Smart venue selection."""
        # Large orders -> dark pool to reduce impact
        if quantity > self.venue_rules["large_order_threshold"]:
            return ExecutionVenue.DARK_POOL

        # Use SOR for complex order types
        if order_type in [OrderType.TWAP, OrderType.VWAP]:
            return ExecutionVenue.SOR

        return ExecutionVenue.PRIMARY

    def submit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None
    ) -> FastOrder:
        """
        Submit order with minimal latency.
        """
        start_time = time.time()

        order = FastOrder(
            order_id=self.generate_order_id(),
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            venue=self.select_venue(symbol, quantity, order_type)
        )

        self.orders[order.order_id] = order

        # Submit asynchronously
        future = self.executor.submit(self._execute_order, order)

        order.submitted_at = time.time()
        submission_latency = (order.submitted_at - start_time) * 1000

        logger.debug(
            f"Order {order.order_id} submitted in {submission_latency:.2f}ms"
        )

        return order

    def submit_parallel(
        self,
        orders: List[Tuple[str, str, float, OrderType]]
    ) -> List[FastOrder]:
        """
        Submit multiple orders in parallel.

        Args:
            orders: List of (symbol, side, quantity, order_type)
        """
        start_time = time.time()

        fast_orders = []
        for symbol, side, quantity, order_type in orders:
            order = FastOrder(
                order_id=self.generate_order_id(),
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=order_type,
                venue=self.select_venue(symbol, quantity, order_type)
            )
            fast_orders.append(order)
            self.orders[order.order_id] = order

        # Submit all in parallel
        futures = [
            self.executor.submit(self._execute_order, order)
            for order in fast_orders
        ]

        elapsed = (time.time() - start_time) * 1000
        logger.info(
            f"Submitted {len(fast_orders)} orders in {elapsed:.2f}ms "
            f"({elapsed/len(fast_orders):.2f}ms/order)"
        )

        return fast_orders

    def _execute_order(self, order: FastOrder) -> FastOrder:
        """Execute order (simulated for demo)."""
        # Simulate execution
        time.sleep(0.001)  # 1ms execution time

        order.filled_at = time.time()
        order.fill_quantity = order.quantity
        order.fill_price = order.limit_price or 100.0  # Simulated
        order.status = "FILLED"

        # Track latency
        if order.submitted_at:
            latency = (order.filled_at - order.submitted_at) * 1000
            self.latency_history.append(latency)
            if len(self.latency_history) > 1000:
                self.latency_history = self.latency_history[-1000:]

        return order

    def get_execution_report(self, order_id: str) -> Optional[ExecutionReport]:
        """Get execution report for an order."""
        order = self.orders.get(order_id)
        if not order or order.status != "FILLED":
            return None

        submission_latency = (
            (order.submitted_at - order.created_at) * 1000
            if order.submitted_at else 0
        )
        total_latency = (
            (order.filled_at - order.created_at) * 1000
            if order.filled_at else 0
        )

        return ExecutionReport(
            order_id=order_id,
            symbol=order.symbol,
            side=order.side,
            requested_quantity=order.quantity,
            filled_quantity=order.fill_quantity,
            average_price=order.fill_price or 0,
            submission_latency_ms=submission_latency,
            total_latency_ms=total_latency,
            slippage_bps=0.0,  # Would calculate from expected price
            market_impact_bps=0.0
        )

    def get_latency_stats(self) -> Dict[str, float]:
        """Get latency statistics."""
        if not self.latency_history:
            return {}

        import numpy as np
        latencies = np.array(self.latency_history)

        return {
            "mean_ms": float(np.mean(latencies)),
            "median_ms": float(np.median(latencies)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "p99_ms": float(np.percentile(latencies, 99)),
            "min_ms": float(np.min(latencies)),
            "max_ms": float(np.max(latencies)),
            "n_orders": len(latencies)
        }

    def shutdown(self):
        """Shutdown executor."""
        self.executor.shutdown(wait=True)


# Global singleton
_fast_router: Optional[UltraFastRouter] = None


def get_ultra_fast_router() -> UltraFastRouter:
    """Get or create global ultra-fast router."""
    global _fast_router
    if _fast_router is None:
        _fast_router = UltraFastRouter()
    return _fast_router
