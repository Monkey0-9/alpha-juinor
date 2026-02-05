"""
Advanced Execution Algorithms
================================

Professional execution algorithms for optimal order execution.

Algorithms:
1. TWAP (Time-Weighted Average Price)
2. VWAP (Volume-Weighted Average Price)
3. Iceberg Orders
4. Sniper (Aggressive Fill)
5. POV (Percentage of Volume)
6. Implementation Shortfall

Minimize slippage. Maximize execution quality.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal, getcontext
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

getcontext().prec = 50


class ExecutionAlgo(Enum):
    """Execution algorithm types."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    TWAP = "TWAP"
    VWAP = "VWAP"
    ICEBERG = "ICEBERG"
    SNIPER = "SNIPER"
    POV = "POV"
    IMPLEMENTATION_SHORTFALL = "IS"


@dataclass
class ExecutionOrder:
    """An execution order slice."""
    timestamp: datetime

    # Order details
    symbol: str
    side: str  # BUY, SELL
    total_quantity: int
    slice_quantity: int
    slice_number: int
    total_slices: int

    # Price
    limit_price: Optional[Decimal]
    order_type: str

    # Timing
    execute_after: datetime
    expires_at: datetime

    # Algorithm
    algorithm: str
    urgency: float  # 0 to 1


@dataclass
class ExecutionPlan:
    """Complete execution plan for an order."""
    symbol: str
    side: str
    total_quantity: int

    # Algorithm
    algorithm: ExecutionAlgo

    # Slices
    orders: List[ExecutionOrder]

    # Estimates
    estimated_avg_price: Decimal
    estimated_slippage: Decimal
    estimated_cost: Decimal

    # Timing
    start_time: datetime
    end_time: datetime
    duration_minutes: int


class TWAPExecutor:
    """
    Time-Weighted Average Price executor.

    Splits order into equal slices over time.
    """

    def __init__(self):
        """Initialize TWAP executor."""
        logger.info("[EXEC] TWAP Executor initialized")

    def create_plan(
        self,
        symbol: str,
        side: str,
        quantity: int,
        current_price: Decimal,
        duration_minutes: int = 60,
        num_slices: int = 10
    ) -> ExecutionPlan:
        """Create TWAP execution plan."""
        orders = []

        slice_qty = quantity // num_slices
        remainder = quantity % num_slices

        start_time = datetime.utcnow()
        interval = timedelta(minutes=duration_minutes / num_slices)

        for i in range(num_slices):
            slice_time = start_time + interval * i

            # Add remainder to last slice
            qty = slice_qty + (remainder if i == num_slices - 1 else 0)

            orders.append(ExecutionOrder(
                timestamp=datetime.utcnow(),
                symbol=symbol,
                side=side,
                total_quantity=quantity,
                slice_quantity=qty,
                slice_number=i + 1,
                total_slices=num_slices,
                limit_price=None,  # Market orders for TWAP
                order_type="MARKET",
                execute_after=slice_time,
                expires_at=slice_time + timedelta(minutes=5),
                algorithm="TWAP",
                urgency=0.5
            ))

        # Estimate cost
        estimated_slippage = Decimal("0.002")  # 0.2% slippage estimate
        slippage_direction = Decimal("1") if side == "BUY" else Decimal("-1")
        estimated_avg = current_price * (1 + estimated_slippage * slippage_direction)

        return ExecutionPlan(
            symbol=symbol,
            side=side,
            total_quantity=quantity,
            algorithm=ExecutionAlgo.TWAP,
            orders=orders,
            estimated_avg_price=estimated_avg.quantize(Decimal("0.01")),
            estimated_slippage=(estimated_slippage * current_price).quantize(Decimal("0.01")),
            estimated_cost=(Decimal(str(quantity)) * estimated_avg).quantize(Decimal("0.01")),
            start_time=start_time,
            end_time=start_time + timedelta(minutes=duration_minutes),
            duration_minutes=duration_minutes
        )


class VWAPExecutor:
    """
    Volume-Weighted Average Price executor.

    Trades in proportion to historical volume profile.
    """

    def __init__(self):
        """Initialize VWAP executor."""
        logger.info("[EXEC] VWAP Executor initialized")

    def create_plan(
        self,
        symbol: str,
        side: str,
        quantity: int,
        current_price: Decimal,
        volume_profile: Optional[List[float]] = None,
        duration_minutes: int = 60,
        num_slices: int = 10
    ) -> ExecutionPlan:
        """Create VWAP execution plan."""
        orders = []

        # Default volume profile (U-shape typical for stocks)
        if volume_profile is None:
            # Higher volume at open and close
            volume_profile = [0.15, 0.12, 0.08, 0.06, 0.05, 0.05, 0.06, 0.08, 0.12, 0.23]

        # Normalize
        total_weight = sum(volume_profile[:num_slices])
        weights = [v / total_weight for v in volume_profile[:num_slices]]

        start_time = datetime.utcnow()
        interval = timedelta(minutes=duration_minutes / num_slices)

        remaining = quantity

        for i in range(num_slices):
            slice_time = start_time + interval * i

            if i == num_slices - 1:
                qty = remaining
            else:
                qty = int(quantity * weights[i])
                remaining -= qty

            if qty <= 0:
                continue

            orders.append(ExecutionOrder(
                timestamp=datetime.utcnow(),
                symbol=symbol,
                side=side,
                total_quantity=quantity,
                slice_quantity=qty,
                slice_number=i + 1,
                total_slices=num_slices,
                limit_price=None,
                order_type="MARKET",
                execute_after=slice_time,
                expires_at=slice_time + timedelta(minutes=5),
                algorithm="VWAP",
                urgency=0.5
            ))

        # Estimate (VWAP typically has lower slippage than TWAP)
        estimated_slippage = Decimal("0.001")
        slippage_direction = Decimal("1") if side == "BUY" else Decimal("-1")
        estimated_avg = current_price * (1 + estimated_slippage * slippage_direction)

        return ExecutionPlan(
            symbol=symbol,
            side=side,
            total_quantity=quantity,
            algorithm=ExecutionAlgo.VWAP,
            orders=orders,
            estimated_avg_price=estimated_avg.quantize(Decimal("0.01")),
            estimated_slippage=(estimated_slippage * current_price).quantize(Decimal("0.01")),
            estimated_cost=(Decimal(str(quantity)) * estimated_avg).quantize(Decimal("0.01")),
            start_time=start_time,
            end_time=start_time + timedelta(minutes=duration_minutes),
            duration_minutes=duration_minutes
        )


class IcebergExecutor:
    """
    Iceberg order executor.

    Shows only small portions of a large order.
    """

    def __init__(self):
        """Initialize Iceberg executor."""
        logger.info("[EXEC] Iceberg Executor initialized")

    def create_plan(
        self,
        symbol: str,
        side: str,
        quantity: int,
        limit_price: Decimal,
        visible_quantity: int = 100
    ) -> ExecutionPlan:
        """Create Iceberg execution plan."""
        orders = []

        num_slices = (quantity + visible_quantity - 1) // visible_quantity

        start_time = datetime.utcnow()
        remaining = quantity

        for i in range(num_slices):
            qty = min(visible_quantity, remaining)
            remaining -= qty

            orders.append(ExecutionOrder(
                timestamp=datetime.utcnow(),
                symbol=symbol,
                side=side,
                total_quantity=quantity,
                slice_quantity=qty,
                slice_number=i + 1,
                total_slices=num_slices,
                limit_price=limit_price,
                order_type="LIMIT",
                execute_after=start_time,
                expires_at=start_time + timedelta(hours=4),
                algorithm="ICEBERG",
                urgency=0.3
            ))

        # Minimal slippage for limit orders
        estimated_avg = limit_price

        return ExecutionPlan(
            symbol=symbol,
            side=side,
            total_quantity=quantity,
            algorithm=ExecutionAlgo.ICEBERG,
            orders=orders,
            estimated_avg_price=estimated_avg,
            estimated_slippage=Decimal("0"),
            estimated_cost=(Decimal(str(quantity)) * estimated_avg).quantize(Decimal("0.01")),
            start_time=start_time,
            end_time=start_time + timedelta(hours=4),
            duration_minutes=240
        )


class SniperExecutor:
    """
    Sniper executor for aggressive fills.

    Takes liquidity quickly when opportunities arise.
    """

    def __init__(self):
        """Initialize Sniper executor."""
        logger.info("[EXEC] Sniper Executor initialized")

    def create_plan(
        self,
        symbol: str,
        side: str,
        quantity: int,
        current_price: Decimal,
        max_slippage_pct: float = 0.005
    ) -> ExecutionPlan:
        """Create Sniper execution plan."""
        orders = []

        start_time = datetime.utcnow()

        # Aggressive limit price
        slippage = Decimal(str(max_slippage_pct))
        if side == "BUY":
            limit = current_price * (1 + slippage)
        else:
            limit = current_price * (1 - slippage)

        # Single aggressive order
        orders.append(ExecutionOrder(
            timestamp=datetime.utcnow(),
            symbol=symbol,
            side=side,
            total_quantity=quantity,
            slice_quantity=quantity,
            slice_number=1,
            total_slices=1,
            limit_price=limit.quantize(Decimal("0.01")),
            order_type="LIMIT",
            execute_after=start_time,
            expires_at=start_time + timedelta(minutes=5),
            algorithm="SNIPER",
            urgency=1.0
        ))

        return ExecutionPlan(
            symbol=symbol,
            side=side,
            total_quantity=quantity,
            algorithm=ExecutionAlgo.SNIPER,
            orders=orders,
            estimated_avg_price=limit.quantize(Decimal("0.01")),
            estimated_slippage=(slippage * current_price).quantize(Decimal("0.01")),
            estimated_cost=(Decimal(str(quantity)) * limit).quantize(Decimal("0.01")),
            start_time=start_time,
            end_time=start_time + timedelta(minutes=5),
            duration_minutes=5
        )


class POVExecutor:
    """
    Percentage of Volume executor.

    Trades as a percentage of market volume.
    """

    def __init__(self):
        """Initialize POV executor."""
        logger.info("[EXEC] POV Executor initialized")

    def create_plan(
        self,
        symbol: str,
        side: str,
        quantity: int,
        current_price: Decimal,
        target_pov: float = 0.10,  # 10% of volume
        avg_volume: int = 1000000,
        duration_minutes: int = 120
    ) -> ExecutionPlan:
        """Create POV execution plan."""
        orders = []

        # Calculate slices based on participation rate
        slices_per_minute = (avg_volume * target_pov) / 390  # Trading day minutes

        num_slices = max(1, min(20, duration_minutes // 10))
        slice_qty = quantity // num_slices

        start_time = datetime.utcnow()
        interval = timedelta(minutes=duration_minutes / num_slices)

        remaining = quantity

        for i in range(num_slices):
            slice_time = start_time + interval * i
            qty = slice_qty if i < num_slices - 1 else remaining
            remaining -= qty

            orders.append(ExecutionOrder(
                timestamp=datetime.utcnow(),
                symbol=symbol,
                side=side,
                total_quantity=quantity,
                slice_quantity=qty,
                slice_number=i + 1,
                total_slices=num_slices,
                limit_price=None,
                order_type="MARKET",
                execute_after=slice_time,
                expires_at=slice_time + timedelta(minutes=15),
                algorithm="POV",
                urgency=0.4
            ))

        estimated_slippage = Decimal("0.0015")
        slippage_direction = Decimal("1") if side == "BUY" else Decimal("-1")
        estimated_avg = current_price * (1 + estimated_slippage * slippage_direction)

        return ExecutionPlan(
            symbol=symbol,
            side=side,
            total_quantity=quantity,
            algorithm=ExecutionAlgo.POV,
            orders=orders,
            estimated_avg_price=estimated_avg.quantize(Decimal("0.01")),
            estimated_slippage=(estimated_slippage * current_price).quantize(Decimal("0.01")),
            estimated_cost=(Decimal(str(quantity)) * estimated_avg).quantize(Decimal("0.01")),
            start_time=start_time,
            end_time=start_time + timedelta(minutes=duration_minutes),
            duration_minutes=duration_minutes
        )


class ExecutionEngine:
    """
    Complete execution engine.

    Selects and executes optimal algorithm based on order characteristics.
    """

    def __init__(self):
        """Initialize the execution engine."""
        self.twap = TWAPExecutor()
        self.vwap = VWAPExecutor()
        self.iceberg = IcebergExecutor()
        self.sniper = SniperExecutor()
        self.pov = POVExecutor()

        self.orders_executed = 0
        self.total_volume = 0

        logger.info("[EXEC] Execution Engine initialized - ALL ALGORITHMS READY")

    def select_algorithm(
        self,
        quantity: int,
        avg_volume: int,
        urgency: float,
        current_price: Decimal
    ) -> ExecutionAlgo:
        """Select optimal execution algorithm."""
        # Order size as % of daily volume
        pov = quantity / avg_volume if avg_volume > 0 else 1

        # Small orders or high urgency -> Sniper
        if pov < 0.01 or urgency > 0.8:
            return ExecutionAlgo.SNIPER

        # Medium orders -> VWAP
        if pov < 0.05:
            return ExecutionAlgo.VWAP

        # Large orders with low urgency -> Iceberg
        if urgency < 0.3:
            return ExecutionAlgo.ICEBERG

        # Large orders with medium urgency -> POV
        if urgency < 0.6:
            return ExecutionAlgo.POV

        # Default to TWAP
        return ExecutionAlgo.TWAP

    def create_execution_plan(
        self,
        symbol: str,
        side: str,
        quantity: int,
        current_price: Decimal,
        algorithm: Optional[ExecutionAlgo] = None,
        avg_volume: int = 1000000,
        urgency: float = 0.5
    ) -> ExecutionPlan:
        """Create execution plan for an order."""
        if algorithm is None:
            algorithm = self.select_algorithm(quantity, avg_volume, urgency, current_price)

        if algorithm == ExecutionAlgo.TWAP:
            return self.twap.create_plan(symbol, side, quantity, current_price)
        elif algorithm == ExecutionAlgo.VWAP:
            return self.vwap.create_plan(symbol, side, quantity, current_price)
        elif algorithm == ExecutionAlgo.ICEBERG:
            return self.iceberg.create_plan(symbol, side, quantity, current_price)
        elif algorithm == ExecutionAlgo.SNIPER:
            return self.sniper.create_plan(symbol, side, quantity, current_price)
        elif algorithm == ExecutionAlgo.POV:
            return self.pov.create_plan(symbol, side, quantity, current_price, avg_volume=avg_volume)
        else:
            return self.sniper.create_plan(symbol, side, quantity, current_price)

    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return {
            "orders_executed": self.orders_executed,
            "total_volume": self.total_volume
        }


# Singleton
_engine: Optional[ExecutionEngine] = None


def get_execution_engine() -> ExecutionEngine:
    """Get or create the Execution Engine."""
    global _engine
    if _engine is None:
        _engine = ExecutionEngine()
    return _engine
