"""
Smart Execution Algorithms - TWAP/VWAP/POV.

Production-grade execution algorithms used by top hedge funds.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class AlgoType(Enum):
    """Execution algorithm types."""
    MARKET = "market"
    TWAP = "twap"  # Time-Weighted Average Price
    VWAP = "vwap"  # Volume-Weighted Average Price
    POV = "pov"    # Percent of Volume
    ICEBERG = "iceberg"
    SNIPER = "sniper"


@dataclass
class ChildOrder:
    """Individual child order from slicing."""
    parent_id: str
    child_id: str
    symbol: str
    side: str
    quantity: int
    price_limit: Optional[float]
    scheduled_time: float
    status: str = "pending"
    filled_qty: int = 0
    avg_price: float = 0.0


@dataclass
class ExecutionPlan:
    """Full execution plan for an order."""
    parent_id: str
    symbol: str
    side: str
    total_quantity: int
    algo_type: AlgoType
    start_time: float
    end_time: float
    child_orders: List[ChildOrder] = field(default_factory=list)
    estimated_cost_bps: float = 0.0
    urgency: str = "low"


@dataclass
class AlgoResult:
    """Result of algorithm execution."""
    parent_id: str
    symbol: str
    algo_type: AlgoType
    target_quantity: int
    filled_quantity: int
    avg_fill_price: float
    benchmark_price: float
    slippage_bps: float
    execution_time_seconds: float
    child_orders_executed: int


class SmartExecutionAlgorithms:
    """
    Production-grade execution algorithms.

    Algorithms:
    1. TWAP - Evenly slice order over time
    2. VWAP - Weight slices by expected volume
    3. POV - Execute as % of market volume
    4. Iceberg - Hide large orders
    5. Sniper - Aggressive liquidity taking
    """

    def __init__(
        self,
        default_duration_minutes: int = 30,
        min_slice_size: int = 100,
        max_slices: int = 20
    ):
        self.default_duration = default_duration_minutes * 60
        self.min_slice_size = min_slice_size
        self.max_slices = max_slices

        # Intraday volume profile (typical US equities)
        self.volume_profile = {
            9: 0.12,   # 9:30 AM
            10: 0.10,
            11: 0.08,
            12: 0.06,
            13: 0.06,
            14: 0.08,
            15: 0.10,
            16: 0.15   # 4:00 PM
        }

    def create_twap_plan(
        self,
        parent_id: str,
        symbol: str,
        side: str,
        quantity: int,
        duration_seconds: Optional[int] = None,
        limit_price: Optional[float] = None
    ) -> ExecutionPlan:
        """Create Time-Weighted Average Price execution plan."""
        duration = duration_seconds or self.default_duration
        start = time.time()
        end = start + duration

        # Calculate number of slices
        num_slices = min(
            max(quantity // self.min_slice_size, 1),
            self.max_slices
        )

        slice_size = quantity // num_slices
        remainder = quantity % num_slices

        # Create evenly spaced child orders
        child_orders = []
        interval = duration / num_slices

        for i in range(num_slices):
            qty = slice_size + (1 if i < remainder else 0)
            scheduled = start + (i + 0.5) * interval  # Middle of each interval

            child_orders.append(ChildOrder(
                parent_id=parent_id,
                child_id=f"{parent_id}_TWAP_{i}",
                symbol=symbol,
                side=side,
                quantity=qty,
                price_limit=limit_price,
                scheduled_time=scheduled
            ))

        # Estimate execution cost (simplified)
        # TWAP typically adds 5-10 bps for liquid stocks
        estimated_cost = 5 + (quantity / 10000) * 2

        return ExecutionPlan(
            parent_id=parent_id,
            symbol=symbol,
            side=side,
            total_quantity=quantity,
            algo_type=AlgoType.TWAP,
            start_time=start,
            end_time=end,
            child_orders=child_orders,
            estimated_cost_bps=estimated_cost,
            urgency="medium"
        )

    def create_vwap_plan(
        self,
        parent_id: str,
        symbol: str,
        side: str,
        quantity: int,
        duration_seconds: Optional[int] = None,
        limit_price: Optional[float] = None,
        volume_forecast: Optional[Dict[int, float]] = None
    ) -> ExecutionPlan:
        """Create Volume-Weighted Average Price execution plan."""
        duration = duration_seconds or self.default_duration
        start = time.time()
        end = start + duration

        # Use provided volume forecast or default profile
        profile = volume_forecast or self.volume_profile

        # Calculate slices weighted by volume
        child_orders = []
        current_hour = datetime.now().hour

        # Normalize profile for remaining hours
        remaining_hours = [h for h in profile.keys() if h >= current_hour]
        if not remaining_hours:
            remaining_hours = list(profile.keys())

        total_vol = sum(profile[h] for h in remaining_hours)

        slice_idx = 0
        for hour in remaining_hours[:self.max_slices]:
            weight = profile[hour] / total_vol
            slice_qty = int(quantity * weight)

            if slice_qty >= self.min_slice_size:
                # Schedule in middle of hour
                minutes_from_now = (hour - current_hour) * 60 + 30
                scheduled = start + max(0, minutes_from_now * 60)

                child_orders.append(ChildOrder(
                    parent_id=parent_id,
                    child_id=f"{parent_id}_VWAP_{slice_idx}",
                    symbol=symbol,
                    side=side,
                    quantity=slice_qty,
                    price_limit=limit_price,
                    scheduled_time=scheduled
                ))
                slice_idx += 1

        # Handle any remainder
        filled_qty = sum(o.quantity for o in child_orders)
        if filled_qty < quantity:
            remainder = quantity - filled_qty
            if child_orders:
                child_orders[-1].quantity += remainder
            else:
                child_orders.append(ChildOrder(
                    parent_id=parent_id,
                    child_id=f"{parent_id}_VWAP_remainder",
                    symbol=symbol,
                    side=side,
                    quantity=remainder,
                    price_limit=limit_price,
                    scheduled_time=start
                ))

        # VWAP typically adds 3-8 bps
        estimated_cost = 4 + (quantity / 10000) * 1.5

        return ExecutionPlan(
            parent_id=parent_id,
            symbol=symbol,
            side=side,
            total_quantity=quantity,
            algo_type=AlgoType.VWAP,
            start_time=start,
            end_time=end,
            child_orders=child_orders,
            estimated_cost_bps=estimated_cost,
            urgency="low"
        )

    def create_pov_plan(
        self,
        parent_id: str,
        symbol: str,
        side: str,
        quantity: int,
        participation_rate: float = 0.10,  # 10% of volume
        max_duration_seconds: int = 3600,
        limit_price: Optional[float] = None
    ) -> ExecutionPlan:
        """Create Percent of Volume execution plan."""
        start = time.time()

        # Estimate time to complete based on ADV
        # Assuming 1M shares ADV, 6.5 hour trading day
        estimated_adv = 1_000_000  # Would come from real data
        shares_per_second = estimated_adv / (6.5 * 3600)
        our_rate = shares_per_second * participation_rate

        estimated_duration = min(
            quantity / max(our_rate, 1),
            max_duration_seconds
        )

        end = start + estimated_duration

        # Create adaptive slices
        num_slices = min(int(estimated_duration / 60), self.max_slices)
        num_slices = max(num_slices, 1)

        child_orders = []
        slice_size = quantity // num_slices

        for i in range(num_slices):
            scheduled = start + i * (estimated_duration / num_slices)
            qty = slice_size + (quantity % num_slices if i == num_slices - 1 else 0)

            child_orders.append(ChildOrder(
                parent_id=parent_id,
                child_id=f"{parent_id}_POV_{i}",
                symbol=symbol,
                side=side,
                quantity=int(qty),
                price_limit=limit_price,
                scheduled_time=scheduled
            ))

        # POV typically adds 2-5 bps for passive participation
        estimated_cost = 3 + participation_rate * 20

        return ExecutionPlan(
            parent_id=parent_id,
            symbol=symbol,
            side=side,
            total_quantity=quantity,
            algo_type=AlgoType.POV,
            start_time=start,
            end_time=end,
            child_orders=child_orders,
            estimated_cost_bps=estimated_cost,
            urgency="low"
        )

    def create_iceberg_plan(
        self,
        parent_id: str,
        symbol: str,
        side: str,
        quantity: int,
        display_size: int = 100,
        limit_price: Optional[float] = None
    ) -> ExecutionPlan:
        """Create Iceberg order plan (hide large orders)."""
        start = time.time()

        # Calculate number of waves
        num_waves = max(quantity // display_size, 1)

        child_orders = []
        remaining = quantity
        wave = 0

        while remaining > 0:
            qty = min(display_size, remaining)

            child_orders.append(ChildOrder(
                parent_id=parent_id,
                child_id=f"{parent_id}_ICE_{wave}",
                symbol=symbol,
                side=side,
                quantity=qty,
                price_limit=limit_price,
                scheduled_time=start + wave * 0.5  # 500ms between waves
            ))

            remaining -= qty
            wave += 1

            if wave >= 100:  # Safety limit
                break

        return ExecutionPlan(
            parent_id=parent_id,
            symbol=symbol,
            side=side,
            total_quantity=quantity,
            algo_type=AlgoType.ICEBERG,
            start_time=start,
            end_time=start + wave * 0.5,
            child_orders=child_orders,
            estimated_cost_bps=8,
            urgency="medium"
        )

    def select_best_algo(
        self,
        symbol: str,
        side: str,
        quantity: int,
        urgency: str = "medium",
        market_conditions: Optional[Dict] = None
    ) -> AlgoType:
        """Intelligently select best algorithm based on conditions."""
        conditions = market_conditions or {}

        adv = conditions.get("adv", 1_000_000)
        spread_bps = conditions.get("spread_bps", 5)
        volatility = conditions.get("volatility", 0.02)

        # Participation rate
        participation = quantity / adv if adv > 0 else 1.0

        # Decision logic
        if urgency == "high" or participation < 0.01:
            return AlgoType.MARKET

        if participation > 0.10:
            # Large order - use POV to minimize impact
            return AlgoType.POV

        if spread_bps > 10 or volatility > 0.03:
            # Wide spread or high vol - use TWAP
            return AlgoType.TWAP

        if participation > 0.02:
            # Medium order - use iceberg
            return AlgoType.ICEBERG

        # Default to VWAP
        return AlgoType.VWAP

    def create_execution_plan(
        self,
        parent_id: str,
        symbol: str,
        side: str,
        quantity: int,
        algo_type: Optional[AlgoType] = None,
        urgency: str = "medium",
        market_conditions: Optional[Dict] = None,
        **kwargs
    ) -> ExecutionPlan:
        """Create execution plan with automatic algo selection."""
        if algo_type is None:
            algo_type = self.select_best_algo(
                symbol, side, quantity, urgency, market_conditions
            )

        if algo_type == AlgoType.TWAP:
            return self.create_twap_plan(parent_id, symbol, side, quantity, **kwargs)
        elif algo_type == AlgoType.VWAP:
            return self.create_vwap_plan(parent_id, symbol, side, quantity, **kwargs)
        elif algo_type == AlgoType.POV:
            return self.create_pov_plan(parent_id, symbol, side, quantity, **kwargs)
        elif algo_type == AlgoType.ICEBERG:
            return self.create_iceberg_plan(parent_id, symbol, side, quantity, **kwargs)
        else:
            # Market order - single child
            return ExecutionPlan(
                parent_id=parent_id,
                symbol=symbol,
                side=side,
                total_quantity=quantity,
                algo_type=AlgoType.MARKET,
                start_time=time.time(),
                end_time=time.time(),
                child_orders=[ChildOrder(
                    parent_id=parent_id,
                    child_id=f"{parent_id}_MKT_0",
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    price_limit=None,
                    scheduled_time=time.time()
                )],
                estimated_cost_bps=10,
                urgency="high"
            )


# Global singleton
_algos: Optional[SmartExecutionAlgorithms] = None


def get_smart_execution() -> SmartExecutionAlgorithms:
    """Get or create global smart execution algorithms."""
    global _algos
    if _algos is None:
        _algos = SmartExecutionAlgorithms()
    return _algos
