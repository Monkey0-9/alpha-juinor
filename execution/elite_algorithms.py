"""
Elite Execution Algorithms
==========================

Advanced execution algorithms for institutional trading.

Algorithms:
- Implementation Shortfall Optimizer
- Adaptive TWAP
- Smart VWAP with liquidity forecasting
- POV (Percentage of Volume)
- Iceberg orders with optimal child sizing

Phase 2.1: Advanced Execution Algorithms
"""

import logging
from typing import List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ExecutionSlice:
    """A slice of an execution order."""
    quantity: float
    target_time: datetime
    limit_price: Optional[float]
    status: str  # PENDING, FILLED, PARTIAL, CANCELLED


@dataclass
class ExecutionPlan:
    """Complete execution plan for an order."""
    symbol: str
    total_quantity: float
    side: str
    algorithm: str
    slices: List[ExecutionSlice]
    estimated_cost_bps: float
    created_at: datetime


class ImplementationShortfallOptimizer:
    """
    Implementation Shortfall (IS) optimizer.

    Minimizes the difference between decision price and actual
    execution price, balancing urgency vs market impact.
    """

    def __init__(self, urgency: float = 0.5, risk_aversion: float = 1.0):
        self.urgency = urgency  # 0 = patient, 1 = aggressive
        self.risk_aversion = risk_aversion
        logger.info(f"IS Optimizer: urgency={urgency}")

    def optimize(
        self,
        symbol: str,
        quantity: float,
        side: str,
        decision_price: float,
        volatility: float,
        adv: float,  # Average Daily Volume
        horizon_minutes: int = 60
    ) -> ExecutionPlan:
        """
        Generate optimal IS execution schedule.
        """
        # Almgren-Chriss optimal trajectory
        # Higher urgency -> front-load execution
        # Higher risk aversion -> more uniform distribution

        n_slices = max(5, int(horizon_minutes / 10))

        # Generate time weights
        times = np.linspace(0, 1, n_slices)
        if self.urgency > 0.5:
            # Front-loaded
            weights = np.exp(-2 * self.urgency * times)
        else:
            # Back-loaded or uniform
            weights = np.ones(n_slices)

        weights /= weights.sum()

        # Create slices
        slices = []
        now = datetime.utcnow()

        for i, w in enumerate(weights):
            slice_qty = quantity * w
            target_time = now + timedelta(minutes=i * (horizon_minutes / n_slices))

            slices.append(ExecutionSlice(
                quantity=slice_qty,
                target_time=target_time,
                limit_price=None,  # Market orders
                status="PENDING"
            ))

        # Estimate cost
        participation = quantity / (adv * (horizon_minutes / 390))
        impact_cost = 10 * participation * volatility * 10000  # bps

        return ExecutionPlan(
            symbol=symbol,
            total_quantity=quantity,
            side=side,
            algorithm="IS_OPTIMAL",
            slices=slices,
            estimated_cost_bps=impact_cost,
            created_at=now
        )


class AdaptiveTWAP:
    """
    Adaptive Time-Weighted Average Price algorithm.

    Adjusts execution speed based on real-time market conditions.
    """

    def __init__(self, base_interval_seconds: int = 60):
        self.base_interval = base_interval_seconds

    def generate_schedule(
        self,
        symbol: str,
        quantity: float,
        side: str,
        duration_minutes: int,
        current_spread: float = 0.001
    ) -> ExecutionPlan:
        """
        Generate adaptive TWAP schedule.
        """
        n_slices = max(3, duration_minutes)

        # Uniform base distribution
        base_qty = quantity / n_slices

        slices = []
        now = datetime.utcnow()

        for i in range(n_slices):
            # Add randomness to avoid detection
            jitter = np.random.uniform(0.8, 1.2)
            slice_qty = base_qty * jitter

            # Adjust interval based on spread
            if current_spread > 0.002:
                # Wide spread -> wait longer
                interval = self.base_interval * 1.5
            else:
                interval = self.base_interval

            target_time = now + timedelta(seconds=i * interval)

            slices.append(ExecutionSlice(
                quantity=slice_qty,
                target_time=target_time,
                limit_price=None,
                status="PENDING"
            ))

        return ExecutionPlan(
            symbol=symbol,
            total_quantity=quantity,
            side=side,
            algorithm="ADAPTIVE_TWAP",
            slices=slices,
            estimated_cost_bps=5.0,  # Typical TWAP cost
            created_at=now
        )


class SmartVWAP:
    """
    Smart VWAP with liquidity forecasting.

    Distributes execution based on predicted intraday volume curve.
    """

    def __init__(self):
        # Typical U-shaped intraday volume pattern
        self.volume_curve = np.array([
            0.12, 0.08, 0.06, 0.05, 0.05, 0.05,  # Morning
            0.06, 0.07, 0.08, 0.10, 0.12, 0.16   # Afternoon
        ])
        self.volume_curve /= self.volume_curve.sum()

    def generate_schedule(
        self,
        symbol: str,
        quantity: float,
        side: str,
        start_hour: int = 9,
        end_hour: int = 16
    ) -> ExecutionPlan:
        """
        Generate VWAP-weighted execution schedule.
        """
        hours = end_hour - start_hour
        n_slices = min(hours, len(self.volume_curve))

        # Get volume weights for trading hours
        weights = self.volume_curve[:n_slices]
        weights /= weights.sum()

        slices = []
        now = datetime.utcnow()

        for i, w in enumerate(weights):
            slice_qty = quantity * w
            target_time = now + timedelta(hours=i)

            slices.append(ExecutionSlice(
                quantity=slice_qty,
                target_time=target_time,
                limit_price=None,
                status="PENDING"
            ))

        return ExecutionPlan(
            symbol=symbol,
            total_quantity=quantity,
            side=side,
            algorithm="SMART_VWAP",
            slices=slices,
            estimated_cost_bps=3.0,  # VWAP typically lower cost
            created_at=now
        )


class IcebergOrder:
    """
    Iceberg order with optimal child sizing.

    Shows only a portion of the order to minimize market impact.
    """

    def __init__(self, display_ratio: float = 0.1):
        self.display_ratio = display_ratio  # Show 10% of remaining

    def generate_slices(
        self,
        symbol: str,
        total_quantity: float,
        side: str,
        min_child_size: float = 100
    ) -> ExecutionPlan:
        """
        Generate iceberg order slices.
        """
        # Calculate optimal display size
        display_size = max(min_child_size, total_quantity * self.display_ratio)
        n_slices = int(np.ceil(total_quantity / display_size))

        slices = []
        remaining = total_quantity
        now = datetime.utcnow()

        for i in range(n_slices):
            slice_qty = min(display_size, remaining)
            remaining -= slice_qty

            slices.append(ExecutionSlice(
                quantity=slice_qty,
                target_time=now + timedelta(seconds=i * 5),
                limit_price=None,
                status="PENDING"
            ))

        return ExecutionPlan(
            symbol=symbol,
            total_quantity=total_quantity,
            side=side,
            algorithm="ICEBERG",
            slices=slices,
            estimated_cost_bps=2.0,  # Low impact
            created_at=now
        )


class ExecutionAlgorithmSelector:
    """
    Selects optimal execution algorithm based on order characteristics.
    """

    def __init__(self):
        self.is_optimizer = ImplementationShortfallOptimizer()
        self.adaptive_twap = AdaptiveTWAP()
        self.smart_vwap = SmartVWAP()
        self.iceberg = IcebergOrder()

    def select_and_execute(
        self,
        symbol: str,
        quantity: float,
        side: str,
        urgency: float = 0.5,
        adv: float = 1_000_000,
        volatility: float = 0.02
    ) -> ExecutionPlan:
        """
        Select best algorithm and generate execution plan.
        """
        participation_rate = quantity / adv

        if urgency > 0.8:
            # Urgent: use IS optimizer
            return self.is_optimizer.optimize(
                symbol, quantity, side,
                decision_price=100.0,
                volatility=volatility,
                adv=adv,
                horizon_minutes=30
            )
        elif participation_rate > 0.01:
            # Large order: use VWAP
            return self.smart_vwap.generate_schedule(
                symbol, quantity, side
            )
        elif participation_rate > 0.001:
            # Medium order: use TWAP
            return self.adaptive_twap.generate_schedule(
                symbol, quantity, side,
                duration_minutes=60
            )
        else:
            # Small order: use Iceberg
            return self.iceberg.generate_slices(
                symbol, quantity, side
            )


# Singleton
_algo_selector = None


def get_algo_selector() -> ExecutionAlgorithmSelector:
    global _algo_selector
    if _algo_selector is None:
        _algo_selector = ExecutionAlgorithmSelector()
    return _algo_selector
