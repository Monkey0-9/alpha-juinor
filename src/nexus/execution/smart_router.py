"""
Smart Order Router (SOR) - Intelligent Venue Selection.

Features:
- Multi-venue routing
- Dark pool access
- Latency arbitrage protection
- Parent/child order management
"""

import logging
import time
import threading
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import random

logger = logging.getLogger(__name__)


class VenueType(Enum):
    LIT = "LIT"  # Public exchange
    DARK = "DARK"  # Dark pool
    MIDPOINT = "MIDPOINT"  # Midpoint matching
    RETAIL = "RETAIL"  # Retail flow


@dataclass
class Venue:
    """Trading venue definition."""
    venue_id: str
    venue_type: VenueType
    avg_latency_ms: float
    fill_rate: float
    rebate_bps: float
    fee_bps: float
    min_size: int
    max_size: int


@dataclass
class ChildOrder:
    """Child order for multi-venue routing."""
    parent_id: str
    child_id: str
    venue_id: str
    symbol: str
    side: str
    quantity: int
    limit_price: Optional[float]
    status: str = "PENDING"
    filled_qty: int = 0
    avg_price: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class RoutingDecision:
    """SOR routing decision."""
    parent_order_id: str
    child_orders: List[ChildOrder]
    total_quantity: int
    routing_strategy: str
    estimated_cost_bps: float


class SmartOrderRouter:
    """
    Smart Order Router for multi-venue execution.

    Strategies:
    - SEEK_LIQUIDITY: Route to venues with best fill rates
    - MINIMIZE_COST: Route to lowest cost venues
    - MINIMIZE_IMPACT: Use dark pools and slice orders
    - SPEED: Route to fastest venues
    """

    def __init__(self):
        self.venues: Dict[str, Venue] = {}
        self.pending_orders: Dict[str, ChildOrder] = {}
        self.executor = ThreadPoolExecutor(max_workers=10)

        self._setup_default_venues()

    def _setup_default_venues(self):
        """Setup default venue configurations."""
        default_venues = [
            Venue("NYSE", VenueType.LIT, 0.5, 0.85, 0.2, 0.3, 100, 100000),
            Venue("NASDAQ", VenueType.LIT, 0.3, 0.88, 0.25, 0.28, 100, 100000),
            Venue("ARCA", VenueType.LIT, 0.4, 0.82, 0.20, 0.32, 100, 50000),
            Venue("IEX", VenueType.LIT, 1.5, 0.75, 0.0, 0.09, 100, 50000),  # Speed bump
            Venue("DARK1", VenueType.DARK, 0.8, 0.40, 0.0, 0.10, 100, 500000),
            Venue("DARK2", VenueType.DARK, 0.6, 0.35, 0.0, 0.12, 100, 500000),
            Venue("MIDPOINT", VenueType.MIDPOINT, 1.0, 0.30, 0.0, 0.05, 100, 100000),
        ]

        for venue in default_venues:
            self.venues[venue.venue_id] = venue

    def route_order(
        self,
        parent_id: str,
        symbol: str,
        side: str,
        quantity: int,
        limit_price: Optional[float] = None,
        strategy: str = "SEEK_LIQUIDITY",
        urgency: str = "MEDIUM"
    ) -> RoutingDecision:
        """
        Route order across venues.
        """
        child_orders = []
        remaining = quantity

        if strategy == "SEEK_LIQUIDITY":
            venues = self._rank_by_fill_rate()
        elif strategy == "MINIMIZE_COST":
            venues = self._rank_by_cost()
        elif strategy == "MINIMIZE_IMPACT":
            venues = self._get_dark_venues()
        elif strategy == "SPEED":
            venues = self._rank_by_latency()
        else:
            venues = list(self.venues.values())

        # Allocate to venues
        for i, venue in enumerate(venues):
            if remaining <= 0:
                break

            # Calculate allocation for this venue
            if strategy == "MINIMIZE_IMPACT":
                # Spread evenly across dark pools
                alloc = min(remaining, quantity // len(venues) + 1)
            else:
                # Prioritize top venues
                alloc = min(
                    remaining,
                    int(remaining * (0.5 ** i) + venue.min_size)
                )

            alloc = min(alloc, venue.max_size)
            alloc = max(alloc, venue.min_size)

            if alloc > 0:
                child = ChildOrder(
                    parent_id=parent_id,
                    child_id=f"{parent_id}_{venue.venue_id}_{i}",
                    venue_id=venue.venue_id,
                    symbol=symbol,
                    side=side,
                    quantity=alloc,
                    limit_price=limit_price
                )
                child_orders.append(child)
                self.pending_orders[child.child_id] = child
                remaining -= alloc

        # Estimate cost
        estimated_cost = self._estimate_routing_cost(child_orders)

        return RoutingDecision(
            parent_order_id=parent_id,
            child_orders=child_orders,
            total_quantity=quantity,
            routing_strategy=strategy,
            estimated_cost_bps=estimated_cost
        )

    def _rank_by_fill_rate(self) -> List[Venue]:
        """Rank venues by fill rate."""
        return sorted(
            self.venues.values(),
            key=lambda v: v.fill_rate,
            reverse=True
        )

    def _rank_by_cost(self) -> List[Venue]:
        """Rank venues by net cost (fee - rebate)."""
        return sorted(
            self.venues.values(),
            key=lambda v: v.fee_bps - v.rebate_bps
        )

    def _rank_by_latency(self) -> List[Venue]:
        """Rank venues by latency."""
        return sorted(
            self.venues.values(),
            key=lambda v: v.avg_latency_ms
        )

    def _get_dark_venues(self) -> List[Venue]:
        """Get dark pool venues."""
        dark = [
            v for v in self.venues.values()
            if v.venue_type in [VenueType.DARK, VenueType.MIDPOINT]
        ]
        return sorted(dark, key=lambda v: v.fill_rate, reverse=True)

    def _estimate_routing_cost(self, orders: List[ChildOrder]) -> float:
        """Estimate total routing cost in bps."""
        if not orders:
            return 0.0

        total_cost = 0.0
        total_qty = sum(o.quantity for o in orders)

        for order in orders:
            venue = self.venues.get(order.venue_id)
            if venue:
                cost = (venue.fee_bps - venue.rebate_bps) * order.quantity
                total_cost += cost

        return total_cost / total_qty if total_qty > 0 else 0.0

    def execute_child_orders(self, decision: RoutingDecision) -> List[ChildOrder]:
        """Execute all child orders."""
        def execute_single(order: ChildOrder) -> ChildOrder:
            venue = self.venues.get(order.venue_id)

            # Simulate execution
            time.sleep(venue.avg_latency_ms / 1000 if venue else 0.001)

            # Simulate fill based on venue fill rate
            fill_rate = venue.fill_rate if venue else 0.8
            if random.random() < fill_rate:
                order.status = "FILLED"
                order.filled_qty = order.quantity
                order.avg_price = order.limit_price or 100.0
            else:
                order.status = "PARTIAL"
                order.filled_qty = int(order.quantity * random.random())
                order.avg_price = order.limit_price or 100.0

            return order

        futures = [
            self.executor.submit(execute_single, order)
            for order in decision.child_orders
        ]

        results = [f.result() for f in futures]
        return results


# Global singleton
_sor: Optional[SmartOrderRouter] = None


def get_smart_router() -> SmartOrderRouter:
    """Get or create global smart order router."""
    global _sor
    if _sor is None:
        _sor = SmartOrderRouter()
    return _sor
