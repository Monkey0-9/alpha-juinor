"""
Execution Engine - Elite Quant Fund System
Closed-form Almgren-Chriss optimal trajectory, adaptive VWAP with intraday volume profile,
Implementation Shortfall, and Smart Order Router across venues with realistic fill probability
Built to Renaissance Technologies / Jane Street standards
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Callable, Any, Literal
from enum import Enum

import numpy as np
from numpy import sinh, cosh, exp, sqrt

from elite_quant_fund.core.types import (
    Order, Fill, ExecutionSchedule, OrderType, Side, Venue,
    MarketImpactEstimate, LiquidityMetrics, MarketBar, Result
)

logger = logging.getLogger(__name__)


# ============================================================================
# ALMGREN-CHRISS OPTIMAL EXECUTION
# ============================================================================

class AlmgrenChrissModel:
    """
    Almgren-Chriss optimal execution model
    
    Minimizes: E[Cost] + lambda * Var[Cost]
    
    Closed-form solution for optimal trading trajectory:
    x(t) = X * sinh(kappa * (T - t)) / sinh(kappa * T)
    
    Where:
    - X: total order size
    - T: execution horizon
    - kappa: urgency parameter = sqrt(lambda * sigma^2 / eta)
    - lambda: risk aversion
    - sigma: volatility
    - eta: temporary impact coefficient
    """
    
    def __init__(
        self,
        permanent_impact_coeff: float = 0.1,  # gamma
        temporary_impact_coeff: float = 0.01,  # eta
        decay_rate: float = 0.5  # rho: decay of temporary impact
    ):
        self.gamma = permanent_impact_coeff
        self.eta = temporary_impact_coeff
        self.rho = decay_rate
    
    def calculate_trajectory(
        self,
        order: Order,
        volatility: float,
        risk_aversion: float = 1e-6,
        num_slices: int = 10
    ) -> ExecutionSchedule:
        """
        Calculate optimal Almgren-Chriss execution trajectory
        
        Args:
            order: Order to execute
            volatility: Annualized volatility of the asset
            risk_aversion: Lambda parameter (risk aversion)
            num_slices: Number of execution slices
        
        Returns:
            ExecutionSchedule with optimal trajectory
        """
        
        X = order.quantity
        side_multiplier = 1 if order.side == Side.BUY else -1
        
        # Execution horizon (default to rest of day)
        now = datetime.now()
        end_of_day = now.replace(hour=16, minute=0, second=0)
        if end_of_day < now:
            end_of_day += timedelta(days=1)
        
        T_hours = (end_of_day - now).total_seconds() / 3600
        T = max(T_hours, 1.0)  # At least 1 hour
        
        # Convert volatility to hourly
        sigma_hourly = volatility / sqrt(252 * 24)
        
        # Calculate kappa (urgency parameter)
        # kappa = sqrt(risk_aversion * sigma^2 / eta)
        if self.eta > 0 and sigma_hourly > 0:
            kappa = sqrt(risk_aversion * sigma_hourly ** 2 / self.eta)
        else:
            kappa = 0.01  # Default low urgency
        
        # Generate execution schedule
        schedule = []
        total_quantity = 0
        
        for i in range(num_slices):
            t = i * T / num_slices  # Current time
            dt = T / num_slices  # Time step
            
            # Optimal position remaining: x(t) = X * sinh(kappa * (T-t)) / sinh(kappa * T)
            if kappa * T > 1e-10:
                remaining_frac = sinh(kappa * (T - t)) / sinh(kappa * T)
            else:
                # Linear for small kappa
                remaining_frac = (T - t) / T
            
            # Position at next step
            t_next = (i + 1) * T / num_slices
            if kappa * T > 1e-10:
                remaining_frac_next = sinh(kappa * (T - t_next)) / sinh(kappa * T)
            else:
                remaining_frac_next = (T - t_next) / T
            
            # Trade size to move from remaining_frac to remaining_frac_next
            trade_size = int(X * (remaining_frac - remaining_frac_next))
            trade_size = max(1, trade_size)  # At least 1 share
            
            exec_time = now + timedelta(hours=t)
            schedule.append((exec_time, trade_size))
            total_quantity += trade_size
        
        # Adjust last slice to match total quantity
        if schedule and total_quantity != X:
            last_time, last_size = schedule[-1]
            adjustment = X - (total_quantity - last_size)
            schedule[-1] = (last_time, max(1, adjustment))
        
        # Calculate expected impact
        expected_impact = self._estimate_impact(X, T, volatility, risk_aversion)
        
        return ExecutionSchedule(
            order_id=order.id,
            symbol=order.symbol,
            total_quantity=X,
            side=order.side,
            start_time=schedule[0][0] if schedule else now,
            end_time=schedule[-1][0] if schedule else end_of_day,
            schedule=schedule,
            expected_impact_bps=expected_impact,
            expected_variance=risk_aversion * volatility ** 2 * X ** 2 * T / 2
        )
    
    def _estimate_impact(
        self,
        quantity: int,
        horizon: float,
        volatility: float,
        risk_aversion: float
    ) -> float:
        """Estimate total market impact in basis points"""
        
        # Simplified impact model
        # Permanent: gamma * X
        permanent = self.gamma * quantity
        
        # Temporary: eta * (X / T)
        temporary = self.eta * (quantity / horizon)
        
        # Convert to bps (assuming price ~ 100)
        price = 100.0
        impact_pct = (permanent + temporary) / price
        impact_bps = impact_pct * 10000
        
        return impact_bps
    
    def calculate_optimal_horizon(
        self,
        quantity: int,
        volatility: float,
        urgency: float = 0.5
    ) -> float:
        """
        Calculate optimal execution horizon given urgency
        Higher urgency = shorter horizon
        """
        # Heuristic: more volume or higher urgency = shorter horizon
        base_horizon = 8.0  # 8 hours
        
        # Scale by quantity (larger orders need more time)
        size_factor = min(1.0, quantity / 10000)
        
        # Urgency factor
        urgency_factor = 1.0 - urgency  # Higher urgency = smaller factor
        
        optimal_horizon = base_horizon * (0.5 + 0.5 * size_factor) * urgency_factor
        
        return max(0.5, min(8.0, optimal_horizon))  # Between 30 min and 8 hours


# ============================================================================
# VWAP EXECUTION
# ============================================================================

class VWAPModel:
    """
    Volume-Weighted Average Price execution with intraday volume profile
    """
    
    def __init__(self):
        # Intraday volume profile (typical shape)
        # Higher volume at open, dip midday, spike at close
        self.volume_profile: List[float] = [
            0.15,  # 9:30-10:00 (high)
            0.12,  # 10:00-10:30
            0.10,  # 10:30-11:00
            0.08,  # 11:00-11:30
            0.07,  # 11:30-12:00
            0.05,  # 12:00-12:30 (midday dip)
            0.06,  # 12:30-13:00
            0.08,  # 13:00-13:30
            0.10,  # 13:30-14:00
            0.11,  # 14:00-14:30
            0.12,  # 14:30-15:00
            0.15,  # 15:00-15:30 (close spike)
            0.08,  # 15:30-16:00
        ]
        
        # Normalize to sum to 1
        total = sum(self.volume_profile)
        self.volume_profile = [v / total for v in self.volume_profile]
    
    def calculate_schedule(
        self,
        order: Order,
        expected_daily_volume: float,
        max_participation: float = 0.10
    ) -> ExecutionSchedule:
        """
        Calculate VWAP execution schedule
        """
        
        X = order.quantity
        now = datetime.now()
        
        # Determine start time
        start_time = order.vwap_start_time or now
        end_time = order.vwap_end_time or now.replace(hour=16, minute=0)
        
        if end_time < start_time:
            end_time += timedelta(days=1)
        
        duration_hours = (end_time - start_time).total_seconds() / 3600
        num_intervals = max(1, int(duration_hours * 2))  # 30-min intervals
        
        # Get volume profile for the time period
        start_idx = self._time_to_index(start_time)
        end_idx = self._time_to_index(end_time)
        
        schedule = []
        remaining_qty = X
        
        for i in range(num_intervals):
            # Map to volume profile index
            profile_idx = (start_idx + i) % len(self.volume_profile)
            interval_volume_frac = self.volume_profile[profile_idx]
            
            # Target participation rate
            participation = min(max_participation, interval_volume_frac * 2)
            
            # Trade size proportional to expected volume in interval
            expected_interval_volume = expected_daily_volume * interval_volume_frac
            trade_size = int(expected_interval_volume * participation)
            trade_size = max(0, min(trade_size, remaining_qty))
            
            if trade_size > 0:
                exec_time = start_time + timedelta(minutes=30 * i)
                schedule.append((exec_time, trade_size))
                remaining_qty -= trade_size
        
        # Handle remaining quantity
        if remaining_qty > 0 and schedule:
            last_time, last_size = schedule[-1]
            schedule[-1] = (last_time, last_size + remaining_qty)
        
        expected_impact = 5.0  # Typical VWAP impact: ~5 bps
        
        return ExecutionSchedule(
            order_id=order.id,
            symbol=order.symbol,
            total_quantity=X,
            side=order.side,
            start_time=schedule[0][0] if schedule else start_time,
            end_time=schedule[-1][0] if schedule else end_time,
            schedule=schedule,
            expected_impact_bps=expected_impact,
            expected_variance=0.0  # VWAP minimizes variance
        )
    
    def _time_to_index(self, time: datetime) -> int:
        """Convert time to volume profile index"""
        hour = time.hour
        minute = time.minute
        
        # Market hours 9:30-16:00
        if hour < 9 or (hour == 9 and minute < 30):
            return 0
        if hour >= 16:
            return len(self.volume_profile) - 1
        
        # Calculate index (30-min intervals)
        total_minutes = (hour - 9) * 60 + minute - 30
        idx = total_minutes // 30
        
        return max(0, min(idx, len(self.volume_profile) - 1))


# ============================================================================
# IMPLEMENTATION SHORTFALL
# ============================================================================

class ImplementationShortfallModel:
    """
    Implementation Shortfall (IS) algorithm
    Minimizes execution cost vs arrival price
    """
    
    def __init__(self):
        self.arrival_price: Optional[float] = None
        self.arrival_time: Optional[datetime] = None
    
    def set_arrival_price(self, price: float, time: datetime) -> None:
        """Set the arrival price (decision time price)"""
        self.arrival_price = price
        self.arrival_time = time
    
    def calculate_schedule(
        self,
        order: Order,
        urgency: float = 0.5,
        market_volatility: float = 0.20
    ) -> ExecutionSchedule:
        """
        Calculate IS execution schedule
        """
        
        X = order.quantity
        now = datetime.now()
        
        # IS trades faster for higher urgency
        base_slices = 5
        num_slices = int(base_slices + urgency * 10)  # 5-15 slices
        
        # Time compression based on urgency
        base_horizon = 4.0  # 4 hours
        horizon = base_horizon * (1.0 - urgency * 0.5)  # 2-4 hours
        
        schedule = []
        
        # Front-loaded execution (more urgent)
        remaining_qty = X
        for i in range(num_slices):
            # Exponential decay of trade size
            frac = exp(-urgency * i)
            trade_size = int(X * frac / sum(exp(-urgency * j) for j in range(num_slices)))
            trade_size = max(1, min(trade_size, remaining_qty))
            
            exec_time = now + timedelta(hours=horizon * i / num_slices)
            schedule.append((exec_time, trade_size))
            remaining_qty -= trade_size
        
        # Handle remainder
        if remaining_qty > 0 and schedule:
            last_time, last_size = schedule[-1]
            schedule[-1] = (last_time, last_size + remaining_qty)
        
        # Expected impact increases with urgency
        expected_impact = 3.0 + urgency * 7.0  # 3-10 bps
        
        return ExecutionSchedule(
            order_id=order.id,
            symbol=order.symbol,
            total_quantity=X,
            side=order.side,
            start_time=schedule[0][0] if schedule else now,
            end_time=schedule[-1][0] if schedule else now + timedelta(hours=horizon),
            schedule=schedule,
            expected_impact_bps=expected_impact,
            expected_variance=market_volatility ** 2 * X ** 2 * horizon / 2
        )


# ============================================================================
# SMART ORDER ROUTER
# ============================================================================

class VenueCharacteristics:
    """Characteristics for each trading venue"""
    
    def __init__(
        self,
        venue: Venue,
        maker_fee_bps: float = -0.2,  # Negative = rebate
        taker_fee_bps: float = 3.0,
        fill_probability: float = 0.8,
        avg_latency_ms: float = 5.0,
        price_improvement_pct: float = 0.0
    ):
        self.venue = venue
        self.maker_fee_bps = maker_fee_bps
        self.taker_fee_bps = taker_fee_bps
        self.fill_probability = fill_probability
        self.avg_latency_ms = avg_latency_ms
        self.price_improvement_pct = price_improvement_pct
    
    def total_cost_bps(self, is_maker: bool = False) -> float:
        """Calculate total cost including fees"""
        fee = self.maker_fee_bps if is_maker else self.taker_fee_bps
        # Latency cost (opportunity cost)
        latency_cost = self.avg_latency_ms * 0.001  # 1ms ~ 0.001 bps
        return fee + latency_cost


class SmartOrderRouter:
    """
    Smart Order Router (SOR) that optimizes venue selection
    Considers: fees, fill probability, latency, price improvement
    """
    
    def __init__(self):
        # Venue characteristics
        self.venues: Dict[Venue, VenueCharacteristics] = {
            Venue.NYSE: VenueCharacteristics(
                venue=Venue.NYSE,
                maker_fee_bps=-0.2,
                taker_fee_bps=3.0,
                fill_probability=0.95,
                avg_latency_ms=3.0
            ),
            Venue.NASDAQ: VenueCharacteristics(
                venue=Venue.NASDAQ,
                maker_fee_bps=-0.2,
                taker_fee_bps=3.0,
                fill_probability=0.95,
                avg_latency_ms=2.5
            ),
            Venue.IEX: VenueCharacteristics(
                venue=Venue.IEX,
                maker_fee_bps=-0.3,
                taker_fee_bps=3.0,
                fill_probability=0.90,
                avg_latency_ms=350.0,  # Higher latency but better for investors
                price_improvement_pct=0.01
            ),
            Venue.BATS: VenueCharacteristics(
                venue=Venue.BATS,
                maker_fee_bps=-0.25,
                taker_fee_bps=2.5,
                fill_probability=0.88,
                avg_latency_ms=2.0
            ),
            Venue.DARK_POOL_SIGMA: VenueCharacteristics(
                venue=Venue.DARK_POOL_SIGMA,
                maker_fee_bps=1.0,  # Higher fees for dark pool
                taker_fee_bps=1.0,
                fill_probability=0.70,  # Lower fill probability
                avg_latency_ms=5.0,
                price_improvement_pct=0.05  # Better for large orders
            ),
        }
        
        # Order splitting configuration
        self.max_venues = 3
        self.min_slice_size = 100
    
    def route_order(
        self,
        order: Order,
        liquidity: LiquidityMetrics
    ) -> List[Tuple[Venue, int]]:
        """
        Determine optimal venue routing for an order
        
        Returns:
            List of (venue, quantity) tuples
        """
        
        total_qty = order.quantity
        
        # Score venues
        venue_scores = []
        for venue, chars in self.venues.items():
            score = self._score_venue(chars, order, liquidity)
            venue_scores.append((venue, score))
        
        # Sort by score
        venue_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select top venues
        selected_venues = venue_scores[:self.max_venues]
        
        # Split order across venues
        # Larger allocation to better venues, but diversify
        total_score = sum(score for _, score in selected_venues)
        
        if total_score == 0:
            # Equal split if no preference
            qty_per_venue = total_qty // len(selected_venues)
            return [(venue, qty_per_venue) for venue, _ in selected_venues]
        
        allocations = []
        remaining = total_qty
        
        for i, (venue, score) in enumerate(selected_venues[:-1]):
            # Proportional allocation
            frac = score / total_score
            qty = int(total_qty * frac)
            qty = max(self.min_slice_size, min(qty, remaining))
            
            allocations.append((venue, qty))
            remaining -= qty
        
        # Last venue gets remainder
        if selected_venues:
            allocations.append((selected_venues[-1][0], remaining))
        
        return allocations
    
    def _score_venue(
        self,
        chars: VenueCharacteristics,
        order: Order,
        liquidity: LiquidityMetrics
    ) -> float:
        """Score a venue for this order"""
        
        # Factors
        # 1. Cost (negative = good)
        cost_score = -chars.total_cost_bps(is_maker=(order.order_type == OrderType.LIMIT))
        
        # 2. Fill probability
        fill_score = chars.fill_probability * 10  # 0-10
        
        # 3. Speed (lower latency = better)
        speed_score = -chars.avg_latency_ms / 10  # Negative penalty
        
        # 4. For large orders, prefer dark pools
        size_score = 0
        if order.quantity * 100 > 100000:  # > $100K
            if 'DARK' in chars.venue.name:
                size_score = 5  # Bonus for dark pools on large orders
        
        # 5. For illiquid stocks, prefer lit markets
        liquidity_score = 0
        if not liquidity.is_liquid:
            if 'DARK' not in chars.venue.name:
                liquidity_score = 3  # Bonus for lit markets
        
        total_score = cost_score + fill_score + speed_score + size_score + liquidity_score
        
        return total_score
    
    def estimate_fill_probability(
        self,
        venue: Venue,
        order: Order,
        liquidity: LiquidityMetrics
    ) -> float:
        """Estimate probability of complete fill"""
        
        chars = self.venues.get(venue)
        if not chars:
            return 0.5
        
        base_prob = chars.fill_probability
        
        # Adjust for order size relative to liquidity
        order_dollar = order.quantity * 100  # Assume $100 price
        liquidity_ratio = order_dollar / (liquidity.adv_20_day + 1)
        
        # Smaller orders more likely to fill
        size_adjustment = max(0.3, 1.0 - liquidity_ratio)
        
        # Limit orders less certain than market
        type_adjustment = 1.0 if order.order_type == OrderType.MARKET else 0.8
        
        return base_prob * size_adjustment * type_adjustment


# ============================================================================
# EXECUTION ORCHESTRATOR
# ============================================================================

class ExecutionEngine:
    """
    Main execution engine orchestrating all algorithms
    """
    
    def __init__(self):
        # Algorithm models
        self.almgren_chriss = AlmgrenChrissModel()
        self.vwap = VWAPModel()
        self.is_model = ImplementationShortfallModel()
        self.sor = SmartOrderRouter()
        
        # Active schedules
        self.active_schedules: Dict[str, ExecutionSchedule] = {}
        
        # Callbacks
        self.slice_callbacks: List[Callable[[str, int, Venue], None]] = []
        
        # Stats
        self.orders_executed = 0
        self.slices_executed = 0
        self.total_impact_bps = 0.0
    
    def register_slice_callback(
        self,
        callback: Callable[[str, int, Venue], None]
    ) -> None:
        """Register callback for execution slices"""
        self.slice_callbacks.append(callback)
    
    def create_schedule(
        self,
        order: Order,
        volatility: float = 0.20,
        expected_volume: float = 1000000,
        liquidity: Optional[LiquidityMetrics] = None
    ) -> ExecutionSchedule:
        """
        Create execution schedule based on order type
        """
        
        if order.order_type == OrderType.ALMGREN_CHRISS:
            schedule = self.almgren_chriss.calculate_trajectory(
                order,
                volatility,
                risk_aversion=1e-6,
                num_slices=10
            )
        
        elif order.order_type == OrderType.VWAP:
            schedule = self.vwap.calculate_schedule(
                order,
                expected_volume,
                max_participation=order.max_participation_pct
            )
        
        elif order.order_type == OrderType.IMPLEMENTATION_SHORTFALL:
            schedule = self.is_model.calculate_schedule(
                order,
                urgency=order.urgency,
                market_volatility=volatility
            )
        
        else:
            # Default to Almgren-Chriss for other types
            schedule = self.almgren_chriss.calculate_trajectory(
                order,
                volatility,
                num_slices=5
            )
        
        self.active_schedules[order.id] = schedule
        
        return schedule
    
    def get_next_slice(
        self,
        order_id: str,
        current_time: datetime
    ) -> Optional[Tuple[int, List[Tuple[Venue, int]]]]:
        """
        Get next execution slice for an order
        
        Returns:
            (total_quantity, [(venue, quantity), ...]) or None if complete
        """
        
        schedule = self.active_schedules.get(order_id)
        if not schedule:
            return None
        
        # Find next slice in schedule
        remaining_slices = [
            (time, qty) for time, qty in schedule.schedule
            if time >= current_time
        ]
        
        if not remaining_slices:
            # Order complete
            del self.active_schedules[order_id]
            return None
        
        # Get next slice
        next_time, next_qty = remaining_slices[0]
        
        # Route through SOR
        # Create dummy order for routing
        dummy_order = Order(
            symbol=schedule.symbol,
            side=schedule.side,
            quantity=next_qty,
            order_type=OrderType.MARKET
        )
        
        # Get venue allocations
        venue_allocations = self.sor.route_order(
            dummy_order,
            LiquidityMetrics(
                symbol=schedule.symbol,
                timestamp=current_time,
                bid_ask_spread_bps=5.0,
                amihud_ratio=0.001,
                kyle_lambda=0.001,
                market_depth_dollars=1000000,
                adv_20_day=10000000
            )
        )
        
        return next_qty, venue_allocations
    
    def record_execution(
        self,
        order_id: str,
        fill: Fill,
        expected_impact_bps: float
    ) -> None:
        """Record execution for analytics"""
        
        self.orders_executed += 1
        self.slices_executed += 1
        self.total_impact_bps += expected_impact_bps
        
        # Notify callbacks
        for callback in self.slice_callbacks:
            try:
                callback(order_id, fill.quantity, fill.venue)
            except Exception as e:
                logger.error(f"Slice callback error: {e}")
    
    def get_average_impact(self) -> float:
        """Get average market impact"""
        if self.slices_executed == 0:
            return 0.0
        return self.total_impact_bps / self.slices_executed
    
    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        return {
            'orders_executed': self.orders_executed,
            'slices_executed': self.slices_executed,
            'average_impact_bps': self.get_average_impact(),
            'active_schedules': len(self.active_schedules)
        }


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'AlmgrenChrissModel',
    'VWAPModel',
    'ImplementationShortfallModel',
    'SmartOrderRouter',
    'VenueCharacteristics',
    'ExecutionEngine',
]
