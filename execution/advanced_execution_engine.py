"""
execution/advanced_execution_engine.py

AdvancedExecutionEngine for professional TWAP, VWAP, and multi-slice order management.
Implements institutional-grade execution algorithms with real-time monitoring.
"""

import asyncio
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import uuid

from monitoring.structured_logger import get_logger
from execution.global_session_tracker import session_tracker

logger = get_logger("execution_engine")


class ExecutionAlgorithm(Enum):
    """Execution algorithm types."""
    IMMEDIATE = "IMMEDIATE"
    TWAP = "TWAP"  # Time-Weighted Average Price
    VWAP = "VWAP"  # Volume-Weighted Average Price
    POV = "POV"  # Percentage of Volume
    IMPLEMENTATION_SHORTFALL = "IMPLEMENTATION_SHORTFALL"
    ARRIVAL_PRICE = "ARRIVAL_PRICE"
    CLOSE_PRICE = "CLOSE_PRICE"


class OrderSide(Enum):
    """Order side enumeration."""
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "PENDING"
    WORKING = "WORKING"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass
class ExecutionSlice:
    """Individual execution slice."""
    slice_id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    algorithm: ExecutionAlgorithm
    start_time: datetime
    end_time: Optional[datetime] = None
    filled_quantity: float = 0.0
    average_price: float = 0.0
    status: OrderStatus = OrderStatus.PENDING
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionOrder:
    """Main execution order with slices."""
    order_id: str
    symbol: str
    side: OrderSide
    total_quantity: float
    algorithm: ExecutionAlgorithm
    urgency: str = "NORMAL"  # LOW, NORMAL, HIGH
    max_participation_rate: float = 0.1  # For POV algorithm
    time_horizon_minutes: int = 30
    max_slice_size: Optional[float] = None
    min_slice_size: Optional[float] = None
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    created_time: datetime = field(default_factory=datetime.utcnow)
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_price: float = 0.0
    slices: List[ExecutionSlice] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MarketDataSnapshot:
    """Real-time market data snapshot."""
    symbol: str
    timestamp: datetime
    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float
    last_price: float
    last_size: float
    volume_today: float
    vwap_today: float
    adv_20d: float  # Average daily volume


class AdvancedExecutionEngine:
    """
    Advanced execution engine with institutional algorithms.
    Supports TWAP, VWAP, POV, and other sophisticated execution strategies.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logger
        self.active_orders: Dict[str, ExecutionOrder] = {}
        self.market_data: Dict[str, MarketDataSnapshot] = {}
        self.execution_history: List[Dict] = []
        
        # Execution parameters
        self.default_slice_size = self.config.get("default_slice_size", 1000)
        self.min_slice_interval = self.config.get("min_slice_interval", 5)  # seconds
        self.max_slippage_bps = self.config.get("max_slippage_bps", 10)
        self.vwap_lookback_periods = self.config.get("vwap_lookback_periods", 20)
        
        # Background tasks
        self._monitoring_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Market data callbacks
        self.market_data_callbacks: List[Callable[[MarketDataSnapshot], None]] = []
    
    async def start(self):
        """Start the execution engine."""
        if self._running:
            return
        
        self._running = True
        self._monitoring_task = asyncio.create_task(self._monitor_executions())
        self.logger.info("Advanced execution engine started")
    
    async def stop(self):
        """Stop the execution engine."""
        self._running = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Advanced execution engine stopped")
    
    async def submit_order(self, order: ExecutionOrder) -> str:
        """Submit a new order for execution."""
        order.order_id = order.order_id or str(uuid.uuid4())
        order.created_time = datetime.utcnow()
        order.status = OrderStatus.WORKING
        
        self.active_orders[order.order_id] = order
        
        # Validate order
        if not self._validate_order(order):
            order.status = OrderStatus.REJECTED
            return order.order_id
        
        # Create execution slices based on algorithm
        slices = self._create_execution_slices(order)
        order.slices = slices
        
        self.logger.info(
            f"Order submitted for execution",
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side.value,
            quantity=order.total_quantity,
            algorithm=order.algorithm.value,
            num_slices=len(slices)
        )
        
        # Start execution for slices
        for slice in slices:
            asyncio.create_task(self._execute_slice(slice))
        
        return order.order_id
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an active order."""
        if order_id not in self.active_orders:
            return False
        
        order = self.active_orders[order_id]
        
        # Cancel all pending slices
        for slice in order.slices:
            if slice.status in [OrderStatus.PENDING, OrderStatus.WORKING]:
                slice.status = OrderStatus.CANCELLED
                slice.end_time = datetime.utcnow()
        
        order.status = OrderStatus.CANCELLED
        order.end_time = datetime.utcnow()
        
        self.logger.info(
            f"Order cancelled",
            order_id=order_id,
            symbol=order.symbol,
            filled_quantity=order.filled_quantity
        )
        
        return True
    
    def _validate_order(self, order: ExecutionOrder) -> bool:
        """Validate order parameters."""
        if not order.symbol or not order.total_quantity > 0:
            return False
        
        if order.max_participation_rate <= 0 or order.max_participation_rate > 1:
            return False
        
        if order.time_horizon_minutes <= 0:
            return False
        
        return True
    
    def _create_execution_slices(self, order: ExecutionOrder) -> List[ExecutionSlice]:
        """Create execution slices based on algorithm."""
        if order.algorithm == ExecutionAlgorithm.IMMEDIATE:
            return self._create_immediate_slices(order)
        elif order.algorithm == ExecutionAlgorithm.TWAP:
            return self._create_twap_slices(order)
        elif order.algorithm == ExecutionAlgorithm.VWAP:
            return self._create_vwap_slices(order)
        elif order.algorithm == ExecutionAlgorithm.POV:
            return self._create_pov_slices(order)
        else:
            # Default to TWAP
            return self._create_twap_slices(order)
    
    def _create_immediate_slices(self, order: ExecutionOrder) -> List[ExecutionSlice]:
        """Create immediate execution slices."""
        slice = ExecutionSlice(
            slice_id=str(uuid.uuid4()),
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.total_quantity,
            algorithm=ExecutionAlgorithm.IMMEDIATE,
            start_time=datetime.utcnow()
        )
        return [slice]
    
    def _create_twap_slices(self, order: ExecutionOrder) -> List[ExecutionSlice]:
        """Create TWAP execution slices."""
        num_slices = max(1, order.time_horizon_minutes // self.min_slice_interval)
        slice_size = order.total_quantity / num_slices
        
        # Apply min/max slice size constraints
        if order.min_slice_size:
            slice_size = max(slice_size, order.min_slice_size)
            num_slices = int(np.ceil(order.total_quantity / slice_size))
        
        if order.max_slice_size:
            slice_size = min(slice_size, order.max_slice_size)
            num_slices = int(np.ceil(order.total_quantity / slice_size))
        
        slices = []
        slice_interval = order.time_horizon_minutes / num_slices * 60  # seconds
        
        for i in range(num_slices):
            start_time = datetime.utcnow() + timedelta(seconds=i * slice_interval)
            
            # Last slice gets remaining quantity
            if i == num_slices - 1:
                remaining = order.total_quantity - sum(s.quantity for s in slices)
                slice_quantity = remaining
            else:
                slice_quantity = min(slice_size, order.total_quantity - sum(s.quantity for s in slices))
            
            slice = ExecutionSlice(
                slice_id=str(uuid.uuid4()),
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                quantity=slice_quantity,
                algorithm=ExecutionAlgorithm.TWAP,
                start_time=start_time
            )
            slices.append(slice)
        
        return slices
    
    def _create_vwap_slices(self, order: ExecutionOrder) -> List[ExecutionSlice]:
        """Create VWAP execution slices based on historical volume profile."""
        # Get market data for volume profile
        market_data = self.market_data.get(order.symbol)
        if not market_data:
            # Fallback to TWAP if no market data
            return self._create_twap_slices(order)
        
        # Simplified volume profile (in production, use historical intraday volume)
        volume_profile = self._get_volume_profile(order.symbol)
        
        slices = []
        cumulative_volume = 0
        target_volume = order.total_quantity
        
        for time_bucket, volume_percent in volume_profile.items():
            if cumulative_volume >= target_volume:
                break
            
            bucket_quantity = min(
                target_volume * volume_percent,
                target_volume - cumulative_volume
            )
            
            if bucket_quantity > 0:
                slice = ExecutionSlice(
                    slice_id=str(uuid.uuid4()),
                    order_id=order.order_id,
                    symbol=order.symbol,
                    side=order.side,
                    quantity=bucket_quantity,
                    algorithm=ExecutionAlgorithm.VWAP,
                    start_time=datetime.utcnow() + timedelta(minutes=time_bucket)
                )
                slices.append(slice)
                cumulative_volume += bucket_quantity
        
        return slices
    
    def _create_pov_slices(self, order: ExecutionOrder) -> List[ExecutionSlice]:
        """Create POV (Percentage of Volume) execution slices."""
        # POV executes based on real-time volume participation
        # Create small slices that will be adjusted based on volume
        num_slices = max(1, order.time_horizon_minutes // self.min_slice_interval)
        slice_size = order.total_quantity / num_slices * 0.1  # Start with 10% participation
        
        slices = []
        for i in range(num_slices):
            start_time = datetime.utcnow() + timedelta(seconds=i * self.min_slice_interval)
            
            slice = ExecutionSlice(
                slice_id=str(uuid.uuid4()),
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                quantity=slice_size,
                algorithm=ExecutionAlgorithm.POV,
                start_time=start_time,
                metadata={"participation_rate": order.max_participation_rate}
            )
            slices.append(slice)
        
        return slices
    
    def _get_volume_profile(self, symbol: str) -> Dict[int, float]:
        """Get simplified volume profile for VWAP execution."""
        # This is a simplified version - in production, use historical intraday volume
        # Typical U-shaped volume profile: higher at open and close
        
        profile = {}
        total_minutes = 390  # 6.5 hours trading day
        
        # Create buckets (30-minute intervals)
        for i in range(0, total_minutes, 30):
            if i < 60:  # First hour - high volume
                weight = 0.15
            elif i < 240:  # Middle of day - lower volume
                weight = 0.05
            else:  # Last hour - high volume
                weight = 0.15
            
            profile[i] = weight
        
        # Normalize weights
        total_weight = sum(profile.values())
        for key in profile:
            profile[key] /= total_weight
        
        return profile
    
    async def _execute_slice(self, slice: ExecutionSlice):
        """Execute an individual slice."""
        slice.status = OrderStatus.WORKING
        
        # Wait until scheduled time
        now = datetime.utcnow()
        if slice.start_time > now:
            wait_seconds = (slice.start_time - now).total_seconds()
            await asyncio.sleep(wait_seconds)
        
        # Check if market is open
        if not session_tracker.is_market_open("NYSE"):
            self.logger.warning(
                f"Market closed, deferring slice execution",
                slice_id=slice.slice_id,
                symbol=slice.symbol
            )
            # Reschedule for next market open
            next_open = session_tracker.get_next_market_open("NYSE")
            if next_open:
                slice.start_time = next_open
                asyncio.create_task(self._execute_slice(slice))
            return
        
        # Execute based on algorithm
        try:
            if slice.algorithm == ExecutionAlgorithm.IMMEDIATE:
                await self._execute_immediate(slice)
            elif slice.algorithm == ExecutionAlgorithm.TWAP:
                await self._execute_twap(slice)
            elif slice.algorithm == ExecutionAlgorithm.VWAP:
                await self._execute_vwap(slice)
            elif slice.algorithm == ExecutionAlgorithm.POV:
                await self._execute_pov(slice)
            
        except Exception as e:
            slice.status = OrderStatus.REJECTED
            self.logger.error(
                f"Slice execution failed",
                slice_id=slice.slice_id,
                error=str(e)
            )
    
    async def _execute_immediate(self, slice: ExecutionSlice):
        """Execute immediate slice."""
        # Simulate immediate market order execution
        market_data = self.market_data.get(slice.symbol)
        if not market_data:
            raise ValueError(f"No market data for {slice.symbol}")
        
        # Use mid-price for immediate execution
        execution_price = (market_data.bid_price + market_data.ask_price) / 2
        
        # Simulate execution
        await self._fill_slice(slice, execution_price, slice.quantity)
    
    async def _execute_twap(self, slice: ExecutionSlice):
        """Execute TWAP slice over time."""
        # Execute proportionally over the slice interval
        slice_duration = self.min_slice_interval
        execution_rate = slice.quantity / slice_duration
        
        executed_quantity = 0
        start_time = time.time()
        
        while executed_quantity < slice.quantity and time.time() - start_time < slice_duration:
            # Execute small chunks
            chunk_size = min(execution_rate, slice.quantity - executed_quantity)
            
            market_data = self.market_data.get(slice.symbol)
            if market_data:
                execution_price = (market_data.bid_price + market_data.ask_price) / 2
                await self._fill_slice(slice, execution_price, chunk_size)
                executed_quantity += chunk_size
            
            await asyncio.sleep(1)  # Execute every second
        
        # Fill any remaining quantity
        if executed_quantity < slice.quantity:
            remaining = slice.quantity - executed_quantity
            market_data = self.market_data.get(slice.symbol)
            if market_data:
                execution_price = (market_data.bid_price + market_data.ask_price) / 2
                await self._fill_slice(slice, execution_price, remaining)
    
    async def _execute_vwap(self, slice: ExecutionSlice):
        """Execute VWAP slice based on volume profile."""
        # Similar to TWAP but considers volume profile
        market_data = self.market_data.get(slice.symbol)
        if not market_data:
            raise ValueError(f"No market data for {slice.symbol}")
        
        # Use current VWAP as reference
        reference_price = market_data.vwap_today
        execution_price = reference_price
        
        await self._fill_slice(slice, execution_price, slice.quantity)
    
    async def _execute_pov(self, slice: ExecutionSlice):
        """Execute POV slice based on real-time volume."""
        participation_rate = slice.metadata.get("participation_rate", 0.1)
        slice_duration = self.min_slice_interval
        
        executed_quantity = 0
        start_time = time.time()
        
        while executed_quantity < slice.quantity and time.time() - start_time < slice_duration:
            market_data = self.market_data.get(slice.symbol)
            if not market_data:
                await asyncio.sleep(1)
                continue
            
            # Calculate allowed participation
            recent_volume = market_data.volume_today * 0.01  # Assume 1% of daily volume is recent
            max_participation = recent_volume * participation_rate
            
            # Execute allowed amount
            chunk_size = min(max_participation, slice.quantity - executed_quantity)
            if chunk_size > 0:
                execution_price = (market_data.bid_price + market_data.ask_price) / 2
                await self._fill_slice(slice, execution_price, chunk_size)
                executed_quantity += chunk_size
            
            await asyncio.sleep(1)
    
    async def _fill_slice(self, slice: ExecutionSlice, price: float, quantity: float):
        """Fill a slice with given price and quantity."""
        slice.filled_quantity += quantity
        slice.average_price = (
            (slice.average_price * (slice.filled_quantity - quantity) + price * quantity) /
            slice.filled_quantity
        )
        
        if slice.filled_quantity >= slice.quantity * 0.999:  # Allow small rounding differences
            slice.status = OrderStatus.FILLED
            slice.end_time = datetime.utcnow()
        
        # Update parent order
        order = self.active_orders.get(slice.order_id)
        if order:
            order.filled_quantity += quantity
            order.average_price = (
                (order.average_price * (order.filled_quantity - quantity) + price * quantity) /
                order.filled_quantity
            )
            
            if order.filled_quantity >= order.total_quantity * 0.999:
                order.status = OrderStatus.FILLED
                order.end_time = datetime.utcnow()
        
        # Log execution
        self.logger.log_trade(
            symbol=slice.symbol,
            side=slice.side.value,
            quantity=quantity,
            price=price,
            order_id=slice.order_id,
            slice_id=slice.slice_id,
            algorithm=slice.algorithm.value,
            remaining_quantity=slice.quantity - slice.filled_quantity
        )
        
        # Record in execution history
        execution_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "order_id": slice.order_id,
            "slice_id": slice.slice_id,
            "symbol": slice.symbol,
            "side": slice.side.value,
            "quantity": quantity,
            "price": price,
            "algorithm": slice.algorithm.value
        }
        self.execution_history.append(execution_record)
    
    async def _monitor_executions(self):
        """Background task to monitor active executions."""
        while self._running:
            try:
                await self._check_slice_timeouts()
                await self._update_market_data()
                await asyncio.sleep(5)  # Check every 5 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Execution monitoring error: {e}")
                await asyncio.sleep(5)
    
    async def _check_slice_timeouts(self):
        """Check for slice timeouts and handle them."""
        current_time = datetime.utcnow()
        
        for order in self.active_orders.values():
            for slice in order.slices:
                if slice.status == OrderStatus.WORKING:
                    # Check if slice has timed out
                    time_elapsed = (current_time - slice.start_time).total_seconds()
                    timeout_seconds = order.time_horizon_minutes * 60
                    
                    if time_elapsed > timeout_seconds:
                        # Force fill remaining quantity
                        remaining = slice.quantity - slice.filled_quantity
                        if remaining > 0:
                            market_data = self.market_data.get(slice.symbol)
                            if market_data:
                                execution_price = (market_data.bid_price + market_data.ask_price) / 2
                                await self._fill_slice(slice, execution_price, remaining)
    
    async def _update_market_data(self):
        """Update market data for all symbols."""
        # This would connect to real market data feeds
        # For now, simulate with random walk
        for symbol in list(self.market_data.keys()):
            current_data = self.market_data[symbol]
            
            # Simulate price movement
            price_change = np.random.normal(0, 0.001)
            new_last = current_data.last_price * (1 + price_change)
            
            # Update market data
            current_data.last_price = new_last
            current_data.bid_price = new_last * 0.999
            current_data.ask_price = new_last * 1.001
            current_data.timestamp = datetime.utcnow()
            
            # Notify callbacks
            for callback in self.market_data_callbacks:
                try:
                    callback(current_data)
                except Exception as e:
                    self.logger.error(f"Market data callback error: {e}")
    
    def update_market_data(self, market_data: MarketDataSnapshot):
        """Update market data for a symbol."""
        self.market_data[market_data.symbol] = market_data
        
        # Notify callbacks
        for callback in self.market_data_callbacks:
            try:
                callback(market_data)
            except Exception as e:
                self.logger.error(f"Market data callback error: {e}")
    
    def add_market_data_callback(self, callback: Callable[[MarketDataSnapshot], None]):
        """Add callback for market data updates."""
        self.market_data_callbacks.append(callback)
    
    def get_order_status(self, order_id: str) -> Optional[ExecutionOrder]:
        """Get status of an order."""
        return self.active_orders.get(order_id)
    
    def get_execution_summary(self, order_id: str = None) -> Dict:
        """Get execution summary for an order or all orders."""
        if order_id:
            orders = {order_id: self.active_orders.get(order_id)} if order_id in self.active_orders else {}
        else:
            orders = self.active_orders
        
        summary = {}
        for oid, order in orders.items():
            if not order:
                continue
            
            summary[oid] = {
                "symbol": order.symbol,
                "side": order.side.value,
                "total_quantity": order.total_quantity,
                "filled_quantity": order.filled_quantity,
                "average_price": order.average_price,
                "status": order.status.value,
                "algorithm": order.algorithm.value,
                "created_time": order.created_time.isoformat(),
                "num_slices": len(order.slices),
                "completed_slices": len([s for s in order.slices if s.status == OrderStatus.FILLED])
            }
        
        return summary
    
    def get_performance_metrics(self, symbol: str = None, 
                             start_time: datetime = None, 
                             end_time: datetime = None) -> Dict:
        """Get execution performance metrics."""
        trades = self.execution_history
        
        # Filter by symbol and time range
        if symbol:
            trades = [t for t in trades if t["symbol"] == symbol]
        
        if start_time:
            trades = [t for t in trades if datetime.fromisoformat(t["timestamp"]) >= start_time]
        
        if end_time:
            trades = [t for t in trades if datetime.fromisoformat(t["timestamp"]) <= end_time]
        
        if not trades:
            return {}
        
        # Calculate metrics
        total_volume = sum(t["quantity"] for t in trades)
        total_notional = sum(t["quantity"] * t["price"] for t in trades)
        vwap = total_notional / total_volume if total_volume > 0 else 0
        
        # Calculate slippage (simplified - would use benchmark prices)
        avg_price = np.mean([t["price"] for t in trades])
        
        return {
            "total_trades": len(trades),
            "total_volume": total_volume,
            "total_notional": total_notional,
            "vwap": vwap,
            "average_price": avg_price,
            "time_range": {
                "start": min(t["timestamp"] for t in trades),
                "end": max(t["timestamp"] for t in trades)
            }
        }


# Global execution engine instance
execution_engine = AdvancedExecutionEngine()
