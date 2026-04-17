"""
orchestration/orchestrator.py

Orchestrator state machine for industrial-scale deployment.
Provides centralized coordination of all trading system components.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid

from monitoring.structured_logger import get_logger
from infrastructure.infrastructure_guard import infrastructure_guard, require_healthy_system
from execution.global_session_tracker import session_tracker, MarketStatus
from execution.advanced_execution_engine import execution_engine, ExecutionOrder, ExecutionAlgorithm, OrderSide

logger = get_logger("orchestrator")


class OrchestratorState(Enum):
    """Orchestrator state machine states."""
    INITIALIZING = "INITIALIZING"
    HEALTH_CHECK = "HEALTH_CHECK"
    PRE_MARKET = "PRE_MARKET"
    MARKET_OPEN = "MARKET_OPEN"
    TRADING_ACTIVE = "TRADING_ACTIVE"
    POST_MARKET = "POST_MARKET"
    ERROR = "ERROR"
    SHUTDOWN = "SHUTDOWN"
    MAINTENANCE = "MAINTENANCE"


class SystemMode(Enum):
    """System operational modes."""
    SIMULATION = "SIMULATION"
    PAPER_TRADING = "PAPER_TRADING"
    LIVE_TRADING = "LIVE_TRADING"
    DISASTER_RECOVERY = "DISASTER_RECOVERY"
    MAINTENANCE = "MAINTENANCE"


@dataclass
class StateTransition:
    """State transition record."""
    from_state: OrchestratorState
    to_state: OrchestratorState
    timestamp: datetime
    reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemMetrics:
    """System performance metrics."""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    active_orders: int
    executed_trades: int
    total_pnl: float
    latency_ms: float
    error_rate: float


class Orchestrator:
    """
    Central orchestrator for industrial-scale trading system.
    Manages state transitions, component lifecycle, and system coordination.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logger
        
        # State management
        self.current_state = OrchestratorState.INITIALIZING
        self.system_mode = SystemMode.SIMULATION
        self.state_history: List[StateTransition] = []
        self.last_state_change = datetime.utcnow()
        
        # Component references
        self.components = {}
        self.task_registry = {}
        
        # Configuration
        self.health_check_interval = self.config.get("health_check_interval", 60)  # seconds
        self.state_check_interval = self.config.get("state_check_interval", 30)  # seconds
        self.max_recovery_attempts = self.config.get("max_recovery_attempts", 3)
        self.emergency_shutdown_threshold = self.config.get("emergency_shutdown_threshold", 0.95)  # CPU/memory
        
        # Background tasks
        self._orchestration_task: Optional[asyncio.Task] = None
        self._health_monitor_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Event callbacks
        self.state_change_callbacks: List[Callable[[StateTransition], None]] = []
        self.error_callbacks: List[Callable[[Exception], None]] = []
        
        # Metrics
        self.metrics_history: List[SystemMetrics] = []
        self.current_metrics: Optional[SystemMetrics] = None
    
    async def start(self):
        """Start the orchestrator."""
        if self._running:
            return
        
        self.logger.info("Starting orchestrator")
        
        try:
            # Initialize components
            await self._initialize_components()
            
            # Start background tasks
            self._running = True
            self._orchestration_task = asyncio.create_task(self._orchestration_loop())
            self._health_monitor_task = asyncio.create_task(self._health_monitoring_loop())
            
            # Transition to initial state
            await self._transition_to(OrchestratorState.HEALTH_CHECK, "System startup")
            
            self.logger.info("Orchestrator started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start orchestrator: {e}")
            await self._transition_to(OrchestratorState.ERROR, f"Startup failed: {e}")
            raise
    
    async def stop(self):
        """Stop the orchestrator gracefully."""
        self.logger.info("Stopping orchestrator")
        
        await self._transition_to(OrchestratorState.SHUTDOWN, "Manual shutdown")
        
        self._running = False
        
        # Cancel background tasks
        tasks = [self._orchestration_task, self._health_monitor_task]
        for task in tasks:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Stop components
        await self._stop_components()
        
        self.logger.info("Orchestrator stopped")
    
    async def _initialize_components(self):
        """Initialize all system components."""
        self.logger.info("Initializing system components")
        
        # Start infrastructure guard
        await infrastructure_guard.pre_flight_check()
        
        # Start session tracker
        await session_tracker.start_monitoring()
        
        # Start execution engine
        await execution_engine.start()
        
        # Register components
        self.components = {
            "infrastructure_guard": infrastructure_guard,
            "session_tracker": session_tracker,
            "execution_engine": execution_engine,
        }
        
        self.logger.info(f"Initialized {len(self.components)} components")
    
    async def _stop_components(self):
        """Stop all system components."""
        self.logger.info("Stopping system components")
        
        # Stop execution engine
        await execution_engine.stop()
        
        # Stop session tracker
        await session_tracker.stop_monitoring()
        
        self.logger.info("All components stopped")
    
    async def _orchestration_loop(self):
        """Main orchestration loop."""
        while self._running:
            try:
                await self._process_current_state()
                await asyncio.sleep(self.state_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Orchestration loop error: {e}")
                await self._handle_error(e)
                await asyncio.sleep(self.state_check_interval)
    
    async def _health_monitoring_loop(self):
        """Background health monitoring loop."""
        while self._running:
            try:
                await self._update_system_metrics()
                await self._check_emergency_conditions()
                await asyncio.sleep(self.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(self.health_check_interval)
    
    async def _process_current_state(self):
        """Process logic for current state."""
        if self.current_state == OrchestratorState.HEALTH_CHECK:
            await self._handle_health_check_state()
        
        elif self.current_state == OrchestratorState.PRE_MARKET:
            await self._handle_pre_market_state()
        
        elif self.current_state == OrchestratorState.MARKET_OPEN:
            await self._handle_market_open_state()
        
        elif self.current_state == OrchestratorState.TRADING_ACTIVE:
            await self._handle_trading_active_state()
        
        elif self.current_state == OrchestratorState.POST_MARKET:
            await self._handle_post_market_state()
        
        elif self.current_state == OrchestratorState.ERROR:
            await self._handle_error_state()
        
        elif self.current_state == OrchestratorState.MAINTENANCE:
            await self._handle_maintenance_state()
    
    async def _handle_health_check_state(self):
        """Handle health check state."""
        # Perform comprehensive health check
        health = await infrastructure_guard.pre_flight_check()
        
        if health.status.value == "HEALTHY":
            # Check market status
            market_status = session_tracker.get_market_status("NYSE")
            
            if market_status == MarketStatus.OPEN:
                await self._transition_to(OrchestratorState.MARKET_OPEN, "Market is open and system healthy")
            elif market_status == MarketStatus.PRE_MARKET:
                await self._transition_to(OrchestratorState.PRE_MARKET, "Pre-market and system healthy")
            else:
                await self._transition_to(OrchestratorState.POST_MARKET, "Market closed and system healthy")
        else:
            await self._transition_to(OrchestratorState.ERROR, f"System unhealthy: {health.summary}")
    
    async def _handle_pre_market_state(self):
        """Handle pre-market state."""
        # Prepare for market open
        market_status = session_tracker.get_market_status("NYSE")
        
        if market_status == MarketStatus.OPEN:
            await self._transition_to(OrchestratorState.MARKET_OPEN, "Market opened")
        elif market_status != MarketStatus.PRE_MARKET:
            await self._transition_to(OrchestratorState.POST_MARKET, "Market status changed")
        
        # Pre-market tasks
        await self._prepare_trading_day()
    
    async def _handle_market_open_state(self):
        """Handle market open state."""
        # Check if we should start trading
        if self.system_mode in [SystemMode.LIVE_TRADING, SystemMode.PAPER_TRADING]:
            await self._transition_to(OrchestratorState.TRADING_ACTIVE, "Starting trading")
        
        # Monitor market status
        market_status = session_tracker.get_market_status("NYSE")
        if market_status != MarketStatus.OPEN:
            await self._transition_to(OrchestratorState.POST_MARKET, "Market closed")
    
    async def _handle_trading_active_state(self):
        """Handle active trading state."""
        # Monitor market status
        market_status = session_tracker.get_market_status("NYSE")
        
        if market_status != MarketStatus.OPEN:
            await self._transition_to(OrchestratorState.POST_MARKET, "Market closed during trading")
        
        # Monitor system health
        if not infrastructure_guard.is_healthy_for_trading():
            await self._transition_to(OrchestratorState.ERROR, "System became unhealthy during trading")
        
        # Trading logic would be implemented here
        await self._process_trading_signals()
    
    async def _handle_post_market_state(self):
        """Handle post-market state."""
        # End of day processing
        await self._end_of_day_processing()
        
        # Check for next market status
        market_status = session_tracker.get_market_status("NYSE")
        
        if market_status == MarketStatus.PRE_MARKET:
            await self._transition_to(OrchestratorState.PRE_MARKET, "Next trading day pre-market")
        elif market_status == MarketStatus.OPEN:
            await self._transition_to(OrchestratorState.MARKET_OPEN, "Market opened unexpectedly")
    
    async def _handle_error_state(self):
        """Handle error state."""
        # Attempt recovery
        recovery_attempts = 0
        
        while recovery_attempts < self.max_recovery_attempts:
            try:
                self.logger.info(f"Attempting recovery (attempt {recovery_attempts + 1})")
                
                # Health check
                health = await infrastructure_guard.pre_flight_check(force=True)
                
                if health.status.value == "HEALTHY":
                    await self._transition_to(OrchestratorState.HEALTH_CHECK, "Recovery successful")
                    return
                
                recovery_attempts += 1
                await asyncio.sleep(30)  # Wait between attempts
                
            except Exception as e:
                self.logger.error(f"Recovery attempt failed: {e}")
                recovery_attempts += 1
                await asyncio.sleep(30)
        
        # If recovery fails, enter maintenance
        await self._transition_to(OrchestratorState.MAINTENANCE, "Recovery failed, entering maintenance")
    
    async def _handle_maintenance_state(self):
        """Handle maintenance state."""
        # Minimal activity during maintenance
        self.logger.info("System in maintenance mode")
        await asyncio.sleep(60)  # Check every minute
    
    async def _prepare_trading_day(self):
        """Prepare for trading day."""
        self.logger.info("Preparing trading day")
        
        # Load trading strategies
        # Calculate risk limits
        # Prepare execution schedules
        
        # This would integrate with strategy modules
        pass
    
    async def _process_trading_signals(self):
        """Process trading signals."""
        # This would integrate with signal generation
        # For now, just log activity
        if time.time() % 60 < 1:  # Log once per minute
            self.logger.info("Trading active - processing signals")
    
    async def _end_of_day_processing(self):
        """End of day processing."""
        self.logger.info("End of day processing")
        
        # Calculate daily P&L
        # Generate reports
        # Archive data
        # Reset daily counters
    
    async def _update_system_metrics(self):
        """Update system performance metrics."""
        try:
            import psutil
            
            # Get system metrics
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Get trading metrics
            execution_summary = execution_engine.get_execution_summary()
            active_orders = len(execution_summary)
            executed_trades = sum(s.get("completed_slices", 0) for s in execution_summary.values())
            
            # Calculate latency (simplified)
            latency_ms = time.time() * 1000 % 1000  # Placeholder
            
            metrics = SystemMetrics(
                timestamp=datetime.utcnow(),
                cpu_usage=cpu_usage,
                memory_usage=memory.percent,
                disk_usage=(disk.used / disk.total) * 100,
                active_orders=active_orders,
                executed_trades=executed_trades,
                total_pnl=0.0,  # Would calculate from actual trades
                latency_ms=latency_ms,
                error_rate=0.0  # Would calculate from error tracking
            )
            
            self.current_metrics = metrics
            self.metrics_history.append(metrics)
            
            # Keep only last 24 hours of metrics
            cutoff_time = datetime.utcnow() - timedelta(hours=24)
            self.metrics_history = [m for m in self.metrics_history if m.timestamp > cutoff_time]
            
        except Exception as e:
            self.logger.error(f"Failed to update system metrics: {e}")
    
    async def _check_emergency_conditions(self):
        """Check for emergency shutdown conditions."""
        if not self.current_metrics:
            return
        
        # Check resource usage
        if (self.current_metrics.cpu_usage > self.emergency_shutdown_threshold * 100 or
            self.current_metrics.memory_usage > self.emergency_shutdown_threshold * 100):
            
            self.logger.critical(
                "Emergency shutdown triggered - resource usage exceeded threshold",
                cpu_usage=self.current_metrics.cpu_usage,
                memory_usage=self.current_metrics.memory_usage,
                threshold=self.emergency_shutdown_threshold * 100
            )
            
            await self._transition_to(OrchestratorState.SHUTDOWN, "Emergency shutdown")
    
    async def _transition_to(self, new_state: OrchestratorState, reason: str, metadata: Dict[str, Any] = None):
        """Transition to a new state."""
        if new_state == self.current_state:
            return
        
        old_state = self.current_state
        transition = StateTransition(
            from_state=old_state,
            to_state=new_state,
            timestamp=datetime.utcnow(),
            reason=reason,
            metadata=metadata or {}
        )
        
        self.current_state = new_state
        self.last_state_change = datetime.utcnow()
        self.state_history.append(transition)
        
        self.logger.info(
            f"State transition",
            from_state=old_state.value,
            to_state=new_state.value,
            reason=reason,
            transition_id=str(uuid.uuid4())
        )
        
        # Notify callbacks
        for callback in self.state_change_callbacks:
            try:
                callback(transition)
            except Exception as e:
                self.logger.error(f"State change callback error: {e}")
    
    async def _handle_error(self, error: Exception):
        """Handle system errors."""
        self.logger.error(f"System error: {error}")
        
        # Notify error callbacks
        for callback in self.error_callbacks:
            try:
                callback(error)
            except Exception as e:
                self.logger.error(f"Error callback error: {e}")
    
    def set_system_mode(self, mode: SystemMode):
        """Set system operational mode."""
        old_mode = self.system_mode
        self.system_mode = mode
        
        self.logger.info(
            f"System mode changed",
            from_mode=old_mode.value,
            to_mode=mode.value
        )
    
    def add_state_change_callback(self, callback: Callable[[StateTransition], None]):
        """Add callback for state changes."""
        self.state_change_callbacks.append(callback)
    
    def add_error_callback(self, callback: Callable[[Exception], None]):
        """Add callback for errors."""
        self.error_callbacks.append(callback)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        health = infrastructure_guard.last_health_check
        session_summary = session_tracker.get_session_summary()
        execution_summary = execution_engine.get_execution_summary()
        
        return {
            "orchestrator": {
                "state": self.current_state.value,
                "mode": self.system_mode.value,
                "uptime_seconds": (datetime.utcnow() - self.last_state_change).total_seconds(),
                "last_state_change": self.last_state_change.isoformat(),
                "state_transitions": len(self.state_history)
            },
            "infrastructure": {
                "health_status": health.status.value if health else "UNKNOWN",
                "health_summary": health.summary if health else "No health check performed",
                "last_health_check": health.timestamp.isoformat() if health else None
            },
            "sessions": session_summary,
            "execution": execution_summary,
            "metrics": {
                "current": self.current_metrics.__dict__ if self.current_metrics else None,
                "metrics_count": len(self.metrics_history)
            }
        }
    
    async def submit_trading_order(self, symbol: str, side: str, quantity: float, 
                                 algorithm: str = "TWAP", **kwargs) -> str:
        """Submit a trading order through the orchestrator."""
        if self.current_state != OrchestratorState.TRADING_ACTIVE:
            raise RuntimeError(f"Cannot submit order in state {self.current_state.value}")
        
        if not infrastructure_guard.is_healthy_for_trading():
            raise RuntimeError("System not healthy for trading")
        
        order = ExecutionOrder(
            order_id=str(uuid.uuid4()),
            symbol=symbol,
            side=OrderSide(side.upper()),
            total_quantity=quantity,
            algorithm=ExecutionAlgorithm(algorithm.upper()),
            **kwargs
        )
        
        order_id = await execution_engine.submit_order(order)
        
        self.logger.info(
            f"Trading order submitted through orchestrator",
            order_id=order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            algorithm=algorithm
        )
        
        return order_id
    
    async def emergency_shutdown(self, reason: str):
        """Perform emergency shutdown."""
        self.logger.critical(f"Emergency shutdown initiated: {reason}")
        
        # Cancel all active orders
        execution_summary = execution_engine.get_execution_summary()
        for order_id in execution_summary.keys():
            await execution_engine.cancel_order(order_id)
        
        # Transition to shutdown
        await self._transition_to(OrchestratorState.SHUTDOWN, f"Emergency shutdown: {reason}")
        
        # Stop orchestrator
        await self.stop()


# Global orchestrator instance
orchestrator = Orchestrator()
