"""
execution/global_session_tracker.py

GlobalSessionTracker for intelligent market-hours execution management.
Manages trading sessions, market hours, and execution scheduling.
"""

import asyncio
import time
from datetime import datetime, timedelta, time as dt_time
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import pytz
import pandas as pd

from monitoring.structured_logger import get_logger

logger = get_logger("session_tracker")


class MarketStatus(Enum):
    """Market status enumeration."""
    PRE_MARKET = "PRE_MARKET"
    OPEN = "OPEN"
    POST_MARKET = "POST_MARKET"
    CLOSED = "CLOSED"
    HOLIDAY = "HOLIDAY"
    WEEKEND = "WEEKEND"


class SessionType(Enum):
    """Trading session types."""
    REGULAR = "REGULAR"
    EXTENDED = "EXTENDED"
    AFTER_HOURS = "AFTER_HOURS"
    PRE_HOURS = "PRE_HOURS"


@dataclass
class MarketHours:
    """Market hours configuration for an exchange."""
    exchange: str
    timezone: str
    regular_open: dt_time  # Time object (HH:MM)
    regular_close: dt_time  # Time object (HH:MM)
    pre_market_open: Optional[dt_time] = None
    post_market_close: Optional[dt_time] = None
    lunch_break_start: Optional[dt_time] = None
    lunch_break_end: Optional[dt_time] = None
    trading_days: Set[int] = field(default_factory=lambda: {0, 1, 2, 3, 4})  # Mon-Fri


@dataclass
class TradingSession:
    """Active trading session information."""
    session_id: str
    exchange: str
    session_type: SessionType
    start_time: datetime
    end_time: datetime
    status: MarketStatus
    active_orders: Set[str] = field(default_factory=set)
    executed_trades: List[Dict] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


@dataclass
class ExecutionSchedule:
    """Execution scheduling information."""
    order_id: str
    symbol: str
    side: str
    quantity: float
    algorithm: str  # TWAP, VWAP, IMMEDIATE, etc.
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    participation_rate: float = 0.1  # For volume algorithms
    max_slice_size: Optional[float] = None
    min_slice_size: Optional[float] = None
    priority: int = 0  # Higher number = higher priority


class GlobalSessionTracker:
    """
    Global session tracker for market-hours execution.
    Manages multiple exchanges, session scheduling, and execution timing.
    """
    
    def __init__(self):
        self.logger = logger
        self.active_sessions: Dict[str, TradingSession] = {}
        self.market_hours: Dict[str, MarketHours] = {}
        self.execution_schedules: Dict[str, List[ExecutionSchedule]] = {}
        self.holidays: Dict[str, Set[datetime]] = {}
        
        # Initialize default market configurations
        self._initialize_market_hours()
        
        # Background task for session monitoring
        self._monitoring_task: Optional[asyncio.Task] = None
        self._running = False
    
    def _initialize_market_hours(self):
        """Initialize default market hours for major exchanges."""
        # NYSE/NASDAQ
        self.market_hours["NYSE"] = MarketHours(
            exchange="NYSE",
            timezone="America/New_York",
            regular_open=dt_time(9, 30),
            regular_close=dt_time(16, 0),
            pre_market_open=dt_time(4, 0),
            post_market_close=dt_time(20, 0)
        )
        
        # London Stock Exchange
        self.market_hours["LSE"] = MarketHours(
            exchange="LSE",
            timezone="Europe/London",
            regular_open=dt_time(8, 0),
            regular_close=dt_time(16, 30)
        )
        
        # Tokyo Stock Exchange
        self.market_hours["TSE"] = MarketHours(
            exchange="TSE",
            timezone="Asia/Tokyo",
            regular_open=dt_time(9, 0),
            regular_close=dt_time(15, 0),
            trading_days={0, 1, 2, 3, 4}  # Mon-Fri
        )
        
        # Hong Kong Stock Exchange
        self.market_hours["HKEX"] = MarketHours(
            exchange="HKEX",
            timezone="Asia/Hong_Kong",
            regular_open=dt_time(9, 30),
            regular_close=dt_time(16, 0),
            lunch_break_start=dt_time(12, 0),
            lunch_break_end=dt_time(13, 0),
            trading_days={0, 1, 2, 3, 4}  # Mon-Fri
        )
    
    async def start_monitoring(self):
        """Start background session monitoring."""
        if self._running:
            return
        
        self._running = True
        self._monitoring_task = asyncio.create_task(self._monitor_sessions())
        self.logger.info("Session monitoring started")
    
    async def stop_monitoring(self):
        """Stop background session monitoring."""
        self._running = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Session monitoring stopped")
    
    async def _monitor_sessions(self):
        """Background task to monitor and update sessions."""
        while self._running:
            try:
                await self._update_all_sessions()
                await self._process_execution_schedules()
                await asyncio.sleep(60)  # Check every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Session monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _update_all_sessions(self):
        """Update status of all active sessions."""
        current_time = datetime.utcnow()
        
        for exchange, session in list(self.active_sessions.items()):
            new_status = self._get_market_status(exchange, current_time)
            
            if new_status != session.status:
                old_status = session.status
                session.status = new_status
                
                self.logger.info(
                    f"Market status changed for {exchange}",
                    exchange=exchange,
                    old_status=old_status.value,
                    new_status=new_status.value,
                    session_id=session.session_id
                )
                
                # Handle status transitions
                await self._handle_status_transition(session, old_status, new_status)
    
    async def _process_execution_schedules(self):
        """Process pending execution schedules."""
        current_time = datetime.utcnow()
        
        for exchange, schedules in self.execution_schedules.items():
            session = self.active_sessions.get(exchange)
            if not session or session.status != MarketStatus.OPEN:
                continue
            
            for schedule in schedules:
                if self._should_execute_schedule(schedule, current_time, session):
                    await self._execute_schedule(schedule, session)
    
    def _should_execute_schedule(self, schedule: ExecutionSchedule, 
                               current_time: datetime, session: TradingSession) -> bool:
        """Determine if a schedule should be executed now."""
        # Check if within time window
        if schedule.start_time and current_time < schedule.start_time:
            return False
        
        if schedule.end_time and current_time > schedule.end_time:
            return False
        
        # Check if market is open
        if session.status != MarketStatus.OPEN:
            return False
        
        # Check if already executed
        if schedule.order_id in session.active_orders:
            return False
        
        return True
    
    async def _execute_schedule(self, schedule: ExecutionSchedule, session: TradingSession):
        """Execute a scheduled order."""
        session.active_orders.add(schedule.order_id)
        
        self.logger.info(
            f"Executing scheduled order",
            order_id=schedule.order_id,
            symbol=schedule.symbol,
            side=schedule.side,
            quantity=schedule.quantity,
            algorithm=schedule.algorithm,
            exchange=session.exchange
        )
        
        # Here you would integrate with the actual execution engine
        # For now, just log the execution
        trade_record = {
            "order_id": schedule.order_id,
            "symbol": schedule.symbol,
            "side": schedule.side,
            "quantity": schedule.quantity,
            "algorithm": schedule.algorithm,
            "execution_time": datetime.utcnow().isoformat(),
            "exchange": session.exchange
        }
        
        session.executed_trades.append(trade_record)
    
    async def _handle_status_transition(self, session: TradingSession, 
                                     old_status: MarketStatus, new_status: MarketStatus):
        """Handle market status transitions."""
        if new_status == MarketStatus.OPEN:
            # Market opened - start execution
            self.logger.info(f"Market opened for {session.exchange}")
            
        elif new_status == MarketStatus.CLOSED:
            # Market closed - cleanup
            self.logger.info(f"Market closed for {session.exchange}")
            session.active_orders.clear()
            
        elif new_status == MarketStatus.HOLIDAY:
            # Holiday - suspend all activity
            self.logger.warning(f"Holiday detected for {session.exchange}")
            session.active_orders.clear()
    
    def get_market_status(self, exchange: str, timestamp: datetime = None) -> MarketStatus:
        """Get current market status for an exchange."""
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        return self._get_market_status(exchange, timestamp)
    
    def _get_market_status(self, exchange: str, timestamp: datetime) -> MarketStatus:
        """Internal method to get market status."""
        if exchange not in self.market_hours:
            return MarketStatus.UNKNOWN
        
        hours = self.market_hours[exchange]
        tz = pytz.timezone(hours.timezone)
        local_time = timestamp.astimezone(tz)
        
        # Check weekend
        if local_time.weekday() not in hours.trading_days:
            return MarketStatus.WEEKEND
        
        # Check holiday
        if exchange in self.holidays:
            holiday_date = local_time.date()
            if any(h.date() == holiday_date for h in self.holidays[exchange]):
                return MarketStatus.HOLIDAY
        
        current_time = local_time.time()
        
        # Check pre-market
        if hours.pre_market_open and current_time < hours.regular_open:
            if current_time >= hours.pre_market_open:
                return MarketStatus.PRE_MARKET
            else:
                return MarketStatus.CLOSED
        
        # Check post-market
        if hours.post_market_close and current_time > hours.regular_close:
            if current_time <= hours.post_market_close:
                return MarketStatus.POST_MARKET
            else:
                return MarketStatus.CLOSED
        
        # Check lunch break
        if hours.lunch_break_start and hours.lunch_break_end:
            if hours.lunch_break_start <= current_time <= hours.lunch_break_end:
                return MarketStatus.CLOSED
        
        # Check regular hours
        if hours.regular_open <= current_time <= hours.regular_close:
            return MarketStatus.OPEN
        
        return MarketStatus.CLOSED
    
    def create_session(self, exchange: str, session_type: SessionType = SessionType.REGULAR) -> str:
        """Create a new trading session."""
        session_id = f"{exchange}_{session_type.value}_{int(time.time())}"
        
        current_time = datetime.utcnow()
        status = self._get_market_status(exchange, current_time)
        
        session = TradingSession(
            session_id=session_id,
            exchange=exchange,
            session_type=session_type,
            start_time=current_time,
            end_time=current_time + timedelta(days=1),  # Default 24-hour session
            status=status
        )
        
        self.active_sessions[exchange] = session
        
        self.logger.info(
            f"Created trading session",
            session_id=session_id,
            exchange=exchange,
            session_type=session_type.value,
            status=status.value
        )
        
        return session_id
    
    def schedule_execution(self, schedule: ExecutionSchedule, exchange: str = "NYSE"):
        """Schedule an order execution."""
        if exchange not in self.execution_schedules:
            self.execution_schedules[exchange] = []
        
        self.execution_schedules[exchange].append(schedule)
        
        # Sort by priority (higher priority first)
        self.execution_schedules[exchange].sort(key=lambda x: x.priority, reverse=True)
        
        self.logger.info(
            f"Scheduled execution",
            order_id=schedule.order_id,
            symbol=schedule.symbol,
            exchange=exchange,
            algorithm=schedule.algorithm,
            priority=schedule.priority
        )
    
    def get_active_session(self, exchange: str) -> Optional[TradingSession]:
        """Get active session for an exchange."""
        return self.active_sessions.get(exchange)
    
    def get_session_summary(self, exchange: str = None) -> Dict:
        """Get summary of active sessions."""
        if exchange:
            sessions = {exchange: self.active_sessions.get(exchange)}
        else:
            sessions = self.active_sessions
        
        summary = {}
        for exch, session in sessions.items():
            if not session:
                continue
            
            summary[exch] = {
                "session_id": session.session_id,
                "status": session.status.value,
                "session_type": session.session_type.value,
                "start_time": session.start_time.isoformat(),
                "active_orders": len(session.active_orders),
                "executed_trades": len(session.executed_trades),
                "current_time": datetime.utcnow().isoformat()
            }
        
        return summary
    
    def is_market_open(self, exchange: str) -> bool:
        """Check if market is currently open for trading."""
        status = self.get_market_status(exchange)
        return status == MarketStatus.OPEN
    
    def get_next_market_open(self, exchange: str) -> Optional[datetime]:
        """Get next market open time for an exchange."""
        if exchange not in self.market_hours:
            return None
        
        hours = self.market_hours[exchange]
        tz = pytz.timezone(hours.timezone)
        
        current_time = datetime.utcnow()
        local_time = current_time.astimezone(tz)
        
        # If market is open now, return current time
        if self.get_market_status(exchange) == MarketStatus.OPEN:
            return current_time
        
        # Find next trading day
        days_ahead = 1
        while days_ahead <= 7:  # Look ahead up to a week
            next_date = (local_time + timedelta(days=days_ahead)).date()
            next_datetime = datetime.combine(next_date, hours.regular_open)
            next_datetime = tz.localize(next_datetime)
            next_datetime = next_datetime.astimezone(pytz.UTC)
            
            if next_date.weekday() in hours.trading_days:
                # Check if it's a holiday
                if exchange not in self.holidays or not any(
                    h.date() == next_date for h in self.holidays[exchange]
                ):
                    return next_datetime
            
            days_ahead += 1
        
        return None
    
    def add_holiday(self, exchange: str, date: datetime):
        """Add a holiday for an exchange."""
        if exchange not in self.holidays:
            self.holidays[exchange] = set()
        
        self.holidays[exchange].add(date)
        
        self.logger.info(
            f"Added holiday",
            exchange=exchange,
            date=date.isoformat()
        )
    
    def get_optimal_execution_window(self, exchange: str, urgency: str = "NORMAL") -> Tuple[datetime, datetime]:
        """Get optimal execution window based on market conditions."""
        if exchange not in self.market_hours:
            raise ValueError(f"Unknown exchange: {exchange}")
        
        hours = self.market_hours[exchange]
        tz = pytz.timezone(hours.timezone)
        current_time = datetime.utcnow()
        local_time = current_time.astimezone(tz)
        
        # Default to regular hours
        open_time = datetime.combine(local_time.date(), hours.regular_open)
        close_time = datetime.combine(local_time.date(), hours.regular_close)
        
        open_time = tz.localize(open_time)
        close_time = tz.localize(close_time)
        
        # Adjust based on urgency
        if urgency == "HIGH":
            # Include pre-market if available
            if hours.pre_market_open:
                open_time = datetime.combine(local_time.date(), hours.pre_market_open)
                open_time = tz.localize(open_time)
        
        elif urgency == "LOW":
            # Restrict to core trading hours (avoid open/close volatility)
            open_time = open_time + timedelta(minutes=30)
            close_time = close_time - timedelta(minutes=30)
        
        return open_time.astimezone(pytz.UTC), close_time.astimezone(pytz.UTC)


# Global session tracker instance
session_tracker = GlobalSessionTracker()
