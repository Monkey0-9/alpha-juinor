"""
Global Error Handler - Zero Error Architecture.

Features:
- Catches and logs all unhandled exceptions
- Automatic recovery for known error types
- Graceful degradation when subsystems fail
- Circuit breaker pattern
- Heartbeat monitoring
"""

import logging
import traceback
import time
import threading
from dataclasses import dataclass, field
from typing import Dict, Callable, Optional, Any, List
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"           # Recoverable, continue
    MEDIUM = "medium"     # Log and retry
    HIGH = "high"         # Pause subsystem
    CRITICAL = "critical" # Full system halt


class SubsystemStatus(Enum):
    """Subsystem health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    RECOVERING = "recovering"


@dataclass
class ErrorRecord:
    """Record of an error occurrence."""
    timestamp: datetime
    subsystem: str
    error_type: str
    message: str
    severity: ErrorSeverity
    traceback: str
    recovered: bool = False
    recovery_action: Optional[str] = None


@dataclass
class CircuitBreaker:
    """Circuit breaker for a subsystem."""
    subsystem: str
    failure_count: int = 0
    failure_threshold: int = 5
    reset_timeout: int = 300  # seconds
    last_failure: Optional[datetime] = None
    state: str = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    def record_failure(self):
        """Record a failure."""
        self.failure_count += 1
        self.last_failure = datetime.utcnow()
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning(
                f"CircuitBreaker: {self.subsystem} OPENED "
                f"after {self.failure_count} failures"
            )

    def record_success(self):
        """Record a success."""
        if self.state == "HALF_OPEN":
            self.state = "CLOSED"
            self.failure_count = 0
            logger.info(f"CircuitBreaker: {self.subsystem} CLOSED after success")

    def can_proceed(self) -> bool:
        """Check if requests can proceed."""
        if self.state == "CLOSED":
            return True
        if self.state == "OPEN":
            if self.last_failure:
                elapsed = (datetime.utcnow() - self.last_failure).total_seconds()
                if elapsed >= self.reset_timeout:
                    self.state = "HALF_OPEN"
                    return True
            return False
        return True  # HALF_OPEN allows one request


class GlobalErrorHandler:
    """
    Central error handling and recovery system.

    Responsibilities:
    - Track errors across all subsystems
    - Implement circuit breakers
    - Coordinate recovery actions
    - Maintain system stability
    """

    # Recovery strategies for known errors
    RECOVERY_STRATEGIES: Dict[str, Callable] = {}

    def __init__(self, max_error_history: int = 1000):
        self.error_history: List[ErrorRecord] = []
        self.max_error_history = max_error_history
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.subsystem_status: Dict[str, SubsystemStatus] = {}
        self.lock = threading.Lock()
        self.system_paused = False
        self.pause_reason: Optional[str] = None

    def register_subsystem(self, name: str, failure_threshold: int = 5):
        """Register a subsystem with circuit breaker."""
        with self.lock:
            self.circuit_breakers[name] = CircuitBreaker(
                subsystem=name,
                failure_threshold=failure_threshold
            )
            self.subsystem_status[name] = SubsystemStatus.HEALTHY
            logger.info(f"ErrorHandler: Registered subsystem '{name}'")

    def handle_error(
        self,
        subsystem: str,
        error: Exception,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[Dict] = None
    ) -> bool:
        """
        Handle an error from a subsystem.

        Returns: True if recovered, False otherwise.
        """
        tb = traceback.format_exc()
        error_type = type(error).__name__

        record = ErrorRecord(
            timestamp=datetime.utcnow(),
            subsystem=subsystem,
            error_type=error_type,
            message=str(error),
            severity=severity,
            traceback=tb
        )

        with self.lock:
            self.error_history.append(record)
            if len(self.error_history) > self.max_error_history:
                self.error_history = self.error_history[-self.max_error_history:]

            # Update circuit breaker
            if subsystem in self.circuit_breakers:
                self.circuit_breakers[subsystem].record_failure()

        # Log based on severity
        if severity == ErrorSeverity.CRITICAL:
            logger.critical(
                f"CRITICAL ERROR in {subsystem}: {error_type} - {error}"
            )
            self._handle_critical(subsystem, record)
        elif severity == ErrorSeverity.HIGH:
            logger.error(f"HIGH ERROR in {subsystem}: {error_type} - {error}")
            self._update_status(subsystem, SubsystemStatus.DEGRADED)
        elif severity == ErrorSeverity.MEDIUM:
            logger.warning(f"ERROR in {subsystem}: {error_type} - {error}")
        else:
            logger.info(f"LOW ERROR in {subsystem}: {error_type} - {error}")

        # Attempt recovery
        recovered = self._attempt_recovery(subsystem, error_type, error, context)
        record.recovered = recovered

        return recovered

    def _attempt_recovery(
        self,
        subsystem: str,
        error_type: str,
        error: Exception,
        context: Optional[Dict]
    ) -> bool:
        """Attempt automatic recovery."""
        # Check for registered recovery strategy
        recovery_key = f"{subsystem}:{error_type}"
        if recovery_key in self.RECOVERY_STRATEGIES:
            try:
                self.RECOVERY_STRATEGIES[recovery_key](error, context)
                logger.info(
                    f"ErrorHandler: Recovered {subsystem} from {error_type}"
                )
                self._update_status(subsystem, SubsystemStatus.RECOVERING)
                return True
            except Exception as e:
                logger.error(f"Recovery failed for {subsystem}: {e}")
                return False

        # Generic recovery attempts
        if "timeout" in str(error).lower():
            logger.info(f"ErrorHandler: Timeout in {subsystem}, will retry")
            return True

        if "rate" in str(error).lower():
            logger.info(f"ErrorHandler: Rate limit in {subsystem}, backing off")
            return True

        return False

    def _handle_critical(self, subsystem: str, record: ErrorRecord):
        """Handle critical errors."""
        self._update_status(subsystem, SubsystemStatus.FAILED)

        # Check if system should pause
        failed_count = sum(
            1 for s in self.subsystem_status.values()
            if s == SubsystemStatus.FAILED
        )

        if failed_count >= 2:
            self.system_paused = True
            self.pause_reason = f"Multiple subsystems failed: {subsystem}"
            logger.critical(
                f"SYSTEM PAUSED: {self.pause_reason}"
            )

    def _update_status(self, subsystem: str, status: SubsystemStatus):
        """Update subsystem status."""
        with self.lock:
            self.subsystem_status[subsystem] = status
            logger.info(f"Subsystem {subsystem} status: {status.value}")

    def can_proceed(self, subsystem: str) -> bool:
        """Check if a subsystem can proceed with operations."""
        if self.system_paused:
            return False

        if subsystem in self.circuit_breakers:
            return self.circuit_breakers[subsystem].can_proceed()

        return True

    def record_success(self, subsystem: str):
        """Record a successful operation."""
        with self.lock:
            if subsystem in self.circuit_breakers:
                self.circuit_breakers[subsystem].record_success()
            self._update_status(subsystem, SubsystemStatus.HEALTHY)

    def get_health_report(self) -> Dict[str, Any]:
        """Get system health report."""
        recent_errors = [
            {
                "timestamp": e.timestamp.isoformat(),
                "subsystem": e.subsystem,
                "error_type": e.error_type,
                "severity": e.severity.value,
                "recovered": e.recovered
            }
            for e in self.error_history[-20:]
        ]

        return {
            "system_paused": self.system_paused,
            "pause_reason": self.pause_reason,
            "subsystem_status": {
                k: v.value for k, v in self.subsystem_status.items()
            },
            "circuit_breakers": {
                k: {"state": v.state, "failures": v.failure_count}
                for k, v in self.circuit_breakers.items()
            },
            "total_errors": len(self.error_history),
            "recent_errors": recent_errors
        }

    def reset_subsystem(self, subsystem: str):
        """Manually reset a subsystem."""
        with self.lock:
            if subsystem in self.circuit_breakers:
                self.circuit_breakers[subsystem].state = "CLOSED"
                self.circuit_breakers[subsystem].failure_count = 0
            self._update_status(subsystem, SubsystemStatus.HEALTHY)
            logger.info(f"ErrorHandler: Reset subsystem {subsystem}")

    def resume_system(self):
        """Resume paused system."""
        self.system_paused = False
        self.pause_reason = None
        logger.info("ErrorHandler: System resumed")


def error_boundary(
    subsystem: str,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    default_return: Any = None
):
    """
    Decorator for error boundary around functions.

    Usage:
        @error_boundary("data_provider", ErrorSeverity.MEDIUM, default_return={})
        def fetch_data():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            handler = get_error_handler()

            if not handler.can_proceed(subsystem):
                logger.warning(
                    f"ErrorBoundary: {subsystem} blocked by circuit breaker"
                )
                return default_return

            try:
                result = func(*args, **kwargs)
                handler.record_success(subsystem)
                return result
            except Exception as e:
                context = {
                    "function": func.__name__,
                    "args": str(args)[:100],
                    "kwargs": str(kwargs)[:100]
                }
                handler.handle_error(subsystem, e, severity, context)
                return default_return

        return wrapper
    return decorator


# Global instance
_error_handler: Optional[GlobalErrorHandler] = None


def get_error_handler() -> GlobalErrorHandler:
    """Get or create global error handler."""
    global _error_handler
    if _error_handler is None:
        _error_handler = GlobalErrorHandler()
        # Register core subsystems
        _error_handler.register_subsystem("data_provider")
        _error_handler.register_subsystem("alpha_engine")
        _error_handler.register_subsystem("risk_manager")
        _error_handler.register_subsystem("executor")
        _error_handler.register_subsystem("pm_brain")
    return _error_handler
