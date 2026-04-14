"""
Enterprise-Grade Structured Logging System

Provides production-ready logging with correlation IDs, structured output,
and comprehensive monitoring capabilities for institutional trading systems.
"""

import json
import logging
import os
import time
import uuid
import threading
from typing import Dict, Any, Optional, Union
from datetime import datetime
from enum import Enum
from contextlib import contextmanager


class LogLevel(Enum):
    """Enterprise log levels with numerical severity."""
    TRACE = 0
    DEBUG = 1
    INFO = 2
    WARN = 3
    ERROR = 4
    FATAL = 5


class EnterpriseLogger:
    """
    Enterprise-grade structured logger for production trading systems.

    Features:
    - Structured JSON output
    - Correlation ID tracking
    - Performance timing
    - Context management
    - Thread-safe operations
    - Audit trail capabilities
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, name: str):
        """Singleton pattern with thread safety."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialize(name)
            return cls._instance

    def _initialize(self, name: str):
        """Initialize logger instance."""
        self.name = name
        self.logger = logging.getLogger(f"ENTERPRISE_{name}")
        self.logger.setLevel(logging.DEBUG)

        # Configure structured JSON formatter
        handler = logging.StreamHandler()
        handler.setFormatter(StructuredFormatter())
        self.logger.addHandler(handler)

        # Thread-local storage for context
        self._context = threading.local()

    @contextmanager
    def context(self, **kwargs):
        """Context manager for adding logging context."""
        old_context = getattr(self._context, 'data', {}).copy()

        # Merge new context
        current_context = getattr(self._context, 'data', {})
        current_context.update(kwargs)
        self._context.data = current_context

        try:
            yield
        finally:
            self._context.data = old_context

    @contextmanager
    def correlation_id(self, correlation_id: Optional[str] = None):
        """Context manager for correlation ID."""
        old_correlation_id = getattr(self._context, 'correlation_id', None)
        self._context.correlation_id = correlation_id or str(uuid.uuid4())

        try:
            yield self._context.correlation_id
        finally:
            self._context.correlation_id = old_correlation_id

    @contextmanager
    def performance_timer(self, operation: str, **metadata):
        """Context manager for performance timing."""
        start_time = time.time()

        try:
            yield
        finally:
            duration_ms = (time.time() - start_time) * 1000
            self.info(
                f"Operation completed: {operation}",
                operation=operation,
                duration_ms=duration_ms,
                **metadata
            )

    def _log(self, level: LogLevel, message: str, **kwargs):
        """Internal logging method with structured output."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": level.name,
            "logger": self.name,
            "message": message,
            "thread_id": threading.current_thread().ident,
            "process_id": os.getpid(),
        }

        # Add correlation ID if available
        correlation_id = getattr(self._context, 'correlation_id', None)
        if correlation_id:
            log_entry["correlation_id"] = correlation_id

        # Add context if available
        context_data = getattr(self._context, 'data', {})
        if context_data:
            log_entry["context"] = context_data

        # Add additional metadata
        log_entry.update(kwargs)

        # Log using standard logger
        log_method = {
            LogLevel.TRACE: self.logger.debug,
            LogLevel.DEBUG: self.logger.debug,
            LogLevel.INFO: self.logger.info,
            LogLevel.WARN: self.logger.warning,
            LogLevel.ERROR: self.logger.error,
            LogLevel.FATAL: self.logger.critical,
        }.get(level, self.logger.info)

        log_method(json.dumps(log_entry, default=str))

    def trace(self, message: str, **kwargs):
        """Log trace level message."""
        self._log(LogLevel.TRACE, message, **kwargs)

    def debug(self, message: str, **kwargs):
        """Log debug level message."""
        self._log(LogLevel.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs):
        """Log info level message."""
        self._log(LogLevel.INFO, message, **kwargs)

    def warn(self, message: str, **kwargs):
        """Log warning level message."""
        self._log(LogLevel.WARN, message, **kwargs)

    def error(self, message: str, **kwargs):
        """Log error level message."""
        self._log(LogLevel.ERROR, message, **kwargs)

    def fatal(self, message: str, **kwargs):
        """Log fatal level message."""
        self._log(LogLevel.FATAL, message, **kwargs)

    # Trading-specific logging methods
    def log_trade(self, symbol: str, side: str, quantity: float,
                 price: float, order_id: str, **kwargs):
        """Log trade execution."""
        self.info(
            f"Trade executed: {side} {quantity} {symbol} @ {price}",
            event_type="trade_execution",
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            order_id=order_id,
            **kwargs
        )

    def log_order(self, symbol: str, side: str, quantity: float,
                 order_type: str, order_id: str, **kwargs):
        """Log order submission."""
        self.info(
            f"Order submitted: {order_type} {side} {quantity} {symbol}",
            event_type="order_submission",
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            order_id=order_id,
            **kwargs
        )

    def log_risk(self, risk_type: str, message: str, severity: str, **kwargs):
        """Log risk event."""
        self.warn(
            f"Risk event: {risk_type} - {message}",
            event_type="risk_event",
            risk_type=risk_type,
            risk_message=message,
            severity=severity,
            **kwargs
        )

    def log_market_data(self, symbol: str, data_type: str,
                      quality: str, **kwargs):
        """Log market data event."""
        self.info(
            f"Market data received: {symbol} {data_type}",
            event_type="market_data",
            symbol=symbol,
            data_type=data_type,
            quality=quality,
            **kwargs
        )

    def log_performance(self, operation: str, duration_ms: float,
                      success: bool, **kwargs):
        """Log performance metric."""
        level = LogLevel.INFO if success else LogLevel.WARN
        self._log(
            level,
            f"Performance: {operation} completed in {duration_ms:.2f}ms",
            event_type="performance",
            operation=operation,
            duration_ms=duration_ms,
            success=success,
            **kwargs
        )

    def log_system_health(self, component: str, status: str,
                        metrics: Dict[str, Any], **kwargs):
        """Log system health event."""
        level = LogLevel.INFO if status == "HEALTHY" else LogLevel.WARN
        self._log(
            level,
            f"System health: {component} is {status}",
            event_type="system_health",
            component=component,
            status=status,
            metrics=metrics,
            **kwargs
        )

    def log_audit(self, action: str, user: str, resource: str,
                 result: str, **kwargs):
        """Log audit event."""
        self.info(
            f"Audit: {user} {action} {resource} - {result}",
            event_type="audit",
            action=action,
            user=user,
            resource=resource,
            result=result,
            **kwargs
        )

    def log_compliance(self, regulation: str, event: str,
                      compliant: bool, **kwargs):
        """Log compliance event."""
        level = LogLevel.INFO if compliant else LogLevel.ERROR
        self._log(
            level,
            f"Compliance: {regulation} - {event} - {'COMPLIANT' if compliant else 'VIOLATION'}",
            event_type="compliance",
            regulation=regulation,
            compliance_event=event,
            compliant=compliant,
            **kwargs
        )


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON output."""

    def format(self, record):
        """Format log record as structured JSON."""
        # Handle structured logs from EnterpriseLogger
        if hasattr(record, 'msg') and record.msg.startswith('{'):
            try:
                # Try to parse as JSON if it looks like structured log
                return record.msg
            except (json.JSONDecodeError, ValueError):
                pass

        # Fallback to standard formatting
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "thread_id": record.thread,
            "process_id": record.process,
        }

        return json.dumps(log_entry, default=str)


def get_enterprise_logger(name: str) -> EnterpriseLogger:
    """Factory function to get enterprise logger instance."""
    return EnterpriseLogger(name)


# Global logger instances for major components
system_logger = get_enterprise_logger("system")
trading_logger = get_enterprise_logger("trading")
risk_logger = get_enterprise_logger("risk")
market_data_logger = get_enterprise_logger("market_data")
execution_logger = get_enterprise_logger("execution")
orchestration_logger = get_enterprise_logger("orchestration")
infrastructure_logger = get_enterprise_logger("infrastructure")
monitoring_logger = get_enterprise_logger("monitoring")
compliance_logger = get_enterprise_logger("compliance")
audit_logger = get_enterprise_logger("audit")
