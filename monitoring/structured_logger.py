"""
monitoring/structured_logger.py

Production-grade structured JSON logging for high-scale analysis.
Provides centralized logging with correlation IDs, performance metrics,
and structured output for observability platforms.
"""

import json
import logging
import time
import uuid
from datetime import datetime
from typing import Dict, Any, Optional
from contextlib import contextmanager
from dataclasses import dataclass, asdict


@dataclass
class LogContext:
    """Structured log context for correlation and tracing."""
    request_id: str
    session_id: str
    user_id: Optional[str] = None
    correlation_id: Optional[str] = None
    component: Optional[str] = None
    operation: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class StructuredLogger:
    """
    Production-grade structured logger with JSON output.
    Supports correlation tracing, performance metrics, and structured fields.
    """
    
    def __init__(self, name: str, level: str = "INFO"):
        self.logger = logging.getLogger(f"STRUCTURED_{name}")
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Remove existing handlers to avoid duplicates
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Create JSON formatter
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        self.logger.addHandler(handler)
        
        self._context_stack = []
    
    def _create_log_record(self, level: str, message: str, **kwargs) -> Dict[str, Any]:
        """Create structured log record with context."""
        context = self._context_stack[-1] if self._context_stack else None
        
        record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": level,
            "logger": self.logger.name,
            "message": message,
            "thread_id": str(time.thread_time_ns()),
        }
        
        # Add context if available
        if context:
            record.update({
                "request_id": context.request_id,
                "session_id": context.session_id,
                "correlation_id": context.correlation_id,
                "component": context.component,
                "operation": context.operation,
            })
            
            if context.user_id:
                record["user_id"] = context.user_id
            
            if context.metadata:
                record["metadata"] = context.metadata
        
        # Add additional fields
        record.update(kwargs)
        
        return record
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        record = self._create_log_record("DEBUG", message, **kwargs)
        self.logger.debug(json.dumps(record))
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        record = self._create_log_record("INFO", message, **kwargs)
        self.logger.info(json.dumps(record))
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        record = self._create_log_record("WARNING", message, **kwargs)
        self.logger.warning(json.dumps(record))
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        record = self._create_log_record("ERROR", message, **kwargs)
        self.logger.error(json.dumps(record))
    
    def critical(self, message: str, **kwargs):
        """Log critical message."""
        record = self._create_log_record("CRITICAL", message, **kwargs)
        self.logger.critical(json.dumps(record))
    
    def log_metric(self, name: str, value: float, unit: str = None, tags: Dict[str, str] = None):
        """Log performance metric."""
        metric_record = {
            "metric_name": name,
            "metric_value": value,
            "metric_unit": unit,
            "metric_tags": tags or {},
        }
        
        context = self._context_stack[-1] if self._context_stack else None
        if context:
            metric_record.update({
                "request_id": context.request_id,
                "session_id": context.session_id,
                "component": context.component,
            })
        
        record = self._create_log_record(
            "METRIC",
            f"Metric: {name}={value}",
            **metric_record
        )
        self.logger.info(json.dumps(record))
    
    def log_trade(self, symbol: str, side: str, quantity: float, price: float, 
                  order_id: str = None, strategy: str = None, **kwargs):
        """Log trade execution."""
        trade_record = {
            "event_type": "TRADE",
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": price,
            "order_id": order_id,
            "strategy": strategy,
            "execution_time": datetime.utcnow().isoformat() + "Z",
        }
        trade_record.update(kwargs)
        
        record = self._create_log_record(
            "TRADE",
            f"Trade executed: {side} {quantity} {symbol} @ {price}",
            **trade_record
        )
        self.logger.info(json.dumps(record))
    
    def log_signal(self, symbol: str, signal: float, confidence: float, 
                  strategy: str = None, timestamp: str = None, **kwargs):
        """Log trading signal."""
        signal_record = {
            "event_type": "SIGNAL",
            "symbol": symbol,
            "signal_value": signal,
            "confidence": confidence,
            "strategy": strategy,
            "signal_time": timestamp or datetime.utcnow().isoformat() + "Z",
        }
        signal_record.update(kwargs)
        
        record = self._create_log_record(
            "SIGNAL",
            f"Signal generated: {symbol}={signal} (conf={confidence})",
            **signal_record
        )
        self.logger.info(json.dumps(record))
    
    def log_risk_event(self, event_type: str, description: str, severity: str = "MEDIUM",
                       exposure: float = None, limit: float = None, **kwargs):
        """Log risk management event."""
        risk_record = {
            "event_type": "RISK",
            "risk_event": event_type,
            "description": description,
            "severity": severity,
            "exposure": exposure,
            "limit": limit,
            "risk_time": datetime.utcnow().isoformat() + "Z",
        }
        risk_record.update(kwargs)
        
        record = self._create_log_record(
            "RISK",
            f"Risk event: {event_type} - {description}",
            **risk_record
        )
        self.logger.warning(json.dumps(record))
    
    @contextmanager
    def context(self, request_id: str = None, session_id: str = None, 
                user_id: str = None, component: str = None, 
                operation: str = None, correlation_id: str = None,
                metadata: Dict[str, Any] = None):
        """Context manager for log context."""
        if not request_id:
            request_id = str(uuid.uuid4())
        if not session_id:
            session_id = str(uuid.uuid4())
        
        context = LogContext(
            request_id=request_id,
            session_id=session_id,
            user_id=user_id,
            correlation_id=correlation_id,
            component=component,
            operation=operation,
            metadata=metadata
        )
        
        self._context_stack.append(context)
        try:
            yield context
        finally:
            self._context_stack.pop()
    
    @contextmanager
    def performance_timer(self, operation: str, tags: Dict[str, str] = None):
        """Context manager for performance timing."""
        start_time = time.time()
        start_perf = time.perf_counter()
        
        try:
            with self.context(operation=operation, metadata=tags):
                yield
        finally:
            duration = time.time() - start_time
            perf_duration = time.perf_counter() - start_perf
            
            self.log_metric(
                f"{operation}_duration_seconds",
                duration,
                "seconds",
                tags
            )
            self.log_metric(
                f"{operation}_duration_perf_counter",
                perf_duration,
                "seconds",
                tags
            )


class JsonFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def format(self, record):
        """Format log record as JSON."""
        # Handle structured records (already JSON)
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            try:
                # Check if it's already a JSON string
                json.loads(record.msg)
                return record.msg
            except (json.JSONDecodeError, TypeError):
                pass
        
        # Handle standard log records
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        if hasattr(record, 'exc_info') and record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)


# Global logger instances
loggers = {}

def get_logger(name: str) -> StructuredLogger:
    """Get or create structured logger instance."""
    if name not in loggers:
        loggers[name] = StructuredLogger(name)
    return loggers[name]

# Convenience functions
def log_trade(symbol: str, side: str, quantity: float, price: float, **kwargs):
    """Global trade logging function."""
    get_logger("trading").log_trade(symbol, side, quantity, price, **kwargs)

def log_signal(symbol: str, signal: float, confidence: float, **kwargs):
    """Global signal logging function."""
    get_logger("signals").log_signal(symbol, signal, confidence, **kwargs)

def log_risk(event_type: str, description: str, **kwargs):
    """Global risk logging function."""
    get_logger("risk").log_risk_event(event_type, description, **kwargs)

def log_metric(name: str, value: float, **kwargs):
    """Global metric logging function."""
    get_logger("metrics").log_metric(name, value, **kwargs)
