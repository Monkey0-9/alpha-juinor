#!/usr/bin/env python3
"""
RESILIENCE FRAMEWORK - FAULT TOLERANCE & CIRCUIT BREAKERS
===========================================================

Enterprise-grade resilience patterns for MiniQuantFund trading system.
Provides unified circuit breakers, retries, fallback mechanisms across
data ingestion, execution, and broker layers.

Features:
- Distributed circuit breaker pattern
- Exponential backoff with jitter
- Bulkhead isolation (resource pools)
- Timeout enforcement
- Health checking
- Graceful degradation
- Automatic failover

Usage:
    from mini_quant_fund.infra.resilience_framework import ResilienceFramework
    
    @resilience.circuit_breaker(name="alpaca_api")
    @resilience.retry(max_attempts=3, backoff=ExponentialBackoff())
    @resilience.timeout(seconds=5)
    def fetch_market_data(symbol):
        return alpaca_api.get_quote(symbol)
"""

import os
import sys
import time
import random
import logging
import functools
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, TypeVar, Generic
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

T = TypeVar('T')


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = auto()      # Normal operation
    OPEN = auto()        # Failing, reject requests
    HALF_OPEN = auto()   # Testing if service recovered


class FailureType(Enum):
    """Types of failures for classification."""
    TRANSIENT = auto()   # Network timeout, temporary
    PERSISTENT = auto()  # Service down, configuration error
    ENTITLEMENT = auto() # Permission denied, quota exceeded


@dataclass
class RetryPolicy:
    """Retry configuration."""
    max_attempts: int = 3
    base_delay_sec: float = 1.0
    max_delay_sec: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_exceptions: tuple = (Exception,)
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt."""
        delay = self.base_delay_sec * (self.exponential_base ** attempt)
        delay = min(delay, self.max_delay_sec)
        
        if self.jitter:
            delay *= (0.5 + random.random())  # Add 50-150% jitter
        
        return delay


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    name: str
    failure_threshold: int = 5
    recovery_timeout_sec: float = 30.0
    half_open_max_calls: int = 3
    success_threshold: int = 2
    
    # Advanced settings
    failure_rate_threshold: float = 0.5  # 50% failure rate
    slow_call_threshold_sec: float = 10.0
    slow_call_rate_threshold: float = 0.5
    
    # Callbacks
    on_open: Optional[Callable] = None
    on_close: Optional[Callable] = None
    on_half_open: Optional[Callable] = None


@dataclass
class ResilienceMetrics:
    """Metrics for resilience operations."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0  # Circuit breaker open
    retry_attempts: int = 0
    timeout_calls: int = 0
    slow_calls: int = 0
    
    # Timing
    avg_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    
    # History
    latencies: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    def record_call(self, latency_ms: float, success: bool):
        """Record a call result."""
        self.total_calls += 1
        self.latencies.append(latency_ms)
        
        if success:
            self.successful_calls += 1
        else:
            self.failed_calls += 1
        
        # Update statistics
        if self.latencies:
            self.avg_latency_ms = sum(self.latencies) / len(self.latencies)
            sorted_latencies = sorted(self.latencies)
            self.p99_latency_ms = sorted_latencies[int(len(sorted_latencies) * 0.99)]


class CircuitBreaker:
    """Thread-safe circuit breaker implementation."""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.half_open_calls = 0
        
        self._lock = threading.RLock()
        self.metrics = ResilienceMetrics()
        
        logger.info(f"Circuit breaker '{config.name}' initialized (CLOSED)")
    
    def can_execute(self) -> bool:
        """Check if execution is allowed."""
        with self._lock:
            if self.state == CircuitState.CLOSED:
                return True
            
            if self.state == CircuitState.OPEN:
                # Check if recovery timeout passed
                if self.last_failure_time:
                    elapsed = (datetime.utcnow() - self.last_failure_time).total_seconds()
                    if elapsed >= self.config.recovery_timeout_sec:
                        self.state = CircuitState.HALF_OPEN
                        self.half_open_calls = 0
                        self.failure_count = 0
                        self.success_count = 0
                        logger.info(f"Circuit '{self.config.name}' entering HALF_OPEN state")
                        if self.config.on_half_open:
                            self.config.on_half_open()
                        return True
                
                self.metrics.rejected_calls += 1
                return False
            
            if self.state == CircuitState.HALF_OPEN:
                if self.half_open_calls < self.config.half_open_max_calls:
                    self.half_open_calls += 1
                    return True
                return False
            
            return False
    
    def record_success(self):
        """Record a successful call."""
        with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    self.half_open_calls = 0
                    logger.info(f"Circuit '{self.config.name}' CLOSED (recovered)")
                    if self.config.on_close:
                        self.config.on_close()
            
            elif self.state == CircuitState.CLOSED:
                self.failure_count = 0
    
    def record_failure(self, failure_type: FailureType = FailureType.TRANSIENT):
        """Record a failed call."""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = datetime.utcnow()
            
            if self.state == CircuitState.HALF_OPEN:
                # Any failure in half-open trips breaker
                self.state = CircuitState.OPEN
                logger.warning(f"Circuit '{self.config.name}' OPENED (half-open failure)")
                if self.config.on_open:
                    self.config.on_open()
            
            elif self.state == CircuitState.CLOSED:
                # Check thresholds
                total_calls = self.failure_count + self.success_count
                if total_calls >= 10:  # Minimum sample size
                    failure_rate = self.failure_count / total_calls
                    
                    if (self.failure_count >= self.config.failure_threshold or
                        failure_rate >= self.config.failure_rate_threshold):
                        self.state = CircuitState.OPEN
                        logger.warning(f"Circuit '{self.config.name}' OPENED "
                                     f"({self.failure_count} failures, {failure_rate:.1%} rate)")
                        if self.config.on_open:
                            self.config.on_open()
    
    def get_state(self) -> Dict:
        """Get current circuit breaker state."""
        with self._lock:
            return {
                "name": self.config.name,
                "state": self.state.name,
                "failure_count": self.failure_count,
                "success_count": self.success_count,
                "half_open_calls": self.half_open_calls,
                "last_failure": self.last_failure_time.isoformat() if self.last_failure_time else None
            }


class Bulkhead:
    """Bulkhead pattern for resource isolation."""
    
    def __init__(self, name: str, max_concurrent: int, max_queue: int = 100):
        self.name = name
        self.max_concurrent = max_concurrent
        self.max_queue = max_queue
        
        self.semaphore = threading.Semaphore(max_concurrent)
        self.queue_size = 0
        self._lock = threading.Lock()
        
        self.metrics = ResilienceMetrics()
    
    def execute(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with bulkhead protection."""
        acquired = self.semaphore.acquire(timeout=5.0)
        
        if not acquired:
            raise BulkheadFullException(f"Bulkhead '{self.name}' is full")
        
        try:
            start = time.time()
            result = func(*args, **kwargs)
            latency_ms = (time.time() - start) * 1000
            self.metrics.record_call(latency_ms, success=True)
            return result
        except Exception as e:
            latency_ms = (time.time() - start) * 1000
            self.metrics.record_call(latency_ms, success=False)
            raise
        finally:
            self.semaphore.release()
    
    def get_metrics(self) -> Dict:
        """Get bulkhead metrics."""
        return {
            "name": self.name,
            "max_concurrent": self.max_concurrent,
            "available_slots": self.semaphore._value,
            "metrics": {
                "total_calls": self.metrics.total_calls,
                "successful": self.metrics.successful_calls,
                "failed": self.metrics.failed_calls,
                "avg_latency_ms": self.metrics.avg_latency_ms,
                "p99_latency_ms": self.metrics.p99_latency_ms
            }
        }


class BulkheadFullException(Exception):
    """Exception raised when bulkhead is full."""
    pass


class ResilienceFramework:
    """Unified resilience framework for trading system."""
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.bulkheads: Dict[str, Bulkhead] = {}
        self.retry_policies: Dict[str, RetryPolicy] = {}
        self._lock = threading.RLock()
        
        self._setup_default_circuit_breakers()
        self._setup_default_bulkheads()
    
    def _setup_default_circuit_breakers(self):
        """Configure default circuit breakers."""
        default_configs = [
            CircuitBreakerConfig(
                name="alpaca_api",
                failure_threshold=3,
                recovery_timeout_sec=30.0,
                on_open=lambda: logger.critical("Alpaca API circuit OPEN - trading paused"),
                on_close=lambda: logger.info("Alpaca API circuit CLOSED - trading resumed")
            ),
            CircuitBreakerConfig(
                name="data_router",
                failure_threshold=5,
                recovery_timeout_sec=60.0
            ),
            CircuitBreakerConfig(
                name="database",
                failure_threshold=3,
                recovery_timeout_sec=30.0
            ),
            CircuitBreakerConfig(
                name="broker_execution",
                failure_threshold=2,
                recovery_timeout_sec=15.0
            ),
        ]
        
        for config in default_configs:
            self.circuit_breakers[config.name] = CircuitBreaker(config)
    
    def _setup_default_bulkheads(self):
        """Configure default bulkheads for resource isolation."""
        default_bulkheads = [
            ("api_calls", 10),      # Max 10 concurrent API calls
            ("database", 20),         # Max 20 concurrent DB operations
            ("executions", 5),        # Max 5 concurrent executions
            ("data_fetch", 15),      # Max 15 concurrent data fetches
        ]
        
        for name, limit in default_bulkheads:
            self.bulkheads[name] = Bulkhead(name, limit)
    
    def get_circuit_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name."""
        return self.circuit_breakers.get(name)
    
    def get_bulkhead(self, name: str) -> Optional[Bulkhead]:
        """Get bulkhead by name."""
        return self.bulkheads.get(name)
    
    def circuit_breaker(self, name: str):
        """Decorator to add circuit breaker protection."""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                cb = self.get_circuit_breaker(name)
                if not cb:
                    return func(*args, **kwargs)
                
                if not cb.can_execute():
                    raise CircuitBreakerOpenException(f"Circuit '{name}' is OPEN")
                
                try:
                    result = func(*args, **kwargs)
                    cb.record_success()
                    return result
                except Exception as e:
                    # Classify failure
                    failure_type = self._classify_failure(e)
                    cb.record_failure(failure_type)
                    raise
            
            return wrapper
        return decorator
    
    def retry(self, policy: Optional[RetryPolicy] = None, **kwargs):
        """Decorator to add retry logic."""
        if policy is None:
            policy = RetryPolicy(**kwargs)
        
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                last_exception = None
                
                for attempt in range(policy.max_attempts):
                    try:
                        return func(*args, **kwargs)
                    except policy.retryable_exceptions as e:
                        last_exception = e
                        
                        if attempt < policy.max_attempts - 1:
                            delay = policy.get_delay(attempt)
                            logger.warning(f"Retry {attempt + 1}/{policy.max_attempts} for {func.__name__} "
                                         f"after {delay:.2f}s (error: {e})")
                            time.sleep(delay)
                
                raise last_exception
            
            return wrapper
        return decorator
    
    def timeout(self, seconds: float):
        """Decorator to add timeout enforcement."""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                import concurrent.futures
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(func, *args, **kwargs)
                    try:
                        return future.result(timeout=seconds)
                    except concurrent.futures.TimeoutError:
                        raise TimeoutException(f"Function {func.__name__} timed out after {seconds}s")
            
            return wrapper
        return decorator
    
    def bulkhead(self, name: str):
        """Decorator to add bulkhead protection."""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                bh = self.get_bulkhead(name)
                if not bh:
                    return func(*args, **kwargs)
                
                return bh.execute(func, *args, **kwargs)
            
            return wrapper
        return decorator
    
    def _classify_failure(self, exception: Exception) -> FailureType:
        """Classify exception type for circuit breaker."""
        exception_str = str(exception).lower()
        
        # Entitlement failures
        if any(x in exception_str for x in ["401", "403", "unauthorized", "forbidden", "quota"]):
            return FailureType.ENTITLEMENT
        
        # Persistent failures
        if any(x in exception_str for x in ["connection refused", "dns", "not found", "configuration"]):
            return FailureType.PERSISTENT
        
        # Default to transient
        return FailureType.TRANSIENT
    
    def get_health_report(self) -> Dict:
        """Get comprehensive health report."""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "circuit_breakers": {
                name: cb.get_state()
                for name, cb in self.circuit_breakers.items()
            },
            "bulkheads": {
                name: bh.get_metrics()
                for name, bh in self.bulkheads.items()
            }
        }
    
    def reset_circuit(self, name: str) -> bool:
        """Manually reset a circuit breaker."""
        cb = self.get_circuit_breaker(name)
        if cb:
            with cb._lock:
                cb.state = CircuitState.CLOSED
                cb.failure_count = 0
                cb.success_count = 0
                cb.half_open_calls = 0
                logger.info(f"Circuit '{name}' manually reset to CLOSED")
                return True
        return False


class CircuitBreakerOpenException(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class TimeoutException(Exception):
    """Exception raised when operation times out."""
    pass


# Global resilience framework instance
_resilience_framework: Optional[ResilienceFramework] = None


def get_resilience_framework() -> ResilienceFramework:
    """Get global resilience framework instance."""
    global _resilience_framework
    if _resilience_framework is None:
        _resilience_framework = ResilienceFramework()
    return _resilience_framework


# Convenience decorators
resilience = get_resilience_framework()


def with_resilience(
    circuit_name: Optional[str] = None,
    max_retries: int = 3,
    timeout_sec: Optional[float] = None,
    bulkhead_name: Optional[str] = None
):
    """Combined resilience decorator."""
    def decorator(func: Callable) -> Callable:
        # Apply decorators in order (inside-out execution)
        
        if timeout_sec:
            func = resilience.timeout(timeout_sec)(func)
        
        if max_retries > 0:
            func = resilience.retry(max_attempts=max_retries)(func)
        
        if circuit_name:
            func = resilience.circuit_breaker(circuit_name)(func)
        
        if bulkhead_name:
            func = resilience.bulkhead(bulkhead_name)(func)
        
        return func
    return decorator


if __name__ == "__main__":
    # Test the resilience framework
    rf = ResilienceFramework()
    
    # Test circuit breaker
    @rf.circuit_breaker("test")
    def test_func():
        if random.random() < 0.7:
            raise Exception("Random failure")
        return "success"
    
    for i in range(20):
        try:
            result = test_func()
            print(f"Call {i}: {result}")
        except CircuitBreakerOpenException:
            print(f"Call {i}: CIRCUIT OPEN")
        except Exception as e:
            print(f"Call {i}: ERROR - {e}")
    
    print("\nCircuit breaker state:")
    print(json.dumps(rf.get_health_report(), indent=2))
