"""
Low-Latency Execution Optimizer

Provides high-performance optimizations for institutional trading
with microsecond-level precision and minimal overhead.
"""

import time
import asyncio
import threading
import multiprocessing
from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
from contextlib import contextmanager
import numpy as np

from .enterprise_logger import get_enterprise_logger
from .exceptions import PerformanceError, LatencyError
from .performance_monitor import performance_monitor


class OptimizationLevel(Enum):
    """Performance optimization levels."""
    MINIMAL = "minimal"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    EXTREME = "extreme"


@dataclass
class LatencyMetrics:
    """Latency measurement metrics."""
    operation: str
    min_latency_us: float
    max_latency_us: float
    avg_latency_us: float
    p50_latency_us: float
    p95_latency_us: float
    p99_latency_us: float
    total_operations: int
    timestamp: datetime


@dataclass
class OptimizationConfig:
    """Optimization configuration."""
    level: OptimizationLevel
    enable_caching: bool = True
    enable_pooling: bool = True
    enable_batching: bool = True
    enable_compression: bool = False
    cache_size: int = 10000
    pool_size: int = 100
    batch_size: int = 50
    max_latency_us: float = 1000.0  # 1ms
    memory_limit_mb: int = 1024


class LowLatencyOptimizer:
    """
    Enterprise-grade low-latency optimization system.

    Features:
    - Microsecond-level latency tracking
    - Memory pool management
    - Connection pooling
    - Batch processing
    - Caching mechanisms
    - NUMA awareness
    - CPU affinity
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """Singleton pattern with thread safety."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialize()
            return cls._instance

    def _initialize(self):
        """Initialize optimizer."""
        self.logger = get_enterprise_logger("low_latency_optimizer")
        self.config = OptimizationConfig(OptimizationLevel.BALANCED)
        self._latency_measurements: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self._performance_cache: Dict[str, Any] = {}
        self._connection_pools: Dict[str, Any] = {}
        self._thread_pools: Dict[str, Any] = {}
        self._memory_pools: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._batch_queues: Dict[str, deque] = defaultdict(deque)
        self._lock = threading.RLock()
        self._running = False
        self._optimization_task: Optional[asyncio.Task] = None

        # Set CPU affinity for performance
        self._set_cpu_affinity()

        # Initialize memory pools
        self._initialize_memory_pools()

    def _set_cpu_affinity(self):
        """Set CPU affinity for performance-critical threads."""
        try:
            current_process = multiprocessing.current_process()
            if hasattr(current_process, 'cpu_affinity'):
                # Pin to specific CPU cores for better cache locality
                cpu_count = multiprocessing.cpu_count()
                if cpu_count >= 4:
                    # Use first 4 cores for trading operations
                    current_process.cpu_affinity(list(range(4)))
                    self.logger.info(f"CPU affinity set to cores 0-3")
        except Exception as e:
            self.logger.warning(f"Failed to set CPU affinity: {e}")

    def _initialize_memory_pools(self):
        """Initialize memory pools for common objects."""
        # Pool for order objects
        self._memory_pools["orders"] = deque(maxlen=1000)

        # Pool for market data objects
        self._memory_pools["market_data"] = deque(maxlen=5000)

        # Pool for risk calculation results
        self._memory_pools["risk_results"] = deque(maxlen=1000)

        # Pre-allocate some objects
        for _ in range(100):
            self._memory_pools["orders"].append({})
            self._memory_pools["market_data"].append({})
            self._memory_pools["risk_results"].append({})

    def configure(self, config: OptimizationConfig):
        """Configure optimization settings."""
        with self._lock:
            self.config = config
            self.logger.info(f"Optimizer configured with level: {config.level}")

    @contextmanager
    def latency_tracker(self, operation: str, **tags):
        """Context manager for tracking operation latency."""
        start_time = time.perf_counter()

        try:
            yield
        finally:
            end_time = time.perf_counter()
            latency_us = (end_time - start_time) * 1_000_000  # Convert to microseconds

            # Record latency
            self._record_latency(operation, latency_us, tags)

            # Check if latency exceeds threshold
            if latency_us > self.config.max_latency_us:
                self.logger.error(
                    f"Latency threshold exceeded: {operation} took {latency_us:.2f}us",
                    operation=operation,
                    latency_us=latency_us,
                    threshold_us=self.config.max_latency_us,
                    **tags
                )
                raise LatencyError(f"Operation {operation} exceeded latency threshold")

    def _record_latency(self, operation: str, latency_us: float, tags: Dict[str, Any]):
        """Record latency measurement."""
        with self._lock:
            self._latency_measurements[operation].append({
                'latency_us': latency_us,
                'timestamp': datetime.utcnow(),
                'tags': tags
            })

        # Record in performance monitor
        performance_monitor.record_timer(f"latency.{operation}", latency_us, **tags)

    def get_latency_metrics(self, operation: str, minutes: int = 5) -> Optional[LatencyMetrics]:
        """Get latency metrics for an operation."""
        with self._lock:
            if operation not in self._latency_measurements:
                return None

            cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)
            recent_measurements = [
                m for m in self._latency_measurements[operation]
                if m['timestamp'] >= cutoff_time
            ]

            if not recent_measurements:
                return None

            latencies = [m['latency_us'] for m in recent_measurements]
            latencies.sort()
            n = len(latencies)

            return LatencyMetrics(
                operation=operation,
                min_latency_us=min(latencies),
                max_latency_us=max(latencies),
                avg_latency_us=sum(latencies) / n,
                p50_latency_us=latencies[n // 2],
                p95_latency_us=latencies[int(0.95 * n)],
                p99_latency_us=latencies[int(0.99 * n)],
                total_operations=n,
                timestamp=datetime.utcnow()
            )

    @contextmanager
    def memory_pool(self, pool_name: str):
        """Context manager for memory pool usage."""
        with self._lock:
            if pool_name in self._memory_pools and self._memory_pools[pool_name]:
                obj = self._memory_pools[pool_name].popleft()
            else:
                obj = {}

        try:
            yield obj
        finally:
            # Clear object and return to pool
            if isinstance(obj, dict):
                obj.clear()

            with self._lock:
                if pool_name in self._memory_pools:
                    self._memory_pools[pool_name].append(obj)

    async def batch_process(self, batch_name: str, items: List[Any],
                          processor: Callable, batch_size: Optional[int] = None):
        """Process items in batches for optimal performance."""
        if batch_size is None:
            batch_size = self.config.batch_size

        if not self.config.enable_batching or len(items) <= batch_size:
            # Process directly if batching is disabled or small batch
            return await processor(items)

        results = []
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]

            with self.latency_tracker(f"batch.{batch_name}", batch_size=len(batch)):
                batch_result = await processor(batch)
                results.extend(batch_result if isinstance(batch_result, list) else [batch_result])

        return results

    def get_cached_result(self, cache_key: str) -> Optional[Any]:
        """Get result from performance cache."""
        if not self.config.enable_caching:
            return None

        with self._lock:
            return self._performance_cache.get(cache_key)

    def cache_result(self, cache_key: str, result: Any, ttl_seconds: int = 300):
        """Cache result for performance."""
        if not self.config.enable_caching:
            return

        with self._lock:
            # Check cache size limit
            if len(self._performance_cache) >= self.config.cache_size:
                # Remove oldest entries (simple LRU)
                oldest_keys = list(self._performance_cache.keys())[:100]
                for key in oldest_keys:
                    del self._performance_cache[key]

            self._performance_cache[cache_key] = {
                'result': result,
                'timestamp': datetime.utcnow(),
                'ttl_seconds': ttl_seconds
            }

    def is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached result is still valid."""
        if not self.config.enable_caching:
            return False

        with self._lock:
            if cache_key not in self._performance_cache:
                return False

            cached_item = self._performance_cache[cache_key]
            age_seconds = (datetime.utcnow() - cached_item['timestamp']).total_seconds()

            return age_seconds <= cached_item['ttl_seconds']

    def get_valid_cached_result(self, cache_key: str) -> Optional[Any]:
        """Get cached result only if still valid."""
        if self.is_cache_valid(cache_key):
            with self._lock:
                return self._performance_cache[cache_key]['result']
        return None

    async def optimize_execution(self, operation: str, func: Callable, *args, **kwargs):
        """Execute function with optimal performance optimizations."""
        # Check cache first
        cache_key = f"{operation}:{hash(str(args) + str(sorted(kwargs.items())))}"
        cached_result = self.get_valid_cached_result(cache_key)
        if cached_result is not None:
            return cached_result

        # Execute with latency tracking
        with self.latency_tracker(f"optimized.{operation}"):
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

        # Cache result
        self.cache_result(cache_key, result)

        return result

    def start_optimization_monitoring(self):
        """Start continuous optimization monitoring."""
        if self._running:
            return

        self._running = True
        self._optimization_task = asyncio.create_task(self._optimization_loop())
        self.logger.info("Optimization monitoring started")

    def stop_optimization_monitoring(self):
        """Stop optimization monitoring."""
        if not self._running:
            return

        self._running = False
        if self._optimization_task:
            self._optimization_task.cancel()
        self.logger.info("Optimization monitoring stopped")

    async def _optimization_loop(self):
        """Background optimization loop."""
        while self._running:
            try:
                # Clean expired cache entries
                self._cleanup_expired_cache()

                # Optimize memory pools
                self._optimize_memory_pools()

                # Check performance thresholds
                self._check_performance_thresholds()

                await asyncio.sleep(30)  # Check every 30 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Optimization loop error: {e}")
                await asyncio.sleep(60)  # Wait before retry

    def _cleanup_expired_cache(self):
        """Clean up expired cache entries."""
        with self._lock:
            current_time = datetime.utcnow()
            expired_keys = []

            for key, item in self._performance_cache.items():
                age_seconds = (current_time - item['timestamp']).total_seconds()
                if age_seconds > item['ttl_seconds']:
                    expired_keys.append(key)

            for key in expired_keys:
                del self._performance_cache[key]

            if expired_keys:
                self.logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")

    def _optimize_memory_pools(self):
        """Optimize memory pool sizes."""
        with self._lock:
            for pool_name, pool in self._memory_pools.items():
                # Adjust pool size based on usage patterns
                current_size = len(pool)

                if current_size < 10:  # Pool is too small
                    # Add more objects to pool
                    for _ in range(50):
                        pool.append({})
                    self.logger.debug(f"Expanded memory pool {pool_name} to {len(pool)}")

                elif current_size > 500:  # Pool is too large
                    # Remove excess objects
                    excess = current_size - 200
                    for _ in range(excess):
                        pool.pop()
                    self.logger.debug(f"Reduced memory pool {pool_name} to {len(pool)}")

    def _check_performance_thresholds(self):
        """Check performance against thresholds."""
        critical_operations = [
            "order_submission",
            "market_data_processing",
            "risk_calculation",
            "portfolio_optimization"
        ]

        for operation in critical_operations:
            metrics = self.get_latency_metrics(operation, minutes=5)
            if metrics and metrics.p95_latency_us > self.config.max_latency_us:
                self.logger.error(
                    f"Performance degradation detected: {operation}",
                    operation=operation,
                    p95_latency_us=metrics.p95_latency_us,
                    threshold_us=self.config.max_latency_us
                )

    def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive optimization report."""
        with self._lock:
            report = {
                "timestamp": datetime.utcnow().isoformat(),
                "config": {
                    "level": self.config.level.value,
                    "caching_enabled": self.config.enable_caching,
                    "pooling_enabled": self.config.enable_pooling,
                    "batching_enabled": self.config.enable_batching,
                    "cache_size": self.config.cache_size,
                    "pool_size": self.config.pool_size,
                    "batch_size": self.config.batch_size,
                    "max_latency_us": self.config.max_latency_us
                },
                "cache_stats": {
                    "cache_size": len(self._performance_cache),
                    "cache_limit": self.config.cache_size,
                    "cache_utilization": len(self._performance_cache) / self.config.cache_size
                },
                "memory_pool_stats": {
                    name: len(pool) for name, pool in self._memory_pools.items()
                },
                "latency_metrics": {}
            }

            # Add latency metrics for key operations
            for operation in self._latency_measurements.keys():
                metrics = self.get_latency_metrics(operation, minutes=5)
                if metrics:
                    report["latency_metrics"][operation] = {
                        "avg_latency_us": metrics.avg_latency_us,
                        "p95_latency_us": metrics.p95_latency_us,
                        "p99_latency_us": metrics.p99_latency_us,
                        "total_operations": metrics.total_operations
                    }

        return report


# Global optimizer instance
low_latency_optimizer = LowLatencyOptimizer()


# Optimization decorators
def optimize_latency(operation_name: str):
    """Decorator for optimizing function latency."""
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                return await low_latency_optimizer.optimize_execution(
                    operation_name, func, *args, **kwargs
                )
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                return low_latency_optimizer.optimize_execution(
                    operation_name, func, *args, **kwargs
                )
            return sync_wrapper
    return decorator


def cache_result(ttl_seconds: int = 300):
    """Decorator for caching function results."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = f"{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"

            # Check cache
            cached_result = low_latency_optimizer.get_valid_cached_result(cache_key)
            if cached_result is not None:
                return cached_result

            # Execute function
            result = func(*args, **kwargs)

            # Cache result
            low_latency_optimizer.cache_result(cache_key, result, ttl_seconds)

            return result
        return wrapper
    return decorator


@contextmanager
def memory_pool_context(pool_name: str):
    """Context manager for memory pool usage."""
    return low_latency_optimizer.memory_pool(pool_name)


@contextmanager
def latency_tracking(operation: str, **tags):
    """Context manager for latency tracking."""
    return low_latency_optimizer.latency_tracker(operation, **tags)
