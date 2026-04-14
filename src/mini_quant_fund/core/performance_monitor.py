"""
Enterprise Performance Monitoring System

Provides comprehensive performance monitoring, metrics collection,
and real-time alerting for institutional trading systems.
"""

import time
import threading
import psutil
import asyncio
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
from contextlib import contextmanager

from .enterprise_logger import get_enterprise_logger
from .exceptions import PerformanceError, MonitoringError


class MetricType(Enum):
    """Types of performance metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Metric:
    """Performance metric data structure."""
    name: str
    value: float
    metric_type: MetricType
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    unit: str = ""


@dataclass
class Alert:
    """Performance alert data structure."""
    name: str
    severity: AlertSeverity
    message: str
    timestamp: datetime
    metric_name: str
    threshold: float
    actual_value: float
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class PerformanceThreshold:
    """Performance threshold configuration."""
    metric_name: str
    warning_threshold: Optional[float] = None
    critical_threshold: Optional[float] = None
    evaluation_window_seconds: int = 60
    consecutive_breaches: int = 1


class PerformanceMonitor:
    """
    Enterprise-grade performance monitoring system.

    Features:
    - Real-time metric collection
    - Threshold-based alerting
    - Performance aggregation
    - Historical data retention
    - Thread-safe operations
    - Custom metric support
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
        """Initialize performance monitor."""
        self.logger = get_enterprise_logger("performance_monitor")
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = defaultdict(float)
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self.timers: Dict[str, List[float]] = defaultdict(list)
        self.thresholds: Dict[str, PerformanceThreshold] = {}
        self.alerts: List[Alert] = []
        self.alert_callbacks: List[Callable[[Alert], None]] = []
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._lock = threading.RLock()

        # System metrics collection
        self.system_metrics_interval = 5  # seconds
        self._system_metrics_task: Optional[asyncio.Task] = None

    def start(self):
        """Start performance monitoring."""
        if self._running:
            return

        self.logger.info("Starting performance monitoring")
        self._running = True
        self._monitor_task = asyncio.create_task(self._monitoring_loop())
        self._system_metrics_task = asyncio.create_task(self._system_metrics_loop())

    def stop(self):
        """Stop performance monitoring."""
        if not self._running:
            return

        self.logger.info("Stopping performance monitoring")
        self._running = False

        if self._monitor_task:
            self._monitor_task.cancel()

        if self._system_metrics_task:
            self._system_metrics_task.cancel()

    def add_threshold(self, threshold: PerformanceThreshold):
        """Add performance threshold for alerting."""
        with self._lock:
            self.thresholds[threshold.metric_name] = threshold
        self.logger.info(f"Added threshold for {threshold.metric_name}")

    def remove_threshold(self, metric_name: str):
        """Remove performance threshold."""
        with self._lock:
            if metric_name in self.thresholds:
                del self.thresholds[metric_name]
        self.logger.info(f"Removed threshold for {metric_name}")

    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Add alert callback function."""
        self.alert_callbacks.append(callback)

    def record_counter(self, name: str, value: float = 1.0, **tags):
        """Record counter metric."""
        with self._lock:
            self.counters[name] += value

        metric = Metric(
            name=name,
            value=self.counters[name],
            metric_type=MetricType.COUNTER,
            timestamp=datetime.utcnow(),
            tags=tags
        )
        self._record_metric(metric)

    def set_gauge(self, name: str, value: float, **tags):
        """Set gauge metric value."""
        with self._lock:
            self.gauges[name] = value

        metric = Metric(
            name=name,
            value=value,
            metric_type=MetricType.GAUGE,
            timestamp=datetime.utcnow(),
            tags=tags
        )
        self._record_metric(metric)

    def record_histogram(self, name: str, value: float, **tags):
        """Record histogram metric value."""
        with self._lock:
            self.histograms[name].append(value)
            # Keep only last 1000 values
            if len(self.histograms[name]) > 1000:
                self.histograms[name] = self.histograms[name][-1000:]

        metric = Metric(
            name=name,
            value=value,
            metric_type=MetricType.HISTOGRAM,
            timestamp=datetime.utcnow(),
            tags=tags
        )
        self._record_metric(metric)

    def record_timer(self, name: str, duration_ms: float, **tags):
        """Record timer metric value."""
        with self._lock:
            self.timers[name].append(duration_ms)
            # Keep only last 1000 values
            if len(self.timers[name]) > 1000:
                self.timers[name] = self.timers[name][-1000:]

        metric = Metric(
            name=name,
            value=duration_ms,
            metric_type=MetricType.TIMER,
            timestamp=datetime.utcnow(),
            unit="ms",
            tags=tags
        )
        self._record_metric(metric)

    def _record_metric(self, metric: Metric):
        """Record metric and check thresholds."""
        self.metrics[metric.name].append(metric)

        # Check thresholds
        if metric.name in self.thresholds:
            self._check_threshold(metric)

    def _check_threshold(self, metric: Metric):
        """Check if metric breaches thresholds."""
        threshold = self.thresholds[metric.name]

        # Get recent values for evaluation
        recent_metrics = [
            m for m in self.metrics[metric.name]
            if (datetime.utcnow() - m.timestamp).total_seconds() <= threshold.evaluation_window_seconds
        ]

        if len(recent_metrics) < threshold.consecutive_breaches:
            return

        # Check consecutive breaches
        recent_values = [m.value for m in recent_metrics[-threshold.consecutive_breaches:]]

        if threshold.critical_threshold and all(v >= threshold.critical_threshold for v in recent_values):
            alert = Alert(
                name=f"{metric.name}_critical",
                severity=AlertSeverity.CRITICAL,
                message=f"Critical threshold breached for {metric.name}: {metric.value} >= {threshold.critical_threshold}",
                timestamp=datetime.utcnow(),
                metric_name=metric.name,
                threshold=threshold.critical_threshold,
                actual_value=metric.value,
                tags=metric.tags
            )
            self._trigger_alert(alert)

        elif threshold.warning_threshold and all(v >= threshold.warning_threshold for v in recent_values):
            alert = Alert(
                name=f"{metric.name}_warning",
                severity=AlertSeverity.WARNING,
                message=f"Warning threshold breached for {metric.name}: {metric.value} >= {threshold.warning_threshold}",
                timestamp=datetime.utcnow(),
                metric_name=metric.name,
                threshold=threshold.warning_threshold,
                actual_value=metric.value,
                tags=metric.tags
            )
            self._trigger_alert(alert)

    def _trigger_alert(self, alert: Alert):
        """Trigger performance alert."""
        self.alerts.append(alert)

        # Keep only last 1000 alerts
        if len(self.alerts) > 1000:
            self.alerts = self.alerts[-1000:]

        # Log alert
        self.logger.error(
            alert.message,
            alert_name=alert.name,
            severity=alert.severity.value,
            metric_name=alert.metric_name,
            threshold=alert.threshold,
            actual_value=alert.actual_value,
            tags=alert.tags
        )

        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Alert callback failed: {e}")

    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                await self._collect_performance_metrics()
                await asyncio.sleep(1)  # Check every second
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(5)  # Wait before retry

    async def _system_metrics_loop(self):
        """System metrics collection loop."""
        while self._running:
            try:
                await self._collect_system_metrics()
                await asyncio.sleep(self.system_metrics_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"System metrics loop error: {e}")
                await asyncio.sleep(10)  # Wait before retry

    async def _collect_performance_metrics(self):
        """Collect application performance metrics."""
        # Memory usage
        memory_info = psutil.virtual_memory()
        self.set_gauge("system.memory.usage_percent", memory_info.percent, unit="percent")
        self.set_gauge("system.memory.used_gb", memory_info.used / (1024**3), unit="GB")
        self.set_gauge("system.memory.available_gb", memory_info.available / (1024**3), unit="GB")

        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        self.set_gauge("system.cpu.usage_percent", cpu_percent, unit="percent")

        # Disk usage
        disk_info = psutil.disk_usage('/')
        self.set_gauge("system.disk.usage_percent", disk_info.percent, unit="percent")
        self.set_gauge("system.disk.used_gb", disk_info.used / (1024**3), unit="GB")
        self.set_gauge("system.disk.free_gb", disk_info.free / (1024**3), unit="GB")

        # Network I/O
        network_io = psutil.net_io_counters()
        self.set_gauge("system.network.bytes_sent", network_io.bytes_sent, unit="bytes")
        self.set_gauge("system.network.bytes_recv", network_io.bytes_recv, unit="bytes")

        # Process info
        process = psutil.Process()
        self.set_gauge("process.memory.rss_mb", process.memory_info().rss / (1024**2), unit="MB")
        self.set_gauge("process.cpu.percent", process.cpu_percent(), unit="percent")
        self.set_gauge("process.num_threads", process.num_threads(), unit="count")
        self.set_gauge("process.open_files", len(process.open_files()), unit="count")

    async def _collect_system_metrics(self):
        """Collect system-level metrics."""
        # System load
        load_avg = psutil.getloadavg()
        self.set_gauge("system.load.1min", load_avg[0])
        self.set_gauge("system.load.5min", load_avg[1])
        self.set_gauge("system.load.15min", load_avg[2])

        # Boot time
        boot_time = psutil.boot_time()
        uptime_seconds = time.time() - boot_time
        self.set_gauge("system.uptime_seconds", uptime_seconds, unit="seconds")

        # Context switches
        self.set_gauge("system.context_switches", psutil.cpu_stats().ctx_switches, unit="count")

        # Interrupts
        self.set_gauge("system.interrupts", psutil.cpu_stats().interrupts, unit="count")

        # System calls
        self.set_gauge("system.syscalls", psutil.cpu_stats().syscalls, unit="count")

    def get_metric_summary(self, metric_name: str,
                        minutes: int = 5) -> Dict[str, Any]:
        """Get metric summary for recent time window."""
        with self._lock:
            if metric_name not in self.metrics:
                return {}

            cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)
            recent_metrics = [
                m for m in self.metrics[metric_name]
                if m.timestamp >= cutoff_time
            ]

            if not recent_metrics:
                return {}

            values = [m.value for m in recent_metrics]

            summary = {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values),
                "latest": values[-1] if values else None,
                "metric_type": recent_metrics[-1].metric_type.value,
                "unit": recent_metrics[-1].unit
            }

            # Add percentiles for numeric metrics
            if recent_metrics[-1].metric_type in [MetricType.GAUGE, MetricType.HISTOGRAM, MetricType.TIMER]:
                sorted_values = sorted(values)
                n = len(sorted_values)
                summary.update({
                    "p50": sorted_values[n//2],
                    "p90": sorted_values[int(0.9*n)],
                    "p95": sorted_values[int(0.95*n)],
                    "p99": sorted_values[int(0.99*n)]
                })

            return summary

    def get_recent_alerts(self, minutes: int = 60) -> List[Alert]:
        """Get recent alerts."""
        cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)
        return [alert for alert in self.alerts if alert.timestamp >= cutoff_time]

    def get_all_metrics(self) -> Dict[str, List[Metric]]:
        """Get all stored metrics."""
        with self._lock:
            return {name: list(metrics) for name, metrics in self.metrics.items()}

    def clear_metrics(self, metric_name: Optional[str] = None):
        """Clear metrics data."""
        with self._lock:
            if metric_name:
                if metric_name in self.metrics:
                    self.metrics[metric_name].clear()
                if metric_name in self.counters:
                    self.counters[metric_name] = 0.0
                if metric_name in self.gauges:
                    del self.gauges[metric_name]
                if metric_name in self.histograms:
                    self.histograms[metric_name].clear()
                if metric_name in self.timers:
                    self.timers[metric_name].clear()
            else:
                self.metrics.clear()
                self.counters.clear()
                self.gauges.clear()
                self.histograms.clear()
                self.timers.clear()

        self.logger.info(f"Cleared metrics: {metric_name or 'all'}")


# Global performance monitor instance
performance_monitor = PerformanceMonitor()


# Performance monitoring context managers
@contextmanager
def performance_timer(operation_name: str, **tags):
    """Context manager for timing operations."""
    start_time = time.time()
    try:
        yield
    finally:
        duration_ms = (time.time() - start_time) * 1000
        performance_monitor.record_timer(f"operation.{operation_name}", duration_ms, **tags)


@contextmanager
def counter_monitor(counter_name: str, **tags):
    """Context manager for counting operations."""
    performance_monitor.record_counter(counter_name, **tags)
    try:
        yield
    finally:
        pass


# Default performance thresholds
DEFAULT_THRESHOLDS = [
    PerformanceThreshold(
        metric_name="system.memory.usage_percent",
        warning_threshold=80.0,
        critical_threshold=90.0
    ),
    PerformanceThreshold(
        metric_name="system.cpu.usage_percent",
        warning_threshold=70.0,
        critical_threshold=85.0
    ),
    PerformanceThreshold(
        metric_name="system.disk.usage_percent",
        warning_threshold=80.0,
        critical_threshold=90.0
    ),
    PerformanceThreshold(
        metric_name="process.memory.rss_mb",
        warning_threshold=1024.0,  # 1GB
        critical_threshold=2048.0   # 2GB
    ),
    PerformanceThreshold(
        metric_name="operation.order_submission",
        warning_threshold=100.0,  # 100ms
        critical_threshold=500.0   # 500ms
    ),
    PerformanceThreshold(
        metric_name="operation.market_data_processing",
        warning_threshold=50.0,   # 50ms
        critical_threshold=200.0   # 200ms
    ),
    PerformanceThreshold(
        metric_name="operation.risk_calculation",
        warning_threshold=10.0,   # 10ms
        critical_threshold=50.0    # 50ms
    )
]
