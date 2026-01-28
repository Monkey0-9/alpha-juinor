"""
monitor/prometheus_exporter.py
P2-2: Prometheus Metrics Exporter
"""
import logging
from typing import Dict, Any

try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server, REGISTRY
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    Counter = Gauge = Histogram = None

logger = logging.getLogger(__name__)


class PrometheusExporter:
    """
    Export metrics to Prometheus.

    Exposes /metrics endpoint on port 8000 by default.

    Usage:
        exporter = PrometheusExporter()
        exporter.start(port=8000)
        exporter.update_metrics(metrics_dict)
    """

    def __init__(self):
        if not PROMETHEUS_AVAILABLE:
            logger.warning(
                "prometheus_client not installed. Install with: pip install prometheus-client"
            )
            self.enabled = False
            return

        self.enabled = True

        # Define metrics
        self.heartbeat_uptime = Gauge(
            'heartbeat_uptime_seconds',
            'System uptime in seconds'
        )

        self.symbols_active = Gauge(
            'symbols_active',
            'Number of active symbols'
        )

        self.cycles_total = Counter(
            'cycles_total',
            'Total number of trading cycles'
        )

        self.model_errors_total = Counter(
            'model_errors_total',
            'Total ML model errors'
        )

        self.arima_fallbacks_total = Counter(
            'arima_fallbacks_total',
            'Total ARIMA fallbacks to EWMA'
        )

        self.inference_latency_seconds = Histogram(
            'inference_latency_seconds',
            'ML inference latency in seconds',
            buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0)
        )

        self.positions_active = Gauge(
            'positions_active',
            'Number of active positions'
        )

        self.system_state = Gauge(
            'system_state',
            'System state (0=OK, 1=DEGRADED, 2=HALTED)'
        )

        self.ml_state = Gauge(
            'ml_state',
            'ML state (0=OK, 1=DEGRADED, 2=DISABLED)'
        )

        logger.info("[PROMETHEUS] Metrics initialized")

    def start(self, port: int = 8000):
        """Start HTTP server for /metrics endpoint."""
        if not self.enabled:
            logger.info("[PROMETHEUS] Disabled - metrics not available")
            return

        try:
            start_http_server(port)
            logger.info(f"[PROMETHEUS] Metrics server started on port {port}")
            logger.info(f"[PROMETHEUS] Access metrics at http://localhost:{port}/metrics")
        except Exception as e:
            logger.error(f"[PROMETHEUS] Failed to start server: {e}")

    def update_metrics(self, metrics: Dict[str, Any]):
        """
        Update Prometheus metrics from system metrics dict.

        Args:
            metrics: Dict with keys like uptime_sec, symbols_count, cycles,
                    model_errors, arima_fallbacks, active_positions, etc.
        """
        if not self.enabled:
            return

        try:
            # Update gauges
            self.heartbeat_uptime.set(metrics.get('uptime_sec', 0))
            self.symbols_active.set(metrics.get('symbols_count', 0))
            self.positions_active.set(metrics.get('active_positions', 0))

            # Update counters (increment by delta)
            # Note: In production, track deltas properly
            # For now, set absolute values (works for demo)

            # Map system state to numeric
            state_map = {"OK": 0, "DEGRADED": 1, "HALTED": 2}
            self.system_state.set(state_map.get(metrics.get('system_state', 'OK'), 0))

            # Map ML state to numeric
            ml_state_str = metrics.get('ml_state', 'ENABLED|OK')
            if 'OK' in ml_state_str:
                ml_val = 0
            elif 'DEGRADED' in ml_state_str:
                ml_val = 1
            else:
                ml_val = 2
            self.ml_state.set(ml_val)

            # Latency histogram (would need actual timing data)
            # Placeholder - in production, use actual measured latencies
            if 'latency_p50' in metrics:
                self.inference_latency_seconds.observe(metrics['latency_p50'] / 1000.0)

        except Exception as e:
            logger.error(f"[PROMETHEUS] Failed to update metrics: {e}")


# Global exporter instance
prometheus_exporter = PrometheusExporter()
