
import logging
import time
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Try importing prometheus_client
try:
    from prometheus_client import Counter, Histogram, Gauge, start_http_server
    PROM_AVAILABLE = True
except ImportError:
    PROM_AVAILABLE = False
    logger.warning("prometheus_client not found. Metrics will be logged only.")

class PrometheusMetrics:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PrometheusMetrics, cls).__new__(cls)
            cls._instance._init_metrics()
        return cls._instance

    def _init_metrics(self):
        if not PROM_AVAILABLE:
            return

        # Counters
        self.decisions_total = Counter('decisions_total', 'Total decisions made', ['status'])
        self.provider_failures = Counter('data_provider_failures_total', 'FAILURES by provider', ['provider'])

        # Gauges
        self.universe_size = Gauge('universe_evaluated', 'Number of symbols in cycle')
        self.quota_usage = Gauge('provider_quota_usage_pct', 'Percentage of monthly quota used', ['provider'])

        # Histograms
        self.cycle_duration = Histogram('cycle_duration_seconds', 'Time taken for full cycle')
        self.conviction = Histogram('decision_conviction', 'Conviction score distribution')
        self.sigma = Histogram('decision_sigma', 'Uncertainty level distribution')

    def start_server(self, port=8000):
        if PROM_AVAILABLE:
            try:
                start_http_server(port)
                logger.info(f"Prometheus metrics exposed on port {port}")
            except Exception as e:
                logger.warning(f"Failed to start Prometheus server: {e}")

    # Helper methods that work regardless of PROM_AVAILABLE
    def inc_decision(self, status: str):
        if PROM_AVAILABLE:
            self.decisions_total.labels(status=status).inc()

    def inc_provider_failure(self, provider: str):
        if PROM_AVAILABLE:
            self.provider_failures.labels(provider=provider).inc()

    def set_universe_size(self, size: int):
        if PROM_AVAILABLE:
            self.universe_size.set(size)

    def observe_cycle_duration(self, seconds: float):
        if PROM_AVAILABLE:
            self.cycle_duration.observe(seconds)

    def set_quota_usage(self, provider: str, pct: float):
        if PROM_AVAILABLE:
            self.quota_usage.labels(provider=provider).set(pct)

# Global Instance
metrics = PrometheusMetrics()
