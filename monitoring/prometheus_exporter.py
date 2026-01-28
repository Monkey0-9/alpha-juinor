
import logging
from prometheus_client import start_http_server, Gauge, Counter, Histogram, CollectorRegistry, REGISTRY
import time
from typing import Dict, Optional

logger = logging.getLogger("PrometheusExporter")

class PrometheusManager:
    """
    Singleton manager for Prometheus metrics.
    Exposes /metrics endpoint on port 8000.
    """
    _instance = None

    def __new__(cls, port: int = 8000):
        if cls._instance is None:
            cls._instance = super(PrometheusManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, port: int = 8000):
        if self._initialized:
            return

        self.port = port
        self.registry = REGISTRY

        # Define Metrics
        self.metrics = {}

        # NAV
        self.metrics['nav'] = Gauge('quant_fund_portfolio_nav', 'Current Portfolio NAV', ['cycle_id'])

        # Order Counts
        self.metrics['orders'] = Counter('quant_fund_order_count', 'Total Orders Generated', ['side', 'type'])

        # Latencies
        self.metrics['cycle_latency'] = Histogram('quant_fund_cycle_latency_seconds', 'Main Loop Cycle Duration')
        self.metrics['decision_latency'] = Histogram('quant_fund_decision_latency_seconds', 'Decision Logic Duration')

        # Provider Health
        self.metrics['provider_health'] = Gauge('quant_fund_provider_health', 'Provider Success Rate', ['provider'])

        try:
            start_http_server(self.port)
            logger.info(f"Prometheus Exporter started on port {self.port}")
        except Exception as e:
            logger.warning(f"Failed to start Prometheus server (maybe already running): {e}")

        self._initialized = True

    def set_nav(self, value: float, cycle_id: str = "latest"):
        self.metrics['nav'].labels(cycle_id=cycle_id).set(value)

    def inc_order(self, side: str, order_type: str = "market"):
        self.metrics['orders'].labels(side=side, type=order_type).inc()

    def set_provider_health(self, provider: str, score: float):
        self.metrics['provider_health'].labels(provider=provider).set(score)

    def observe_latency(self, metric_name: str, seconds: float):
        if metric_name in self.metrics:
            self.metrics[metric_name].observe(seconds)

# Global Instance
metrics = PrometheusManager()
