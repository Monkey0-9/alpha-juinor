"""
Elite Monitoring Dashboard Backend
==================================

Serves real-time metrics for:
- P&L and Risk
- System Latency & Health
- Alpha & Signal Performance
- Execution Quality

Integrates with Prometheus/Grafana via metrics exposure.
"""

import time
import logging
import threading
from typing import Dict, List, Any
from dataclasses import dataclass, field
from datetime import datetime
import json
import random

logger = logging.getLogger(__name__)

@dataclass
class SystemMetrics:
    """Real-time system health metrics."""
    cpu_usage: float
    memory_usage: float
    db_latency_ms: float
    order_latency_ms: float
    active_threads: int
    error_rate: float
    last_heartbeat: float

@dataclass
class BusinessMetrics:
    """Real-time trading metrics."""
    pnl_daily: float
    sharpe_running: float
    exposure_gross: float
    exposure_net: float
    risk_utilization: float
    active_orders: int
    open_positions: int

class MonitoringService:
    """
    Central monitoring service for the trading platform.
    """

    def __init__(self):
        self.system_metrics = SystemMetrics(0,0,0,0,0,0,0)
        self.business_metrics = BusinessMetrics(0,0,0,0,0,0,0)
        self.alerts = []
        self._running = False
        self._lock = threading.Lock()

    def start(self):
        """Start monitoring background threads."""
        self._running = True
        logger.info("Monitoring service started")
        # In a real system, would start Prometheus server here
        # start_http_server(8000)

    def record_latency(self, component: str, latency_ms: float):
        """Record latency metric."""
        with self._lock:
            if component == 'db':
                self.system_metrics.db_latency_ms = latency_ms
            elif component == 'order':
                self.system_metrics.order_latency_ms = latency_ms

    def update_pnl(self, pnl: float):
        """Update P&L metric."""
        with self._lock:
            self.business_metrics.pnl_daily = pnl

    def add_alert(self, level: str, message: str, component: str):
        """Register a new alert."""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'message': message,
            'component': component
        }
        self.alerts.append(alert)
        if len(self.alerts) > 100:
            self.alerts.pop(0)
        logger.warning(f"ALERT [{level}]: {message}")

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get snapshot for frontend dashboard."""
        with self._lock:
            return {
                'system': self.system_metrics.__dict__,
                'business': self.business_metrics.__dict__,
                'recent_alerts': self.alerts[-5:],
                'timestamp': time.time()
            }

# Global singleton
_monitor: MonitoringService = MonitoringService()

def get_monitor() -> MonitoringService:
    return _monitor
