"""
Prometheus Metrics & Grafana Dashboard
=======================================
Production observability for trading systems:
- Prometheus metrics exporter
- Grafana dashboard config generator
- Key performance indicators
- Real-time alerting rules
"""

import logging
import time
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class TradingMetrics:
    """
    Prometheus-compatible metrics for trading system.

    Metrics:
    - Order latency (histogram)
    - Fill rate (gauge)
    - Slippage (histogram)
    - P&L (gauge)
    - Position count (gauge)
    - Risk utilization (gauge)
    - Data feed health (gauge)
    - Exchange connectivity (gauge)
    """

    def __init__(self):
        self._counters: Dict[str, float] = defaultdict(
            float
        )
        self._gauges: Dict[str, float] = defaultdict(
            float
        )
        self._histograms: Dict[str, list] = defaultdict(
            list
        )
        self._labels: Dict[str, Dict[str, str]] = {}
        self._started = datetime.utcnow()

    # Counters
    def inc_orders_submitted(self, broker: str = ""):
        self._counters[f"orders_submitted_{broker}"] += 1

    def inc_orders_filled(self, broker: str = ""):
        self._counters[f"orders_filled_{broker}"] += 1

    def inc_orders_rejected(self, broker: str = ""):
        self._counters[f"orders_rejected_{broker}"] += 1

    def inc_risk_gate_blocked(self, gate: str = ""):
        self._counters[f"risk_gate_blocked_{gate}"] += 1

    # Gauges
    def set_total_equity(self, value: float):
        self._gauges["total_equity"] = value

    def set_daily_pnl(self, value: float):
        self._gauges["daily_pnl"] = value

    def set_position_count(self, value: int):
        self._gauges["position_count"] = value

    def set_gross_leverage(self, value: float):
        self._gauges["gross_leverage"] = value

    def set_net_leverage(self, value: float):
        self._gauges["net_leverage"] = value

    def set_max_drawdown(self, value: float):
        self._gauges["max_drawdown"] = value

    def set_sharpe_ratio(self, value: float):
        self._gauges["sharpe_ratio"] = value

    def set_exchange_connected(
        self, exchange: str, connected: bool
    ):
        val = 1.0 if connected else 0.0
        self._gauges[f"exchange_connected_{exchange}"] = val

    def set_data_feed_healthy(
        self, feed: str, healthy: bool
    ):
        val = 1.0 if healthy else 0.0
        self._gauges[f"data_feed_healthy_{feed}"] = val

    # Histograms
    def observe_order_latency(self, ms: float):
        self._histograms["order_latency_ms"].append(ms)

    def observe_slippage(self, bps: float):
        self._histograms["slippage_bps"].append(bps)

    def observe_fill_ratio(self, ratio: float):
        self._histograms["fill_ratio"].append(ratio)

    # Export
    def to_prometheus(self) -> str:
        """Export metrics in Prometheus text format."""
        lines = []
        lines.append(
            f"# HELP uptime_seconds "
            f"System uptime in seconds"
        )
        uptime = (
            datetime.utcnow() - self._started
        ).total_seconds()
        lines.append(f"uptime_seconds {uptime:.0f}")

        for name, value in self._counters.items():
            lines.append(f"# TYPE {name} counter")
            lines.append(f"{name} {value}")

        for name, value in self._gauges.items():
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name} {value}")

        for name, values in self._histograms.items():
            if values:
                avg = sum(values) / len(values)
                lines.append(f"# TYPE {name} summary")
                lines.append(
                    f'{name}{{quantile="0.5"}} '
                    f"{sorted(values)[len(values)//2]}"
                )
                lines.append(
                    f'{name}{{quantile="0.99"}} '
                    f"{sorted(values)[int(len(values)*0.99)]}"
                )
                lines.append(f"{name}_count {len(values)}")
                lines.append(f"{name}_sum {sum(values):.2f}")

        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """Export as dictionary."""
        return {
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "histograms": {
                k: {
                    "count": len(v),
                    "avg": sum(v) / len(v) if v else 0,
                    "min": min(v) if v else 0,
                    "max": max(v) if v else 0,
                }
                for k, v in self._histograms.items()
            },
        }


def generate_grafana_dashboard() -> Dict:
    """
    Generate Grafana dashboard JSON for import.
    Covers: P&L, risk, execution, connectivity.
    """
    return {
        "dashboard": {
            "title": "Mini-Quant-Fund | Live Trading",
            "uid": "mqf-live-v1",
            "timezone": "utc",
            "refresh": "10s",
            "panels": [
                {
                    "title": "Total Equity",
                    "type": "stat",
                    "gridPos": {
                        "x": 0, "y": 0,
                        "w": 6, "h": 4,
                    },
                    "targets": [{
                        "expr": "total_equity",
                    }],
                },
                {
                    "title": "Daily P&L",
                    "type": "stat",
                    "gridPos": {
                        "x": 6, "y": 0,
                        "w": 6, "h": 4,
                    },
                    "targets": [{
                        "expr": "daily_pnl",
                    }],
                },
                {
                    "title": "Gross Leverage",
                    "type": "gauge",
                    "gridPos": {
                        "x": 12, "y": 0,
                        "w": 6, "h": 4,
                    },
                    "targets": [{
                        "expr": "gross_leverage",
                    }],
                    "fieldConfig": {
                        "defaults": {
                            "max": 3,
                            "thresholds": {
                                "steps": [
                                    {"value": 0, "color": "green"},
                                    {"value": 1, "color": "yellow"},
                                    {"value": 2, "color": "red"},
                                ],
                            },
                        },
                    },
                },
                {
                    "title": "Max Drawdown",
                    "type": "gauge",
                    "gridPos": {
                        "x": 18, "y": 0,
                        "w": 6, "h": 4,
                    },
                    "targets": [{
                        "expr": "max_drawdown",
                    }],
                },
                {
                    "title": "Order Latency (ms)",
                    "type": "timeseries",
                    "gridPos": {
                        "x": 0, "y": 4,
                        "w": 12, "h": 6,
                    },
                    "targets": [{
                        "expr": (
                            'order_latency_ms'
                            '{quantile="0.99"}'
                        ),
                    }],
                },
                {
                    "title": "Slippage (bps)",
                    "type": "timeseries",
                    "gridPos": {
                        "x": 12, "y": 4,
                        "w": 12, "h": 6,
                    },
                    "targets": [{
                        "expr": (
                            'slippage_bps'
                            '{quantile="0.5"}'
                        ),
                    }],
                },
                {
                    "title": "Exchange Connectivity",
                    "type": "table",
                    "gridPos": {
                        "x": 0, "y": 10,
                        "w": 12, "h": 4,
                    },
                    "targets": [{
                        "expr": (
                            'exchange_connected_'
                            '{exchange=~".+"}'
                        ),
                    }],
                },
                {
                    "title": "Risk Gate Blocks",
                    "type": "barchart",
                    "gridPos": {
                        "x": 12, "y": 10,
                        "w": 12, "h": 4,
                    },
                    "targets": [{
                        "expr": (
                            'risk_gate_blocked_'
                            '{gate=~".+"}'
                        ),
                    }],
                },
            ],
        },
        "overwrite": True,
    }


ALERTING_RULES = {
    "groups": [{
        "name": "trading_alerts",
        "interval": "30s",
        "rules": [
            {
                "alert": "HighDrawdown",
                "expr": "max_drawdown > 0.05",
                "for": "2m",
                "labels": {"severity": "critical"},
                "annotations": {
                    "summary": (
                        "Drawdown exceeds 5% limit"
                    ),
                },
            },
            {
                "alert": "ExchangeDown",
                "expr": (
                    'exchange_connected_'
                    '{exchange=~".+"} == 0'
                ),
                "for": "1m",
                "labels": {"severity": "critical"},
                "annotations": {
                    "summary": "Exchange disconnected",
                },
            },
            {
                "alert": "HighSlippage",
                "expr": (
                    'slippage_bps{quantile="0.99"} > 50'
                ),
                "for": "5m",
                "labels": {"severity": "warning"},
                "annotations": {
                    "summary": (
                        "P99 slippage exceeds 50bps"
                    ),
                },
            },
            {
                "alert": "OrderLatencyHigh",
                "expr": (
                    'order_latency_ms'
                    '{quantile="0.99"} > 1000'
                ),
                "for": "2m",
                "labels": {"severity": "warning"},
                "annotations": {
                    "summary": (
                        "P99 order latency > 1s"
                    ),
                },
            },
        ],
    }],
}


# Singleton
_metrics: Optional[TradingMetrics] = None


def get_metrics() -> TradingMetrics:
    """Get or create metrics instance."""
    global _metrics
    if _metrics is None:
        _metrics = TradingMetrics()
    return _metrics
