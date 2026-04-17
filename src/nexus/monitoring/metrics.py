import time
from typing import Dict, Any
from collections import deque

class MetricsCollector:
    """
    In-memory metrics accumulator for real-time monitoring and statistics.
    Provides snapshots of system health and trading performance.
    """
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.latencies = deque(maxlen=window_size)
        self.order_outcomes = {"success": 0, "failure": 0}
        self.cumulative_pnl = 0.0
        self.start_time = time.time()

    def record_latency(self, latency_ms: float):
        self.latencies.append(latency_ms)

    def record_order(self, success: bool):
        key = "success" if success else "failure"
        self.order_outcomes[key] += 1

    def update_pnl(self, pnl: float):
        self.cumulative_pnl += pnl

    def get_snapshot(self) -> Dict[str, Any]:
        """Returns a snapshot of current system metrics."""
        uptime = time.time() - self.start_time
        avg_latency = sum(self.latencies) / len(self.latencies) if self.latencies else 0
        
        total_orders = sum(self.order_outcomes.values())
        success_rate = (self.order_outcomes["success"] / total_orders) if total_orders > 0 else 1.0

        return {
            "uptime_seconds": round(uptime, 2),
            "avg_latency_ms": round(avg_latency, 4),
            "order_success_rate": round(success_rate, 4),
            "cumulative_pnl": round(self.cumulative_pnl, 2),
            "total_orders": total_orders
        }
