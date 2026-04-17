import json
from datetime import datetime
from typing import Any, Dict
from ..core.context import engine_context

class TelemetrySystem:
    """
    Standardized telemetry system for institutional observability.
    Collects heartbeats, performance metrics, and trading events.
    """
    def __init__(self):
        self.logger = engine_context.get_logger("telemetry")

    def emit_heartbeat(self, status: str = "HEALTHY", metadata: Dict[str, Any] = None):
        """Emits a system heartbeat with health status."""
        self.logger.info(
            f"Heartbeat: {status}",
            event_type="heartbeat",
            status=status,
            **(metadata or {})
        )

    def log_trade_event(self, symbol: str, side: str, quantity: float, price: float):
        """Logs a standardized trade event for audit and real-time monitoring."""
        self.logger.info(
            f"TELEMETRY_TRADE: {side} {quantity} {symbol} @ {price}",
            event_type="trade",
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price
        )

    def log_performance(self, metric_name: str, value: float, units: str = "ms"):
        """Logs a performance metric (e.g., latency, execution time)."""
        self.logger.info(
            f"METRIC {metric_name}: {value}{units}",
            event_type="performance_metric",
            metric_name=metric_name,
            value=value,
            units=units
        )

    def log_pnl(self, equity: float, pnl_day: float):
        """Logs real-time PnL and equity for dashboard hooks."""
        self.logger.info(
            f"TELEMETRY_PNL: Equity={equity:,.2f} DayPnL={pnl_day:,.2f}",
            event_type="pnl_update",
            equity=equity,
            pnl_day=pnl_day
        )
