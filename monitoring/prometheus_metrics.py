"""
Prometheus Metrics for Institutional Trading System.
Tracks cycles, decisions, latency, provider failures, data quality, and performance.
"""

import logging
import time
from typing import Dict, Any, Optional
from collections import defaultdict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class MetricsCollector:
    """Collects and aggregates metrics for Prometheus export"""

    # Counters
    cycles_total: int = 0
    decisions_execute: int = 0
    decisions_hold: int = 0
    decisions_reject: int = 0
    decisions_error: int = 0

    # Provider failures
    provider_failures: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # Histograms (store raw values for percentile calculation)
    cycle_durations: list = field(default_factory=list)

    # Gauges
    avg_conviction: float = 0.0
    avg_quality_score: float = 0.0
    trade_fill_rate: float = 0.0
    avg_slippage_pct: float = 0.0

    # Data Quality Metrics (NEW)
    data_missing_days_total: int = 0
    price_history_rows_total: int = 0
    quality_failures_total: int = 0
    symbols_with_data: int = 0

    def record_cycle(self, duration_sec: float):
        """Record a completed cycle"""
        self.cycles_total += 1
        self.cycle_durations.append(duration_sec)
        # Keep only last 100 for memory efficiency
        if len(self.cycle_durations) > 100:
            self.cycle_durations.pop(0)

    def record_decision(self, decision_type: str):
        """Record a decision"""
        if decision_type == "EXECUTE":
            self.decisions_execute += 1
        elif decision_type == "HOLD":
            self.decisions_hold += 1
        elif decision_type == "REJECT":
            self.decisions_reject += 1
        elif decision_type == "ERROR":
            self.decisions_error += 1

    def record_provider_failure(self, provider: str):
        """Record a provider failure"""
        self.provider_failures[provider] += 1

    def update_gauges(self, conviction: float = None, quality: float = None,
                     fill_rate: float = None, slippage: float = None):
        """Update gauge metrics"""
        if conviction is not None:
            self.avg_conviction = conviction
        if quality is not None:
            self.avg_quality_score = quality
        if fill_rate is not None:
            self.trade_fill_rate = fill_rate
        if slippage is not None:
            self.avg_slippage_pct = slippage

    # Data Quality Metrics Methods (NEW)
    def update_quality_metrics(
        self,
        avg_quality_score: Optional[float] = None,
        data_missing_days: Optional[int] = None,
        price_history_rows: Optional[int] = None,
        symbols_with_data: Optional[int] = None,
        quality_failures: Optional[int] = None
    ):
        """Update data quality related metrics"""
        if avg_quality_score is not None:
            self.avg_quality_score = avg_quality_score
        if data_missing_days is not None:
            self.data_missing_days_total = data_missing_days
        if price_history_rows is not None:
            self.price_history_rows_total = price_history_rows
        if symbols_with_data is not None:
            self.symbols_with_data = symbols_with_data
        if quality_failures is not None:
            self.quality_failures_total = quality_failures

    def increment_data_missing_days(self, count: int = 1):
        """Increment missing days counter"""
        self.data_missing_days_total += count

    def add_price_history_rows(self, count: int):
        """Add to total price history rows"""
        self.price_history_rows_total += count

    def increment_quality_failures(self, count: int = 1):
        """Increment quality failures counter"""
        self.quality_failures_total += count

    def get_metrics_text(self) -> str:
        """
        Generate Prometheus text format metrics.
        Can be written to file or exposed via HTTP endpoint.
        """
        lines = []

        # Counters - Cycles
        lines.append(f"# HELP quant_cycles_total Total number of trading cycles")
        lines.append(f"# TYPE quant_cycles_total counter")
        lines.append(f"quant_cycles_total {self.cycles_total}")

        # Counters - Decisions
        lines.append(f"# HELP quant_decisions_total Total decisions by type")
        lines.append(f"# TYPE quant_decisions_total counter")
        lines.append(f'quant_decisions_total{{decision="EXECUTE"}} {self.decisions_execute}')
        lines.append(f'quant_decisions_total{{decision="HOLD"}} {self.decisions_hold}')
        lines.append(f'quant_decisions_total{{decision="REJECT"}} {self.decisions_reject}')
        lines.append(f'quant_decisions_total{{decision="ERROR"}} {self.decisions_error}')

        # Provider failures
        lines.append(f"# HELP quant_provider_failures_total Provider failures by provider")
        lines.append(f"# TYPE quant_provider_failures_total counter")
        for provider, count in self.provider_failures.items():
            lines.append(f'quant_provider_failures_total{{provider="{provider}"}} {count}')

        # Histogram (simplified - just show latest)
        if self.cycle_durations:
            latest_duration = self.cycle_durations[-1]
            lines.append(f"# HELP quant_cycle_duration_seconds Cycle duration")
            lines.append(f"# TYPE quant_cycle_duration_seconds gauge")
            lines.append(f"quant_cycle_duration_seconds {latest_duration:.3f}")

        # Gauges - Performance
        lines.append(f"# HELP quant_avg_conviction Average conviction score")
        lines.append(f"# TYPE quant_avg_conviction gauge")
        lines.append(f"quant_avg_conviction {self.avg_conviction:.4f}")

        lines.append(f"# HELP quant_trade_fill_rate Trade fill rate")
        lines.append(f"# TYPE quant_trade_fill_rate gauge")
        lines.append(f"quant_trade_fill_rate {self.trade_fill_rate:.4f}")

        lines.append(f"# HELP quant_avg_slippage_pct Average slippage percentage")
        lines.append(f"# TYPE quant_avg_slippage_pct gauge")
        lines.append(f"quant_avg_slippage_pct {self.avg_slippage_pct:.4f}")

        # Data Quality Metrics (NEW)
        lines.append(f"# HELP quant_avg_quality_score Average data quality score (0-1)")
        lines.append(f"# TYPE quant_avg_quality_score gauge")
        lines.append(f"quant_avg_quality_score {self.avg_quality_score:.4f}")

        lines.append(f"# HELP quant_data_missing_days_total Total missing trading days across all symbols")
        lines.append(f"# TYPE quant_data_missing_days_total counter")
        lines.append(f"quant_data_missing_days_total {self.data_missing_days_total}")

        lines.append(f"# HELP quant_price_history_rows_total Total rows in price_history_daily table")
        lines.append(f"# TYPE quant_price_history_rows_total counter")
        lines.append(f"quant_price_history_rows_total {self.price_history_rows_total}")

        lines.append(f"# HELP quant_quality_failures_total Total symbols with quality score < 0.6")
        lines.append(f"# TYPE quant_quality_failures_total counter")
        lines.append(f"quant_quality_failures_total {self.quality_failures_total}")

        lines.append(f"# HELP quant_symbols_with_data Number of symbols with price data")
        lines.append(f"# TYPE quant_symbols_with_data gauge")
        lines.append(f"quant_symbols_with_data {self.symbols_with_data}")

        return "\n".join(lines)

    def write_to_file(self, filepath: str = "runtime/metrics.prom"):
        """Write metrics to file in Prometheus format"""
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            f.write(self.get_metrics_text())
        logger.info(f"Metrics written to {filepath}")

    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary as dict"""
        return {
            "cycles_total": self.cycles_total,
            "decisions": {
                "EXECUTE": self.decisions_execute,
                "HOLD": self.decisions_hold,
                "REJECT": self.decisions_reject,
                "ERROR": self.decisions_error
            },
            "provider_failures": dict(self.provider_failures),
            "avg_conviction": self.avg_conviction,
            "avg_quality_score": self.avg_quality_score,
            "trade_fill_rate": self.trade_fill_rate,
            "avg_slippage_pct": self.avg_slippage_pct,
            # Data Quality Metrics
            "data_missing_days_total": self.data_missing_days_total,
            "price_history_rows_total": self.price_history_rows_total,
            "quality_failures_total": self.quality_failures_total,
            "symbols_with_data": self.symbols_with_data
        }


# Global metrics collector
_metrics = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector"""
    return _metrics


def record_cycle(duration_sec: float):
    """Convenience function to record a cycle"""
    _metrics.record_cycle(duration_sec)


def record_decision(decision_type: str):
    """Convenience function to record a decision"""
    _metrics.record_decision(decision_type)


def record_provider_failure(provider: str):
    """Convenience function to record a provider failure"""
    _metrics.record_provider_failure(provider)


def update_quality_metrics(
    avg_quality_score: float = None,
    data_missing_days: int = None,
    price_history_rows: int = None,
    symbols_with_data: int = None,
    quality_failures: int = None
):
    """Convenience function to update quality metrics"""
    _metrics.update_quality_metrics(
        avg_quality_score=avg_quality_score,
        data_missing_days=data_missing_days,
        price_history_rows=price_history_rows,
        symbols_with_data=symbols_with_data,
        quality_failures=quality_failures
    )

