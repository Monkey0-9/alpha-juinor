"""
monitoring/trading_monitor.py

Comprehensive monitoring and logging for live and paper trading.
Tracks:
- Real-time trading metrics
- Order execution tracking
- Performance metrics
- Risk metrics
- System health
"""

import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TradeEvent:
    """Single trade event for logging."""

    timestamp: str
    symbol: str
    side: str  # BUY/SELL
    quantity: int
    price: float
    status: str  # SUBMITTED/FILLED/REJECTED
    reason: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PerformanceMetrics:
    """Performance metrics snapshot."""

    timestamp: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    max_drawdown: float
    sharpe_ratio: float
    current_portfolio_value: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SystemHealth:
    """System health metrics."""

    timestamp: str
    is_running: bool
    mode: str  # PAPER/LIVE
    active_connections: int
    data_health: float  # 0-1
    last_order_time: Optional[str]
    errors_last_hour: int

    def to_dict(self) -> dict:
        return asdict(self)


class TradingMonitor:
    """
    Real-time monitoring system for trading operations.
    """

    def __init__(
        self,
        log_dir: str = "runtime/monitoring",
        trading_mode: str = "paper",
        buffer_size: int = 10000,
    ):
        self.log_dir = Path(log_dir)
        self.trading_mode = trading_mode
        self.buffer_size = buffer_size

        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # In-memory buffers
        self.trades: List[TradeEvent] = []
        self.metrics_history: List[PerformanceMetrics] = []
        self.health_history: List[SystemHealth] = []

        # File paths
        self.trades_log = self.log_dir / f"trades_{trading_mode}.jsonl"
        self.metrics_log = self.log_dir / f"metrics_{trading_mode}.jsonl"
        self.health_log = self.log_dir / f"health_{trading_mode}.jsonl"
        self.summary_log = self.log_dir / f"summary_{trading_mode}.json"

        logger.info(f"[Monitor] Initialized: mode={trading_mode}, log_dir={log_dir}")

    # =========================================================================
    # TRADE LOGGING
    # =========================================================================

    def log_trade(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: float,
        status: str = "SUBMITTED",
        reason: Optional[str] = None,
    ):
        """Log a trade event."""
        event = TradeEvent(
            timestamp=datetime.utcnow().isoformat(),
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            status=status,
            reason=reason,
        )

        self.trades.append(event)

        # Trim buffer if too large
        if len(self.trades) > self.buffer_size:
            self.trades = self.trades[-self.buffer_size :]

        # Write to log file
        with open(self.trades_log, "a") as f:
            f.write(json.dumps(event.to_dict()) + "\n")

        logger.info(
            f"[Monitor] Trade {side} {quantity} {symbol} @ ${price:.2f} - {status}"
        )

    def log_rejected_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        reason: str,
    ):
        """Log a rejected order."""
        self.log_trade(
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=0.0,
            status="REJECTED",
            reason=reason,
        )
        logger.warning(f"[Monitor] Order rejected: {symbol} - {reason}")

    # =========================================================================
    # PERFORMANCE METRICS
    # =========================================================================

    def log_performance(
        self,
        total_trades: int,
        winning_trades: int,
        losing_trades: int,
        total_pnl: float,
        max_drawdown: float,
        sharpe_ratio: float,
        current_value: float,
    ):
        """Log performance metrics."""
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

        metrics = PerformanceMetrics(
            timestamp=datetime.utcnow().isoformat(),
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_pnl=total_pnl,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            current_portfolio_value=current_value,
        )

        self.metrics_history.append(metrics)

        # Trim buffer
        if len(self.metrics_history) > self.buffer_size // 100:
            self.metrics_history = self.metrics_history[-(self.buffer_size // 100) :]

        # Write to log
        with open(self.metrics_log, "a") as f:
            f.write(json.dumps(metrics.to_dict()) + "\n")

        logger.info(
            f"[Monitor] Performance: {total_trades} trades, "
            f"Win Rate: {win_rate:.1%}, PnL: ${total_pnl:,.0f}, "
            f"Sharpe: {sharpe_ratio:.2f}"
        )

    # =========================================================================
    # SYSTEM HEALTH
    # =========================================================================

    def log_health(
        self,
        is_running: bool,
        active_connections: int,
        data_health: float,
        last_order_time: Optional[str] = None,
        errors_last_hour: int = 0,
    ):
        """Log system health metrics."""
        health = SystemHealth(
            timestamp=datetime.utcnow().isoformat(),
            is_running=is_running,
            mode=self.trading_mode,
            active_connections=active_connections,
            data_health=data_health,
            last_order_time=last_order_time,
            errors_last_hour=errors_last_hour,
        )

        self.health_history.append(health)

        # Trim buffer
        if len(self.health_history) > self.buffer_size // 100:
            self.health_history = self.health_history[-(self.buffer_size // 100) :]

        # Write to log
        with open(self.health_log, "a") as f:
            f.write(json.dumps(health.to_dict()) + "\n")

        status_emoji = "✓" if is_running else "✗"
        logger.info(
            f"[Monitor] {status_emoji} Health: connections={active_connections}, "
            f"data_health={data_health:.1%}, errors_1h={errors_last_hour}"
        )

    # =========================================================================
    # STATISTICS & SUMMARIES
    # =========================================================================

    def get_trade_count(self) -> int:
        """Get total number of trades logged."""
        return len(self.trades)

    def get_filled_trades(self) -> List[TradeEvent]:
        """Get all filled trades."""
        return [t for t in self.trades if t.status == "FILLED"]

    def get_rejected_trades(self) -> List[TradeEvent]:
        """Get all rejected trades."""
        return [t for t in self.trades if t.status == "REJECTED"]

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        filled = self.get_filled_trades()
        rejected = self.get_rejected_trades()

        summary = {
            "timestamp": datetime.utcnow().isoformat(),
            "mode": self.trading_mode,
            "total_logged_trades": len(self.trades),
            "filled_trades": len(filled),
            "rejected_trades": len(rejected),
            "rejection_rate": len(rejected) / len(self.trades) if self.trades else 0.0,
            "latest_metrics": (
                asdict(self.metrics_history[-1]) if self.metrics_history else None
            ),
            "latest_health": (
                asdict(self.health_history[-1]) if self.health_history else None
            ),
        }

        # Write summary
        with open(self.summary_log, "w") as f:
            json.dump(summary, f, indent=2)

        return summary

    def print_summary(self):
        """Print human-readable summary to console."""
        summary = self.get_summary()

        print("\n" + "=" * 70)
        print(f"TRADING MONITOR - {self.trading_mode.upper()} MODE")
        print("=" * 70)
        print(f"Timestamp: {summary['timestamp']}")
        print(f"\nTrade Statistics:")
        print(f"  Total Trades: {summary['total_logged_trades']}")
        print(f"  Filled: {summary['filled_trades']}")
        print(f"  Rejected: {summary['rejected_trades']}")
        print(f"  Rejection Rate: {summary['rejection_rate']:.1%}")

        if summary["latest_metrics"]:
            m = summary["latest_metrics"]
            print(f"\nPerformance:")
            print(f"  Win Rate: {m['win_rate']:.1%}")
            print(f"  Total PnL: ${m['total_pnl']:,.0f}")
            print(f"  Sharpe Ratio: {m['sharpe_ratio']:.2f}")
            print(f"  Max Drawdown: {m['max_drawdown']:.2%}")
            print(f"  Portfolio Value: ${m['current_portfolio_value']:,.0f}")

        if summary["latest_health"]:
            h = summary["latest_health"]
            print(f"\nSystem Health:")
            print(f"  Running: {h['is_running']}")
            print(f"  Connections: {h['active_connections']}")
            print(f"  Data Health: {h['data_health']:.1%}")
            print(f"  Errors (1h): {h['errors_last_hour']}")

        print("=" * 70 + "\n")
