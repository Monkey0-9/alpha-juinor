"""
Learning Feedback Loop - Trade→Research Connection
=====================================================

The CRITICAL missing link: connect trade results back to research.

This module:
1. Tags every trade with its triggering alpha signal
2. Tracks P&L attribution per signal
3. Detects signal decay in real-time
4. Feeds back to strategy_validator for kill switch decisions

The loop: Trade → Log → Analyze → Learn → Improve → Trade
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple, Set
from enum import Enum
from collections import defaultdict
import numpy as np
import threading
import json

logger = logging.getLogger(__name__)


class SignalCategory(Enum):
    """Categories of alpha signals."""
    OPTIONS_FLOW = "OPTIONS_FLOW"
    INSIDER_ACTIVITY = "INSIDER_ACTIVITY"
    ORDER_FLOW = "ORDER_FLOW"
    TECHNICAL = "TECHNICAL"
    REGIME = "REGIME"
    SMART_MONEY = "SMART_MONEY"
    PATTERN = "PATTERN"
    FUNDAMENTAL = "FUNDAMENTAL"
    ML_MODEL = "ML_MODEL"
    ENSEMBLE = "ENSEMBLE"


@dataclass
class SignalAttribution:
    """Attribution of a signal to a trade."""
    signal_name: str
    signal_category: SignalCategory
    signal_strength: float  # 0-1
    signal_timestamp: datetime

    # Additional context
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TradeRecord:
    """Complete record of a trade for feedback learning."""
    trade_id: str
    symbol: str
    side: str  # BUY, SELL

    # Entry
    entry_price: float
    entry_time: datetime
    quantity: int

    # Exit
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None

    # P&L
    realized_pnl: float = 0.0
    realized_pnl_pct: float = 0.0

    # Signal attribution
    primary_signal: Optional[SignalAttribution] = None
    secondary_signals: List[SignalAttribution] = field(default_factory=list)

    # Regime at entry
    regime_at_entry: str = "UNKNOWN"

    # Status
    is_closed: bool = False

    # Trade quality
    grade: str = "C"  # A, B, C, D, F


@dataclass
class SignalPerformance:
    """Performance metrics for a signal."""
    signal_name: str
    category: SignalCategory

    # Trade counts
    total_trades: int
    winning_trades: int
    losing_trades: int

    # P&L
    total_pnl: float
    avg_pnl_per_trade: float
    avg_win: float
    avg_loss: float

    # Ratios
    win_rate: float
    profit_factor: float
    expectancy: float

    # Trend
    recent_win_rate_30d: float
    recent_pnl_30d: float
    performance_trend: str  # IMPROVING, STABLE, DECLINING

    # Health
    health_score: float  # 0-100
    needs_review: bool
    should_disable: bool


class LearningFeedback:
    """
    Learning feedback loop connecting trades to research.

    Every closed trade teaches us something.
    This module ensures we learn.
    """

    def __init__(self):
        """Initialize the feedback loop."""
        self.trades: Dict[str, TradeRecord] = {}
        self.closed_trades: List[TradeRecord] = []

        # Performance by signal
        self.signal_pnl: Dict[str, List[float]] = defaultdict(list)
        self.signal_trades: Dict[str, List[TradeRecord]] = defaultdict(list)

        # Regime performance
        self.regime_performance: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )

        self._lock = threading.Lock()

        logger.info(
            "[FEEDBACK] Learning Feedback Loop initialized - "
            "EVERY TRADE TEACHES US"
        )

    def open_trade(
        self,
        trade_id: str,
        symbol: str,
        side: str,
        entry_price: float,
        quantity: int,
        primary_signal: SignalAttribution,
        secondary_signals: Optional[List[SignalAttribution]] = None,
        regime: str = "UNKNOWN",
        grade: str = "C"
    ) -> TradeRecord:
        """Record a new trade with its signal attribution."""
        record = TradeRecord(
            trade_id=trade_id,
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            entry_time=datetime.utcnow(),
            quantity=quantity,
            primary_signal=primary_signal,
            secondary_signals=secondary_signals or [],
            regime_at_entry=regime,
            grade=grade
        )

        with self._lock:
            self.trades[trade_id] = record

        logger.info(
            f"[FEEDBACK] Trade opened: {trade_id} | "
            f"Signal: {primary_signal.signal_name} | "
            f"Regime: {regime}"
        )

        return record

    def close_trade(
        self,
        trade_id: str,
        exit_price: float,
        realized_pnl: Optional[float] = None
    ) -> Optional[TradeRecord]:
        """Close a trade and record P&L attribution."""
        with self._lock:
            if trade_id not in self.trades:
                logger.warning(f"[FEEDBACK] Trade not found: {trade_id}")
                return None

            record = self.trades[trade_id]

            # Calculate P&L
            if record.side == "BUY":
                pnl = (exit_price - record.entry_price) * record.quantity
                pnl_pct = (exit_price - record.entry_price) / record.entry_price
            else:
                pnl = (record.entry_price - exit_price) * record.quantity
                pnl_pct = (record.entry_price - exit_price) / record.entry_price

            record.exit_price = exit_price
            record.exit_time = datetime.utcnow()
            record.realized_pnl = realized_pnl if realized_pnl else pnl
            record.realized_pnl_pct = pnl_pct
            record.is_closed = True

            # Attribute P&L to signal
            if record.primary_signal:
                signal_name = record.primary_signal.signal_name
                self.signal_pnl[signal_name].append(pnl_pct)
                self.signal_trades[signal_name].append(record)

                # Regime-specific attribution
                self.regime_performance[record.regime_at_entry][signal_name].append(pnl_pct)

            # Move to closed trades
            self.closed_trades.append(record)
            del self.trades[trade_id]

        logger.info(
            f"[FEEDBACK] Trade closed: {trade_id} | "
            f"P&L: {pnl_pct:.2%} | "
            f"Signal: {record.primary_signal.signal_name if record.primary_signal else 'N/A'}"
        )

        return record

    def get_signal_performance(self, signal_name: str) -> Optional[SignalPerformance]:
        """Get performance metrics for a signal."""
        with self._lock:
            if signal_name not in self.signal_trades:
                return None

            trades = self.signal_trades[signal_name]
            if not trades:
                return None

            # Calculate metrics
            pnls = [t.realized_pnl_pct for t in trades if t.is_closed]
            if not pnls:
                return None

            winning = [p for p in pnls if p > 0]
            losing = [p for p in pnls if p < 0]

            win_rate = len(winning) / len(pnls) if pnls else 0
            avg_win = np.mean(winning) if winning else 0
            avg_loss = np.mean(losing) if losing else 0

            total_pnl = sum(pnls)
            avg_pnl = np.mean(pnls)

            # Profit factor
            gross_profit = sum(winning) if winning else 0
            gross_loss = abs(sum(losing)) if losing else 0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 10

            # Expectancy
            expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

            # Recent performance (30 days)
            cutoff = datetime.utcnow() - timedelta(days=30)
            recent_trades = [t for t in trades if t.exit_time and t.exit_time >= cutoff]
            recent_pnls = [t.realized_pnl_pct for t in recent_trades]

            recent_win_rate = sum(1 for p in recent_pnls if p > 0) / len(recent_pnls) if recent_pnls else 0
            recent_pnl = sum(recent_pnls)

            # Trend detection
            if len(pnls) >= 20:
                first_half = pnls[:len(pnls)//2]
                second_half = pnls[len(pnls)//2:]

                first_avg = np.mean(first_half)
                second_avg = np.mean(second_half)

                if second_avg > first_avg * 1.2:
                    trend = "IMPROVING"
                elif second_avg < first_avg * 0.8:
                    trend = "DECLINING"
                else:
                    trend = "STABLE"
            else:
                trend = "STABLE"

            # Health score
            health = self._calculate_signal_health(
                win_rate, profit_factor, expectancy, trend, len(trades)
            )

            # Flags
            needs_review = health < 50 or trend == "DECLINING"
            should_disable = health < 30 or (win_rate < 0.3 and len(trades) > 20)

            category = trades[0].primary_signal.signal_category if trades[0].primary_signal else SignalCategory.TECHNICAL

            return SignalPerformance(
                signal_name=signal_name,
                category=category,
                total_trades=len(trades),
                winning_trades=len(winning),
                losing_trades=len(losing),
                total_pnl=total_pnl,
                avg_pnl_per_trade=avg_pnl,
                avg_win=avg_win,
                avg_loss=avg_loss,
                win_rate=win_rate,
                profit_factor=min(profit_factor, 10),
                expectancy=expectancy,
                recent_win_rate_30d=recent_win_rate,
                recent_pnl_30d=recent_pnl,
                performance_trend=trend,
                health_score=health,
                needs_review=needs_review,
                should_disable=should_disable
            )

    def _calculate_signal_health(
        self,
        win_rate: float,
        profit_factor: float,
        expectancy: float,
        trend: str,
        n_trades: int
    ) -> float:
        """Calculate signal health score (0-100)."""
        score = 50  # Base

        # Win rate contribution
        if win_rate >= 0.6:
            score += 20
        elif win_rate >= 0.5:
            score += 10
        elif win_rate >= 0.4:
            score += 0
        else:
            score -= 15

        # Profit factor
        if profit_factor >= 2.0:
            score += 15
        elif profit_factor >= 1.5:
            score += 10
        elif profit_factor >= 1.0:
            score += 0
        else:
            score -= 20

        # Expectancy
        if expectancy > 0.02:
            score += 15
        elif expectancy > 0.01:
            score += 10
        elif expectancy > 0:
            score += 5
        else:
            score -= 15

        # Trend
        if trend == "IMPROVING":
            score += 10
        elif trend == "DECLINING":
            score -= 15

        # Sample size penalty
        if n_trades < 10:
            score -= 10

        return max(0, min(100, score))

    def get_all_signal_performances(self) -> Dict[str, SignalPerformance]:
        """Get performance for all signals."""
        with self._lock:
            signals = list(self.signal_trades.keys())

        performances = {}
        for signal in signals:
            perf = self.get_signal_performance(signal)
            if perf:
                performances[signal] = perf

        return performances

    def get_regime_insights(self) -> Dict[str, Dict[str, Any]]:
        """Get insights about signal performance by regime."""
        insights = {}

        with self._lock:
            for regime, signal_perfs in self.regime_performance.items():
                regime_insights = {}

                for signal, pnls in signal_perfs.items():
                    if not pnls:
                        continue

                    regime_insights[signal] = {
                        "trades": len(pnls),
                        "win_rate": sum(1 for p in pnls if p > 0) / len(pnls),
                        "avg_pnl": np.mean(pnls),
                        "total_pnl": sum(pnls)
                    }

                if regime_insights:
                    insights[regime] = {
                        "signals": regime_insights,
                        "best_signal": max(
                            regime_insights.keys(),
                            key=lambda s: regime_insights[s]["avg_pnl"]
                        ) if regime_insights else None
                    }

        return insights

    def get_signals_to_disable(self) -> List[str]:
        """Get list of signals that should be disabled."""
        to_disable = []

        performances = self.get_all_signal_performances()
        for signal, perf in performances.items():
            if perf.should_disable:
                to_disable.append(signal)
                logger.warning(
                    f"[FEEDBACK] Signal should be DISABLED: {signal} | "
                    f"Health: {perf.health_score:.0f} | "
                    f"Win rate: {perf.win_rate:.0%}"
                )

        return to_disable

    def get_top_signals(self, n: int = 10) -> List[Tuple[str, SignalPerformance]]:
        """Get top performing signals."""
        performances = self.get_all_signal_performances()

        sorted_perfs = sorted(
            performances.items(),
            key=lambda x: (x[1].health_score, x[1].expectancy),
            reverse=True
        )

        return sorted_perfs[:n]

    def export_learning_data(self) -> Dict[str, Any]:
        """Export learning data for external analysis."""
        with self._lock:
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "total_trades": len(self.closed_trades),
                "active_trades": len(self.trades),
                "signals_tracked": len(self.signal_trades),
                "signal_performances": {
                    name: {
                        "trades": perf.total_trades,
                        "win_rate": perf.win_rate,
                        "expectancy": perf.expectancy,
                        "health": perf.health_score,
                        "trend": perf.performance_trend
                    }
                    for name, perf in self.get_all_signal_performances().items()
                },
                "regime_insights": self.get_regime_insights()
            }


# Singleton
_feedback: Optional[LearningFeedback] = None


def get_learning_feedback() -> LearningFeedback:
    """Get or create the Learning Feedback loop."""
    global _feedback
    if _feedback is None:
        _feedback = LearningFeedback()
    return _feedback
