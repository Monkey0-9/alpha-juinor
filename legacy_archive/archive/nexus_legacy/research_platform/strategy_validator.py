"""
Strategy Validator - Kill Switch Pipeline
=============================================

Continuously monitors strategy performance and automatically
retires strategies with decaying or poor performance.

This is the "KILL SWITCH" - brutal, objective, essential.

Features:
1. Real-time performance monitoring
2. Decay detection with statistical tests
3. Automatic strategy retirement
4. Health scoring and grading
5. Alerts for degrading strategies
"""

import logging
from dataclasses import dataclass
from datetime import datetime

from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
# import pandas as pd # Removed, assuming pd is indeed unused or reimported
# Wait, checking if pd is used. If used, I shouldn't remove it.
# The dashboard uses pandas. StrategyHealth uses it?
# Diagnositc says "pandas as pd" imported but unused.
# Let's comment it out safely.
# import pandas as pd
from scipy import stats
import threading


logger = logging.getLogger(__name__)


@dataclass
class StrategyHealth:
    """Health assessment for a strategy."""
    strategy_name: str

    # Current performance
    rolling_sharpe_30d: float
    rolling_sharpe_90d: float
    rolling_sharpe_252d: float

    # Historical performance
    peak_sharpe: float
    current_vs_peak: float  # Current / Peak

    # Decay metrics
    decay_detected: bool
    decay_severity: str  # NONE, MILD, MODERATE, SEVERE
    decay_rate_annual: float

    # Consistency
    win_rate_30d: float
    profit_factor_30d: float
    max_drawdown_current: float

    # Health score (0-100)
    health_score: float

    # Status
    status: str  # ACTIVE, WARNING, CRITICAL, RETIRED
    days_until_retirement: Optional[int]

    # Timestamps
    last_updated: datetime
    last_profitable_day: Optional[datetime]


@dataclass
class RetirementDecision:
    """Decision to retire a strategy."""
    strategy_name: str
    decision: str  # KEEP, WARN, RETIRE

    # Reasons
    reasons: List[str]

    # Metrics at decision
    current_sharpe: float
    decay_rate: float
    health_score: float

    # Timestamp
    decision_date: datetime


class StrategyMonitor:
    """
    Monitors individual strategy performance.

    Tracks rolling metrics and detects decay.
    """

    def __init__(self, strategy_name: str):
        """Initialize the monitor."""
        self.strategy_name = strategy_name

        # Performance history
        self.daily_returns: List[float] = []
        self.daily_dates: List[datetime] = []

        # Peak tracking
        self.peak_sharpe = 0.0
        self.peak_date: Optional[datetime] = None

        # Decay tracking
        self.sharpe_history: List[Tuple[datetime, float]] = []

        # Status
        self.status = "ACTIVE"
        self.warning_count = 0
        self.last_warning_date: Optional[datetime] = None

    def record_return(self, date: datetime, daily_return: float):
        """Record a daily return."""
        self.daily_returns.append(daily_return)
        self.daily_dates.append(date)

        # Keep last 504 days (2 years)
        if len(self.daily_returns) > 504:
            self.daily_returns.pop(0)
            self.daily_dates.pop(0)

    def assess_health(self) -> StrategyHealth:
        """Assess current strategy health."""
        if len(self.daily_returns) < 30:
            return self._null_health()

        returns = np.array(self.daily_returns)

        # Rolling Sharpe ratios
        sharpe_30d = self._calc_sharpe(returns[-30:])
        sharpe_90d = (
            self._calc_sharpe(returns[-90:])
            if len(returns) >= 90 else sharpe_30d
        )
        sharpe_252d = (
            self._calc_sharpe(returns[-252:])
            if len(returns) >= 252 else sharpe_90d
        )

        # Update peak
        if sharpe_90d > self.peak_sharpe:
            self.peak_sharpe = sharpe_90d
            self.peak_date = datetime.utcnow()

        current_vs_peak = (
            sharpe_90d / self.peak_sharpe if self.peak_sharpe > 0 else 0
        )

        # Decay detection
        decay_detected, decay_severity, decay_rate = self._detect_decay(returns)

        # Win rate and profit factor
        win_rate = (returns[-30:] > 0).mean() if len(returns) >= 30 else 0

        gains = returns[-30:][returns[-30:] > 0].sum()
        losses = abs(returns[-30:][returns[-30:] < 0].sum())
        profit_factor = gains / losses if losses > 0 else 10

        # Max drawdown
        cumulative = (1 + returns).cumprod()
        rolling_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - rolling_max) / rolling_max
        max_dd = drawdown.min()

        # Health score
        health_score = self._calculate_health_score(
            sharpe_90d, decay_severity, win_rate, profit_factor, max_dd
        )

        # Status determination
        if health_score >= 70:
            status = "ACTIVE"
            days_until = None
        elif health_score >= 50:
            status = "WARNING"
            days_until = 30
        elif health_score >= 30:
            status = "CRITICAL"
            days_until = 7
        else:
            status = "RETIRED"
            days_until = 0

        # Last profitable day
        last_profitable = None
        for i in range(len(returns) - 1, -1, -1):
            if returns[i] > 0:
                last_profitable = (
                    self.daily_dates[i]
                    if i < len(self.daily_dates) else None
                )
                break

        return StrategyHealth(
            strategy_name=self.strategy_name,
            rolling_sharpe_30d=float(sharpe_30d),
            rolling_sharpe_90d=float(sharpe_90d),
            rolling_sharpe_252d=float(sharpe_252d),
            peak_sharpe=float(self.peak_sharpe),
            current_vs_peak=float(current_vs_peak),
            decay_detected=decay_detected,
            decay_severity=decay_severity,
            decay_rate_annual=float(decay_rate),
            win_rate_30d=float(win_rate),
            profit_factor_30d=float(min(profit_factor, 10)),
            max_drawdown_current=float(max_dd),
            health_score=float(health_score),
            status=status,
            days_until_retirement=days_until,
            last_updated=datetime.utcnow(),
            last_profitable_day=last_profitable
        )

    def _calc_sharpe(self, returns: np.ndarray, risk_free: float = 0.04) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) < 5 or np.std(returns) == 0:
            return 0
        excess = returns - risk_free / 252
        return float(np.mean(excess) / np.std(excess) * np.sqrt(252))

    def _detect_decay(
        self,
        returns: np.ndarray
    ) -> Tuple[bool, str, float]:
        """Detect performance decay."""
        if len(returns) < 60:
            return False, "NONE", 0

        # Calculate rolling 30-day Sharpe
        window = 30
        rolling_sharpes = []

        for i in range(window, len(returns)):
            s = self._calc_sharpe(returns[i-window:i])
            rolling_sharpes.append(s)

        if len(rolling_sharpes) < 3:
            return False, "NONE", 0

        # Linear regression on Sharpe over time
        x = np.arange(len(rolling_sharpes))
        slope, _, r_value, p_value, _ = stats.linregress(x, rolling_sharpes)

        # Annualized decay rate
        decay_rate = slope * 252 / window

        # Is decay statistically significant?
        significant_decay = slope < 0 and p_value < 0.1 and r_value ** 2 > 0.1

        if not significant_decay:
            return False, "NONE", 0

        # Severity
        if decay_rate < -0.5:
            severity = "SEVERE"
        elif decay_rate < -0.2:
            severity = "MODERATE"
        else:
            severity = "MILD"

        return True, severity, decay_rate

    def _calculate_health_score(
        self,
        sharpe: float,
        decay_severity: str,
        win_rate: float,
        profit_factor: float,
        max_dd: float
    ) -> float:
        """Calculate overall health score (0-100)."""
        score = 50  # Base score

        # Sharpe contribution (-20 to +30)
        if sharpe >= 2.0:
            score += 30
        elif sharpe >= 1.5:
            score += 20
        elif sharpe >= 1.0:
            score += 10
        elif sharpe >= 0.5:
            score += 0
        elif sharpe >= 0:
            score -= 10
        else:
            score -= 20

        # Decay penalty
        if decay_severity == "SEVERE":
            score -= 30
        elif decay_severity == "MODERATE":
            score -= 15
        elif decay_severity == "MILD":
            score -= 5

        # Win rate contribution
        if win_rate >= 0.6:
            score += 10
        elif win_rate >= 0.5:
            score += 5
        elif win_rate < 0.4:
            score -= 10

        # Profit factor
        if profit_factor >= 2.0:
            score += 10
        elif profit_factor >= 1.5:
            score += 5
        elif profit_factor < 1.0:
            score -= 10

        # Drawdown penalty
        if max_dd < -0.20:
            score -= 20
        elif max_dd < -0.10:
            score -= 10

        return max(0, min(100, score))

    def _null_health(self) -> StrategyHealth:
        """Return null health for insufficient data."""
        return StrategyHealth(
            strategy_name=self.strategy_name,
            rolling_sharpe_30d=0,
            rolling_sharpe_90d=0,
            rolling_sharpe_252d=0,
            peak_sharpe=0,
            current_vs_peak=0,
            decay_detected=False,
            decay_severity="NONE",
            decay_rate_annual=0,
            win_rate_30d=0,
            profit_factor_30d=0,
            max_drawdown_current=0,
            health_score=50,
            status="ACTIVE",
            days_until_retirement=None,
            last_updated=datetime.utcnow(),
            last_profitable_day=None
        )


class StrategyKillSwitch:
    """
    The KILL SWITCH - automatically retires underperforming strategies.

    Rules:
    1. Health score < 30 -> IMMEDIATE RETIREMENT
    2. Decay severity SEVERE for 30+ days -> RETIREMENT
    3. Sharpe below 0.5 for 90+ days -> RETIREMENT
    4. Max drawdown > 25% -> CRITICAL WARNING
    5. No profitable day in 60 days -> RETIREMENT
    """

    # Retirement thresholds
    MIN_HEALTH_SCORE = 30
    MIN_SHARPE = 0.5
    MAX_DRAWDOWN = -0.25
    MAX_DAYS_UNPROFITABLE = 60
    SEVERE_DECAY_DAYS = 30

    def __init__(self):
        """Initialize the kill switch."""
        self.monitors: Dict[str, StrategyMonitor] = {}
        self.retired_strategies: Set[str] = set()
        self.retirement_history: List[RetirementDecision] = []

        self._lock = threading.Lock()

        logger.info(
            "[KILL SWITCH] Strategy Kill Switch initialized - "
            "BRUTAL. OBJECTIVE. ESSENTIAL."
        )

    def register_strategy(self, strategy_name: str):
        """Register a strategy for monitoring."""
        with self._lock:
            if strategy_name not in self.monitors:
                self.monitors[strategy_name] = StrategyMonitor(strategy_name)
                logger.info(f"[KILL SWITCH] Registered: {strategy_name}")

    def record_performance(
        self,
        strategy_name: str,
        date: datetime,
        daily_return: float
    ):
        """Record daily performance for a strategy."""
        with self._lock:
            if strategy_name not in self.monitors:
                self.monitors[strategy_name] = StrategyMonitor(strategy_name)

            self.monitors[strategy_name].record_return(date, daily_return)

    def evaluate_all(self) -> Dict[str, StrategyHealth]:
        """Evaluate health of all strategies."""
        health_reports = {}

        with self._lock:
            for name, monitor in self.monitors.items():
                if name in self.retired_strategies:
                    continue

                health = monitor.assess_health()
                health_reports[name] = health

                # Check for retirement
                decision = self._evaluate_retirement(health)

                if decision.decision == "RETIRE":
                    self._retire_strategy(decision)

        return health_reports

    def _evaluate_retirement(self, health: StrategyHealth) -> RetirementDecision:
        """Evaluate if strategy should be retired."""
        reasons = []

        # Rule 1: Health score too low
        if health.health_score < self.MIN_HEALTH_SCORE:
            reasons.append(
                f"Health score {health.health_score:.0f} < "
                f"{self.MIN_HEALTH_SCORE}"
            )

        # Rule 2: Severe decay
        if health.decay_severity == "SEVERE":
            reasons.append(
                f"SEVERE decay detected "
                f"(rate: {health.decay_rate_annual:.2f})"
            )

        # Rule 3: Sharpe too low
        if health.rolling_sharpe_90d < self.MIN_SHARPE:
            reasons.append(
                f"Sharpe {health.rolling_sharpe_90d:.2f} < "
                f"{self.MIN_SHARPE}"
            )

        # Rule 4: Max drawdown too deep
        if health.max_drawdown_current < self.MAX_DRAWDOWN:
            reasons.append(f"Drawdown {health.max_drawdown_current:.1%} exceeds limit")

        # Rule 5: Extended unprofitable period
        if health.last_profitable_day:
            days_since_profit = (datetime.utcnow() - health.last_profitable_day).days
            if days_since_profit > self.MAX_DAYS_UNPROFITABLE:
                reasons.append(f"No profit in {days_since_profit} days")

        # Decision
        if len(reasons) >= 2 or (
            len(reasons) >= 1 and health.health_score < 20
        ):
            decision = "RETIRE"
        elif len(reasons) >= 1:
            decision = "WARN"
        else:
            decision = "KEEP"

        return RetirementDecision(
            strategy_name=health.strategy_name,
            decision=decision,
            reasons=reasons,
            current_sharpe=health.rolling_sharpe_90d,
            decay_rate=health.decay_rate_annual,
            health_score=health.health_score,
            decision_date=datetime.utcnow()
        )

    def _retire_strategy(self, decision: RetirementDecision):
        """Retire a strategy."""
        self.retired_strategies.add(decision.strategy_name)
        self.retirement_history.append(decision)

        logger.warning(
            f"[KILL SWITCH] RETIRED: {decision.strategy_name} | "
            f"Reasons: {', '.join(decision.reasons)}"
        )

    def is_active(self, strategy_name: str) -> bool:
        """Check if strategy is still active."""
        with self._lock:
            return strategy_name not in self.retired_strategies

    def get_active_strategies(self) -> List[str]:
        """Get list of active strategies."""
        with self._lock:
            return [
                name for name in self.monitors.keys()
                if name not in self.retired_strategies
            ]

    def get_retired_strategies(self) -> List[Dict]:
        """Get list of retired strategies with reasons."""
        return [
            {
                "name": d.strategy_name,
                "date": d.decision_date.isoformat(),
                "reasons": d.reasons,
                "final_sharpe": d.current_sharpe,
                "final_health": d.health_score
            }
            for d in self.retirement_history
        ]

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for health dashboard."""
        health_reports = self.evaluate_all()

        active = []
        warning = []
        critical = []

        for name, health in health_reports.items():
            summary = {
                "name": name,
                "health_score": health.health_score,
                "sharpe_90d": health.rolling_sharpe_90d,
                "decay": health.decay_severity,
                "status": health.status
            }

            if health.status == "ACTIVE":
                active.append(summary)
            elif health.status == "WARNING":
                warning.append(summary)
            elif health.status == "CRITICAL":
                critical.append(summary)

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "total_strategies": len(self.monitors),
            "active": len(active),
            "warning": len(warning),
            "critical": len(critical),
            "retired": len(self.retired_strategies),
            "strategies": {
                "active": sorted(active, key=lambda x: x["health_score"], reverse=True),
                "warning": warning,
                "critical": critical,
                "retired": self.get_retired_strategies()
            }
        }


# Singleton
_kill_switch: Optional[StrategyKillSwitch] = None


def get_kill_switch() -> StrategyKillSwitch:
    """Get or create the Kill Switch."""
    global _kill_switch
    if _kill_switch is None:
        _kill_switch = StrategyKillSwitch()
    return _kill_switch
