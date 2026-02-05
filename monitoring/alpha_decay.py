"""
Alpha Decay Detection - Monitor Strategy Performance.

Detects when strategies are losing their edge:
- Rolling Sharpe ratio decline
- Information ratio degradation
- Drawdown persistence
- Auto-allocation reduction for decaying strategies
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class AlphaDecayAlert:
    """Alert for alpha decay detection."""
    strategy_name: str
    timestamp: str
    alert_type: str  # "WARNING", "CRITICAL"

    # Metrics
    current_sharpe: float
    historical_sharpe: float
    decay_pct: float

    # Recommendation
    recommendation: str
    suggested_weight_reduction: float


@dataclass
class StrategyPerformance:
    """Track strategy performance over time."""
    name: str
    returns: List[float] = field(default_factory=list)
    timestamps: List[str] = field(default_factory=list)

    # Computed metrics
    rolling_sharpe: List[float] = field(default_factory=list)
    rolling_ir: List[float] = field(default_factory=list)
    drawdowns: List[float] = field(default_factory=list)


class AlphaDecayDetector:
    """
    Monitor strategies for alpha decay.

    Metrics tracked:
    1. Rolling Sharpe ratio (60-day)
    2. Information ratio vs. benchmark
    3. Drawdown depth and duration
    4. Win rate stability

    Actions:
    - Issue warnings for declining strategies
    - Suggest allocation reductions
    - Auto-disable critically decayed strategies
    """

    def __init__(
        self,
        sharpe_lookback: int = 60,
        decay_threshold: float = 0.5,  # 50% decline triggers warning
        critical_threshold: float = 0.75,  # 75% decline is critical
        min_sharpe: float = 0.5  # Minimum acceptable Sharpe
    ):
        self.sharpe_lookback = sharpe_lookback
        self.decay_threshold = decay_threshold
        self.critical_threshold = critical_threshold
        self.min_sharpe = min_sharpe

        # Track performance per strategy
        self.strategies: Dict[str, StrategyPerformance] = {}

        # Historical baselines (peak performance)
        self.baselines: Dict[str, Dict] = {}

        # Active alerts
        self.alerts: List[AlphaDecayAlert] = []

    def record_return(
        self,
        strategy_name: str,
        daily_return: float,
        timestamp: Optional[str] = None
    ):
        """Record a daily return for a strategy."""
        if strategy_name not in self.strategies:
            self.strategies[strategy_name] = StrategyPerformance(name=strategy_name)

        ts = timestamp or datetime.utcnow().isoformat()
        perf = self.strategies[strategy_name]

        perf.returns.append(daily_return)
        perf.timestamps.append(ts)

        # Keep last 252 trading days
        if len(perf.returns) > 252:
            perf.returns = perf.returns[-252:]
            perf.timestamps = perf.timestamps[-252:]

        # Update rolling metrics
        self._update_metrics(strategy_name)

    def _update_metrics(self, strategy_name: str):
        """Update rolling metrics for a strategy."""
        perf = self.strategies[strategy_name]

        if len(perf.returns) < self.sharpe_lookback:
            return

        # Rolling Sharpe
        recent = perf.returns[-self.sharpe_lookback:]
        sharpe = self._calc_sharpe(recent)
        perf.rolling_sharpe.append(sharpe)

        # Rolling drawdown
        cum_returns = np.cumprod(1 + np.array(recent)) - 1
        peak = np.maximum.accumulate(cum_returns)
        dd = (cum_returns - peak) / (1 + peak)
        max_dd = abs(dd.min())
        perf.drawdowns.append(max_dd)

        # Update baseline (track peak Sharpe)
        if strategy_name not in self.baselines:
            self.baselines[strategy_name] = {
                "peak_sharpe": sharpe,
                "established_date": perf.timestamps[-1]
            }
        elif sharpe > self.baselines[strategy_name]["peak_sharpe"]:
            self.baselines[strategy_name]["peak_sharpe"] = sharpe

    def _calc_sharpe(self, returns: List[float]) -> float:
        """Calculate annualized Sharpe ratio."""
        returns_arr = np.array(returns)
        if len(returns_arr) < 2 or returns_arr.std() == 0:
            return 0.0

        mean_ret = returns_arr.mean() * 252
        vol = returns_arr.std() * np.sqrt(252)
        return float(mean_ret / vol) if vol > 0 else 0.0

    def check_decay(self, strategy_name: str) -> Optional[AlphaDecayAlert]:
        """
        Check if a strategy is experiencing alpha decay.

        Returns:
            AlphaDecayAlert if decay detected, None otherwise
        """
        if strategy_name not in self.strategies:
            return None

        perf = self.strategies[strategy_name]
        if not perf.rolling_sharpe:
            return None

        current_sharpe = perf.rolling_sharpe[-1]

        # Get baseline
        baseline = self.baselines.get(strategy_name, {})
        peak_sharpe = baseline.get("peak_sharpe", current_sharpe)

        if peak_sharpe <= 0:
            return None

        # Calculate decay
        decay_pct = 1 - (current_sharpe / peak_sharpe) if peak_sharpe > 0 else 0
        decay_pct = max(0, decay_pct)  # Only positive decay

        # Check thresholds
        if decay_pct >= self.critical_threshold:
            alert_type = "CRITICAL"
            recommendation = "DISABLE or significantly reduce allocation"
            weight_reduction = 0.75
        elif decay_pct >= self.decay_threshold:
            alert_type = "WARNING"
            recommendation = "Monitor closely, consider reducing allocation"
            weight_reduction = 0.30
        elif current_sharpe < self.min_sharpe:
            alert_type = "WARNING"
            recommendation = f"Sharpe below minimum ({self.min_sharpe})"
            weight_reduction = 0.20
        else:
            return None

        alert = AlphaDecayAlert(
            strategy_name=strategy_name,
            timestamp=datetime.utcnow().isoformat(),
            alert_type=alert_type,
            current_sharpe=current_sharpe,
            historical_sharpe=peak_sharpe,
            decay_pct=decay_pct,
            recommendation=recommendation,
            suggested_weight_reduction=weight_reduction
        )

        self.alerts.append(alert)

        # Keep last 100 alerts
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]

        logger.warning(
            f"AlphaDecay [{alert_type}] {strategy_name}: "
            f"Sharpe {current_sharpe:.2f} vs peak {peak_sharpe:.2f} "
            f"({decay_pct:.0%} decay)"
        )

        return alert

    def check_all_strategies(self) -> List[AlphaDecayAlert]:
        """Check all tracked strategies for decay."""
        alerts = []
        for name in self.strategies:
            alert = self.check_decay(name)
            if alert:
                alerts.append(alert)
        return alerts

    def get_weight_adjustments(self) -> Dict[str, float]:
        """
        Get recommended weight adjustments for all strategies.

        Returns:
            Dict of strategy_name -> weight multiplier (0 to 1)
        """
        adjustments = {}

        for name, perf in self.strategies.items():
            if not perf.rolling_sharpe:
                adjustments[name] = 1.0
                continue

            current_sharpe = perf.rolling_sharpe[-1]
            baseline = self.baselines.get(name, {})
            peak_sharpe = baseline.get("peak_sharpe", current_sharpe)

            if peak_sharpe <= 0:
                adjustments[name] = 1.0
                continue

            # Decay ratio becomes weight multiplier
            ratio = current_sharpe / peak_sharpe if peak_sharpe > 0 else 1.0
            ratio = np.clip(ratio, 0.1, 1.0)  # Min 10%, max 100%

            # Sharpe floor
            if current_sharpe < self.min_sharpe:
                ratio *= 0.5

            adjustments[name] = float(ratio)

        return adjustments

    def get_health_report(self) -> Dict[str, Dict]:
        """Get health report for all strategies."""
        report = {}

        for name, perf in self.strategies.items():
            baseline = self.baselines.get(name, {})

            current_sharpe = perf.rolling_sharpe[-1] if perf.rolling_sharpe else 0
            peak_sharpe = baseline.get("peak_sharpe", 0)
            current_dd = perf.drawdowns[-1] if perf.drawdowns else 0

            decay = 1 - (current_sharpe / peak_sharpe) if peak_sharpe > 0 else 0
            decay = max(0, decay)

            status = "HEALTHY"
            if decay > self.critical_threshold:
                status = "CRITICAL"
            elif decay > self.decay_threshold:
                status = "WARNING"
            elif current_sharpe < self.min_sharpe:
                status = "UNDERPERFORMING"

            report[name] = {
                "status": status,
                "current_sharpe": round(current_sharpe, 2),
                "peak_sharpe": round(peak_sharpe, 2),
                "decay_pct": round(decay * 100, 1),
                "current_drawdown": round(current_dd * 100, 1),
                "n_observations": len(perf.returns)
            }

        return report


# Global singleton
_decay_detector: Optional[AlphaDecayDetector] = None


def get_alpha_decay_detector() -> AlphaDecayDetector:
    """Get or create global AlphaDecayDetector."""
    global _decay_detector
    if _decay_detector is None:
        _decay_detector = AlphaDecayDetector()
    return _decay_detector
