"""
ml/alpha_decay.py

Alpha Decay & Strategy Death Detection.
Prevents strategies from "slowly rotting" via:
1. Rolling IC (Information Coefficient) monitoring
2. Statistical decay detection (t-test vs baseline)
3. Capacity saturation curves
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from scipy import stats
import logging

logger = logging.getLogger("ALPHA_DECAY")

@dataclass
class DecayMetrics:
    strategy_id: str
    date: str
    rolling_ic_30d: float
    rolling_ic_60d: float
    rolling_ic_90d: float
    decay_score: float  # 0-1, higher = more decay
    capacity_utilization: float  # 0-1, current AUM / estimated capacity
    status: str  # HEALTHY, DEGRADED, CRITICAL
    recommendation: str  # CONTINUE, REDUCE_ALLOCATION, QUARANTINE, RETIRE

class AlphaDecayMonitor:
    def __init__(self,
                 decay_threshold: float = 0.3,  # IC drop > 30% triggers DEGRADED
                 critical_threshold: float = 0.5,  # IC drop > 50% triggers CRITICAL
                 min_ic_baseline: float = 0.02):  # Minimum acceptable IC
        self.decay_threshold = decay_threshold
        self.critical_threshold = critical_threshold
        self.min_ic_baseline = min_ic_baseline

    def compute_rolling_ic(self,
                          signals: pd.Series,
                          forward_returns: pd.Series,
                          window: int = 30) -> pd.Series:
        """
        Compute rolling Information Coefficient (Spearman correlation).

        Args:
            signals: Alpha signal values (indexed by date)
            forward_returns: Forward N-day returns (indexed by date)
            window: Rolling window in days

        Returns:
            Rolling IC series
        """
        # Align series
        df = pd.concat([signals, forward_returns], axis=1).dropna()
        if len(df) < window:
            return pd.Series(dtype=float)

        df.columns = ['signal', 'return']

        # Rolling Spearman correlation
        rolling_ic = df['signal'].rolling(window).corr(df['return'], method='spearman')

        return rolling_ic

    def detect_decay(self,
                    current_ic: float,
                    historical_ic: pd.Series,
                    lookback_days: int = 252) -> Tuple[bool, float, str]:
        """
        Detect statistically significant decay via t-test.

        Args:
            current_ic: Most recent IC value
            historical_ic: Historical IC series
            lookback_days: Days to use for baseline

        Returns:
            (is_decayed, decay_score, status)
        """
        if historical_ic.empty or len(historical_ic) < 30:
            return False, 0.0, "INSUFFICIENT_DATA"

        # Baseline: mean IC over lookback period
        baseline = historical_ic.tail(lookback_days).mean()
        baseline_std = historical_ic.tail(lookback_days).std()

        if baseline <= 0 or baseline_std == 0:
            return True, 1.0, "CRITICAL"  # Never had positive IC

        # Decay score: normalized drop from baseline
        ic_drop = max(0, baseline - current_ic)
        decay_score = min(1.0, ic_drop / baseline)

        # Statistical test: is current IC significantly below baseline?
        # Using simple z-score (could use t-test with recent window)
        z_score = (current_ic - baseline) / (baseline_std + 1e-9)

        # Status determination
        if current_ic < self.min_ic_baseline:
            status = "CRITICAL"
        elif decay_score > self.critical_threshold:
            status = "CRITICAL"
        elif decay_score > self.decay_threshold:
            status = "DEGRADED"
        elif z_score < -2.0:  # 2 std below baseline
            status = "DEGRADED"
        else:
            status = "HEALTHY"

        is_decayed = status in ["DEGRADED", "CRITICAL"]

        return is_decayed, decay_score, status

    def estimate_capacity(self,
                         strategy_sharpe: float,
                         avg_position_size: float,
                         avg_daily_volume: float,
                         participation_rate: float = 0.05) -> float:
        """
        Estimate strategy capacity using square-root impact model.

        Capacity = (Sharpe^2 / Impact_Coefficient) * ADV * participation_rate

        Simplified: Capacity ≈ ADV * participation_rate / sqrt(turnover)
        """
        if avg_daily_volume <= 0 or strategy_sharpe <= 0:
            return 0.0

        # Simplified capacity estimate
        # Assumes strategy can trade up to participation_rate of ADV
        # without significant impact degradation
        base_capacity = avg_daily_volume * participation_rate

        # Adjust for Sharpe (higher Sharpe = more capacity before decay)
        sharpe_factor = min(2.0, strategy_sharpe / 1.0)  # Cap at 2x

        estimated_capacity = base_capacity * sharpe_factor

        return estimated_capacity

    def recommend_retirement(self, metrics: DecayMetrics) -> str:
        """
        Recommend action based on decay metrics.

        Returns: CONTINUE, REDUCE_ALLOCATION, QUARANTINE, RETIRE
        """
        if metrics.status == "CRITICAL":
            if metrics.rolling_ic_90d < 0:
                return "RETIRE"  # Negative IC over 90 days
            else:
                return "QUARANTINE"  # Critical but not negative yet

        elif metrics.status == "DEGRADED":
            if metrics.capacity_utilization > 0.8:
                return "REDUCE_ALLOCATION"  # Near capacity limit
            else:
                return "QUARANTINE"  # Degraded performance

        elif metrics.capacity_utilization > 0.9:
            return "REDUCE_ALLOCATION"  # Approaching capacity

        else:
            return "CONTINUE"

    def analyze_strategy(self,
                        strategy_id: str,
                        signals: pd.Series,
                        forward_returns: pd.Series,
                        current_aum: float,
                        avg_daily_volume: float,
                        strategy_sharpe: float = 1.0) -> DecayMetrics:
        """
        Full analysis pipeline for a strategy.
        """
        date = signals.index[-1].strftime("%Y-%m-%d") if not signals.empty else "UNKNOWN"

        # Compute rolling ICs
        ic_30 = self.compute_rolling_ic(signals, forward_returns, window=30)
        ic_60 = self.compute_rolling_ic(signals, forward_returns, window=60)
        ic_90 = self.compute_rolling_ic(signals, forward_returns, window=90)

        if ic_30.empty or ic_60.empty or ic_90.empty:
            logger.warning(f"[{strategy_id}] Insufficient data for IC calculation")
            return DecayMetrics(
                strategy_id=strategy_id,
                date=date,
                rolling_ic_30d=0.0,
                rolling_ic_60d=0.0,
                rolling_ic_90d=0.0,
                decay_score=1.0,
                capacity_utilization=0.0,
                status="INSUFFICIENT_DATA",
                recommendation="QUARANTINE"
            )

        current_ic_30 = ic_30.iloc[-1]
        current_ic_60 = ic_60.iloc[-1]
        current_ic_90 = ic_90.iloc[-1]

        # Detect decay using 60-day IC
        is_decayed, decay_score, status = self.detect_decay(current_ic_60, ic_60)

        # Estimate capacity
        estimated_capacity = self.estimate_capacity(
            strategy_sharpe,
            current_aum / 252,  # Rough avg position
            avg_daily_volume
        )

        capacity_util = current_aum / estimated_capacity if estimated_capacity > 0 else 0.0

        metrics = DecayMetrics(
            strategy_id=strategy_id,
            date=date,
            rolling_ic_30d=float(current_ic_30),
            rolling_ic_60d=float(current_ic_60),
            rolling_ic_90d=float(current_ic_90),
            decay_score=float(decay_score),
            capacity_utilization=float(capacity_util),
            status=status,
            recommendation=self.recommend_retirement(DecayMetrics(
                strategy_id=strategy_id,
                date=date,
                rolling_ic_30d=current_ic_30,
                rolling_ic_60d=current_ic_60,
                rolling_ic_90d=current_ic_90,
                decay_score=decay_score,
                capacity_utilization=capacity_util,
                status=status,
                recommendation=""
            ))
        )

        logger.info(f"[{strategy_id}] IC: 30d={current_ic_30:.3f}, 60d={current_ic_60:.3f}, 90d={current_ic_90:.3f} | "
                   f"Decay: {decay_score:.2f} | Status: {status} | Rec: {metrics.recommendation}")

        return metrics
