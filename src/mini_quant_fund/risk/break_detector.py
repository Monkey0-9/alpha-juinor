"""
risk/break_detector.py

Structural Break Detection Agent
Detects regime changes and applies hard capital reduction.
"""

import numpy as np
import pandas as pd
import logging
from typing import Tuple, Dict, Optional

logger = logging.getLogger(__name__)

class StructuralBreakDetector:
    """
    Detects structural breaks using multiple signals:
    - Volatility regime shifts
    - Return distribution changes
    - Rolling likelihood collapse
    """

    def __init__(self,
                 vol_threshold: float = 2.0,
                 lookback_window: int = 20,
                 baseline_window: int = 252):
        """
        Args:
            vol_threshold: Volatility ratio triggering break (2.0 = doubled)
            lookback_window: Recent window for break detection
            baseline_window: Historical baseline for comparison
        """
        self.vol_threshold = vol_threshold
        self.lookback_window = lookback_window
        self.baseline_window = baseline_window

    def detect_break(self,
                    returns: pd.Series,
                    prices: Optional[pd.Series] = None) -> Tuple[bool, float, str]:
        """
        Detect if a structural break occurred.

        Returns:
            (break_detected: bool, severity: float, reason: str)
        """
        if len(returns) < self.baseline_window + self.lookback_window:
            logger.warning("[BREAK] Insufficient data for break detection")
            return False, 1.0, "INSUFFICIENT_DATA"

        # Signal 1: Volatility regime shift
        vol_break, vol_severity = self._detect_volatility_break(returns)

        # Signal 2: Return distribution change
        dist_break, dist_severity = self._detect_distribution_break(returns)

        # Signal 3: Drawdown acceleration
        dd_break, dd_severity = self._detect_drawdown_break(prices) if prices is not None else (False, 1.0)

        # Aggregate signals
        max_severity = max(vol_severity, dist_severity, dd_severity)

        if vol_break:
            return True, vol_severity, f"VOLATILITY_SPIKE_{vol_severity:.2f}x"
        elif dist_break:
            return True, dist_severity, f"DISTRIBUTION_SHIFT_{dist_severity:.2f}"
        elif dd_break:
            return True, dd_severity, f"DRAWDOWN_ACCELERATION_{dd_severity:.2f}"

        return False, 1.0, "NORMAL"

    def _detect_volatility_break(self, returns: pd.Series) -> Tuple[bool, float]:
        """Detect volatility regime shift"""
        recent_vol = returns.tail(self.lookback_window).std()
        baseline_vol = returns.head(self.baseline_window).std()

        if baseline_vol == 0:
            return False, 1.0

        vol_ratio = recent_vol / baseline_vol

        if vol_ratio > self.vol_threshold:
            logger.warning(f"[BREAK] Volatility break detected: {vol_ratio:.2f}x baseline")
            return True, vol_ratio

        return False, vol_ratio

    def _detect_distribution_break(self, returns: pd.Series) -> Tuple[bool, float]:
        """Detect return distribution change using Kolmogorov-Smirnov test"""
        from scipy import stats

        recent = returns.tail(self.lookback_window)
        baseline = returns.head(self.baseline_window)

        # KS test for distribution change
        ks_stat, p_value = stats.ks_2samp(recent, baseline)

        # Significant difference if p < 0.01
        if p_value < 0.01:
            logger.warning(f"[BREAK] Distribution break detected: KS={ks_stat:.3f}, p={p_value:.4f}")
            return True, ks_stat * 10  # Scale for severity

        return False, 1.0

    def _detect_drawdown_break(self, prices: pd.Series) -> Tuple[bool, float]:
        """Detect drawdown acceleration"""
        # Compute cumulative max and drawdown
        cummax = prices.expanding().max()
        drawdown = (prices - cummax) / cummax

        recent_dd = drawdown.tail(self.lookback_window).min()
        baseline_dd = drawdown.head(self.baseline_window).min()

        # Check if recent DD is significantly worse
        if recent_dd < baseline_dd * 1.5:  # 50% worse
            severity = abs(recent_dd / baseline_dd)
            logger.warning(f"[BREAK] Drawdown acceleration: {recent_dd:.2%} vs {baseline_dd:.2%}")
            return True, severity

        return False, 1.0

    def apply_break_response(self,
                            severity: float,
                            reason: str,
                            base_config: Dict) -> Dict:
        """
        Apply hard capital reduction based on break severity.

        Severity levels:
        - 2.0-3.0: Moderate (50% reduction)
        - 3.0-5.0: Severe (75% reduction)
        - 5.0+: Extreme (90% reduction)
        """
        config = base_config.copy()

        if severity >= 5.0:
            # Extreme: Near-total shutdown
            reduction_factor = 0.10  # 90% reduction
            disabled_families = ["momentum", "mean_reversion", "value", "carry"]
            logger.critical(f"[BREAK_RESPONSE] EXTREME break ({severity:.2f}): 90% capital reduction")

        elif severity >= 3.0:
            # Severe: Major reduction
            reduction_factor = 0.25  # 75% reduction
            disabled_families = ["momentum", "mean_reversion"]
            logger.error(f"[BREAK_RESPONSE] SEVERE break ({severity:.2f}): 75% capital reduction")

        elif severity >= 2.0:
            # Moderate: Significant reduction
            reduction_factor = 0.50  # 50% reduction
            disabled_families = ["momentum"]
            logger.warning(f"[BREAK_RESPONSE] MODERATE break ({severity:.2f}): 50% capital reduction")

        else:
            # No break response needed
            return base_config

        # Apply reductions
        config["risk"] = config.get("risk", {}).copy()
        config["risk"]["max_gross_leverage"] = config["risk"].get("max_gross_leverage", 1.0) * reduction_factor
        config["risk"]["max_position_size"] = config["risk"].get("max_position_size", 0.1) * reduction_factor
        config["disabled_families"] = disabled_families
        config["break_reason"] = reason
        config["break_severity"] = severity

        return config

    def get_recovery_condition(self, severity: float) -> int:
        """
        Determine number of calm periods needed before recovery.

        Returns: Number of consecutive calm cycles needed
        """
        if severity >= 5.0:
            return 20  # 20 calm cycles for extreme breaks
        elif severity >= 3.0:
            return 10  # 10 cycles for severe
        elif severity >= 2.0:
            return 5   # 5 cycles for moderate
        else:
            return 1
