import numpy as np
import pandas as pd
import structlog
from typing import Dict, Any

logger = structlog.get_logger()

class StructuralBreakDetector:
    """
    Institutional Model Governance.
    Uses CUSUM (Cumulative Sum) tests to detect structural breaks/regime shifts.
    Triggers 'MODEL_DECAY' alert if cumulative residuals exceed threshold.
    """
    def __init__(self, threshold: float = 5.0):
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.s_pos = 0.0
        self.s_neg = 0.0

    def detect_break(self, series: pd.Series) -> bool:
        """
        Standard CUSUM test for mean shift.
        """
        if len(series) < 30:
            return False

        returns = series.pct_change().dropna()
        mu = returns.mean()
        sigma = returns.std() + 1e-9

        # Z-scores of returns
        z = (returns - mu) / sigma

        # Cumulative Sums
        for val in z.tail(10): # Look at recent window
            self.s_pos = max(0, self.s_pos + val - 0.5)
            self.s_neg = min(0, self.s_neg + val + 0.5)

            if self.s_pos > self.threshold or abs(self.s_neg) > self.threshold:
                logger.critical("STRUCTURAL_BREAK detected",
                                s_pos=self.s_pos,
                                s_neg=self.s_neg)
                return True
        return False

    def get_status(self) -> str:
        if self.s_pos > self.threshold or abs(self.s_neg) > self.threshold:
             return "SYSTEM_STRESSED"
        return "MODEL_STABLE"
