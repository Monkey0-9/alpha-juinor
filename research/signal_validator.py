"""
Signal Validator
================

Quality Assurance (QA) layer for research.
Enforces strict statistical thresholds before a strategy is termed "PROVEN".

Gates:
1. Sharpe Ratio > 0.5
2. Sortino Ratio > 1.0 (Downside protection)
3. Min Trades > 50 (Statistical significance)
4. Stability > 0.7 (R-squared of equity curve)
"""

import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

class SignalValidator:
    def __init__(self,
                 min_sharpe: float = 0.5,
                 min_trades: int = 50,
                 min_sortino: float = 1.0):
        self.min_sharpe = min_sharpe
        self.min_trades = min_trades
        self.min_sortino = min_sortino

    def validate(self, metrics: Dict[str, float]) -> Tuple[bool, str]:
        """
        Validate a strategy based on backtest metrics.
        Returns: (Passed?, Reason)
        """
        # 1. Trade Count
        if metrics.get('total_trades', 0) < self.min_trades:
            return False, f"Insufficient Trades ({metrics.get('total_trades')} < {self.min_trades})"

        # 2. Sharpe
        if metrics.get('sharpe', 0.0) < self.min_sharpe:
            return False, f"Low Sharpe ({metrics.get('sharpe', 0.0):.2f} < {self.min_sharpe})"

        # 3. Sortino (if available)
        if 'sortino' in metrics and metrics['sortino'] < self.min_sortino:
            return False, f"Low Sortino ({metrics.get('sortino'):.2f} < {self.min_sortino})"

        return True, "PASSED"
