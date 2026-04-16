"""
execution/regime_aware_executor.py

Execution Regime Coupling
Adjusts execution strategy based on market regime.
"""

import logging
from typing import Dict, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)

class ExecutionUrgency(Enum):
    """Execution urgency levels"""
    PASSIVE = "passive"
    PATIENT = "patient"
    NO_EXECUTION = "no_execution"

class RegimeAwareExecutor:
    """
    Adapts execution aggressiveness to market regime.

    Profiles:
    - NORMAL → Passive (low urgency, multi-slice)
    - VOLATILE → Patient (very low urgency, many slices)
    - CRISIS → No execution (do not trade)
    """

    EXECUTION_PROFILES = {
        "NORMAL": {
            "urgency": ExecutionUrgency.PASSIVE,
            "max_slices": 10,
            "participation_rate": 0.05,  # 5% of volume
            "time_horizon_minutes": 60
        },
        "STRESS": {
            "urgency": ExecutionUrgency.PATIENT,
            "max_slices": 15,
            "participation_rate": 0.03,  # 3% of volume
            "time_horizon_minutes": 120
        },
        "VOLATILE": {
            "urgency": ExecutionUrgency.PATIENT,
            "max_slices": 20,
            "participation_rate": 0.02,  # 2% of volume
            "time_horizon_minutes": 180
        },
        "CRISIS": {
            "urgency": ExecutionUrgency.NO_EXECUTION,
            "max_slices": 0,
            "participation_rate": 0.0,
            "time_horizon_minutes": 0
        },
        "CROWDED": {
            "urgency": ExecutionUrgency.PATIENT,
            "max_slices": 12,
            "participation_rate": 0.04,
            "time_horizon_minutes": 90
        }
    }

    def __init__(self):
        self.current_regime = "NORMAL"

    def set_regime(self, regime: str):
        """Update current market regime"""
        if regime not in self.EXECUTION_PROFILES:
            logger.warning(f"[REG_EXEC] Unknown regime {regime}, defaulting to NORMAL")
            regime = "NORMAL"

        self.current_regime = regime
        logger.info(f"[REG_EXEC] Regime set to {regime}")

    def should_execute(self, regime: Optional[str] = None) -> bool:
        """Check if execution is allowed in current regime"""
        regime = regime or self.current_regime
        profile = self.EXECUTION_PROFILES[regime]

        if profile["urgency"] == ExecutionUrgency.NO_EXECUTION:
            logger.warning(f"[REG_EXEC] Execution BLOCKED in {regime} regime")
            return False

        return True

    def execute_order(self,
                     symbol: str,
                     qty: float,
                     side: str,
                     regime: Optional[str] = None) -> Dict:
        """
        Execute order with regime-appropriate strategy.

        Returns:
            {
                "status": str,
                "slices": List[Dict],
                "profile": Dict
            }
        """
        regime = regime or self.current_regime
        profile = self.EXECUTION_PROFILES[regime]

        # Check if execution allowed
        if not self.should_execute(regime):
            return {
                "status": "SKIP_CRISIS_MODE",
                "reason": f"{regime} regime blocks execution",
                "slices": [],
                "profile": profile
            }

        # Create slices based on regime profile
        slices = self._create_slices(
            symbol, qty, side, profile["max_slices"]
        )

        logger.info(
            f"[REG_EXEC] {symbol} {side} {qty}: "
            f"{len(slices)} slices over {profile['time_horizon_minutes']}m "
            f"({profile['urgency'].value} urgency)"
        )

        return {
            "status": "SLICED",
            "slices": slices,
            "profile": profile,
            "regime": regime
        }

    def _create_slices(self,
                      symbol: str,
                      qty: float,
                      side: str,
                      max_slices: int) -> List[Dict]:
        """
        Slice order into smaller parts for execution.

        Returns list of slice specifications.
        """
        if max_slices <= 1:
            # Single execution
            return [{
                "symbol": symbol,
                "qty": qty,
                "side": side,
                "slice_num": 1,
                "total_slices": 1
            }]

        # Equal slicing
        slice_size = qty / max_slices
        slices = []

        for i in range(max_slices):
            slices.append({
                "symbol": symbol,
                "qty": slice_size,
                "side": side,
                "slice_num": i + 1,
                "total_slices": max_slices
            })

        return slices

    def estimate_market_impact(self,
                               qty: float,
                               profile: Dict) -> float:
        """
        Estimate market impact cost based on execution profile.

        Returns: Estimated impact as fraction of notional
        """
        # Simplified impact model: inverse of participation rate
        # More patient = lower impact

        base_impact = 0.001  # 10 bps base
        participation_penalty = (1.0 / profile["participation_rate"]) * 0.0001

        estimated_impact = base_impact + participation_penalty

        return estimated_impact
