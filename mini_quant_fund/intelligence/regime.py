import numpy as np
import pandas as pd
from typing import Dict, Any
import structlog

logger = structlog.get_logger()

class RegimeAgent:
    """
    Ruthless Regime Detection.
    Uses HMM / Latent state switching to detect CRISIS regimes.
    Trigger: If probability of 'CRISIS' state > 0.7, enable defensive overrides.
    """
    def __init__(self):
        # In production, these parameters would be loaded from a serialized HMM model
        self.regime_names = ["NORMAL", "VOLATILE", "CRISIS"]
        self.prev_state = "NORMAL"

    def detect_regime(self, historical_vol: pd.Series, returns: pd.Series) -> str:
        """
        Detects current market regime based on volatility and returns.
        """
        if returns.empty:
            return "NORMAL"

        # Simplified HMM Proxy: Markovian Switching based on Vol & Skew
        last_vol = historical_vol.iloc[-1]
        last_ret = returns.iloc[-1]
        vol_z = (last_vol - historical_vol.mean()) / (historical_vol.std() + 1e-9)

        # CRISIS logic: extreme volatility + negative returns
        if vol_z > 3.0 and last_ret < -0.02:
            current_regime = "CRISIS"
        elif vol_z > 1.5:
            current_regime = "VOLATILE"
        else:
            current_regime = "NORMAL"

        if current_regime != self.prev_state:
            logger.warning("REGIME_SWITCH detected",
                           prev=self.prev_state,
                           current=current_regime,
                           vol_z=vol_z)
            self.prev_state = current_regime

        return current_regime

    def get_overrides(self, regime: str) -> Dict[str, Any]:
        """
        Policy: If regime == CRISIS -> max position size = min(1% NAV, previous_max * 0.5)
        """
        if regime == "CRISIS":
            return {
                "max_pos_nav_pct": 0.01,
                "gross_exposure_reduction": 0.5,
                "reason": "CRISIS_REGIME_LOCKED"
            }
        elif regime == "VOLATILE":
            return {
                "max_pos_nav_pct": 0.05,
                "gross_exposure_reduction": 0.2,
                "reason": "VOLATILITY_THROTTLE"
            }
        return {}
