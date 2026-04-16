"""
risk/regime_controller.py

Global Regime Controller - applies hard overrides during extreme conditions.
"""

import logging
from enum import Enum
from typing import Dict, Any, Optional
import pandas as pd

logger = logging.getLogger(__name__)

import yaml
from pathlib import Path

class MarketRegime(Enum):
    """Market regime classifications"""
    NORMAL = "NORMAL"
    RISK_OFF = "RISK_OFF"  # Replaces STRESS
    CRISIS = "CRISIS"
    CROWDED = "CROWDED"
    LIQUIDITY_STRESS = "LIQUIDITY_STRESS"
    RISK_ON = "RISK_ON"

class RegimeController:
    """
    Detects market regimes and applies hard overrides from YAML.
    """

    def __init__(self, override_file: str = "risk/regime_overrides.yaml"):
        self.current_regime = MarketRegime.NORMAL
        self.vix_crisis_threshold = 30.0
        self.dd_stress_threshold = 0.10
        self.corr_crowded_threshold = 0.7
        self.overrides = {}
        self._load_overrides(override_file)

    def _load_overrides(self, filepath: str):
        """Load regime overrides from YAML."""
        try:
            path = Path(__file__).parent / "regime_overrides.yaml" # Default relative
            if not path.exists():
                path = Path(filepath)

            if path.exists():
                with open(path, 'r') as f:
                    self.overrides = yaml.safe_load(f)
                logger.info(f"[REGIME] Loaded overrides from {path}")
            else:
                logger.warning(f"[REGIME] Override file {path} not found. Using defaults.")
        except Exception as e:
            logger.error(f"[REGIME] Failed to load overrides: {e}")

    def detect_regime(self, vix: Optional[float] = None,
                     drawdown: Optional[float] = None,
                     avg_correlation: Optional[float] = None) -> MarketRegime:
        """
        Detect current market regime based on indicators.
        """
        # VIX-based detection (highest priority)
        if vix and vix > self.vix_crisis_threshold:
            self.current_regime = MarketRegime.CRISIS
            logger.critical(f"[REGIME] CRISIS detected: VIX={vix:.1f}")
            return self.current_regime

        # Drawdown-based detection
        if drawdown and abs(drawdown) > self.dd_stress_threshold:
            self.current_regime = MarketRegime.RISK_OFF
            logger.warning(f"[REGIME] RISK_OFF (Drawdown) detected: DD={drawdown:.2%}")
            return self.current_regime

        # Correlation-based detection
        if avg_correlation and avg_correlation > self.corr_crowded_threshold:
            self.current_regime = MarketRegime.CROWDED
            logger.warning(f"[REGIME] CROWDED detected: avg_corr={avg_correlation:.2f}")
            return self.current_regime

        # Default to normal
        self.current_regime = MarketRegime.NORMAL
        return self.current_regime

    def get_current_overrides(self) -> Dict[str, Any]:
        """Get overrides for the current active regime."""
        regime_key = self.current_regime.value
        return self.overrides.get(regime_key, {})

    def apply_overrides(self, regime: MarketRegime,
                       config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply hard overrides from loaded YAML config.
        """
        overridden_config = config.copy()
        regime_key = regime.value

        # Get override params for this regime
        params = self.overrides.get(regime_key, {})

        if not params:
            return overridden_config

        logger.info(f"[REGIME_OVERRIDE] Applying {regime_key} restrictions: {params}")

        # Apply parameters to config structure
        if "max_gross_exposure" in params:
            overridden_config["risk"] = overridden_config.get("risk", {}).copy()
            overridden_config["risk"]["max_gross_leverage"] = params["max_gross_exposure"]

        if "max_position_size" in params:
            overridden_config["risk"] = overridden_config.get("risk", {}).copy()
            overridden_config["risk"]["max_position_size"] = params["max_position_size"]

        if params.get("execution_mode") == "DISABLED":
             overridden_config["trading_halted"] = True

        # Add metadata about override
        overridden_config["regime_override"] = regime_key

        return overridden_config

    def get_sizing_multiplier(self, regime: MarketRegime) -> float:
        """Get sizing multiplier (derived from max_gross_exposure vs normal 1.0)."""
        params = self.overrides.get(regime.value, {})
        # Heuristic: If max_gross is defined, take ratio to 1.0
        max_gross = params.get("max_gross_exposure", 1.0)
        return min(max_gross, 1.0)
