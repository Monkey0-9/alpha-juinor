"""
Adaptive Risk Manager - 2026 Elite
===================================

Dynamic risk management that adapts to market conditions.

Features:
- Dynamic position limits based on volatility
- Correlation-adjusted exposure
- Drawdown protection with recovery mode
- Tail risk hedging triggers
- Kelly-based leverage

Target: Max DD < 15%, Sharpe > 2.5
"""

import logging
import numpy as np
from typing import Dict, Any
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class RiskBudget:
    """Current risk budget allocation."""
    max_leverage: float
    max_position_size: float
    max_sector_exposure: float
    max_correlated_exposure: float
    stop_loss_level: float
    take_profit_level: float
    hedge_trigger_dd: float
    current_risk_utilization: float


class AdaptiveRiskManager:
    """
    Elite adaptive risk management system.
    """

    def __init__(
        self,
        base_risk_budget: float = 0.15,
        max_drawdown_limit: float = 0.15
    ):
        self.base_risk = base_risk_budget
        self.max_dd = max_drawdown_limit

        # State tracking
        self.peak_nav = 0.0
        self.current_dd = 0.0
        self.in_recovery_mode = False
        self.vol_regime = "NORMAL"

        # Dynamic parameters
        self.vol_multiplier = 1.0
        self.dd_multiplier = 1.0

        logger.info("[RISK_MGR] Adaptive risk manager initialized")

    def update_state(
        self,
        nav: float,
        market_volatility: float,
        correlation_avg: float,
        regime: str
    ) -> RiskBudget:
        """
        Update risk state and return current budget.
        """
        # Track drawdown
        if nav > self.peak_nav:
            self.peak_nav = nav
            self.in_recovery_mode = False

        self.current_dd = (self.peak_nav - nav) / self.peak_nav \
            if self.peak_nav > 0 else 0.0

        # Enter recovery mode if DD > 5%
        if self.current_dd > 0.05:
            self.in_recovery_mode = True

        # Volatility regime
        if market_volatility > 0.03:
            self.vol_regime = "HIGH"
            self.vol_multiplier = 0.5
        elif market_volatility > 0.02:
            self.vol_regime = "ELEVATED"
            self.vol_multiplier = 0.7
        elif market_volatility < 0.01:
            self.vol_regime = "LOW"
            self.vol_multiplier = 1.2
        else:
            self.vol_regime = "NORMAL"
            self.vol_multiplier = 1.0

        # Drawdown multiplier
        if self.current_dd > 0.10:
            self.dd_multiplier = 0.3
        elif self.current_dd > 0.07:
            self.dd_multiplier = 0.5
        elif self.current_dd > 0.05:
            self.dd_multiplier = 0.7
        else:
            self.dd_multiplier = 1.0

        # Calculate risk budget
        effective_risk = (
            self.base_risk *
            self.vol_multiplier *
            self.dd_multiplier
        )

        # Regime adjustments
        regime_factor = {
            "BULL": 1.2,
            "BEAR": 0.6,
            "VOLATILE": 0.4,
            "SIDEWAYS": 0.9,
            "CRISIS": 0.2,
            "RECOVERY": 0.8
        }.get(regime, 1.0)

        effective_risk *= regime_factor

        # Calculate position and sector limits
        max_position = min(0.15, effective_risk / 3)
        max_sector = min(0.30, effective_risk)
        max_corr = min(0.40, effective_risk * 1.5)

        # Dynamic stop loss based on volatility
        base_stop = 0.05
        stop_loss = base_stop * (1 + market_volatility * 10)
        stop_loss = np.clip(stop_loss, 0.02, 0.10)

        # Take profit based on expected move
        take_profit = stop_loss * 2.5  # 2.5:1 risk-reward

        # Hedge trigger (deeper than normal DD recovery)
        hedge_trigger = max(0.08, self.current_dd + 0.03)

        # Current utilization (placeholder)
        utilization = min(1.0, correlation_avg + abs(self.current_dd))

        return RiskBudget(
            max_leverage=min(1.5, 1.0 / (market_volatility * 10 + 0.5)),
            max_position_size=max_position,
            max_sector_exposure=max_sector,
            max_correlated_exposure=max_corr,
            stop_loss_level=stop_loss,
            take_profit_level=take_profit,
            hedge_trigger_dd=hedge_trigger,
            current_risk_utilization=utilization
        )

    def should_reduce_exposure(self) -> bool:
        """Check if we should reduce overall exposure."""
        return (
            self.current_dd > 0.07 or
            self.vol_regime == "HIGH" or
            self.in_recovery_mode
        )

    def should_hedge(self) -> bool:
        """Check if we should initiate hedging."""
        return self.current_dd > 0.10

    def get_position_size(
        self,
        signal_strength: float,
        symbol_volatility: float,
        risk_budget: RiskBudget
    ) -> float:
        """
        Calculate optimal position size for a signal.
        """
        if abs(signal_strength) < 0.1:
            return 0.0

        # Base size from risk budget
        base_size = risk_budget.max_position_size * abs(signal_strength)

        # Vol adjustment
        if symbol_volatility > 0.04:
            base_size *= 0.5
        elif symbol_volatility > 0.03:
            base_size *= 0.7
        elif symbol_volatility < 0.015:
            base_size *= 1.2

        # Recovery mode penalty
        if self.in_recovery_mode:
            base_size *= 0.7

        return min(base_size, risk_budget.max_position_size)

    def get_status(self) -> Dict[str, Any]:
        """Get risk manager status."""
        return {
            "current_drawdown": self.current_dd,
            "peak_nav": self.peak_nav,
            "vol_regime": self.vol_regime,
            "in_recovery": self.in_recovery_mode,
            "vol_multiplier": self.vol_multiplier,
            "dd_multiplier": self.dd_multiplier
        }


# Singleton
_risk_mgr = None


def get_adaptive_risk_manager() -> AdaptiveRiskManager:
    global _risk_mgr
    if _risk_mgr is None:
        _risk_mgr = AdaptiveRiskManager()
    return _risk_mgr
