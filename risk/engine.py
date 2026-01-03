# risk/engine.py
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

from enum import Enum

class RiskDecision(Enum):
    ALLOW = "ALLOW"
    SCALE = "SCALE"
    REJECT = "REJECT" # Block this trade
    FREEZE = "FREEZE" # Stop all trading
    LIQUIDATE = "LIQUIDATE" # Close all positions
    RECOVERY = "RECOVERY" # Controlled re-entry phase

class RiskManager:
    """
    Institutional RiskManager with Pre-trade scaling and Post-trade circuit breakers.
    
    Refined for:
    - Elite Drawdown Control (6-18% range)
    - Predictive Regime Awareness (0.7x Risk-Off Haircut)
    - Controlled Recovery (20% -> 40% -> 60% -> 80% -> 100% ramp)
    """

    def __init__(
        self,
        max_leverage: float = 1.0,
        target_vol_limit: float = 0.15,
        var_limit: float = 0.04, 
        cvar_limit: float = 0.06,
        max_drawdown_limit: float = 0.18,
        bootstrap_days: int = 60, 
    ):
        self.max_leverage = float(max_leverage)
        self.target_vol_limit = float(target_vol_limit)
        self.var_limit = float(var_limit)
        self.cvar_limit = float(cvar_limit)
        self.max_drawdown_limit = float(max_drawdown_limit)
        self.bootstrap_days = bootstrap_days
        
        # Stateful Risk Management
        self.state = RiskDecision.ALLOW
        self.cooldown_days = 20
        self.cooldown_counter = 0
        
        # Recovery tracking
        self.recovery_level = 0 # Phase 1 to 5 (20% steps)
        self._current_realized_vol = 0.0
        
        self._max_equity = -1.0
        self._last_logged_cooldown = -1
        
        # Hysteresis tracking
        self._last_scale_factor = 1.0

    def compute_var(self, returns: pd.Series, confidence: float = 0.95) -> float:
        if returns.empty or len(returns) < 20: 
            return 0.0
        cutoff = np.percentile(returns, 100 * (1 - confidence))
        return -cutoff if cutoff < 0 else 0.0

    def compute_cvar(self, returns: pd.Series, confidence: float = 0.95) -> float:
        if returns.empty or len(returns) < 20:
            return 0.0
        var = self.compute_var(returns, confidence)
        tail_losses = returns[returns <= -var]
        if tail_losses.empty:
            return var
        return -tail_losses.mean()

    def check_pre_trade(
        self, 
        target_weights: Dict[str, float], 
        baskets_returns: pd.DataFrame,
        timestamp: pd.Timestamp,
        current_equity: float = 1.0,
        is_risk_on: bool = True
    ) -> Dict[str, Any]:
        """
        PRE-TRADE: Scaling trade intent.
        Includes Elite Drawdown Control, Regime Awareness, and Controlled Recovery.
        """
        if self.state == RiskDecision.FREEZE:
            return {"ok": False, "decision": RiskDecision.REJECT}

        # 0. Proactive Drawdown Scaler (Elite Control)
        if current_equity > self._max_equity:
            self._max_equity = current_equity
            
        dd = (self._max_equity - current_equity) / self._max_equity if self._max_equity > 0 else 0.0
        
        dd_scalar = 1.0
        dd_violation = None
        if dd > self.max_drawdown_limit:
             return {"ok": False, "decision": RiskDecision.REJECT, "violations": [f"Drawdown {dd:.1%} > {self.max_drawdown_limit:.1%} Limit"]}
        elif dd > 0.05:
             # Fix #1: Start scaling at 5%. Stiffer cut: 5% -> 1.0, 18% -> 0.0
             # Range is 13%
             dd_scalar = max(0.0, (self.max_drawdown_limit - dd) / 0.13)
             dd_violation = f"Drawdown {dd:.1%} > 5% (Scaler {dd_scalar:.2f})"

        # 1. Recovery Scalar (Fix: Controlled Re-entry)
        # Progresses across REBALANCE cycles
        recovery_scalar = 1.0
        if self.state == RiskDecision.RECOVERY:
            # Check stability/trend to progress to NEXT level
            is_stable = self._current_realized_vol < self.target_vol_limit * 1.5
            if is_risk_on and is_stable:
                 if self.recovery_level < 5:
                      self.recovery_level += 1
                      logger.info(f"RECOVERY PROGRESS: Level {self.recovery_level} ({self.recovery_level*20}% cap)")
                 
                 if self.recovery_level >= 5:
                      logger.info("RECOVERY COMPLETE. Resuming normal operations.")
                      self.state = RiskDecision.ALLOW
            
            # Apply the CAP for the current phase (before progression)
            # Level 1=20%, 2=40%, 3=60%, 4=80%, 5=100%
            recovery_scalar = self.recovery_level * 0.20
            if recovery_scalar < 1.0:
                 dd_violation = f"RECOVERY PHASE {self.recovery_level} (Cap {recovery_scalar:.0%})"

        gross_leverage = sum(abs(w) for w in target_weights.values())
        if gross_leverage < 0.01:
            return {"ok": True, "decision": RiskDecision.ALLOW, "scale_factor": 1.0}

        # Historical Returns Simulation
        valid_weights = {k: v for k, v in target_weights.items() if k in baskets_returns.columns}
        if not valid_weights or baskets_returns.empty:
            portfolio_returns = pd.Series(0.0, index=baskets_returns.index)
        else:
            w_vector = pd.Series(valid_weights)
            portfolio_returns = baskets_returns[w_vector.index].dot(w_vector)

        current_var = self.compute_var(portfolio_returns)
        current_cvar = self.compute_cvar(portfolio_returns)
        
        violations = []
        if dd_violation: violations.append(dd_violation)
        
        raw_scale = 1.0

        if gross_leverage > self.max_leverage + 0.01:
            violations.append(f"Leverage {gross_leverage:.3f} > {self.max_leverage}")
            raw_scale = min(raw_scale, self.max_leverage / (gross_leverage + 1e-9))

        # Fix #2: Predictive Deleveraging (Trend + Vol Proxy)
        # 0.7x (30% haircut) during Risk-Off
        regime_scalar = 1.0
        if not is_risk_on:
             regime_scalar = 0.70
             violations.append("Market Regime Risk-Off (Predictive De-risk 0.7x)")

        if current_var > self.var_limit:
            violations.append(f"VaR {current_var:.2%} > {self.var_limit:.2%}")
            raw_scale = min(raw_scale, self.var_limit / (current_var + 1e-9))

        if current_cvar > self.cvar_limit:
            violations.append(f"CVaR {current_cvar:.2%} > {self.cvar_limit:.2%}")
            raw_scale = min(raw_scale, self.cvar_limit / (current_cvar + 1e-9))

        # Apply Proactive Scaler combining structural risk + drawdown + regime + recovery
        final_scale = raw_scale * dd_scalar * regime_scalar * recovery_scalar

        # Hysteresis logic
        if abs(final_scale - self._last_scale_factor) < 0.05 and self._last_scale_factor < 1.0:
            applied_scale = self._last_scale_factor
        else:
            applied_scale = final_scale
        
        self._last_scale_factor = applied_scale

        if violations or applied_scale < 0.98:
            return {
                "ok": False,
                "decision": RiskDecision.SCALE,
                "scale_factor": applied_scale,
                "violations": violations
            }
        
        return {"ok": True, "decision": RiskDecision.ALLOW, "scale_factor": 1.0}

    def check_circuit_breaker(self, current_equity: float, realized_returns: pd.Series) -> RiskDecision:
        if current_equity > self._max_equity:
            self._max_equity = current_equity
        
        drawdown = (self._max_equity - current_equity) / (self._max_equity + 1e-9) if self._max_equity > 0 else 0.0
        
        # Track realized vol for stability check
        self._current_realized_vol = realized_returns.std() * np.sqrt(252) if len(realized_returns) > 10 else 0.0

        # Hard stop limits
        if drawdown > self.max_drawdown_limit:
            if self.state != RiskDecision.FREEZE:
                logger.error(f"CIRCUIT BREAKER: Drawdown {drawdown:.2%} > {self.max_drawdown_limit:.2%}. FREEZING.")
                self.state = RiskDecision.FREEZE
                self.cooldown_counter = self.cooldown_days
                self.recovery_level = 0
            return RiskDecision.FREEZE

        # Volatility spike breaker
        if self._current_realized_vol > self.target_vol_limit * 2.5: 
            if self.state != RiskDecision.FREEZE:
                logger.error(f"CIRCUIT BREAKER: Realized Vol {self._current_realized_vol:.2%} > 2.5x Target. FREEZING.")
                self.state = RiskDecision.FREEZE
                self.cooldown_counter = self.cooldown_days
                self.recovery_level = 0
            return RiskDecision.FREEZE

        return self.state

    def process_state_daily(self, is_risk_on: bool):
        """
        DAILY Risk State Machine Progression.
        Handles:
        - Cooldown countdown during FREEZE.
        RECOVERY progression is handled during REBALANCE in check_pre_trade.
        """
        if self.state == RiskDecision.FREEZE:
            self.cooldown_counter -= 1
            if self.cooldown_counter <= 0:
                logger.info("Risk Cooldown COMPLETE. Entering Gradual RECOVERY Phase.")
                self.state = RiskDecision.RECOVERY
                self.recovery_level = 1 # Start at Phase 1 (20%)
                self._max_equity = -1.0 
            else:
                if self.cooldown_counter % 5 == 0 and self.cooldown_counter != self._last_logged_cooldown:
                    logger.info(f"Risk Cooldown: {self.cooldown_counter} days remaining.")
                    self._last_logged_cooldown = self.cooldown_counter
