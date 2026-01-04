# risk/engine.py
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional, Any
from dataclasses import dataclass, field
import logging
from risk.factor_model import StatisticalRiskModel
from risk.factor_exposure import FactorExposureEngine

logger = logging.getLogger(__name__)

from enum import Enum

class RiskDecision(Enum):
    ALLOW = "ALLOW"
    SCALE = "SCALE"
    REJECT = "REJECT" # Block this trade
    FREEZE = "FREEZE" # Stop all trading
    LIQUIDATE = "LIQUIDATE" # Close all positions
    RECOVERY = "RECOVERY" # Controlled re-entry phase

class RiskRegime(Enum):
    BULL_QUIET = "BULL_QUIET"       # Low Vol, Uptrend (Risk-On)
    BULL_VOLATILE = "BULL_VOLATILE" # High Vol, Uptrend (Caution)
    BEAR_QUIET = "BEAR_QUIET"       # Low Vol, Downtrend (Mean Reversion?)
    BEAR_CRISIS = "BEAR_CRISIS"     # High Vol, Downtrend (Risk-Off)
    UNCERTAIN = "UNCERTAIN"

@dataclass
class RiskCheckResult:
    ok: bool
    decision: RiskDecision
    scale_factor: float = 1.0
    violations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RiskSignalResult:
    adjusted_conviction: pd.Series
    estimated_leverage: float
    violations: List[str] = field(default_factory=list)

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
        
        # Market Regime
        self.regime = RiskRegime.BULL_QUIET
        self.is_risk_on = True
        self._spy_ma200 = 0.0
        self._last_spy_price = 0.0
        self._vol_history: List[float] = [] # Track rolling realized vol for percentile
        
        # Factor Risk
        self.pca = StatisticalRiskModel(n_components=3)
        self.factor_engine = FactorExposureEngine(self.pca)
        self.factor_limit = 0.50 # Max exposure to any single factor
        
        # Stress Scenarios (Institutional Shock Scenarios)
        self.stress_scenarios = {
            "Black Monday": -0.22,
            "2008 Crisis": -0.08,
            "2020 Covid": -0.12,
            "Inflation Shock": -0.05
        }
        self.stress_limit = 0.25 # Max allowable loss under extreme scenario

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

    def calculate_stress_loss(self, target_weights: Union[Dict, pd.Series]) -> Dict[str, float]:
        """
        Calculate potential portfolio loss under institutional stress scenarios.
        Uses a simplified beta-multiplier model.
        """
        # Generic beta of 1.0 if not calculated; in production this uses factor exposure
        portfolio_beta = 1.0 
        losses = {}
        
        # Robustly extract weights (handle Dict or Series)
        if isinstance(target_weights, dict):
             weights_iter = target_weights.values()
        else:
             # Assume Pandas Series or similar
             weights_iter = target_weights.values

        # Convert to list/array once to avoid multiple property accesses or generator exhaustion
        weights_array = list(weights_iter)
        
        for name, shock in self.stress_scenarios.items():
            expected_loss = sum(abs(w) for w in weights_array) * portfolio_beta * abs(shock)
            losses[name] = expected_loss
            
        return losses

    def update_regime(self, spy_history: pd.Series):
        """
        Update market regime state using Trend (MA200) and Volatility Regime (Percentile).
        """
        if spy_history.empty or len(spy_history) < 200:
            self.regime = RiskRegime.UNCERTAIN
            self.is_risk_on = True # Default to allow
            return

        # 1. Trend Component (MA200)
        # Using rolling mean over last 200 bars
        ma200 = spy_history.rolling(200).mean().iloc[-1]
        current_spy = spy_history.iloc[-1]
        
        # 2. Volatility Component
        # 21-day realized vol
        returns = spy_history.pct_change().dropna()
        if len(returns) < 21:
            current_vol = 0.01
        else:
            current_vol = returns.tail(21).std() * np.sqrt(252)
            
        # Update vol history for percentile (keep last 500 days approx)
        self._vol_history.append(current_vol)
        if len(self._vol_history) > 500:
            self._vol_history.pop(0)
            
        # Compute Vol Percentile
        if len(self._vol_history) > 50:
            vol_percentile = pd.Series(self._vol_history).rank(pct=True).iloc[-1]
        else:
            vol_percentile = 0.5 # Neutral
            
        # 3. Regime Classification
        is_uptrend = current_spy > ma200
        is_high_vol = vol_percentile > 0.80 # 80th percentile threshold for High Vol
        
        if is_uptrend and not is_high_vol:
            self.regime = RiskRegime.BULL_QUIET
            self.is_risk_on = True
        elif is_uptrend and is_high_vol:
            self.regime = RiskRegime.BULL_VOLATILE
            self.is_risk_on = True # Still Risk-On but maybe scaled down?
        elif not is_uptrend and not is_high_vol:
             self.regime = RiskRegime.BEAR_QUIET
             self.is_risk_on = False # Risk-Off or Mean Reversion only
        else:
             self.regime = RiskRegime.BEAR_CRISIS
             self.is_risk_on = False # Hard Risk-Off
             
        self._spy_ma200 = ma200
        self._last_spy_price = current_spy
        logger.info(f"REGIME UPDATE: {self.regime.value} | Vol: {current_vol:.1%} ({vol_percentile:.0%}ile) | Price: {current_spy:.2f} vs MA200: {ma200:.2f}")

    def check_pre_trade(
        self, 
        target_weights: Dict[str, float], 
        baskets_returns: pd.DataFrame,
        timestamp: pd.Timestamp,
        current_equity: float = 1.0
    ) -> RiskCheckResult:
        """
        PRE-TRADE: Scaling trade intent.
        Includes Elite Drawdown Control, Regime Awareness, and Controlled Recovery.
        """
        if self.state == RiskDecision.FREEZE:
            return RiskCheckResult(ok=False, decision=RiskDecision.REJECT, violations=["Risk State: FREEZE"])

        violations = []

        # 0. Proactive Drawdown Scaler (Elite Control)
        if current_equity > self._max_equity:
            self._max_equity = current_equity
            
        dd = (self._max_equity - current_equity) / self._max_equity if self._max_equity > 0 else 0.0
        
        dd_scalar = 1.0
        dd_violation = None
        if dd > self.max_drawdown_limit:
             return RiskCheckResult(
                 ok=False, 
                 decision=RiskDecision.REJECT, 
                 violations=[f"Drawdown {dd:.1%} > {self.max_drawdown_limit:.1%} Limit"]
             )
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
            if self.is_risk_on and is_stable:
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

        # 3. Stress Test Overlay (Tail Risk Guard)
        stress_losses = self.calculate_stress_loss(target_weights)
        max_stress_loss = max(stress_losses.values()) if stress_losses else 0.0
        
        if max_stress_loss > self.stress_limit:
            return RiskCheckResult(
                ok=False, 
                decision=RiskDecision.REJECT, 
                violations=[f"Stress Loss {max_stress_loss:.1%} > {self.stress_limit:.1%} Limit"]
            )

        # 4. Factor Exposure Guard (Systematic Risk Control)
        if not baskets_returns.empty:
            portfolio_exposures = self.factor_engine.calculate_exposures(target_weights, baskets_returns)
            factor_violations = self.factor_engine.check_exposure_limits(portfolio_exposures, limit=self.factor_limit)
            if factor_violations:
                 # Factor violations usually trigger scaling rather than rejection, 
                 # but for strict compliance we can return them as violations for scaling.
                 violations.extend(factor_violations)

        gross_leverage = sum(abs(w) for w in target_weights.values())
        if gross_leverage < 0.01:
            return RiskCheckResult(ok=True, decision=RiskDecision.ALLOW, scale_factor=1.0)

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

        # 2. Predictive Deleveraging (Trend + Vol Proxy)
        # 0.7x (30% haircut) during Risk-Off
        regime_scalar = 1.0
        if not self.is_risk_on:
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
            return RiskCheckResult(
                ok=False,
                decision=RiskDecision.SCALE,
                scale_factor=applied_scale,
                violations=violations
            )
        
        return RiskCheckResult(ok=True, decision=RiskDecision.ALLOW, scale_factor=1.0)

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

    def enforce_limits(self, conviction: pd.Series, prices: pd.Series, volumes: pd.Series) -> RiskSignalResult:
        """
        Institutional Per-Asset Limit Enforcement.
        
        This method is called by the strategy to apply vol-scaling and risk haircuts
        at the signal level, ensuring all trade intent is pre-vetted against the 
        current risk manifold.
        """
        if self.state == RiskDecision.FREEZE:
            return RiskSignalResult(
                adjusted_conviction=pd.Series([0.0], index=conviction.index), 
                estimated_leverage=0.0,
                violations=["Risk State: FREEZE"]
            )

        # 1. Realized Volatility Calculation
        returns = prices.pct_change().dropna()
        realized_vol = returns.std() * np.sqrt(252) if len(returns) > 10 else self.target_vol_limit
        realized_vol = max(realized_vol, 0.05) # Floor at 5% to avoid division by zero/extreme scaling

        # 2. Volatility Scaler (Target Vol Targeting)
        # MANDATORY FORMULA: scale = clip(target_vol / realized_vol, 0.3, 1.2)
        raw_vol_scale = self.target_vol_limit / realized_vol
        vol_scaler = np.clip(raw_vol_scale, 0.3, 1.2)
        
        # 3. Liquidity Haircut (Institutional Guard)
        # Penalize conviction if it implies > 5% of ADV
        adv = volumes.mean() if not volumes.empty else 0.0
        liq_scalar = 1.0
        
        # 4. Structural Risk Haircuts
        equity_nav = 1.0 
        
        # Calculate DD Scaler (Elite Control)
        dd = (self._max_equity - equity_nav) / self._max_equity if self._max_equity > 0 else 0.0
        dd_scalar = 1.0
        violations = []
        if dd > 0.05:
            dd_scalar = max(0.0, (self.max_drawdown_limit - dd) / 0.13)
            violations.append(f"Drawdown {dd:.1%} (Scaler {dd_scalar:.2f})")

        # 4b. Regime Scalar (Institutional Guard)
        regime_scalar = 1.0
        if not self.is_risk_on:
             regime_scalar = 0.50 # Risk-Off: Cut exposure by half (Aggressive Protection)
             violations.append(f"Regime: Risk-Off ({self.regime.value}) -> 0.5x")
             
        if self.regime == RiskRegime.BEAR_CRISIS:
             regime_scalar = 0.0 # Hard Block in Crisis
             violations.append("Regime: CRISIS -> BLOCK")
            
        # 5. Apply Scaling
        # Combined scaler using the strict vol_scaler
        applied_scaler = vol_scaler * dd_scalar * liq_scalar * regime_scalar
        
        # If in recovery, apply recovery scalar
        if self.state == RiskDecision.RECOVERY:
            rec_scalar = (self.recovery_level * 0.20)
            applied_scaler *= rec_scalar
            violations.append(f"RECOVERY Level {self.recovery_level} (scaler {rec_scalar:.2f})")
            
        # 6. Apply Liquidity Penalty AFTER vol-scaling
        # Correctly calculate ADV in DOLLARS
        current_price = prices.iloc[-1] if not prices.empty else 0.0
        adv_dollars = adv * current_price
        
        # If adjusted conviction (potential leverage) is > 10% of ADV Dollars, penalize
        if adv_dollars > 0 and abs(conviction.iloc[-1]) > (adv_dollars * 0.10):
             liq_scalar = 0.8
             applied_scaler *= liq_scalar
             violations.append(f"Liquidity Haircut (ADV ${adv_dollars:,.0f})")

        adjusted_conviction = conviction * applied_scaler
        
        # 7. Return adjusted signal and estimated leverage
        # Lev is estimate of notional for this asset
        est_lev = float(adjusted_conviction.iloc[-1]) if not adjusted_conviction.empty else 0.0
        
        if applied_scaler < 0.98:
            violations.append(f"Total Risk Scalar: {applied_scaler:.2f}")

        return RiskSignalResult(
            adjusted_conviction=adjusted_conviction,
            estimated_leverage=est_lev,
            violations=violations
        )

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
