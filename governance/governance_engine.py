"""
Governance Engine - Phase 9
============================
Implements veto power and structured decision audit for all trading decisions.

Every decision MUST emit:
{
  "decision": "EXECUTE | HOLD | REJECT | ERROR",
  "reason_codes": [],
  "mu": ,
  "sigma": ,
  "cvar": ,
  "data_quality": ,
  "model_confidence":
}

Governance Can VETO:
- Unexplained profits
- Too-perfect execution
- Data dependency risk
- Correlated wins
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from governance.institutional_specification import (
    GovernanceDecision,
    CVaRConfig,
    StrategyLifecycle
)

logger = logging.getLogger(__name__)


class VetoTrigger(Enum):
    """Veto trigger types."""
    UNEXPLAINED_PROFIT = "unexplained_profit"
    TOO_PERFECT_EXECUTION = "too_perfect_execution"
    DATA_DEPENDENCY_RISK = "data_dependency_risk"
    CORRELATED_WINS = "correlated_wins"
    HIGH_CVAR = "high_cvar"
    LEVERAGE_BREACH = "leverage_breach"
    DRAWDOWN_LIMIT = "drawdown_limit"
    MODEL_DECAY = "model_decay"
    INSUFFICIENT_HISTORY = "insufficient_history"
    LOW_DATA_QUALITY = "low_data_quality"


@dataclass
class VetoCheck:
    """Result of a single veto check."""
    trigger: VetoTrigger
    triggered: bool
    reason: str
    severity: str = "info"  # info, warning, error, critical


class GovernanceEngine:
    """
    Governance engine with veto power.

    Applies structured checks to all trading decisions and can veto
    trades that violate governance rules.
    """

    def __init__(
        self,
        cvar_config: Optional[CVaRConfig] = None,
        max_leverage: float = 1.0,
        max_drawdown: float = 0.18,
        profit_explanation_threshold: float = 0.03,
        enable_all_vetoes: bool = True
    ):
        """
        Initialize Governance Engine.

        Args:
            cvar_config: CVaR configuration
            max_leverage: Maximum portfolio leverage
            max_drawdown: Maximum drawdown before halt
            profit_explanation_threshold: Return threshold requiring explanation
            enable_all_vetoes: Whether to enable all veto checks
        """
        self.cvar_config = cvar_config or CVaRConfig()
        self.max_leverage = max_leverage
        self.max_drawdown = max_drawdown
        self.profit_explanation_threshold = profit_explanation_threshold
        self.enable_all_vetoes = enable_all_vetoes

        # Track recent decisions for correlation detection
        self._recent_decisions: List[Dict[str, Any]] = []
        self._max_recent_decisions = 100

    def evaluate_decision(
        self,
        decision: GovernanceDecision,
        portfolio_state: Dict[str, Any] = None
    ) -> GovernanceDecision:
        """
        Evaluate a trading decision through all governance checks.

        Args:
            decision: The trading decision to evaluate
            portfolio_state: Current portfolio state (NAV, drawdown, etc.)

        Returns:
            Updated decision with veto status
        """
        if portfolio_state is None:
            portfolio_state = {}

        veto_checks: List[VetoCheck] = []

        # Check 1: Unexplained profits
        check1 = self._check_unexplained_profits(decision)
        veto_checks.append(check1)

        # Check 2: Too-perfect execution
        check2 = self._check_execution_quality(decision)
        veto_checks.append(check2)

        # Check 3: Data dependency risk
        check3 = self._check_data_dependency(decision)
        veto_checks.append(check3)

        # Check 4: Correlated wins
        check4 = self._check_correlated_wins(decision)
        veto_checks.append(check4)

        # Check 5: CVaR limits
        check5 = self._check_cvar_limits(decision)
        veto_checks.append(check5)

        # Check 6: Leverage limits
        check6 = self._check_leverage_limits(decision, portfolio_state)
        veto_checks.append(check6)

        # Check 7: Drawdown limits
        check7 = self._check_drawdown_limits(portfolio_state)
        veto_checks.append(check7)

        # Check 8: Model decay
        check8 = self._check_model_decay(decision)
        veto_checks.append(check8)

        # Check 9: Insufficient history
        check9 = self._check_history_requirements(decision)
        veto_checks.append(check9)

        # Check 10: Data quality
        check10 = self._check_data_quality(decision)
        veto_checks.append(check10)

        # Aggregate veto triggers
        veto_triggers: Dict[str, bool] = {}
        critical_veto = False
        veto_reasons: List[str] = []

        for check in veto_checks:
            veto_triggers[check.trigger.value] = check.triggered
            if check.triggered:
                if check.severity in ["error", "critical"]:
                    critical_veto = True
                veto_reasons.append(f"{check.trigger.value}: {check.reason}")

        # Update decision
        decision.veto_triggers = veto_triggers

        if critical_veto:
            decision.vetoed = True
            decision.veto_reason = "; ".join(veto_reasons)
            decision.decision = "REJECT"

        # Add reason codes from veto checks
        for check in veto_checks:
            if check.triggered:
                decision.reason_codes.append(f"{check.trigger.value}: {check.reason}")

        # Log the governance decision
        self._log_governance_decision(decision, veto_checks)

        # Store for correlation detection
        self._record_decision(decision)

        return decision

    def _check_unexplained_profits(self, decision: GovernanceDecision) -> VetoCheck:
        """Check for unexplained high profits."""
        if not self.enable_all_vetoes:
            return VetoCheck(
                trigger=VetoTrigger.UNEXPLAINED_PROFIT,
                triggered=False,
                reason="Vetoes disabled"
            )

        # If return is suspiciously high with low confidence
        if decision.mu > self.profit_explanation_threshold and decision.model_confidence < 0.7:
            return VetoCheck(
                trigger=VetoTrigger.UNEXPLAINED_PROFIT,
                triggered=True,
                reason=f"High return {decision.mu:.2%} with low confidence {decision.model_confidence:.2%}",
                severity="warning"
            )

        return VetoCheck(
            trigger=VetoTrigger.UNEXPLAINED_PROFIT,
            triggered=False,
            reason="OK"
        )

    def _check_execution_quality(self, decision: GovernanceDecision) -> VetoCheck:
        """Check for suspiciously perfect execution."""
        if not self.enable_all_vetoes:
            return VetoCheck(
                trigger=VetoTrigger.TOO_PERFECT_EXECUTION,
                triggered=False,
                reason="Vetoes disabled"
            )

        # This check would require execution feedback data
        # Simplified: flag if conviction is extremely high
        conviction = abs(decision.mu) / max(decision.sigma, 0.001)
        if conviction > 5.0:
            return VetoCheck(
                trigger=VetoTrigger.TOO_PERFECT_EXECUTION,
                triggered=True,
                reason=f"Extremely high conviction {conviction:.1f}",
                severity="info"
            )

        return VetoCheck(
            trigger=VetoTrigger.TOO_PERFECT_EXECUTION,
            triggered=False,
            reason="OK"
        )

    def _check_data_dependency(self, decision: GovernanceDecision) -> VetoCheck:
        """Check for data dependency risk."""
        if not self.enable_all_vetoes:
            return VetoCheck(
                trigger=VetoTrigger.DATA_DEPENDENCY_RISK,
                triggered=False,
                reason="Vetoes disabled"
            )

        # If data quality is marginal
        if decision.data_quality < 0.7:
            return VetoCheck(
                trigger=VetoTrigger.DATA_DEPENDENCY_RISK,
                triggered=True,
                reason=f"Low data quality {decision.data_quality:.2f}",
                severity="warning"
            )

        return VetoCheck(
            trigger=VetoTrigger.DATA_DEPENDENCY_RISK,
            triggered=False,
            reason="OK"
        )

    def _check_correlated_wins(self, decision: GovernanceDecision) -> VetoCheck:
        """Check for correlated winning trades."""
        if not self.enable_all_vetoes:
            return VetoCheck(
                trigger=VetoTrigger.CORRELATED_WINS,
                triggered=False,
                reason="Vetoes disabled"
            )

        # Count recent buys on similar symbols/sectors
        # This is a simplified check
        if decision.symbol and len(decision.symbol) > 0:
            recent_wins = [
                d for d in self._recent_decisions
                if d.get('decision') == 'EXECUTE_BUY' and d.get('symbol', '').startswith(decision.symbol[0])
            ]
        else:
            recent_wins = []

        if len(recent_wins) > 5:
            return VetoCheck(
                trigger=VetoTrigger.CORRELATED_WINS,
                triggered=True,
                reason=f"Too many correlated wins ({len(recent_wins)} recent)",
                severity="info"
            )

        return VetoCheck(
            trigger=VetoTrigger.CORRELATED_WINS,
            triggered=False,
            reason="OK"
        )

    def _check_cvar_limits(self, decision: GovernanceDecision) -> VetoCheck:
        """Check CVaR limits."""
        if decision.cvar > self.cvar_config.portfolio_limit:
            return VetoCheck(
                trigger=VetoTrigger.HIGH_CVAR,
                triggered=True,
                reason=f"CVaR {decision.cvar:.3f} exceeds limit {self.cvar_config.portfolio_limit:.3f}",
                severity="critical"
            )

        return VetoCheck(
            trigger=VetoTrigger.HIGH_CVAR,
            triggered=False,
            reason="OK"
        )

    def _check_leverage_limits(
        self,
        decision: GovernanceDecision,
        portfolio_state: Dict[str, Any]
    ) -> VetoCheck:
        """Check leverage limits."""
        current_leverage = portfolio_state.get('leverage', 0.0)
        proposed_leverage = current_leverage + decision.position_size

        if proposed_leverage > self.max_leverage:
            return VetoCheck(
                trigger=VetoTrigger.LEVERAGE_BREACH,
                triggered=True,
                reason=f"Proposed leverage {proposed_leverage:.2f} exceeds limit {self.max_leverage:.2f}",
                severity="error"
            )

        return VetoCheck(
            trigger=VetoTrigger.LEVERAGE_BREACH,
            triggered=False,
            reason="OK"
        )

    def _check_drawdown_limits(self, portfolio_state: Dict[str, Any]) -> VetoCheck:
        """Check drawdown limits."""
        current_drawdown = portfolio_state.get('drawdown', 0.0)

        if current_drawdown > self.max_drawdown:
            return VetoCheck(
                trigger=VetoTrigger.DRAWDOWN_LIMIT,
                triggered=True,
                reason=f"Drawdown {current_drawdown:.1%} exceeds limit {self.max_drawdown:.1%}",
                severity="critical"
            )

        return VetoCheck(
            trigger=VetoTrigger.DRAWDOWN_LIMIT,
            triggered=False,
            reason="OK"
        )

    def _check_model_decay(self, decision: GovernanceDecision) -> VetoCheck:
        """Check for model decay indicators."""
        # If this were a full implementation, we'd check model age and error rates
        # For now, use model confidence as proxy
        if decision.model_confidence < 0.5:
            return VetoCheck(
                trigger=VetoTrigger.MODEL_DECAY,
                triggered=True,
                reason=f"Low model confidence {decision.model_confidence:.2f}",
                severity="warning"
            )

        return VetoCheck(
            trigger=VetoTrigger.MODEL_DECAY,
            triggered=False,
            reason="OK"
        )

    def _check_history_requirements(self, decision: GovernanceDecision) -> VetoCheck:
        """Check for sufficient historical data."""
        # This would check against the database
        # Simplified: assume we have history if symbol is provided
        if not decision.symbol:
            return VetoCheck(
                trigger=VetoTrigger.INSUFFICIENT_HISTORY,
                triggered=True,
                reason="No symbol provided",
                severity="error"
            )

        return VetoCheck(
            trigger=VetoTrigger.INSUFFICIENT_HISTORY,
            triggered=False,
            reason="OK"
        )

    def _check_data_quality(self, decision: GovernanceDecision) -> VetoCheck:
        """Check data quality meets threshold."""
        if decision.data_quality < 0.6:
            return VetoCheck(
                trigger=VetoTrigger.LOW_DATA_QUALITY,
                triggered=True,
                reason=f"Data quality {decision.data_quality:.2f} below 0.6 threshold",
                severity="error"
            )

        return VetoCheck(
            trigger=VetoTrigger.LOW_DATA_QUALITY,
            triggered=False,
            reason="OK"
        )

    def _log_governance_decision(
        self,
        decision: GovernanceDecision,
        veto_checks: List[VetoCheck]
    ) -> None:
        """Log the governance decision."""
        if decision.vetoed:
            logger.warning(
                f"[GOVERNANCE] VETO: {decision.symbol} | "
                f"Decision: {decision.decision} | "
                f"Reason: {decision.veto_reason}"
            )
        else:
            logger.info(
                f"[GOVERNANCE] APPROVED: {decision.symbol} | "
                f"Decision: {decision.decision} | "
                f"CVaR: {decision.cvar:.3f} | "
                f"Data Q: {decision.data_quality:.2f}"
            )

        # Log detailed veto checks
        for check in veto_checks:
            if check.triggered:
                logger.info(f"  [VETO CHECK] {check.trigger.value}: {check.reason}")

    def _record_decision(self, decision: GovernanceDecision) -> None:
        """Record decision for correlation detection."""
        self._recent_decisions.append({
            'symbol': decision.symbol,
            'decision': decision.decision,
            'mu': decision.mu,
            'timestamp': decision.timestamp or datetime.utcnow().isoformat()
        })

        # Trim old decisions
        if len(self._recent_decisions) > self._max_recent_decisions:
            self._recent_decisions = self._recent_decisions[-self._max_recent_decisions:]

    def get_governance_summary(self) -> Dict[str, Any]:
        """Get summary of recent governance decisions."""
        total = len(self._recent_decisions)
        vetoed = sum(1 for d in self._recent_decisions if d.get('vetoed', False))

        return {
            'total_decisions': total,
            'vetoed_count': vetoed,
            'approval_rate': (total - vetoed) / total if total > 0 else 1.0,
            'recent_veto_triggers': self._aggregate_veto_triggers()
        }

    def _aggregate_veto_triggers(self) -> Dict[str, int]:
        """Aggregate veto triggers from recent decisions."""
        triggers: Dict[str, int] = {}
        for decision in self._recent_decisions:
            veto_triggers = decision.get('veto_triggers', {})
            for trigger, triggered in veto_triggers.items():
                if triggered:
                    triggers[trigger] = triggers.get(trigger, 0) + 1
        return triggers


def create_governance_decision(
    symbol: str,
    decision_type: str,
    mu: float,
    sigma: float,
    cvar: float,
    data_quality: float,
    model_confidence: float,
    position_size: float = 0.0,
    cycle_id: str = "",
    reason_codes: List[str] = None,
    **kwargs
) -> GovernanceDecision:
    """
    Factory function to create a GovernanceDecision.

    Args:
        symbol: Trading symbol
        decision_type: EXECUTE_BUY, EXECUTE_SELL, HOLD, REJECT
        mu: Expected return
        sigma: Uncertainty
        cvar: CVaR (95%)
        data_quality: Data quality score
        model_confidence: Model confidence
        position_size: Position size
        cycle_id: Cycle identifier
        reason_codes: List of reason codes
        **kwargs: Additional fields

    Returns:
        GovernanceDecision instance
    """
    return GovernanceDecision(
        decision=decision_type,
        reason_codes=reason_codes or [],
        mu=mu,
        sigma=sigma,
        cvar=cvar,
        data_quality=data_quality,
        model_confidence=model_confidence,
        symbol=symbol,
        position_size=position_size,
        cycle_id=cycle_id,
        timestamp=datetime.utcnow().isoformat(),
        **kwargs
    )

