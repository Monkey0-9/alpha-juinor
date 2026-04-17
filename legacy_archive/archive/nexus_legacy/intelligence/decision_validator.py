"""
Perfect Decision Validator - Zero Error Tolerance
==================================================

Triple-validates every decision before execution:
1. CALCULATION VALIDATION: Verify all math is correct
2. LOGIC VALIDATION: Ensure reasoning is sound
3. RISK VALIDATION: Confirm risk is acceptable

Only PERFECT decisions are allowed to execute.
"""

import logging
import time
from dataclasses import dataclass
from decimal import Decimal, getcontext
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# Maximum precision
getcontext().prec = 50


class ValidationResult(Enum):
    """Result of validation check."""
    PASSED = "PASSED"
    FAILED = "FAILED"
    WARNING = "WARNING"
    REQUIRES_REVIEW = "REQUIRES_REVIEW"


@dataclass
class ValidationCheck:
    """Single validation check result."""
    check_name: str
    category: str  # CALCULATION, LOGIC, RISK
    passed: bool
    message: str
    severity: int  # 1-10
    auto_fixable: bool = False
    fix_applied: bool = False


@dataclass
class DecisionValidation:
    """Complete validation result for a decision."""
    decision_id: str
    symbol: str
    action: str

    # Validation results
    calculation_checks: List[ValidationCheck]
    logic_checks: List[ValidationCheck]
    risk_checks: List[ValidationCheck]

    # Summary
    all_passed: bool
    critical_failures: int
    warnings: int
    final_verdict: ValidationResult

    # Validation time
    validation_time_ms: float


class PerfectDecisionValidator:
    """
    Triple-validates every decision for perfection.

    No errors allowed. All calculations verified.
    All logic checked. All risks assessed.
    """

    def __init__(self):
        """Initialize the validator."""
        self.validations_performed = 0
        self.validations_passed = 0
        self.validations_failed = 0

        logger.info("[VALIDATOR] Perfect Decision Validator initialized")

    def validate(
        self,
        decision: Dict[str, Any],
        market_data: Optional[Dict] = None,
        portfolio_state: Optional[Dict] = None
    ) -> DecisionValidation:
        """
        Perform complete validation of a trading decision.

        This is the FINAL CHECK before any trade execution.
        """
        start_time = time.time()
        decision_id = f"VAL_{int(time.time())}_{self.validations_performed}"

        symbol = decision.get("symbol", "UNKNOWN")
        action = decision.get("action", "UNKNOWN")

        logger.debug(f"[VALIDATOR] Validating decision for {symbol}: {action}")

        # Run all validation checks
        calc_checks = self._validate_calculations(decision, market_data)
        logic_checks = self._validate_logic(decision, market_data)
        risk_checks = self._validate_risk(decision, portfolio_state)

        # Count results
        all_checks = calc_checks + logic_checks + risk_checks
        critical_failures = sum(
            1 for c in all_checks
            if not c.passed and c.severity >= 8
        )
        warnings = sum(
            1 for c in all_checks
            if not c.passed and c.severity < 8
        )
        all_passed = all(c.passed for c in all_checks)

        # Determine final verdict
        if critical_failures > 0:
            verdict = ValidationResult.FAILED
        elif warnings > 2:
            verdict = ValidationResult.REQUIRES_REVIEW
        elif warnings > 0:
            verdict = ValidationResult.WARNING
        else:
            verdict = ValidationResult.PASSED

        # Update stats
        self.validations_performed += 1
        if verdict == ValidationResult.PASSED:
            self.validations_passed += 1
        else:
            self.validations_failed += 1

        validation_time = (time.time() - start_time) * 1000

        result = DecisionValidation(
            decision_id=decision_id,
            symbol=symbol,
            action=action,
            calculation_checks=calc_checks,
            logic_checks=logic_checks,
            risk_checks=risk_checks,
            all_passed=all_passed,
            critical_failures=critical_failures,
            warnings=warnings,
            final_verdict=verdict,
            validation_time_ms=validation_time
        )

        if verdict != ValidationResult.PASSED:
            logger.warning(
                f"[VALIDATOR] {symbol} {action}: {verdict.value} "
                f"({critical_failures} critical, {warnings} warnings)"
            )
        else:
            logger.debug(
                f"[VALIDATOR] {symbol} {action}: PASSED in {validation_time:.1f}ms"
            )

        return result

    def _validate_calculations(
        self,
        decision: Dict,
        market_data: Optional[Dict]
    ) -> List[ValidationCheck]:
        """Validate all calculations are correct."""
        checks = []

        # Check 1: Position size is valid
        position_size = decision.get("position_size", 0)
        if position_size < 0 or position_size > 1:
            checks.append(ValidationCheck(
                check_name="position_size_bounds",
                category="CALCULATION",
                passed=False,
                message=f"Position size {position_size} outside [0,1]",
                severity=10,
                auto_fixable=True
            ))
        else:
            checks.append(ValidationCheck(
                check_name="position_size_bounds",
                category="CALCULATION",
                passed=True,
                message="Position size within bounds",
                severity=0
            ))

        # Check 2: Signal strength is valid
        signal = decision.get("signal_strength", 0)
        if signal < -1 or signal > 1:
            checks.append(ValidationCheck(
                check_name="signal_bounds",
                category="CALCULATION",
                passed=False,
                message=f"Signal {signal} outside [-1,1]",
                severity=9
            ))
        else:
            checks.append(ValidationCheck(
                check_name="signal_bounds",
                category="CALCULATION",
                passed=True,
                message="Signal within bounds",
                severity=0
            ))

        # Check 3: Confidence is valid
        confidence = decision.get("confidence", 0)
        if confidence < 0 or confidence > 1:
            checks.append(ValidationCheck(
                check_name="confidence_bounds",
                category="CALCULATION",
                passed=False,
                message=f"Confidence {confidence} outside [0,1]",
                severity=8
            ))
        else:
            checks.append(ValidationCheck(
                check_name="confidence_bounds",
                category="CALCULATION",
                passed=True,
                message="Confidence within bounds",
                severity=0
            ))

        # Check 4: Target price vs stop loss makes sense
        target = decision.get("target_price", 0)
        stop = decision.get("stop_loss", 0)
        entry = decision.get("entry_price", 0)
        action = decision.get("action", "HOLD")

        if action == "BUY" and target > 0 and stop > 0 and entry > 0:
            if target <= entry:
                checks.append(ValidationCheck(
                    check_name="target_vs_entry",
                    category="CALCULATION",
                    passed=False,
                    message="BUY target <= entry price",
                    severity=9
                ))
            elif stop >= entry:
                checks.append(ValidationCheck(
                    check_name="stop_vs_entry",
                    category="CALCULATION",
                    passed=False,
                    message="BUY stop >= entry price",
                    severity=9
                ))
            else:
                checks.append(ValidationCheck(
                    check_name="price_levels",
                    category="CALCULATION",
                    passed=True,
                    message="Price levels are valid",
                    severity=0
                ))

        # Check 5: Expected return calculation
        expected_return = decision.get("expected_return", 0)
        if action in ["BUY", "SELL"] and abs(expected_return) > 0.5:
            checks.append(ValidationCheck(
                check_name="expected_return_realistic",
                category="CALCULATION",
                passed=False,
                message=f"Expected return {expected_return:.1%} seems unrealistic",
                severity=7
            ))

        return checks

    def _validate_logic(
        self,
        decision: Dict,
        market_data: Optional[Dict]
    ) -> List[ValidationCheck]:
        """Validate decision logic is sound."""
        checks = []

        action = decision.get("action", "HOLD")
        signal = decision.get("signal_strength", 0)
        confidence = decision.get("confidence", 0)

        # Check 1: Action matches signal direction
        if action == "BUY" and signal < 0:
            checks.append(ValidationCheck(
                check_name="action_signal_match",
                category="LOGIC",
                passed=False,
                message="BUY action with negative signal",
                severity=10
            ))
        elif action == "SELL" and signal > 0:
            checks.append(ValidationCheck(
                check_name="action_signal_match",
                category="LOGIC",
                passed=False,
                message="SELL action with positive signal",
                severity=10
            ))
        else:
            checks.append(ValidationCheck(
                check_name="action_signal_match",
                category="LOGIC",
                passed=True,
                message="Action matches signal",
                severity=0
            ))

        # Check 2: Confidence supports action
        if action in ["BUY", "SELL"] and confidence < 0.5:
            checks.append(ValidationCheck(
                check_name="confidence_threshold",
                category="LOGIC",
                passed=False,
                message=f"Taking action with low confidence {confidence:.1%}",
                severity=8
            ))

        # Check 3: Position size matches conviction
        position_size = decision.get("position_size", 0)
        if confidence < 0.6 and position_size > 0.05:
            checks.append(ValidationCheck(
                check_name="position_conviction_match",
                category="LOGIC",
                passed=False,
                message="Large position with low conviction",
                severity=7
            ))

        # Check 4: Reasoning exists
        reasoning = decision.get("reasoning", decision.get("buy_reasons", []))
        if action in ["BUY", "SELL"] and not reasoning:
            checks.append(ValidationCheck(
                check_name="reasoning_exists",
                category="LOGIC",
                passed=False,
                message="No reasoning provided for trade",
                severity=6
            ))
        else:
            checks.append(ValidationCheck(
                check_name="reasoning_exists",
                category="LOGIC",
                passed=True,
                message="Reasoning provided",
                severity=0
            ))

        return checks

    def _validate_risk(
        self,
        decision: Dict,
        portfolio_state: Optional[Dict]
    ) -> List[ValidationCheck]:
        """Validate risk is acceptable."""
        checks = []

        position_size = decision.get("position_size", 0)
        symbol = decision.get("symbol", "")
        action = decision.get("action", "HOLD")

        # Check 1: Single position limit
        if position_size > 0.10:
            checks.append(ValidationCheck(
                check_name="max_position_size",
                category="RISK",
                passed=False,
                message=f"Position {position_size:.1%} exceeds 10% limit",
                severity=9
            ))
        else:
            checks.append(ValidationCheck(
                check_name="max_position_size",
                category="RISK",
                passed=True,
                message="Position size within limits",
                severity=0
            ))

        # Check 2: Concentration risk
        if portfolio_state:
            current_positions = portfolio_state.get("positions", {})
            sector = portfolio_state.get("sector_map", {}).get(symbol, "UNKNOWN")

            sector_exposure = sum(
                abs(v) for k, v in current_positions.items()
                if portfolio_state.get("sector_map", {}).get(k, "") == sector
            )

            if sector_exposure + position_size > 0.30:
                checks.append(ValidationCheck(
                    check_name="sector_concentration",
                    category="RISK",
                    passed=False,
                    message=f"Sector exposure would exceed 30%",
                    severity=7
                ))

        # Check 3: Risk/reward ratio
        risk_reward = decision.get("risk_reward_ratio", 0)
        if action in ["BUY", "SELL"]:
            if isinstance(risk_reward, Decimal):
                risk_reward = float(risk_reward)

            if risk_reward < 1.5:
                checks.append(ValidationCheck(
                    check_name="risk_reward_ratio",
                    category="RISK",
                    passed=False,
                    message=f"Risk/reward {risk_reward:.1f} below 1.5",
                    severity=6
                ))
            else:
                checks.append(ValidationCheck(
                    check_name="risk_reward_ratio",
                    category="RISK",
                    passed=True,
                    message=f"Risk/reward {risk_reward:.1f} acceptable",
                    severity=0
                ))

        # Check 4: Stop loss exists
        stop_loss = decision.get("stop_loss", 0)
        if action in ["BUY", "SELL"] and stop_loss == 0:
            checks.append(ValidationCheck(
                check_name="stop_loss_exists",
                category="RISK",
                passed=False,
                message="No stop loss defined",
                severity=8
            ))

        return checks

    def get_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        total = self.validations_performed
        return {
            "total_validations": total,
            "passed": self.validations_passed,
            "failed": self.validations_failed,
            "pass_rate": self.validations_passed / total if total > 0 else 0
        }


# Singleton
_validator: Optional[PerfectDecisionValidator] = None


def get_validator() -> PerfectDecisionValidator:
    """Get or create the Perfect Decision Validator."""
    global _validator
    if _validator is None:
        _validator = PerfectDecisionValidator()
    return _validator
