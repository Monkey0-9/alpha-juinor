"""
governance/explainer.py

Operator Trust & Explainability.
Makes the system "comfortable" not just correct:
1. Human-readable decision explanations
2. "Why NOT trade" logic
3. Confidence bands on all decisions
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger("EXPLAINER")

@dataclass
class DecisionExplanation:
    symbol: str
    action: str  # BUY, SELL, SKIP
    reasoning: List[str]  # List of human-readable reasons
    confidence_low: float  # Lower bound of confidence interval
    confidence_high: float  # Upper bound
    risk_checks: Dict[str, bool]  # {check_name: passed}
    governance_status: str  # APPROVED, BLOCKED, MANUAL_REVIEW

class DecisionExplainer:
    def __init__(self):
        self.explanation_templates = {
            "alpha_signal": "✓ Alpha Signal: {signal_strength:.1f}σ ({components})",
            "risk_pass": "✓ Risk Check: {metric} {value} {comparison} {limit}",
            "risk_fail": "✗ Risk Limit: {metric} {value} {comparison} {limit}",
            "liquidity_pass": "✓ Liquidity: {size} shares = {pct_adv:.2%} ADV (safe)",
            "liquidity_fail": "✗ Liquidity: {size} shares = {pct_adv:.2%} ADV exceeds {limit:.1%}",
            "governance_pass": "✓ Governance: Approved by PM Brain",
            "governance_block": "✗ Governance Block: {reason}",
            "low_confidence": "⚠ Low Confidence: {reason}",
        }

    def explain_trade(self,
                     symbol: str,
                     action: str,
                     signal_strength: float,
                     signal_components: Dict[str, float],
                     position_size: int,
                     adv: float,
                     risk_metrics: Dict[str, Tuple[float, float, str]],  # {metric: (value, limit, comparison)}
                     governance_approved: bool,
                     confidence_interval: Tuple[float, float]) -> DecisionExplanation:
        """
        Generate human-readable explanation for a trade decision.

        Args:
            symbol: Trading symbol
            action: BUY or SELL
            signal_strength: Alpha signal in standard deviations
            signal_components: Dict of {component_name: contribution}
            position_size: Number of shares
            adv: Average daily volume
            risk_metrics: Risk check results
            governance_approved: Whether governance approved
            confidence_interval: (low, high) confidence bounds

        Returns:
            DecisionExplanation
        """
        reasoning = []
        risk_checks = {}

        # 1. Alpha Signal
        components_str = " + ".join([f"{k}" for k, v in signal_components.items() if v > 0])
        reasoning.append(self.explanation_templates["alpha_signal"].format(
            signal_strength=signal_strength,
            components=components_str or "Mixed"
        ))

        # 2. Risk Checks
        all_risk_pass = True
        for metric, (value, limit, comparison) in risk_metrics.items():
            passed = self._evaluate_comparison(value, limit, comparison)
            risk_checks[metric] = passed

            if passed:
                reasoning.append(self.explanation_templates["risk_pass"].format(
                    metric=metric,
                    value=f"{value:.2f}",
                    comparison=comparison,
                    limit=f"{limit:.2f}"
                ))
            else:
                reasoning.append(self.explanation_templates["risk_fail"].format(
                    metric=metric,
                    value=f"{value:.2f}",
                    comparison=comparison,
                    limit=f"{limit:.2f}"
                ))
                all_risk_pass = False

        # 3. Liquidity
        pct_adv = position_size / adv if adv > 0 else 999
        if pct_adv < 0.05:  # 5% threshold
            reasoning.append(self.explanation_templates["liquidity_pass"].format(
                size=position_size,
                pct_adv=pct_adv
            ))
            risk_checks["liquidity"] = True
        else:
            reasoning.append(self.explanation_templates["liquidity_fail"].format(
                size=position_size,
                pct_adv=pct_adv,
                limit=0.05
            ))
            risk_checks["liquidity"] = False
            all_risk_pass = False

        # 4. Governance
        if governance_approved:
            reasoning.append(self.explanation_templates["governance_pass"])
            risk_checks["governance"] = True
        else:
            reasoning.append(self.explanation_templates["governance_block"].format(
                reason="Manual review required"
            ))
            risk_checks["governance"] = False

        # 5. Confidence Warning
        if signal_strength < 1.5:
            reasoning.append(self.explanation_templates["low_confidence"].format(
                reason=f"Alpha signal {signal_strength:.1f}σ (threshold 1.5σ)"
            ))

        governance_status = "APPROVED" if all_risk_pass and governance_approved else "BLOCKED"

        return DecisionExplanation(
            symbol=symbol,
            action=action,
            reasoning=reasoning,
            confidence_low=confidence_interval[0],
            confidence_high=confidence_interval[1],
            risk_checks=risk_checks,
            governance_status=governance_status
        )

    def explain_skip(self,
                    symbol: str,
                    skip_reasons: List[Tuple[str, str]]) -> DecisionExplanation:
        """
        Generate "Why NOT trade" explanation.

        Args:
            symbol: Symbol
            skip_reasons: List of (category, reason) tuples

        Returns:
            DecisionExplanation
        """
        reasoning = [f"Decision: SKIP {symbol}", "Reasoning:"]
        risk_checks = {}

        for category, reason in skip_reasons:
            if category == "governance":
                reasoning.append(self.explanation_templates["governance_block"].format(reason=reason))
                risk_checks["governance"] = False
            elif category == "risk":
                reasoning.append(f"✗ {reason}")
                risk_checks[reason.split(":")[0]] = False
            elif category == "confidence":
                reasoning.append(self.explanation_templates["low_confidence"].format(reason=reason))
            else:
                reasoning.append(f"⚠ {reason}")

        return DecisionExplanation(
            symbol=symbol,
            action="SKIP",
            reasoning=reasoning,
            confidence_low=0.0,
            confidence_high=0.0,
            risk_checks=risk_checks,
            governance_status="BLOCKED"
        )

    def compute_confidence_bands(self,
                                signal_strength: float,
                                historical_ic: float,
                                regime_uncertainty: float = 0.1) -> Tuple[float, float]:
        """
        Compute confidence interval for decision.

        Confidence = f(signal_strength, historical_IC, regime_uncertainty)

        Args:
            signal_strength: Alpha signal in σ
            historical_ic: Historical information coefficient
            regime_uncertainty: Current regime uncertainty (0-1)

        Returns:
            (confidence_low, confidence_high)
        """
        # Base confidence from signal strength
        # Strong signal (>2σ) → high confidence
        # Weak signal (<1σ) → low confidence
        base_confidence = min(0.95, max(0.05, signal_strength / 3.0))

        # Adjust for historical IC
        # High IC → tighter bands
        # Low IC → wider bands
        ic_factor = max(0.5, min(1.5, historical_ic / 0.05))  # Normalize around 0.05

        # Adjust for regime uncertainty
        uncertainty_factor = 1.0 + regime_uncertainty

        # Compute interval width
        interval_width = (1.0 - base_confidence) * uncertainty_factor / ic_factor

        confidence_low = max(0.0, base_confidence - interval_width / 2)
        confidence_high = min(1.0, base_confidence + interval_width / 2)

        return (confidence_low, confidence_high)

    def _evaluate_comparison(self, value: float, limit: float, comparison: str) -> bool:
        """Evaluate comparison operator"""
        if comparison == "<":
            return value < limit
        elif comparison == "<=":
            return value <= limit
        elif comparison == ">":
            return value > limit
        elif comparison == ">=":
            return value >= limit
        else:
            return True

    def format_explanation(self, explanation: DecisionExplanation) -> str:
        """Format explanation as human-readable string"""
        lines = [
            f"Decision: {explanation.action} {explanation.symbol}",
            f"Confidence: [{explanation.confidence_low:.0%} - {explanation.confidence_high:.0%}]",
            "Reasoning:"
        ]
        lines.extend([f"  {r}" for r in explanation.reasoning])
        lines.append(f"Status: {explanation.governance_status}")

        return "\n".join(lines)
