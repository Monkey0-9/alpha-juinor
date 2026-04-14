
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("EXPLAINER")

@dataclass
class DecisionExplanation:
    """Structured explanation for a decision."""
    symbol: str
    action: str
    reasoning: List[str] = field(default_factory=list)
    confidence_low: float = 0.0
    confidence_high: float = 0.0
    risk_checks: Dict[str, bool] = field(default_factory=dict)
    governance_status: str = "PENDING"
    summary: str = ""
    factors: List[Dict[str, Any]] = field(default_factory=list)
    confidence_breakdown: Dict[str, float] = field(default_factory=dict)
    risk_flags: List[str] = field(default_factory=list)
    regime_context: str = "UNKNOWN"

class DecisionExplainer:
    """
    Generate institutional-grade explanations for trading decisions.
    """

    def __init__(self):
        logger.info("[EXPLAINER] Initialized")

    def explain_trade(
        self,
        symbol: str,
        action: str,
        signal_strength: float,
        signal_components: Dict[str, float],
        position_size: float,
        adv: float,
        risk_metrics: Dict[str, Tuple[float, float, str]],
        governance_approved: bool,
        confidence_interval: Tuple[float, float]
    ) -> DecisionExplanation:
        """Detailed trade explanation for operators."""
        
        reasoning = []
        risk_checks = {"governance": governance_approved}
        
        # 1. Alpha Reasoning
        direction = "bullish" if signal_strength > 0 else "bearish"
        reasoning.append(f"✓ Alpha Signal: {abs(signal_strength):.2f}σ ({direction})")
        for comp, val in signal_components.items():
            reasoning.append(f"  - {comp}: {val:+.2f}")

        # 2. Liquidity Reasoning
        participation = (position_size / adv) if adv > 0 else 0
        liquidity_ok = participation < 0.1 # Institutional rule: < 10% ADV
        risk_checks["liquidity"] = liquidity_ok
        reasoning.append(f"{'✓' if liquidity_ok else '✗'} Liquidity: {participation:.2%} of ADV")

        # 3. Risk Reasoning
        for metric, (val, threshold, op) in risk_metrics.items():
            passed = False
            if op == "<": passed = val < threshold
            elif op == ">": passed = val > threshold
            risk_checks[metric] = passed
            reasoning.append(f"{'✓' if passed else '✗'} Risk Check ({metric}): {val:.2f} {op} {threshold}")

        status = "APPROVED" if governance_approved and all(risk_checks.values()) else "BLOCKED"

        return DecisionExplanation(
            symbol=symbol,
            action=action,
            reasoning=reasoning,
            confidence_low=confidence_interval[0],
            confidence_high=confidence_interval[1],
            risk_checks=risk_checks,
            governance_status=status
        )

    def explain_skip(self, symbol: str, skip_reasons: List[Tuple[str, str]]) -> DecisionExplanation:
        """Why did we NOT trade?"""
        reasoning = [f"✗ {cat.upper()}: {msg}" for cat, msg in skip_reasons]
        return DecisionExplanation(
            symbol=symbol,
            action="SKIP",
            reasoning=reasoning,
            governance_status="BLOCKED"
        )

    def compute_confidence_bands(
        self, 
        signal_strength: float, 
        historical_ic: float = 0.05, 
        regime_uncertainty: float = 0.1
    ) -> Tuple[float, float]:
        """Compute Bayesian confidence intervals for the signal."""
        base_conf = min(0.9, abs(signal_strength) * historical_ic * 10)
        uncertainty = regime_uncertainty * (1.0 / (abs(signal_strength) + 1e-6))
        return (max(0, base_conf - uncertainty), min(1.0, base_conf + uncertainty))

    def format_explanation(self, exp: DecisionExplanation) -> str:
        """Format explanation into a readable string."""
        lines = [
            f"--- DECISION EXPLANATION: {exp.action} {exp.symbol} ---",
            f"STATUS: {exp.governance_status}",
            f"Confidence: [{exp.confidence_low:.2f} - {exp.confidence_high:.2f}]",
            "REASONING:"
        ]
        lines.extend(exp.reasoning)
        return "\n".join(lines)

    def explain(
        self,
        symbol: str,
        decision: str,
        alpha_outputs: Dict[str, Any],
        regime: str = "UNKNOWN",
        risk_checks: Optional[Dict[str, bool]] = None,
    ) -> DecisionExplanation:
        """Legacy compatibility wrapper for older agent loops."""
        factors = []
        conf_breakdown = {}

        for source, output in alpha_outputs.items():
            mu = output.get("mu", 0.0) if isinstance(output, dict) else getattr(output, "mu", 0.0)
            conf = output.get("confidence", 0.0) if isinstance(output, dict) else getattr(output, "confidence", 0.0)
            factors.append({"source": source, "signal": mu, "confidence": conf, "contribution": mu * conf})
            conf_breakdown[source] = conf

        risk_flags = [f"FAILED: {k}" for k, v in (risk_checks or {}).items() if not v]
        
        return DecisionExplanation(
            symbol=symbol,
            action=decision,
            factors=factors,
            confidence_breakdown=conf_breakdown,
            risk_flags=risk_flags,
            regime_context=f"Market regime: {regime}",
            governance_status="APPROVED" if not risk_flags else "BLOCKED"
        )

def get_explainer() -> DecisionExplainer:
    return DecisionExplainer()
