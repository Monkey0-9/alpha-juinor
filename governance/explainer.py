"""
governance/explainer.py

Decision Explainability Module.
Generates human-readable explanations for trading decisions.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

logger = logging.getLogger("EXPLAINER")


@dataclass
class Explanation:
    """Structured explanation for a decision."""
    symbol: str
    decision: str
    summary: str
    factors: List[Dict[str, Any]]
    confidence_breakdown: Dict[str, float]
    risk_flags: List[str]
    regime_context: str


class DecisionExplainer:
    """
    Generate explanations for trading decisions.

    Features:
    - Factor attribution
    - Risk flag highlighting
    - Regime context
    - Human-readable summaries
    """

    FACTOR_WEIGHTS = {
        "momentum": 0.25,
        "mean_reversion": 0.20,
        "sentiment": 0.15,
        "ml_signal": 0.25,
        "regime": 0.15
    }

    def __init__(self):
        logger.info("[EXPLAINER] Initialized")

    def explain(
        self,
        symbol: str,
        decision: str,
        alpha_outputs: Dict[str, Any],
        regime: str = "UNKNOWN",
        risk_checks: Optional[Dict[str, bool]] = None
    ) -> Explanation:
        """Generate explanation for a trading decision."""
        factors = []
        conf_breakdown = {}

        # Parse alpha outputs into factors
        for source, output in alpha_outputs.items():
            if isinstance(output, dict):
                mu = output.get("mu", 0.0)
                conf = output.get("confidence", 0.0)
            else:
                mu = getattr(output, "mu", 0.0)
                conf = getattr(output, "confidence", 0.0)

            factors.append({
                "source": source,
                "signal": mu,
                "confidence": conf,
                "contribution": mu * conf
            })
            conf_breakdown[source] = conf

        # Identify risk flags
        risk_flags = []
        if risk_checks:
            for check, passed in risk_checks.items():
                if not passed:
                    risk_flags.append(f"FAILED: {check}")

        # Generate summary
        summary = self._generate_summary(
            symbol, decision, factors, regime, risk_flags
        )

        return Explanation(
            symbol=symbol,
            decision=decision,
            summary=summary,
            factors=factors,
            confidence_breakdown=conf_breakdown,
            risk_flags=risk_flags,
            regime_context=f"Market regime: {regime}"
        )

    def _generate_summary(
        self,
        symbol: str,
        decision: str,
        factors: List[Dict],
        regime: str,
        risk_flags: List[str]
    ) -> str:
        """Generate human-readable summary."""
        if not factors:
            return f"{symbol}: {decision} (No alpha signals)"

        # Find dominant factor
        sorted_factors = sorted(
            factors, key=lambda x: abs(x["contribution"]), reverse=True
        )
        dominant = sorted_factors[0] if sorted_factors else None

        if dominant:
            direction = "bullish" if dominant["signal"] > 0 else "bearish"
            summary = (
                f"{symbol}: {decision} driven by {dominant['source']} "
                f"({direction}, conf={dominant['confidence']:.0%})"
            )
        else:
            summary = f"{symbol}: {decision}"

        if risk_flags:
            summary += f" [RISK: {len(risk_flags)} flags]"

        if regime != "UNKNOWN":
            summary += f" [Regime: {regime}]"

        return summary


# Singleton
_instance: Optional[DecisionExplainer] = None


def get_explainer() -> DecisionExplainer:
    global _instance
    if _instance is None:
        _instance = DecisionExplainer()
    return _instance
