"""
execution/impact_gate.py

Market Impact Gate (Ticket 14)

Comprehensive pre-trade impact checks with regime awareness.
Blocks or reduces orders that would have excessive market impact.

Features:
- Almgren-Chriss impact estimation
- Regime-adaptive impact thresholds
- ADV participation limits
- Slippage estimation
- Execution recommendation (REJECT/REDUCE/SLICE/APPROVE)
"""

import json
import logging
from datetime import datetime
from typing import Dict, Optional, Any, List, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

import numpy as np

logger = logging.getLogger("IMPACT_GATE")


class ImpactDecision(str, Enum):
    """Impact gate decision types."""
    APPROVE = "APPROVE"          # Trade within limits
    REDUCE = "REDUCE"            # Reduce size to fit
    SLICE = "SLICE"              # Execute in slices
    REJECT = "REJECT"            # Reject entirely


@dataclass
class ImpactGateResult:
    """Result of impact gate check."""
    decision: ImpactDecision
    symbol: str
    proposed_qty: float
    approved_qty: float
    estimated_impact_bps: float
    participation_rate: float
    slippage_estimate: float
    execution_tactic: str        # "IMMEDIATE", "TWAP", "VWAP", "PATIENTLY"
    reason: str
    slice_count: Optional[int]   # If SLICE, number of slices
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['decision'] = self.decision.value
        return d


class ImpactGate:
    """
    Market Impact Gate.

    Evaluates pre-trade market impact and recommends execution approach.
    Integrates with regime controller for dynamic thresholds.

    Impact Thresholds (regime-dependent):
    - RISK_ON: Max 30 bps impact, 15% ADV
    - RISK_OFF: Max 20 bps impact, 10% ADV
    - CRISIS: Max 10 bps impact, 5% ADV
    - LIQUIDITY_STRESS: Max 15 bps impact, 3% ADV, SLICE only

    Impact Model: Almgren-Chriss square-root formula
        Impact (bps) = k * sigma * sqrt(Q / ADV)
        where k ≈ 0.5 (temporary) + 0.1 (permanent)
    """

    # Base thresholds (adjusted by regime)
    BASE_MAX_IMPACT_BPS = 25.0
    BASE_MAX_ADV_PCT = 0.10

    # Impact model coefficients
    TEMP_IMPACT_COEFF = 0.5      # Temporary impact
    PERM_IMPACT_COEFF = 0.1     # Permanent impact
    SPREAD_DEFAULT_BPS = 5.0    # Default spread

    # Slicing parameters
    MIN_SLICE_SIZE_USD = 10000
    MAX_SLICES = 20

    # Regime multipliers
    REGIME_MULTIPLIERS = {
        "RISK_ON": {"impact": 1.2, "adv": 1.5},
        "RISK_OFF": {"impact": 0.8, "adv": 1.0},
        "CRISIS": {"impact": 0.4, "adv": 0.5},
        "LIQUIDITY_STRESS": {"impact": 0.6, "adv": 0.3},
        "UNKNOWN": {"impact": 0.7, "adv": 0.8}
    }

    def __init__(
        self,
        max_impact_bps: float = None,
        max_adv_pct: float = None,
        regime_controller = None
    ):
        """
        Initialize ImpactGate.

        Args:
            max_impact_bps: Maximum allowed impact in bps
            max_adv_pct: Maximum ADV participation
            regime_controller: RegimeController instance
        """
        self.max_impact_bps = max_impact_bps or self.BASE_MAX_IMPACT_BPS
        self.max_adv_pct = max_adv_pct or self.BASE_MAX_ADV_PCT
        self._regime_controller = regime_controller

        self._approval_count = 0
        self._rejection_count = 0
        self._reduction_count = 0
        self._decisions: List[ImpactGateResult] = []

    @property
    def regime_controller(self):
        if self._regime_controller is None:
            try:
                from regime.controller import get_regime_controller
                self._regime_controller = get_regime_controller()
            except Exception:
                pass
        return self._regime_controller

    def check_impact(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        volatility: float,
        adv: float,
        spread_bps: float = None
    ) -> ImpactGateResult:
        """
        Check market impact for a proposed trade.

        Args:
            symbol: Stock symbol
            side: "BUY" or "SELL"
            quantity: Number of shares
            price: Current price
            volatility: Daily volatility (decimal)
            adv: Average daily volume (shares)
            spread_bps: Bid-ask spread in bps

        Returns:
            ImpactGateResult with decision and details
        """
        now = datetime.utcnow().isoformat() + 'Z'
        spread_bps = spread_bps or self.SPREAD_DEFAULT_BPS

        # Get regime-adjusted thresholds
        max_impact, max_adv = self._get_regime_thresholds()

        # Calculate participation rate
        participation_rate = quantity / max(1, adv)

        # Estimate market impact using Almgren-Chriss
        impact_bps = self._estimate_impact(quantity, volatility, adv, spread_bps)

        # Estimate slippage
        slippage_est = self._estimate_slippage(quantity, price, volatility, adv)

        # ========== DECISION LOGIC ==========

        # Check ADV limit
        if participation_rate > max_adv:
            # Can we reduce to fit?
            reduced_qty = adv * max_adv * 0.9  # 90% of limit
            reduced_impact = self._estimate_impact(reduced_qty, volatility, adv, spread_bps)

            if reduced_impact < max_impact:
                result = ImpactGateResult(
                    decision=ImpactDecision.REDUCE,
                    symbol=symbol,
                    proposed_qty=quantity,
                    approved_qty=reduced_qty,
                    estimated_impact_bps=reduced_impact,
                    participation_rate=reduced_qty / max(1, adv),
                    slippage_estimate=slippage_est * (reduced_qty / quantity),
                    execution_tactic="TWAP",
                    reason=f"Reduced from {quantity:.0f} to {reduced_qty:.0f} (ADV limit {max_adv*100:.1f}%)",
                    slice_count=None,
                    timestamp=now
                )
                self._reduction_count += 1
            else:
                # Need to slice
                slice_result = self._calculate_slicing(
                    quantity, price, volatility, adv, max_impact, max_adv
                )
                result = ImpactGateResult(
                    decision=ImpactDecision.SLICE,
                    symbol=symbol,
                    proposed_qty=quantity,
                    approved_qty=quantity,
                    estimated_impact_bps=slice_result['per_slice_impact'],
                    participation_rate=slice_result['per_slice_participation'],
                    slippage_estimate=slippage_est,
                    execution_tactic="TWAP",
                    reason=f"Execute in {slice_result['n_slices']} slices",
                    slice_count=slice_result['n_slices'],
                    timestamp=now
                )
                self._approval_count += 1

        # Check impact limit
        elif impact_bps > max_impact:
            # Can we reduce to fit impact limit?
            # Impact ~ sqrt(Q/ADV), so Q ~ (Impact / k)^2 * ADV
            impact_ratio = (max_impact * 0.9) / max(1, impact_bps)
            reduced_qty = quantity * (impact_ratio ** 2)  # Quadratic scaling

            if reduced_qty >= quantity * 0.2:  # At least 20% of original
                result = ImpactGateResult(
                    decision=ImpactDecision.REDUCE,
                    symbol=symbol,
                    proposed_qty=quantity,
                    approved_qty=reduced_qty,
                    estimated_impact_bps=max_impact * 0.9,
                    participation_rate=reduced_qty / max(1, adv),
                    slippage_estimate=slippage_est * (reduced_qty / quantity),
                    execution_tactic="PATIENTLY",
                    reason=f"Reduced to fit impact limit ({impact_bps:.1f} -> {max_impact*0.9:.1f} bps)",
                    slice_count=None,
                    timestamp=now
                )
                self._reduction_count += 1
            else:
                # Reject - can't execute safely
                result = ImpactGateResult(
                    decision=ImpactDecision.REJECT,
                    symbol=symbol,
                    proposed_qty=quantity,
                    approved_qty=0,
                    estimated_impact_bps=impact_bps,
                    participation_rate=participation_rate,
                    slippage_estimate=slippage_est,
                    execution_tactic="NONE",
                    reason=f"Impact {impact_bps:.1f} bps exceeds limit {max_impact:.1f} bps",
                    slice_count=None,
                    timestamp=now
                )
                self._rejection_count += 1

                logger.warning(json.dumps({
                    "event": "IMPACT_GATE_REJECTION",
                    "symbol": symbol,
                    "side": side,
                    "quantity": quantity,
                    "impact_bps": impact_bps,
                    "limit_bps": max_impact
                }))

        else:
            # Approved - within all limits
            tactic = self._recommend_tactic(participation_rate, impact_bps, volatility)

            result = ImpactGateResult(
                decision=ImpactDecision.APPROVE,
                symbol=symbol,
                proposed_qty=quantity,
                approved_qty=quantity,
                estimated_impact_bps=impact_bps,
                participation_rate=participation_rate,
                slippage_estimate=slippage_est,
                execution_tactic=tactic,
                reason="Within impact and ADV limits",
                slice_count=None,
                timestamp=now
            )
            self._approval_count += 1

        self._decisions.append(result)

        logger.info(json.dumps({
            "event": "IMPACT_GATE_DECISION",
            "symbol": symbol,
            "decision": result.decision.value,
            "proposed_qty": quantity,
            "approved_qty": result.approved_qty,
            "impact_bps": result.estimated_impact_bps
        }))

        return result

    def _get_regime_thresholds(self) -> Tuple[float, float]:
        """Get regime-adjusted thresholds."""
        if self.regime_controller:
            state = self.regime_controller.get_current_state()
            if state:
                regime = state.regime.value
                mult = self.REGIME_MULTIPLIERS.get(regime, self.REGIME_MULTIPLIERS["UNKNOWN"])
                return (
                    self.max_impact_bps * mult["impact"],
                    self.max_adv_pct * mult["adv"]
                )
        return self.max_impact_bps, self.max_adv_pct

    def _estimate_impact(
        self,
        quantity: float,
        volatility: float,
        adv: float,
        spread_bps: float
    ) -> float:
        """
        Estimate market impact using Almgren-Chriss.

        Impact (bps) = spread + k * sigma * sqrt(Q/ADV) * 10000
        """
        if adv <= 0:
            return 100.0  # Very high impact for zero liquidity

        participation = quantity / adv
        k = self.TEMP_IMPACT_COEFF + self.PERM_IMPACT_COEFF

        # Square-root impact
        impact = k * volatility * np.sqrt(participation) * 10000  # Convert to bps

        # Add spread
        total_impact = spread_bps + impact

        return round(total_impact, 2)

    def _estimate_slippage(
        self,
        quantity: float,
        price: float,
        volatility: float,
        adv: float
    ) -> float:
        """Estimate total slippage in dollars."""
        impact_bps = self._estimate_impact(quantity, volatility, adv, self.SPREAD_DEFAULT_BPS)
        notional = quantity * price
        slippage = notional * (impact_bps / 10000)
        return round(slippage, 2)

    def _calculate_slicing(
        self,
        quantity: float,
        price: float,
        volatility: float,
        adv: float,
        max_impact: float,
        max_adv: float
    ) -> Dict[str, Any]:
        """Calculate optimal slicing strategy."""
        notional = quantity * price

        # Determine slice size based on impact limit
        # Impact ~ sqrt(Q/ADV), so Q ~ (Impact/k/sigma)^2 * ADV
        k = self.TEMP_IMPACT_COEFF + self.PERM_IMPACT_COEFF
        target_impact = max_impact * 0.7  # Target 70% of limit

        max_slice_qty = ((target_impact / 10000) / (k * volatility)) ** 2 * adv
        max_slice_qty = max(max_slice_qty, 1)

        # Also respect ADV limit
        max_slice_by_adv = adv * max_adv * 0.8
        slice_qty = min(max_slice_qty, max_slice_by_adv)

        # Calculate number of slices
        n_slices = int(np.ceil(quantity / max(slice_qty, 1)))
        n_slices = min(n_slices, self.MAX_SLICES)

        # Adjust slice size
        slice_qty = quantity / n_slices
        slice_impact = self._estimate_impact(slice_qty, volatility, adv, self.SPREAD_DEFAULT_BPS)

        return {
            "n_slices": n_slices,
            "slice_qty": slice_qty,
            "per_slice_impact": slice_impact,
            "per_slice_participation": slice_qty / max(1, adv)
        }

    def _recommend_tactic(
        self,
        participation: float,
        impact_bps: float,
        volatility: float
    ) -> str:
        """Recommend execution tactic based on order characteristics."""
        if participation < 0.01 and impact_bps < 5:
            return "IMMEDIATE"
        elif participation < 0.05:
            return "VWAP"
        elif volatility > 0.03:  # High vol
            return "TWAP"
        else:
            return "PATIENTLY"

    def get_stats(self) -> Dict[str, Any]:
        """Get gate statistics."""
        total = self._approval_count + self._rejection_count + self._reduction_count
        return {
            "approvals": self._approval_count,
            "rejections": self._rejection_count,
            "reductions": self._reduction_count,
            "total": total,
            "rejection_rate": self._rejection_count / max(1, total),
            "thresholds": {
                "max_impact_bps": self.max_impact_bps,
                "max_adv_pct": self.max_adv_pct
            }
        }

    def get_recent_decisions(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent gate decisions."""
        return [d.to_dict() for d in self._decisions[-limit:]]


# Singleton instance
_instance: Optional[ImpactGate] = None


def get_impact_gate() -> ImpactGate:
    """Get singleton ImpactGate instance."""
    global _instance
    if _instance is None:
        _instance = ImpactGate()
    return _instance
