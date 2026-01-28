"""
risk/cvar_gate.py

CVaR Blocking Gate (Ticket 12)

Blocks trades that would violate CVaR limits at symbol or portfolio level.
This is a HARD GATE - no trade can bypass CVaR limits.

Limits:
- Symbol CVaR_95: -5% max per position
- Portfolio CVaR_95: -2% max (entire portfolio)
- Portfolio CVaR_99: -5% max

If any limit would be breached, the trade is REJECTED.
"""

import json
import logging
from datetime import datetime
from typing import Dict, Optional, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

logger = logging.getLogger("CVAR_GATE")


class RejectionReason(str, Enum):
    """Reason for trade rejection."""
    SYMBOL_CVAR_EXCEEDED = "SYMBOL_CVAR_EXCEEDED"
    PORTFOLIO_CVAR95_EXCEEDED = "PORTFOLIO_CVAR95_EXCEEDED"
    PORTFOLIO_CVAR99_EXCEEDED = "PORTFOLIO_CVAR99_EXCEEDED"
    MAX_LOSS_EXCEEDED = "MAX_LOSS_EXCEEDED"
    CONCENTRATION_LIMIT = "CONCENTRATION_LIMIT"
    APPROVED = "APPROVED"


@dataclass
class GateDecision:
    """CVaR gate decision."""
    approved: bool
    reason: RejectionReason
    symbol: str
    proposed_weight: float
    adjusted_weight: Optional[float]
    symbol_cvar: float
    portfolio_cvar: float
    explanation: str
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "approved": self.approved,
            "reason": self.reason.value,
            "symbol": self.symbol,
            "proposed_weight": self.proposed_weight,
            "adjusted_weight": self.adjusted_weight,
            "symbol_cvar": self.symbol_cvar,
            "portfolio_cvar": self.portfolio_cvar,
            "explanation": self.explanation,
            "timestamp": self.timestamp
        }


class CVaRGate:
    """
    CVaR Blocking Gate.

    All trades MUST pass through this gate.
    Enforces hard limits on tail risk.

    Limits (configurable):
    - Per-symbol CVaR_95: -5% (max loss per position at 95% confidence)
    - Portfolio CVaR_95: -2% (max loss for entire portfolio)
    - Portfolio CVaR_99: -5% (extreme tail limit)

    Behavior:
    - REJECT: Trade blocked entirely
    - REDUCE: Trade size reduced to fit limits
    - APPROVE: Trade within limits
    """

    # Default limits (configurable)
    DEFAULT_SYMBOL_CVAR_LIMIT = -0.05      # -5% per symbol
    DEFAULT_PORTFOLIO_CVAR95_LIMIT = -0.02  # -2% portfolio
    DEFAULT_PORTFOLIO_CVAR99_LIMIT = -0.05  # -5% extreme
    DEFAULT_CONCENTRATION_LIMIT = 0.15      # 15% max single position

    def __init__(
        self,
        symbol_cvar_limit: float = None,
        portfolio_cvar95_limit: float = None,
        portfolio_cvar99_limit: float = None,
        concentration_limit: float = None,
        cvar_engine = None
    ):
        """
        Initialize CVaRGate.

        Args:
            symbol_cvar_limit: Max CVaR per symbol (negative)
            portfolio_cvar95_limit: Max portfolio CVaR at 95%
            portfolio_cvar99_limit: Max portfolio CVaR at 99%
            concentration_limit: Max single position weight
            cvar_engine: CVaREngine instance
        """
        self.symbol_cvar_limit = symbol_cvar_limit or self.DEFAULT_SYMBOL_CVAR_LIMIT
        self.portfolio_cvar95_limit = portfolio_cvar95_limit or self.DEFAULT_PORTFOLIO_CVAR95_LIMIT
        self.portfolio_cvar99_limit = portfolio_cvar99_limit or self.DEFAULT_PORTFOLIO_CVAR99_LIMIT
        self.concentration_limit = concentration_limit or self.DEFAULT_CONCENTRATION_LIMIT

        self._cvar_engine = cvar_engine
        self._rejection_count = 0
        self._approval_count = 0
        self._decisions: List[GateDecision] = []

    @property
    def cvar_engine(self):
        if self._cvar_engine is None:
            from risk.cvar_engine import get_cvar_engine
            self._cvar_engine = get_cvar_engine()
        return self._cvar_engine

    def check_trade(
        self,
        symbol: str,
        proposed_weight: float,
        symbol_returns: pd.Series,
        current_portfolio_weights: Dict[str, float],
        portfolio_returns: Dict[str, pd.Series],
        allow_reduction: bool = True
    ) -> GateDecision:
        """
        Check if a proposed trade passes CVaR limits.

        Args:
            symbol: Symbol to trade
            proposed_weight: Proposed portfolio weight
            symbol_returns: Historical returns for the symbol
            current_portfolio_weights: Current portfolio weights
            portfolio_returns: Historical returns for all positions
            allow_reduction: If True, may suggest reduced size

        Returns:
            GateDecision with approval status and details
        """
        now = datetime.utcnow().isoformat() + 'Z'

        # ========== SYMBOL-LEVEL CHECK ==========

        # Compute symbol CVaR
        symbol_cvar_result = self.cvar_engine.compute_cvar(
            symbol_returns,
            confidence=0.95,
            symbol=symbol
        )
        symbol_cvar = symbol_cvar_result.cvar

        # Weight-adjusted symbol CVaR contribution
        weighted_symbol_cvar = symbol_cvar * abs(proposed_weight)

        if weighted_symbol_cvar < self.symbol_cvar_limit:
            # Symbol CVaR exceeded
            if allow_reduction:
                # Calculate max weight that fits within limit
                if symbol_cvar != 0:
                    max_weight = abs(self.symbol_cvar_limit / symbol_cvar)
                    max_weight = min(max_weight, self.concentration_limit)
                    adjusted_weight = np.sign(proposed_weight) * max_weight * 0.9  # 90% of limit
                else:
                    adjusted_weight = 0.0

                logger.warning(json.dumps({
                    "event": "CVAR_GATE_REDUCTION",
                    "symbol": symbol,
                    "proposed": proposed_weight,
                    "adjusted": adjusted_weight,
                    "symbol_cvar": symbol_cvar,
                    "limit": self.symbol_cvar_limit
                }))

                # Recheck with adjusted weight
                weighted_cvar_adj = symbol_cvar * abs(adjusted_weight)
                if weighted_cvar_adj >= self.symbol_cvar_limit:
                    decision = GateDecision(
                        approved=True,
                        reason=RejectionReason.APPROVED,
                        symbol=symbol,
                        proposed_weight=proposed_weight,
                        adjusted_weight=adjusted_weight,
                        symbol_cvar=symbol_cvar,
                        portfolio_cvar=0.0,  # Will compute below
                        explanation=f"Reduced from {proposed_weight:.4f} to {adjusted_weight:.4f} to fit CVaR limit",
                        timestamp=now
                    )
                    proposed_weight = adjusted_weight  # Use adjusted for portfolio check
                else:
                    decision = GateDecision(
                        approved=False,
                        reason=RejectionReason.SYMBOL_CVAR_EXCEEDED,
                        symbol=symbol,
                        proposed_weight=proposed_weight,
                        adjusted_weight=None,
                        symbol_cvar=symbol_cvar,
                        portfolio_cvar=0.0,
                        explanation=f"Symbol CVaR {symbol_cvar:.4f} * weight {proposed_weight:.4f} = {weighted_symbol_cvar:.4f} exceeds limit {self.symbol_cvar_limit}",
                        timestamp=now
                    )
                    self._rejection_count += 1
                    self._decisions.append(decision)
                    return decision
            else:
                decision = GateDecision(
                    approved=False,
                    reason=RejectionReason.SYMBOL_CVAR_EXCEEDED,
                    symbol=symbol,
                    proposed_weight=proposed_weight,
                    adjusted_weight=None,
                    symbol_cvar=symbol_cvar,
                    portfolio_cvar=0.0,
                    explanation=f"Symbol CVaR {weighted_symbol_cvar:.4f} exceeds limit {self.symbol_cvar_limit}",
                    timestamp=now
                )
                self._rejection_count += 1
                self._decisions.append(decision)
                return decision

        # ========== CONCENTRATION CHECK ==========

        if abs(proposed_weight) > self.concentration_limit:
            if allow_reduction:
                adjusted_weight = np.sign(proposed_weight) * self.concentration_limit * 0.95
                proposed_weight = adjusted_weight
                logger.info(f"Reduced {symbol} weight to {adjusted_weight:.4f} for concentration limit")
            else:
                decision = GateDecision(
                    approved=False,
                    reason=RejectionReason.CONCENTRATION_LIMIT,
                    symbol=symbol,
                    proposed_weight=proposed_weight,
                    adjusted_weight=None,
                    symbol_cvar=symbol_cvar,
                    portfolio_cvar=0.0,
                    explanation=f"Weight {proposed_weight:.4f} exceeds concentration limit {self.concentration_limit}",
                    timestamp=now
                )
                self._rejection_count += 1
                self._decisions.append(decision)
                return decision

        # ========== PORTFOLIO-LEVEL CHECK ==========

        # Create pro-forma portfolio with new trade
        proforma_weights = current_portfolio_weights.copy()
        proforma_weights[symbol] = proposed_weight

        # Compute portfolio CVaR
        portfolio_cvar_result = self.cvar_engine.compute_portfolio_cvar(
            portfolio_returns,
            proforma_weights
        )

        portfolio_cvar_95 = portfolio_cvar_result.cvar_95
        portfolio_cvar_99 = portfolio_cvar_result.cvar_99

        # Check 95% limit
        if portfolio_cvar_95 < self.portfolio_cvar95_limit:
            decision = GateDecision(
                approved=False,
                reason=RejectionReason.PORTFOLIO_CVAR95_EXCEEDED,
                symbol=symbol,
                proposed_weight=proposed_weight,
                adjusted_weight=None,
                symbol_cvar=symbol_cvar,
                portfolio_cvar=portfolio_cvar_95,
                explanation=f"Portfolio CVaR_95 {portfolio_cvar_95:.4f} would exceed limit {self.portfolio_cvar95_limit}",
                timestamp=now
            )
            self._rejection_count += 1
            self._decisions.append(decision)

            logger.error(json.dumps({
                "event": "CVAR_GATE_REJECTION",
                "reason": "PORTFOLIO_CVAR95_EXCEEDED",
                "symbol": symbol,
                "portfolio_cvar": portfolio_cvar_95,
                "limit": self.portfolio_cvar95_limit
            }))

            return decision

        # Check 99% limit
        if portfolio_cvar_99 < self.portfolio_cvar99_limit:
            decision = GateDecision(
                approved=False,
                reason=RejectionReason.PORTFOLIO_CVAR99_EXCEEDED,
                symbol=symbol,
                proposed_weight=proposed_weight,
                adjusted_weight=None,
                symbol_cvar=symbol_cvar,
                portfolio_cvar=portfolio_cvar_99,
                explanation=f"Portfolio CVaR_99 {portfolio_cvar_99:.4f} would exceed limit {self.portfolio_cvar99_limit}",
                timestamp=now
            )
            self._rejection_count += 1
            self._decisions.append(decision)
            return decision

        # ========== APPROVED ==========

        decision = GateDecision(
            approved=True,
            reason=RejectionReason.APPROVED,
            symbol=symbol,
            proposed_weight=proposed_weight,
            adjusted_weight=proposed_weight if proposed_weight != proposed_weight else None,
            symbol_cvar=symbol_cvar,
            portfolio_cvar=portfolio_cvar_95,
            explanation="Trade within all CVaR limits",
            timestamp=now
        )

        self._approval_count += 1
        self._decisions.append(decision)

        logger.info(json.dumps({
            "event": "CVAR_GATE_APPROVED",
            "symbol": symbol,
            "weight": proposed_weight,
            "symbol_cvar": symbol_cvar,
            "portfolio_cvar": portfolio_cvar_95
        }))

        return decision

    def check_portfolio_risk(
        self,
        portfolio_weights: Dict[str, float],
        portfolio_returns: Dict[str, pd.Series]
    ) -> Dict[str, Any]:
        """
        Check current portfolio risk against limits.

        Returns dict with:
        - within_limits: bool
        - cvar_95: float
        - cvar_99: float
        - headroom_95: float (remaining capacity)
        - violations: list of violated limits
        """
        portfolio_cvar = self.cvar_engine.compute_portfolio_cvar(
            portfolio_returns,
            portfolio_weights
        )

        violations = []

        if portfolio_cvar.cvar_95 < self.portfolio_cvar95_limit:
            violations.append({
                "limit": "PORTFOLIO_CVAR95",
                "value": portfolio_cvar.cvar_95,
                "threshold": self.portfolio_cvar95_limit
            })

        if portfolio_cvar.cvar_99 < self.portfolio_cvar99_limit:
            violations.append({
                "limit": "PORTFOLIO_CVAR99",
                "value": portfolio_cvar.cvar_99,
                "threshold": self.portfolio_cvar99_limit
            })

        return {
            "within_limits": len(violations) == 0,
            "cvar_95": portfolio_cvar.cvar_95,
            "cvar_99": portfolio_cvar.cvar_99,
            "headroom_95": self.portfolio_cvar95_limit - portfolio_cvar.cvar_95,
            "headroom_99": self.portfolio_cvar99_limit - portfolio_cvar.cvar_99,
            "gross_exposure": portfolio_cvar.gross_exposure,
            "net_exposure": portfolio_cvar.net_exposure,
            "violations": violations,
            "position_cvar": portfolio_cvar.positions
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get gate statistics."""
        total = self._approval_count + self._rejection_count
        return {
            "approvals": self._approval_count,
            "rejections": self._rejection_count,
            "rejection_rate": self._rejection_count / max(1, total),
            "total_decisions": total,
            "limits": {
                "symbol_cvar": self.symbol_cvar_limit,
                "portfolio_cvar95": self.portfolio_cvar95_limit,
                "portfolio_cvar99": self.portfolio_cvar99_limit,
                "concentration": self.concentration_limit
            }
        }

    def get_recent_decisions(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent gate decisions."""
        return [d.to_dict() for d in self._decisions[-limit:]]


# Singleton instance
_instance: Optional[CVaRGate] = None


def get_cvar_gate() -> CVaRGate:
    """Get singleton CVaRGate instance."""
    global _instance
    if _instance is None:
        _instance = CVaRGate()
    return _instance
