"""
audit/decision_recorder.py

Decision Audit Trail - Records all trading decisions for compliance.
"""

import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger("DECISION_RECORDER")


@dataclass
class DecisionRecord:
    """Immutable record of a trading decision."""
    timestamp: str
    symbol: str
    decision: str  # BUY, SELL, HOLD, ABSTAIN
    signal_strength: float
    confidence: float
    source_alpha: str
    portfolio_weight: float
    risk_check_passed: bool
    regime: str
    execution_tactic: str
    rationale: str
    meta: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DecisionType(str, Enum):
    """Types of trading decisions (backward-compatible)."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    ABSTAIN = "ABSTAIN"
    SKIP = "SKIP"
    # Back-compat aliases
    EXECUTE = "EXECUTE"
    REJECT = "REJECT"
    ERROR = "ERROR"

    @classmethod
    def normalize(cls, v: str) -> "DecisionType":
        mapping = {
            "EXECUTE": cls.BUY,
            "REJECT": cls.ABSTAIN,
            "ERROR": cls.ABSTAIN,
        }
        try:
            return cls(v)
        except ValueError: # Changed from generic Exception to ValueError for precision
            return mapping.get(v, cls.HOLD)


@dataclass
class AlphaContribution:
    """Alpha source contribution to a decision."""
    source: str
    signal: float
    confidence: float
    weight: float



class DecisionRecorder:
    """
    Append-only decision audit trail.

    Features:
    - JSON-line format for easy parsing
    - Rotation support (daily files)
    - Query interface for recent decisions
    """

    def __init__(self, log_dir: str = "logs/decisions"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._buffer: List[DecisionRecord] = []
        logger.info(f"[DECISION_RECORDER] Initialized at {self.log_dir}")

    def record(
        self,
        symbol: str,
        decision: str,
        signal_strength: float,
        confidence: float,
        source_alpha: str,
        portfolio_weight: float = 0.0,
        risk_check_passed: bool = True,
        regime: str = "UNKNOWN",
        execution_tactic: str = "NORMAL",
        rationale: str = "",
        meta: Optional[Dict[str, Any]] = None
    ) -> DecisionRecord:
        """Record a trading decision."""
        record = DecisionRecord(
            timestamp=datetime.utcnow().isoformat() + "Z",
            symbol=symbol,
            decision=decision,
            signal_strength=signal_strength,
            confidence=confidence,
            source_alpha=source_alpha,
            portfolio_weight=portfolio_weight,
            risk_check_passed=risk_check_passed,
            regime=regime,
            execution_tactic=execution_tactic,
            rationale=rationale,
            meta=meta or {}
        )

        # Append to buffer
        self._buffer.append(record)

        # Write to file
        self._write_to_file(record)

        logger.debug(f"[DECISION] {symbol} {decision} @ {confidence:.2f}")
        return record

    def _write_to_file(self, record: DecisionRecord):
        """Write decision to daily log file."""
        today = datetime.utcnow().strftime("%Y-%m-%d")
        filepath = self.log_dir / f"decisions_{today}.jsonl"

        with open(filepath, "a") as f:
            f.write(json.dumps(record.to_dict()) + "\n")

    def get_recent(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent decisions from buffer."""
        return [r.to_dict() for r in self._buffer[-limit:]]

    def query_by_symbol(self, symbol: str) -> List[Dict[str, Any]]:
        """Query decisions for a specific symbol."""
        return [r.to_dict() for r in self._buffer if r.symbol == symbol]


# Singleton
_instance: Optional[DecisionRecorder] = None


def get_decision_recorder() -> DecisionRecorder:
    global _instance
    if _instance is None:
        _instance = DecisionRecorder()
    return _instance
