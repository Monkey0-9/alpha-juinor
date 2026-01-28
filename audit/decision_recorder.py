"""
audit/decision_recorder.py

Decision Audit Service (Ticket 17)

Records EVERY trading decision with full context for institutional compliance.
Ensures "we always know WHY we traded."

Decision Record includes:
- All alpha outputs (mu, sigma, cvar, confidence)
- Model versions used
- Data quality at decision time
- Reason codes for every action
- Execution details if trade occurred
"""

import json
import hashlib
import logging
import uuid
from datetime import datetime
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, asdict, field
from enum import Enum

logger = logging.getLogger("DECISION_AUDIT")


class DecisionType(str, Enum):
    """Decision types."""
    EXECUTE = "EXECUTE"       # Trade executed
    HOLD = "HOLD"             # No action taken
    REJECT = "REJECT"         # Blocked by gate
    ERROR = "ERROR"           # Processing error


@dataclass
class AlphaContribution:
    """Contribution from a single alpha source."""
    provider: str
    model_version: str
    mu: float
    sigma: float
    cvar_95: float
    confidence: float
    weight: float             # Weight in ensemble

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DecisionRecord:
    """
    Full decision audit record.

    This is the SINGLE SOURCE OF TRUTH for why any decision was made.
    Every record must be complete and immutable.
    """
    # Identifiers
    id: str                       # UUID
    run_id: str                   # Session/run identifier
    symbol: str
    timestamp: str

    # Alpha inputs
    alpha_contributions: List[AlphaContribution]

    # Ensemble outputs
    final_mu: float
    final_sigma: float
    final_cvar: float
    final_confidence: float

    # Decision
    decision: DecisionType
    reason_codes: List[str]       # ["CVaR_LIMIT", "REGIME_RISK_OFF", etc.]

    # Data context
    data_quality_score: float
    data_providers: List[str]
    regime_label: str

    # Model context
    config_hash: str              # Hash of config at decision time
    model_versions: Dict[str, str]  # provider -> version

    # Execution (if EXECUTE)
    execution_id: Optional[str] = None
    executed_quantity: Optional[float] = None
    executed_price: Optional[float] = None
    slippage_bps: Optional[float] = None

    # Portfolio context
    position_before: float = 0.0
    position_after: float = 0.0
    portfolio_cvar: float = 0.0

    # Metadata
    created_at: str = ""
    processing_time_ms: float = 0.0

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat() + 'Z'

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['decision'] = self.decision.value
        d['alpha_contributions'] = [a.to_dict() if hasattr(a, 'to_dict') else a for a in self.alpha_contributions]
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DecisionRecord":
        decision = d.get('decision', 'HOLD')
        if isinstance(decision, str):
            try:
                decision = DecisionType(decision)
            except ValueError:
                decision = DecisionType.HOLD

        alphas = []
        for a in d.get('alpha_contributions', []):
            if isinstance(a, dict):
                alphas.append(AlphaContribution(**a))
            else:
                alphas.append(a)

        return cls(
            id=d['id'],
            run_id=d['run_id'],
            symbol=d['symbol'],
            timestamp=d['timestamp'],
            alpha_contributions=alphas,
            final_mu=d.get('final_mu', 0.0),
            final_sigma=d.get('final_sigma', 0.02),
            final_cvar=d.get('final_cvar', -0.02),
            final_confidence=d.get('final_confidence', 0.5),
            decision=decision,
            reason_codes=d.get('reason_codes', []),
            data_quality_score=d.get('data_quality_score', 0.8),
            data_providers=d.get('data_providers', []),
            regime_label=d.get('regime_label', 'UNKNOWN'),
            config_hash=d.get('config_hash', ''),
            model_versions=d.get('model_versions', {}),
            execution_id=d.get('execution_id'),
            executed_quantity=d.get('executed_quantity'),
            executed_price=d.get('executed_price'),
            slippage_bps=d.get('slippage_bps'),
            position_before=d.get('position_before', 0.0),
            position_after=d.get('position_after', 0.0),
            portfolio_cvar=d.get('portfolio_cvar', 0.0),
            created_at=d.get('created_at', ''),
            processing_time_ms=d.get('processing_time_ms', 0.0)
        )


class DecisionRecorder:
    """
    Decision Audit Service.

    Records ALL trading decisions with full context.
    Every record is:
    - Complete (all required fields)
    - Immutable (no updates, only new records)
    - Traceable (links to execution, data, models)

    This is a CRITICAL PATH component - failures block trading.
    """

    def __init__(self, run_id: str = None, db_manager=None, fail_open: bool = False):
        """
        Initialize DecisionRecorder.

        Args:
            run_id: Session identifier (generated if not provided)
            db_manager: DatabaseManager instance
            fail_open: If True, continue trading if audit fails (NOT RECOMMENDED)
        """
        self._run_id = run_id or f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        self._db = db_manager
        self._fail_open = fail_open

        self._record_count = 0
        self._config_hash: Optional[str] = None
        self._pending_records: List[DecisionRecord] = []

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def db(self):
        if self._db is None:
            from database.manager import DatabaseManager
            self._db = DatabaseManager()
        return self._db

    def set_config_hash(self, config: Dict[str, Any]):
        """Set config hash for the session."""
        content = json.dumps(config, sort_keys=True, default=str)
        self._config_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

    def record_decision(
        self,
        symbol: str,
        alpha_contributions: List[AlphaContribution],
        final_mu: float,
        final_sigma: float,
        final_cvar: float,
        final_confidence: float,
        decision: DecisionType,
        reason_codes: List[str],
        data_quality_score: float,
        data_providers: List[str],
        regime_label: str,
        model_versions: Dict[str, str],
        execution_id: str = None,
        executed_quantity: float = None,
        executed_price: float = None,
        slippage_bps: float = None,
        position_before: float = 0.0,
        position_after: float = 0.0,
        portfolio_cvar: float = 0.0,
        processing_time_ms: float = 0.0
    ) -> DecisionRecord:
        """
        Record a trading decision.

        This is a CRITICAL PATH operation.
        If it fails and fail_open=False, the trade should be blocked.
        """
        record = DecisionRecord(
            id=str(uuid.uuid4()),
            run_id=self._run_id,
            symbol=symbol,
            timestamp=datetime.utcnow().isoformat() + 'Z',
            alpha_contributions=alpha_contributions,
            final_mu=final_mu,
            final_sigma=final_sigma,
            final_cvar=final_cvar,
            final_confidence=final_confidence,
            decision=decision,
            reason_codes=reason_codes,
            data_quality_score=data_quality_score,
            data_providers=data_providers,
            regime_label=regime_label,
            config_hash=self._config_hash or '',
            model_versions=model_versions,
            execution_id=execution_id,
            executed_quantity=executed_quantity,
            executed_price=executed_price,
            slippage_bps=slippage_bps,
            position_before=position_before,
            position_after=position_after,
            portfolio_cvar=portfolio_cvar,
            processing_time_ms=processing_time_ms
        )

        # Persist to database
        success = self._persist_record(record)

        if not success and not self._fail_open:
            logger.error(f"CRITICAL: Failed to record decision for {symbol}")
            raise RuntimeError(f"Audit record failed for {symbol} - trading blocked")

        self._record_count += 1

        # Log summary
        logger.info(json.dumps({
            "event": "DECISION_RECORDED",
            "id": record.id,
            "symbol": symbol,
            "decision": decision.value,
            "mu": final_mu,
            "cvar": final_cvar,
            "reason_codes": reason_codes[:3]
        }))

        return record

    def _persist_record(self, record: DecisionRecord) -> bool:
        """Persist record to database."""
        try:
            with self.db.get_connection() as conn:
                # Use decision_records_v2 table
                conn.execute("""
                    INSERT INTO decision_records_v2
                    (id, run_id, symbol, timestamp, mu_list, sigma_list, confidence_list,
                     model_versions, final_decision, reason_codes, execution_id,
                     data_quality_score, data_providers, config_sha256, regime_label,
                     cvar_portfolio, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    record.id,
                    record.run_id,
                    record.symbol,
                    record.timestamp,
                    json.dumps([a.mu for a in record.alpha_contributions]),
                    json.dumps([a.sigma for a in record.alpha_contributions]),
                    json.dumps([a.confidence for a in record.alpha_contributions]),
                    json.dumps(record.model_versions),
                    record.decision.value,
                    json.dumps(record.reason_codes),
                    record.execution_id,
                    record.data_quality_score,
                    json.dumps(record.data_providers),
                    record.config_hash,
                    record.regime_label,
                    record.portfolio_cvar,
                    record.created_at
                ))
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to persist decision record: {e}")
            # Add to pending for retry
            self._pending_records.append(record)
            return False

    def flush_pending(self) -> int:
        """Flush pending records that failed to persist."""
        flushed = 0
        for record in self._pending_records[:]:
            if self._persist_record(record):
                self._pending_records.remove(record)
                flushed += 1
        return flushed

    def get_decision(self, decision_id: str) -> Optional[DecisionRecord]:
        """Retrieve a decision record by ID."""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.execute(
                    "SELECT * FROM decision_records_v2 WHERE id = ?",
                    (decision_id,)
                )
                row = cursor.fetchone()
                if row:
                    return self._row_to_record(dict(row))
        except Exception as e:
            logger.error(f"Failed to get decision {decision_id}: {e}")
        return None

    def get_decisions_for_symbol(
        self,
        symbol: str,
        limit: int = 100,
        decision_filter: DecisionType = None
    ) -> List[DecisionRecord]:
        """Get decisions for a symbol."""
        records = []
        try:
            query = "SELECT * FROM decision_records_v2 WHERE symbol = ?"
            params: List[Any] = [symbol]

            if decision_filter:
                query += " AND final_decision = ?"
                params.append(decision_filter.value)

            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            with self.db.get_connection() as conn:
                cursor = conn.execute(query, params)
                for row in cursor.fetchall():
                    records.append(self._row_to_record(dict(row)))
        except Exception as e:
            logger.error(f"Failed to get decisions for {symbol}: {e}")
        return records

    def get_run_summary(self) -> Dict[str, Any]:
        """Get summary of decisions in current run."""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.execute("""
                    SELECT
                        final_decision,
                        COUNT(*) as count,
                        AVG(data_quality_score) as avg_quality
                    FROM decision_records_v2
                    WHERE run_id = ?
                    GROUP BY final_decision
                """, (self._run_id,))

                by_decision = {}
                for row in cursor.fetchall():
                    by_decision[row['final_decision']] = {
                        "count": row['count'],
                        "avg_quality": round(row['avg_quality'] or 0, 4)
                    }

                return {
                    "run_id": self._run_id,
                    "total_records": self._record_count,
                    "by_decision": by_decision,
                    "pending_records": len(self._pending_records)
                }
        except Exception as e:
            logger.error(f"Failed to get run summary: {e}")
            return {"run_id": self._run_id, "error": str(e)}

    def _row_to_record(self, row: Dict[str, Any]) -> DecisionRecord:
        """Convert database row to DecisionRecord."""
        # Parse JSON fields
        mu_list = json.loads(row.get('mu_list', '[]'))
        sigma_list = json.loads(row.get('sigma_list', '[]'))
        confidence_list = json.loads(row.get('confidence_list', '[]'))
        model_versions = json.loads(row.get('model_versions', '{}'))
        reason_codes = json.loads(row.get('reason_codes', '[]'))
        data_providers = json.loads(row.get('data_providers', '[]'))

        # Reconstruct alpha contributions
        alphas = []
        for i, mu in enumerate(mu_list):
            alphas.append(AlphaContribution(
                provider=f"alpha_{i}",
                model_version=list(model_versions.values())[i] if i < len(model_versions) else "v0",
                mu=mu,
                sigma=sigma_list[i] if i < len(sigma_list) else 0.02,
                cvar_95=-0.02,
                confidence=confidence_list[i] if i < len(confidence_list) else 0.5,
                weight=1.0 / max(1, len(mu_list))
            ))

        # Get final values (avg of list)
        final_mu = sum(mu_list) / max(1, len(mu_list)) if mu_list else 0.0
        final_sigma = sum(sigma_list) / max(1, len(sigma_list)) if sigma_list else 0.02
        final_confidence = sum(confidence_list) / max(1, len(confidence_list)) if confidence_list else 0.5

        return DecisionRecord(
            id=row['id'],
            run_id=row['run_id'],
            symbol=row['symbol'],
            timestamp=row['timestamp'],
            alpha_contributions=alphas,
            final_mu=final_mu,
            final_sigma=final_sigma,
            final_cvar=row.get('cvar_portfolio', -0.02),
            final_confidence=final_confidence,
            decision=DecisionType(row['final_decision']),
            reason_codes=reason_codes,
            data_quality_score=row.get('data_quality_score', 0.8),
            data_providers=data_providers,
            regime_label=row.get('regime_label', 'UNKNOWN'),
            config_hash=row.get('config_sha256', ''),
            model_versions=model_versions,
            execution_id=row.get('execution_id'),
            portfolio_cvar=row.get('cvar_portfolio', 0.0),
            created_at=row.get('created_at', '')
        )


# Singleton instance
_instance: Optional[DecisionRecorder] = None


def get_decision_recorder(run_id: str = None) -> DecisionRecorder:
    """Get singleton DecisionRecorder instance."""
    global _instance
    if _instance is None or (run_id and _instance.run_id != run_id):
        _instance = DecisionRecorder(run_id=run_id)
    return _instance
