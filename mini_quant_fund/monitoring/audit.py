from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from mini_quant_fund.db.models import Decision, Base
import structlog
from typing import List, Dict, Any
import datetime
import uuid

logger = structlog.get_logger()

class AuditManager:
    """
    Institutional Audit Layer.
    Mandate: Atomic transactional write to audit DB.
    If write fails, system must throw CRITICAL and STOP.
    """
    def __init__(self, db_url: str = "sqlite:///mini_quant.db"):
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)

    def write_audit(self,
                    run_id: str,
                    symbol: str,
                    decision: str,
                    forecasts: Dict[str, Any],
                    weight: float,
                    reason_codes: List[str],
                    price_at_decision: float = 0.0,
                    config_sha256: str = "N/A",
                    execution_id: str = None):
        """
        Atomic transactional write to audit DB.
        """
        try:
            with Session(self.engine) as session:
                record = Decision(
                    id=str(uuid.uuid4()),
                    run_id=run_id,
                    symbol=symbol,
                    timestamp_utc=datetime.datetime.utcnow(),
                    decision=decision,
                    weight=weight,
                    price_at_decision=price_at_decision,
                    mu=forecasts.get("mu", 0.0),
                    sigma=forecasts.get("sigma", 0.0),
                    mu_adjusted=forecasts.get("mu_adj", 0.0),
                    data_quality=forecasts.get("data_quality", 0.0),
                    reason_codes=reason_codes,
                    model_versions=forecasts.get("model_versions", {}),
                    config_sha256=config_sha256,
                    execution_id=execution_id
                )
                session.add(record)
                session.commit()
        except Exception as e:
            logger.critical("AUDIT_WRITE_FAILURE",
                            error=str(e),
                            symbol=symbol,
                            run_id=run_id)
            # BRUTAL MANDATE: STOP EVERYTHING
            raise SystemExit(f"CRITICAL: Audit write failure for {symbol}. Halting system.")

def write_audit(run_id: str, symbol: str, decision: str, forecasts: dict, weight: float, reason_codes: List[str]):
    """
    Convenience wrapper for the global AuditManager.
    """
    # For demo/standalone usage. In production, we'd use a singleton or context.
    manager = AuditManager()
    manager.write_audit(run_id, symbol, decision, forecasts, weight, reason_codes)
