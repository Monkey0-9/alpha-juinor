import structlog
import datetime
from typing import List, Dict, Any

logger = structlog.get_logger()

class MonitoringService:
    """
    Institutional Telemetry & Alerting.
    Tracks NAV, Drawdown, and Model Disagreement.
    """
    def __init__(self):
        self.alerts = []

    def log_alert(self, level: str, category: str, message: str, metadata: Dict[str, Any]):
        alert = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "level": level,
            "category": category,
            "message": message,
            "metadata": metadata
        }
        self.alerts.append(alert)

        # Enforce CRITICAL alerts to structured logs
        if level == "CRITICAL":
            logger.critical(f"ALERT: {category}", **alert)
        else:
            logger.warning(f"ALERT: {category}", **alert)

    def check_thresholds(self, run_results: Dict[str, Any]):
        # Data Quality Alert
        avg_dq = run_results.get("avg_dq_score", 1.0)
        if avg_dq < 0.7:
             self.log_alert("WARNING", "DATA_QUALITY_DROP", "Universe data quality below optimal threshold", {"avg_dq": avg_dq})

        # Disagreement Alert
        disagreement_var = run_results.get("disagreement_variance", 0.0)
        if disagreement_var > 0.01:
             self.log_alert("CRITICAL", "HIGH_DISAGREEMENT", "Extreme agent disagreement detected", {"var": disagreement_var})

        # Tail Risk Alert
        max_cvar = run_results.get("max_cvar", 0.0)
        if max_cvar < -0.08:
             self.log_alert("CRITICAL", "TAIL_RISK_EXCEED", "CVaR(95%) threshold exceeded", {"cvar": max_cvar})
