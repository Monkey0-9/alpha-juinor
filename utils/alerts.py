# utils/alerts.py
import logging
import json
from datetime import datetime
from utils.metrics import metrics

logger = logging.getLogger("ALERTS")

def escalate_slack(message: str, level: str = "WARNING"):
    """Simulation of Slack escalation via structured logs."""
    record = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "type": "SLACK_ALERT",
        "level": level,
        "message": message
    }
    logger.warning(json.dumps(record))

def page_on_call(message: str):
    """Simulation of PagerDuty escalation via structured logs."""
    record = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "type": "PAGER_DUTY",
        "level": "CRITICAL",
        "message": message
    }
    logger.critical(json.dumps(record))

def check_system_thresholds():
    """Run periodic checks against institutional thresholds."""

    # 1. Feature Mismatches
    # We should probably track this per symbol, but global count is a start.
    # Actually metrics.get("ml_feature_mismatch_count")
    mismatches = metrics.get("ml_feature_mismatch_total")
    if mismatches > 0:
        page_on_call(f"ML Feature Schema Mismatch detected! Total={mismatches}")
        metrics.reset("ml_feature_mismatch_total")

    # 2. Data Quality
    low_quality = metrics.get("low_data_quality_count")
    if low_quality > 10:
        escalate_slack(f"High volume of low quality data symbols: {low_quality}", "ERROR")
        metrics.reset("low_data_quality_count")

    # 3. Impact Rejections
    rejections = metrics.get("impact_gate_rejections")
    if rejections > 5:
        escalate_slack(f"Excessive impact rejections in cycle: {rejections}. Review universe liquidity.", "WARNING")
        metrics.reset("impact_gate_rejections")
