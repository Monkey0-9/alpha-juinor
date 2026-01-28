"""
scripts/health_report.py
Generates reports/health_report.json with institutional metrics.
"""

import os
import sys
import json
import logging
from datetime import datetime
from collections import Counter

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.manager import DatabaseManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HealthReport")

def generate_report():
    db = DatabaseManager()
    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "system_status": "OPERATIONAL",
        "governance": {},
        "data_quality": {},
        "ml_status": {},
        "risk_metrics": {}
    }

    # 1. Governance Distribution
    conn = db.get_connection()
    cursor = conn.execute("SELECT state, COUNT(*) FROM symbol_governance GROUP BY state")
    report["governance"] = dict(cursor.fetchall())

    # 2. Data Quality Distribution
    cursor = conn.execute("SELECT AVG(data_quality), MIN(data_quality), MAX(data_quality) FROM symbol_governance WHERE state = 'ACTIVE'")
    dq = cursor.fetchone()
    report["data_quality"] = {
        "avg_active_quality": dq[0] or 0.0,
        "min_active_quality": dq[1] or 0.0,
        "max_active_quality": dq[2] or 0.0
    }

    # 3. ML Model Status
    model_dir = "models/ml_alpha"
    if os.path.exists(model_dir):
        models = [f for f in os.listdir(model_dir) if f.endswith(".joblib") and not f.endswith("_latest.joblib")]
        report["ml_status"] = {
            "total_models": len(models),
            "directory": model_dir
        }

    # 4. Risk Vetoes
    cursor = conn.execute("SELECT COUNT(*) FROM governance_decisions WHERE vetoed = 1")
    report["risk_metrics"] = {
        "total_vetoes": cursor.fetchone()[0]
    }

    # Save Report
    os.makedirs("reports", exist_ok=True)
    report_path = "reports/health_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)

    logger.info(f"Health report generated at {report_path}")
    print(json.dumps(report, indent=4))

if __name__ == "__main__":
    generate_report()
