"""
ops/postmortem_generator.py

Generates forensic post-mortem reports for incidents or routine audits.
Collects logs, audit DB records, and PnL data.
"""

import sys
import os
import argparse
import json
import logging
from datetime import datetime, timedelta
import pandas as pd
import sqlite3

# Import our log config if available, else basic
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("POSTMORTEM")

def generate_report(run_id: str, since: str, output_path: str):
    logger.info(f"Generating postmortem for RunID={run_id}, Since={since}")

    report = {
        "meta": {
            "generated_at": datetime.utcnow().isoformat(),
            "run_id": run_id,
            "period": since
        },
        "incidents": [],
        "audit_trail": [],
        "risk_breaches": [],
        "system_health": {}
    }

    # 1. Parse 'since' (e.g. "-72h", "-7d")
    # Simple parse: assuming format -NumberUnit
    try:
        if since.startswith("-"):
            unit = since[-1]
            val = int(since[1:-1])
            now = datetime.utcnow()
            if unit == 'h':
                start_dt = now - timedelta(hours=val)
            elif unit == 'd':
                start_dt = now - timedelta(days=val)
            else:
                start_dt = now - timedelta(days=1)
        else:
            start_dt = datetime.utcnow() - timedelta(days=1)
    except Exception:
        start_dt = datetime.utcnow() - timedelta(days=1)

    start_str = start_dt.isoformat()

    # 2. Query Audit DB
    db_path = "runtime/audit.db"
    if os.path.exists(db_path):
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Fetch recent decisions (limit 1000)
            cursor.execute("SELECT * FROM decisions WHERE timestamp > ? ORDER BY timestamp DESC LIMIT 1000", (start_str,))
            rows = cursor.fetchall()

            for row in rows:
                rec = dict(row)
                report["audit_trail"].append(rec)

                # Check for "REJECT" or overrides
                if rec.get("final_decision") == "REJECT" or rec.get("pm_override") != "NONE":
                    report["incidents"].append({
                        "type": "DECISION_REJECT_OR_OVERRIDE",
                        "symbol": rec.get("symbol"),
                        "timestamp": rec.get("timestamp"),
                        "reason": rec.get("reason_codes"),
                        "override": rec.get("pm_override")
                    })

            conn.close()
        except Exception as e:
            report["system_health"]["audit_db_error"] = str(e)
    else:
        report["system_health"]["audit_db_status"] = "MISSING"

    # 3. Analyze Logs (mock implementation, typically grep logs)
    # We could scan log files in logs/ directory for "ERROR" or "CRITICAL"
    log_dir = "logs"
    if os.path.exists(log_dir):
        errors = []
        for f in os.listdir(log_dir):
            if f.endswith(".log"):
                path = os.path.join(log_dir, f)
                try:
                    with open(path, "r", encoding="utf-8") as lf:
                        # Tail last 1000 lines? or scan whole?
                        # Scan for ERROR
                        for line in lf:
                            if "ERROR" in line or "CRITICAL" in line:
                                errors.append(f"{f}: {line.strip()}")
                                if len(errors) > 50: break
                except:
                    pass
        report["system_health"]["recent_errors"] = errors

    # 4. Save
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"Report saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", default="manual")
    parser.add_argument("--since", default="-24h")
    parser.add_argument("--output", default="postmortem.json")
    args = parser.parse_args()

    generate_report(args.run_id, args.since, args.output)
