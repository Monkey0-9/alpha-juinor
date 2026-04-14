
import json
import os
from datetime import datetime

ALERTS_LOG = "runtime/alerts.jsonl"
LIVE_LOG = "runtime/logs/live.jsonl"
OUTPUT_DIR = "docs/postmortems"

def generate_postmortem(incident_id=None):
    incident_id = incident_id or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Collate events
    events = []
    if os.path.exists(ALERTS_LOG):
        with open(ALERTS_LOG, 'r') as f:
            for line in f:
                events.append(json.loads(line))

    # Basic Report Template
    report = [
        f"# Incident Postmortem: {incident_id}",
        f"**Date Generated**: {datetime.utcnow().isoformat()}",
        "",
        "## Summary",
        "TBD: Provide a brief summary of the event.",
        "",
        "## Timeline",
    ]

    for event in events[-10:]: # Last 10 alerts
        report.append(f"- **{event['timestamp']}**: [{event['level']}] {event['message']}")

    report.extend([
        "",
        "## Root Cause Analysis",
        "TBD: What failed and why?",
        "",
        "## Impact Assessment",
        "- PnL Impact: TBD",
        "- System Downtime: TBD",
        "",
        "## Corrective Actions",
        "- [ ] Action Item 1",
        "- [ ] Action Item 2",
    ])

    report_path = os.path.join(OUTPUT_DIR, f"INCIDENT_{incident_id}.md")
    with open(report_path, 'w') as f:
        f.write("\n".join(report))

    print(f"Postmortem generated at {report_path}")

if __name__ == "__main__":
    generate_postmortem()
