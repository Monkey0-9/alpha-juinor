# ops/checklists.py
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

def generate_daily_checklist(config_hash: str, output_path: str = "output/ops/daily_checklist.md"):
    """
    Generates a fresh markdown checklist for the day's operations.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    checklist = f"""# Daily Trading Operations Checklist
Generated: {timestamp}

## 1. Configuration & Integrity
- [ ] Config SHA256 Hash Verified: `{config_hash}`
- [ ] No unauthorized changes to `configs/golden_config.yaml`
- [ ] Environment variables (.env) loaded correctly

## 2. Environment & Connectivity
- [ ] Broker API (Paper/Live) Connection Test: [ ]
- [ ] Market Data Datafeed Status: [ ]
- [ ] Local Database / Cache Consistency: [ ]

## 3. Pre-Trade Reconciliation
- [ ] Internal Ledger vs. Broker Position Sync
- [ ] Cash Balance Match (< 0.1% drift)
- [ ] No pending orders from previous session

## 4. Safety & Risk
- [ ] Kill-Switch Status: ACTIVE/LOCKED: [ ]
- [ ] Daily Loss Limit set to: [ ]
- [ ] Volatility circuit breakers functional

## 5. Execution Monitoring
- [ ] Alert Dispatcher Heartbeat Sent
- [ ] Monitoring Dashboard (Streamlit) Launched

---
**Operator Signature**: _________________
"""
    
    with open(path, "w") as f:
        f.write(checklist)
    
    logger.info(f"Daily Checklist generated at {output_path}")

if __name__ == "__main__":
    generate_daily_checklist("MANUAL_TEST_HASH")
