# Daily Trading Operations Checklist
Generated: 2026-01-04 13:28:21

## 1. Configuration & Integrity
- [ ] Config SHA256 Hash Verified: `f6e51d16316f0d54ed7c9a921d9e0d680ec0aac0db4bb02f435ed6a402b2569f`
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
