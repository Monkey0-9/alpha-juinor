# Institutional Capital Escalation Policy

## Overview
This policy defines the rigorous transition path from paper trading to full live capital deployment. Graduation between stages requires meeting both performance and operational stability criteria.

## Stage 1: Paper Trading (The "Sanity" Phase)
- **Capital**: $1,000,000 (Virtual)
- **Duration**: Minimum 30 Trading Days
- **Graduation Criteria**:
  - Sharpe Ratio > 0.5 (Annualized)
  - Zero "Critical" Alerts related to data gaps or hangs.
  - Zero position drift incidents (>0.1% drift).
  - Config hash remained immutable for the entire duration.

## Stage 2: Live-Low (The "Execution" Phase)
- **Capital**: $10,000 (Real)
- **Duration**: 60 Trading Days
- **Graduation Criteria**:
  - Realized Slippage within 1.5x of Model expectations.
  - System successfully handled at least 3 "Chaos" events (e.g., API disconnects).
  - No single-day loss exceeding 3% of NAV.
  - All trades successfully reconciled within 60 seconds of fill.

## Stage 3: Live-Full (The "Scale" Phase)
- **Capital**: Graduated scaling ($25K -> $50K -> $100K...)
- **Duration**: Indefinite
- **Governance**:
  - Monthly operational audits of `BacktestRegistry` artifacts.
  - Quarterly review of `AttributionEngine` outputs.
  - Immediate Kill-Switch activation on >15% drawdown from peak.

---
**Approved By**: Operations Oversight
**Version**: 1.0 (Golden Baseline)
