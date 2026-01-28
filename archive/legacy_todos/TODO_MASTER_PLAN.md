# Mini-Quant-Fund Master Implementation Plan

## Current Status (Post-Interruption)
- **Overall Progress**: 40%
- **Phase 1 (Test Infrastructure)**: IN PROGRESS (25%)
- **Phase 2 (Database Schema)**: COMPLETE ✅
- **Phase 3 (Historical Backfill)**: COMPLETE ✅
- **Phase 4 (Code TODOs)**: IN PROGRESS (40%)
- **Phase 5 (Data Quality & Alerts)**: PENDING
- **Phase 6 (Governance Gates)**: IN PROGRESS (25%)
- **Phase 7 (Paper Mode Test)**: NEXT
- **Phase 8 (Test Suite)**: PENDING
- **Phase 9 (Documentation)**: IN PROGRESS (10%)
- **Phase 10 (Final Verification)**: PENDING

---

## Phase 1: Test Infrastructure Fixes (Priority: HIGH)

### 1.1 Fix Test Collection Errors
- [ ] Identify all failing test files
- [ ] Run: `python -m pytest --collect-only tests/`
- [ ] Fix import errors in test files

### 1.2 Fix test_phase2.py Imports
- [ ] Update imports to use correct module paths:
  - `from risk.engine import RiskManager, RiskRegime` → Verify path
  - `from strategies.alpha import CompositeAlpha, TrendAlpha` → Verify path
  - `from strategies.ml_models.ml_alpha import MLAlpha` → Verify path
  - `from data.processors.features import FeatureEngineer` → Verify path
- [ ] Run test to verify fixes

### 1.3 Fix Broken Test Paths/References
- [ ] Check all test files for incorrect module references
- [ ] Update conftest.py fixtures if needed
- [ ] Verify test collection works

### 1.4 Verify All Tests Collect Successfully
- [ ] Run full test collection
- [ ] Fix any remaining issues
- [ ] Target: 90%+ pass rate

---

## Phase 4: Complete Remaining Code TODOs (Priority: MEDIUM)

### 4.3 Add Opportunity Cost Calculation
- **File**: `meta_intelligence/opportunity_cost.py:54`
- **Status**: Implementation exists, needs integration
- **Actions**:
  - [ ] Review existing implementation (OpportunityCostEngine class)
  - [ ] Integrate with PM Brain (meta_intelligence/pm_brain.py)
  - [ ] Add opportunity_cost check to symbol_worker.py action determination
  - [ ] Add reason_code: OPPORTUNITY_COST_FAIL when triggered

### 4.4 Enhance Health Monitoring
- **File**: `monitoring/health.py:101`
- **Status**: Implementation exists, needs connection
- **Actions**:
  - [ ] Review HealthMonitor class implementation
  - [ ] Integrate with cycle_runner.py for periodic health checks
  - [ ] Add health check calls in symbol_worker.py
  - [ ] Wire alerts to alert system

### 4.5 Implement Paper Reconciliation
- **File**: `scripts/paper_reconciliation.py:101`
- **Status**: Implementation exists, needs wiring
- **Actions**:
  - [ ] Review reconciliation logic
  - [ ] Add database queries to get actual orders/fills
  - [ ] Create reconciliation report generator
  - [ ] Add to run_cycle.py as post-cycle step

---

## Phase 5: Wire Data Quality & Alerts (Priority: HIGH)

### 5.1 Update Quality Agent to Export Metrics
- **File**: `data_intelligence/quality_agent.py`
- **Status**: Partial implementation
- **Actions**:
  - [ ] Ensure `get_prometheus_metrics()` returns correct metrics
  - [ ] Add `quality_score` gauge
  - [ ] Add `missing_days` counter
  - [ ] Add `rows_total` counter

### 5.2 Add Prometheus Metrics
- **File**: `monitoring/prometheus_metrics.py`
- **Status**: Implementation exists
- **Actions**:
  - [ ] Verify `update_quality_metrics()` method works
  - [ ] Add metrics for avg_quality_score, data_missing_days_total, price_history_rows_total
  - [ ] Test metrics export

### 5.3 Enforce Quality in Cycle Runner
- **File**: `orchestration/cycle_runner.py`
- **Status**: Quality enforcement exists (_apply_quality_enforcement)
- **Actions**:
  - [ ] Verify quality enforcement is called in run_cycle()
  - [ ] Ensure REJECT on quality_score < 0.6
  - [ ] Add reason_code: "data_quality"

### 5.4 Add Alert Firing Logic
- **File**: `monitoring/alerts.py`
- **Status**: Alert methods exist
- **Actions**:
  - [ ] Add alert trigger for `data_missing_days_total > 0`
  - [ ] Add alert for low quality scores (< 0.6)
  - [ ] Integrate alerts with cycle_runner.py
  - [ ] Test alert firing

---

## Phase 6: Live Governance Gates (Priority: HIGH)

### 6.1 Test Startup Gates - PASSED ✅
- [ ] Verify test_startup_gates.py passes
- [ ] Document results

### 6.2 Align LiveSignal with GovernanceDecision
- **Files**: `governance/`, `agents/`
- **Actions**:
  - [ ] Review LiveSignal class
  - [ ] Ensure GovernanceDecision includes all required fields
  - [ ] Align decision reasons and metadata

### 6.3 Add CVaR Pre-Trade Gates
- **File**: `risk/engine.py`
- **Actions**:
  - [ ] Implement CVaR calculation
  - [ ] Add pre-trade CVaR check
  - [ ] Add reason_code: cvar_breach

### 6.4 Implement Position Scaling on Breach
- **File**: `risk/engine.py`
- **Actions**:
  - [ ] Scale positions when risk limits breached
  - [ ] Add reason_code: position_scaled
  - [ ] Log scaling events

---

## Phase 7: Paper Mode End-to-End Test (Priority: HIGH)

### 7.1 Run Small Universe Smoke Test
- **Command**: `python run_cycle.py --paper --workers 10 --symbols AAPL,MSFT,GOOG`
- **Actions**:
  - [ ] Execute smoke test
  - [ ] Monitor logs for errors
  - [ ] Verify cycle completes

### 7.2 Verify Cycle Meta, Decisions, Orders Tables
- **Actions**:
  - [ ] Query: `SELECT * FROM cycle_meta LIMIT 5`
  - [ ] Query: `SELECT * FROM decisions LIMIT 5`
  - [ ] Query: `SELECT * FROM orders LIMIT 5`
  - [ ] Verify data persistence

### 7.3 Check Audit Logs
- **Actions**:
  - [ ] Query: `SELECT * FROM audit_log ORDER BY timestamp DESC LIMIT 10`
  - [ ] Verify audit entries for each stage
  - [ ] Check for missing stages

### 7.4 Verify No Duplicates
- **Actions**:
  - [ ] Query: `SELECT symbol, COUNT(*) FROM decisions GROUP BY symbol HAVING COUNT(*) > 1`
  - [ ] Should return 0 rows
  - [ ] If duplicates, investigate idempotency

### 7.5 Check Decision Quality and Reason Codes
- **Actions**:
  - [ ] Review decision distribution (BUY/SELL/HOLD/REJECT)
  - [ ] Check reason_codes for validity
  - [ ] Verify data_quality reason used correctly

---

## Phase 8: Complete Test Suite (Priority: MEDIUM)

### 8.1 Fix test_phase2.py Imports
- **Status**: Known issue from Phase 1
- **Actions**:
  - [ ] Fix broken imports
  - [ ] Update module paths
  - [ ] Run test to verify

### 8.2 Run Full Test Suite
- **Command**: `python -m pytest tests/ -v --tb=short`
- **Actions**:
  - [ ] Run all tests
  - [ ] Record pass/fail status
  - [ ] Identify failing tests

### 8.3 Achieve 90%+ Pass Rate
- **Actions**:
  - [ ] Fix critical failing tests
  - [ ] Skip flaky tests if needed
  - [ ] Document known issues

### 8.4 Document Known Issues
- **Actions**:
  - [ ] Create KNOWN_ISSUES.md
  - [ ] List all failing tests with explanations
  - [ ] Note any intentional skips

---

## Phase 9: Documentation Update (Priority: MEDIUM)

### 9.1 Update PROJECT_COMPLETION.md
- **Actions**:
  - [ ] Mark Phase 1-3 as complete
  - [ ] Update progress percentages
  - [ ] Add latest achievements

### 9.2 Update COMPLETE_IMPLEMENTATION_REPORT.md
- **Actions**:
  - [ ] Add Phase 4-7 completion status
  - [ ] Update architecture diagrams
  - [ ] Add new table schemas

### 9.3 Mark Completed Items in TODO_*.md Files
- **Actions**:
  - [ ] Update TODO_5Y_BACKFILL.md
  - [ ] Update TODO_LIVE_TRADING.md
  - [ ] Update TODO_INSTITUTIONAL_IMPLEMENTATION.md

### 9.4 Update README with Getting Started
- **Actions**:
  - [ ] Add quick start section
  - [ ] Document prerequisites
  - [ ] Add troubleshooting tips

### 9.5 Update Operational Runbooks
- **File**: `RUNBOOK.md`
- **Actions**:
  - [ ] Add paper trading section
  - [ ] Add live trading section
  - [ ] Document emergency procedures

---

## Phase 10: Final Verification (Priority: HIGH)

### 10.1 Run System Diagnostics
- **Commands**:
  - [ ] `python verify_db_status.py`
  - [ ] `python verify_backfill.py`
  - [ ] `python check_backfill.py --symbols AAPL,MSFT`
- **Actions**:
  - [ ] Verify all checks pass
  - [ ] Document any warnings

### 10.2 Verify All Acceptance Criteria
- **Criteria**:
  - [ ] 225 symbols with >= 1260 rows
  - [ ] Database schema v2.4.0 complete
  - [ ] Governance gates pass
  - [ ] Paper mode functional
  - [ ] Test suite runs with 90%+ pass rate

### 10.3 Create Final Status Report
- **Actions**:
  - [ ] Generate status report
  - [ ] Include metrics and counts
  - [ ] Document completion percentage

### 10.4 Document Known Limitations
- **Actions**:
  - [ ] List any missing features
  - [ ] Document workarounds
  - [ ] Note technical debt

### 10.5 Create Deployment Checklist
- **Actions**:
  - [ ] Pre-deployment checks
  - [ ] Deployment steps
  - [ ] Post-deployment verification
  - [ ] Rollback procedures

---

## Implementation Order (Recommended)

1. **Week 1**: Test Infrastructure & Code TODOs
   - Fix test imports (Phase 1)
   - Complete Phase 4 TODOs (4.3-4.5)

2. **Week 2**: Data Quality & Paper Testing
   - Wire data quality alerts (Phase 5)
   - Run paper mode E2E test (Phase 7)

3. **Week 3**: Governance & Test Suite
   - Complete governance gates (Phase 6)
   - Fix test suite (Phase 8)

4. **Week 4**: Documentation & Final Verification
   - Update documentation (Phase 9)
   - Final verification (Phase 10)

---

## Key Commands Reference

```bash
# Test Commands
python -m pytest tests/test_phase2.py -v
python -m pytest tests/ --collect-only
python -m pytest tests/ -v --tb=short

# Backfill Commands
python verify_backfill.py --json
python check_backfill.py --symbols AAPL,MSFT,GOOG

# Paper Trading
python run_cycle.py --paper --workers 10 --symbols AAPL,MSFT,GOOG
python main.py --mode=paper

# Live Trading (when ready)
python main.py --mode=live

# Verification
python verify_db_status.py --json
python test_startup_gates.py

# Database Queries
sqlite3 mini_quant.db "SELECT COUNT(*) FROM price_history"
sqlite3 mini_quant.db "SELECT symbol, COUNT(*) FROM price_history GROUP BY symbol"
```

---

## Dependencies Summary

| Component | Status | Notes |
|-----------|--------|-------|
| Database (SQLite) | ✅ Ready | mini_quant.db exists |
| 225 Symbols | ✅ Ready | All have >= 1260 rows |
| Quality Agent | ⚠️ Partial | Needs Prometheus integration |
| Alert System | ⚠️ Partial | Needs data quality alerts |
| Test Suite | ❌ Broken | Imports need fixing |
| Paper Mode | ⚠️ Untested | E2E test pending |

---

## Progress Tracking

| Phase | Status | Completion | Dependencies |
|-------|--------|------------|--------------|
| Phase 1 | 🔄 IN PROGRESS | 25% | None |
| Phase 2 | ✅ COMPLETE | 100% | None |
| Phase 3 | ✅ COMPLETE | 100% | None |
| Phase 4 | 🔄 IN PROGRESS | 40% | Phase 1 |
| Phase 5 | ⏳ PENDING | 0% | Phase 4.3 |
| Phase 6 | 🔄 IN PROGRESS | 25% | Phase 2 |
| Phase 7 | 🔄 NEXT | 0% | Phase 5 |
| Phase 8 | ⏳ PENDING | 0% | Phase 1 |
| Phase 9 | 🔄 IN PROGRESS | 10% | All |
| Phase 10 | ⏳ PENDING | 0% | All |

---

## Next Immediate Actions

1. **Run test collection** to identify all broken tests
2. **Fix test_phase2.py imports** as priority
3. **Complete Phase 4 TODOs** (4.3-4.5)
4. **Wire data quality alerts** to cycle_runner
5. **Run paper mode E2E test**

---
Generated: 2026-01-22
Last Updated: 2026-01-22

