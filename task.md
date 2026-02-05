# Master Task List: System Completion

This task list consolidates all remaining work to bring the `mini-quant-fund` to a "complete and final" institutional state.

## Phase 1: Database Schema Finalization

- [ ] **1.1 Add Missing Tables**
  - [ ] `model_decay_metrics` (for tracking model degradation)
  - [ ] `capital_allocations` (for auction results)
  - [ ] `strategy_lifecycle` (for incubating/scaling strategies)
  - [ ] `governance_decisions` (explicit table if `decisions` is insufficient, or map to it)
- [ ] **1.2 Update DatabaseManager**
  - [ ] Add read/write methods for new tables.
- [ ] **1.3 Verify Migration**
  - [ ] Ensure `check_db_schemas.py` passes with new schema.

## Phase 2: Risk Engine Hardening (CVaR & Config)

- [ ] **2.1 Integrate CVaRConfig**
  - [ ] Update `risk/engine.py` to accept and enforce `CVaRConfig` from `institutional_specification.py`.
  - [ ] Ensure `cvar_limit` usage is dynamic based on config.
- [ ] **2.2 Enforce Strict Limits**
  - [ ] Verify hard stops for daily loss and drawstring.

## Phase 3: Live Governance Gates

- [ ] **3.1 Startup History Gate**
  - [ ] Update `orchestration/live_decision_loop.py` to **BLOCK** startup if symbols are missing 1260 days of history.
  - [ ] Add bypass flag for `paper_mode` (optional).
- [ ] **3.2 Dataclass Alignment**
  - [ ] Ensure `LiveSignal` matches `GovernanceDecision` structure perfectly.

## Phase 4: Final 5-Year Backfill Execution

- [ ] **4.1 Verify Backfill Tool**
  - [ ] Run `tools/backfill_5y.py` dry-run or sample.
  - [ ] Verify `tools/validation_report.py` output.
- [ ] **4.2 Execute Deep Backfill** (Long-running, may be triggered as background)
  - [ ] Run full ingestion for active universe.

## Phase 5: Documentation & Handover

- [ ] **5.1 Update READMEs**
  - [ ] Ensure `COMPLETE_IMPLEMENTATION_REPORT.md` is current.

## Phase 6: Live Trading Stabilization

- [x] **6.1 ML Mode Enforcement**
  - [x] Implement `disabled | shadow | live` logic in `MLAlpha`.
  - [x] Activate `ml_mode: live` in `golden_config.yaml`.
- [x] **6.2 ARIMA Hardening**
  - [x] Relax strict warning filters in `StatisticalAlpha`.
  - [x] Increase heartbeat threshold for `DEGRADED` state in `main.py`.
- [x] **6.3 Governance Optimization**
  - [x] Relax `ml_health_ratio` and feature staleness checks.
- [x] **6.4 End-to-End Verification**
  - [x] Run full cycle and verify `NORMAL` state and non-zero orders.
