# 5-Year Backfill & Validation - Implementation Plan

## Mission: Complete canonical 5-year backfill with full auditability and data quality enforcement

---

## Phase 1: Harden Data Quality & Alerts (Priority: HIGH)

### 1.1 Wire Quality Agent to Alerts System
- [ ] Update `data_intelligence/quality_agent.py` to:
  - Export `quality_score` for Prometheus metrics
  - Emit alerts when quality_score < 0.6
  - Add `data_missing_days_total` metric

### 1.2 Add Prometheus Metrics
- [ ] Update `monitoring/prometheus_metrics.py`:
  - Add `avg_quality_score` gauge
  - Add `data_missing_days_total` gauge
  - Add `price_history_rows_total` counter

### 1.3 Enforce REJECT on Low Quality
- [ ] Update `orchestration/cycle_runner.py`:
  - Import QualityAgent
  - For symbols with quality_score < 0.6, set final_decision = REJECT
  - Add reason_code: "data_quality"

### 1.4 Alert Firing Logic
- [ ] Update `monitoring/alerts.py`:
  - Add alert trigger for `data_missing_days_total > 0`
  - Add alert for low quality scores

---

## Phase 2: Paper Mode Smoke Run (Priority: HIGH)

### 2.1 Verify Pipeline End-to-End
- [ ] Run: `python run_cycle.py --paper --workers 10 --symbols AAPL,MSFT,GOOG`
- [ ] Verify cycle_meta row created with durations and counts
- [ ] Verify audit_log has entries for fetch→features→agents→meta→risk→orders

### 2.2 Check Decision Outcomes
- [ ] Query: `SELECT symbol, final_decision, reason_codes FROM decisions LIMIT 5`
- [ ] Verify no duplicates: `SELECT count(*) FROM decisions WHERE symbol='AAPL'`

### 2.3 Check Order Generation
- [ ] Verify PENDING orders created
- [ ] Verify idempotent order_id generation

---

## Phase 3: Execute 5-Year Backfill (Priority: CRITICAL)

### 3.1 Prepare Backfill Environment
- [ ] Verify `configs/universe.json` has full symbol list
- [ ] Check provider API keys in environment:
  - `ALPHA_VANTAGE_API_KEY`
  - `POLYGON_API_KEY`
- [ ] Ensure `runtime/raw/` directory exists for raw payloads

### 3.2 Run Full Backfill
- [ ] Execute: `python tools/backfill_5y.py --start 2021-01-19 --end 2026-01-19 --symbols all --workers 20`
- [ ] Monitor progress (logs/backfill_5y_*.log)
- [ ] Track: success/failed/invalid counts

### 3.3 Verify Idempotency
- [ ] Re-run backfill command
- [ ] Query: `SELECT count(*) FROM price_history_daily WHERE symbol='AAPL' AND date='2025-12-31'`
- [ ] Verify result is 1 (not 2)

### 3.4 Check Data Coverage
- [ ] Query: `SELECT symbol, COUNT(*) as cnt FROM price_history_daily GROUP BY symbol ORDER BY cnt LIMIT 5`
- [ ] Expected: ~1260 trading days per symbol (5 years)

---

## Phase 4: Validation & Audit (Priority: HIGH)

### 4.1 Generate Validation Report
- [ ] Execute: `python tools/validation_report.py --start 2021-01-19 --end 2026-01-19 --output validation_report.json`
- [ ] Review:
  - [ ] Overall quality score ≥ 0.8
  - [ ] Low quality symbols count = 0
  - [ ] Missing trading days < 5%

### 4.2 Check Audit JSONL Files
- [ ] Verify: `raw/jobmanifests/backfill_*.jsonl` exists
- [ ] Each file should contain per-symbol audit records
- [ ] Check raw_hash present in each record

### 4.3 Verify Provider Metrics
- [ ] Query: `SELECT * FROM provider_metrics ORDER BY date DESC LIMIT 10`
- [ ] Verify avg_quality_score populated
- [ ] Verify pulls and successes counts

### 4.4 Check Data Quality Log
- [ ] Query: `SELECT symbol, quality_score, issues_json FROM data_quality_log WHERE quality_score < 0.6`
- [ ] Should return 0 rows
- [ ] If any rows, investigate and re-fetch those symbols

---

## Phase 5: Wire Features to Canonical DB (Priority: MEDIUM)

### 5.1 Verify Feature Compute Reads from Canonical DB
- [ ] Update `features/compute.py`:
  - Add method to read price data from `price_history_daily`
  - Remove direct provider calls in favor of DB reads

### 5.2 Ensure Feature Versioning
- [ ] Verify `features` table has `version` column
- [ ] Verify `features/compute.py` writes `feature_version`

### 5.3 Test Feature Pipeline
- [ ] Run: `python run_cycle.py --paper --symbols 5`
- [ ] Verify features written to DB
- [ ] Query: `SELECT symbol, date, version FROM features LIMIT 5`

---

## Phase 6: Final Acceptance Checklist (Priority: BLOCKER)

### 6.1 Data Quality Acceptance
- [ ] price_history_daily has continuous rows for trading days
- [ ] OR data_quality_log entry explaining gaps
- [ ] quality_score ≥ 0.6 for all symbols

### 6.2 Audit Acceptance
- [ ] Audit JSONL exists in `raw/jobmanifests/`
- [ ] Each symbol/job has audit record
- [ ] raw_hash, source_provider, pulled_at present

### 6.3 Metrics Acceptance
- [ ] provider_metrics shows pulls
- [ ] provider_metrics shows avg_quality_score
- [ ] Prometheus metrics exposed

### 6.4 Idempotency Acceptance
- [ ] Re-running backfill does not create duplicates
- [ ] Query returns exactly 1 row for any symbol/date combination

---

## Commands Reference

```bash
# Paper smoke run
python run_cycle.py --paper --workers 10 --symbols AAPL,MSFT,GOOG

# Full 5-year backfill
python tools/backfill_5y.py --start 2021-01-19 --end 2026-01-19 --symbols all --workers 20

# Idempotency test (run twice)
python tools/backfill_5y.py --start 2021-01-19 --end 2026-01-19 --symbols AAPL --workers 1

# Check for duplicates
sqlite3 runtime/institutional_trading.db "SELECT symbol, date, COUNT(*) FROM price_history_daily WHERE symbol='AAPL' GROUP BY symbol, date HAVING COUNT(*) > 1"

# Generate validation report
python tools/validation_report.py --start 2021-01-19 --end 2026-01-19 --output validation_report.json

# Check data coverage
sqlite3 runtime/institutional_trading.db "SELECT COUNT(*) as total_rows FROM price_history_daily"
sqlite3 runtime/institutional_trading.db "SELECT COUNT(DISTINCT symbol) as symbols FROM price_history_daily"

# Check quality summary
sqlite3 runtime/institutional_trading.db "SELECT AVG(quality_score) as avg_score FROM data_quality_log"

# Check provider metrics
sqlite3 runtime/institutional_trading.db "SELECT provider_name, SUM(pulls) as total_pulls, AVG(avg_quality_score) as avg_q FROM provider_metrics GROUP BY provider_name"

# View audit JSONL
cat raw/jobmanifests/backfill_*.jsonl | head -100
```

---

## Progress Tracking

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 1: Data Quality & Alerts | ⏳ PENDING | Wire quality_agent to alerts |
| Phase 2: Paper Smoke Run | ⏳ PENDING | Verify E2E pipeline |
| Phase 3: 5-Year Backfill | ⏳ PENDING | Execute full backfill |
| Phase 4: Validation & Audit | ⏳ PENDING | Generate reports, verify audit |
| Phase 5: Feature Wiring | ⏳ PENDING | Wire features to canonical DB |
| Phase 6: Final Acceptance | ⏳ PENDING | Verify all acceptance criteria |

---

## Dependencies

1. **Provider APIs**: AlphaVantage, Polygon (configured in environment)
2. **Database**: SQLite at `runtime/institutional_trading.db`
3. **Storage**: `runtime/raw/` for raw payloads, `raw/jobmanifests/` for audit JSONL
4. **Monitoring**: Prometheus metrics file at `runtime/metrics.prom`

## Risk Mitigation

1. **API Rate Limits**: Provider bandit handles fallback automatically
2. **Data Gaps**: Quality agent flags symbols with gaps
3. **Duplicates**: UPSERT ensures idempotency
4. **Failures**: Backfill failures logged to `backfill_failures` table

## Next Steps (After Completion)

After the 5-year backfill is complete and validated, proceed with:
- Factor Attribution service (MVP)
- Agent weights store + online update
- Portfolio targets table + optimizer
- Execution feedback ingestion

