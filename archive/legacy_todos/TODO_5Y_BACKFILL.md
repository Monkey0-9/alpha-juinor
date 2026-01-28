# 5-Year Market Data Backfill - Implementation Tracker

## Task: Fetch, persist, validate 5 years of market data (2021-01-19 → 2026-01-19)

---

## Phase 1: Database Schema Setup ✅ COMPLETE
- [x] 1.1 Update database/schema.py with required tables:
  - [x] price_history_daily (with source_provider, raw_hash, pulled_at)
  - [x] price_history_intraday (partitioned by date)
  - [x] execution_feedback (order_id primary key)
  - [x] provider_metrics (provider_name, date composite PK)
  - [x] data_quality_log (symbol, date composite PK)
  - [x] ingestion_audit (audit JSONL records)

## Phase 2: Provider Implementation ✅ COMPLETE
- [x] 2.1 Add AlphaVantage provider with daily + intraday support
- [x] 2.2 Add Stooq provider (free fallback)
- [x] 2.3 Update provider_bandit.py with quota integration
- [x] 2.4 Update provider_quota_manager.py with persistence
- [x] 2.5 Create provider registry (__init__.py)

## Phase 3: Data Backfill Orchestrator ✅ COMPLETE
- [x] 3.1 Create tools/backfill_5y.py:
  - [x] Job driver with atomic operations
  - [x] Provider bandit selection per symbol
  - [x] Daily bars fetch for 5-year window
  - [x] Raw payload persistence to runtime/raw/
  - [x] SHA256 raw_hash computation
  - [x] UPSERT to price_history_daily
  - [x] Per-symbol validation checklist
  - [x] JSONL audit record emission
  - [x] Retry logic (exponential backoff, max 5 attempts)

## Phase 4: Validation System ✅ COMPLETE
- [x] 4.1 Implement validate_daily_data() with:
  - [x] Row count vs trading day calendar
  - [x] Null cells ≤ 5% check
  - [x] No negative volumes/prices
  - [x] Price continuity check (log-return sanity)
  - [x] Quality score computation (0.0-1.0)
  - [x] tools/validation_report.py for reporting

## Phase 5: Incremental Sync ✅ COMPLETE
- [x] 5.1 Create tools/data_sync.py:
  - [x] Daily job (00:15 UTC): fetch yesterday + missing 7 days
  - [x] Intraday streaming support (30s polling)
  - [x] Validate, upsert, update quality log

## Phase 6: Integration & Testing ⏳ IN PROGRESS
- [ ] 6.1 Wire canonical DB to feature pipeline
- [ ] 6.2 Run full backfill for 249+ symbols
- [ ] 6.3 Generate validation report
- [ ] 6.4 Verify all downstream pipelines use canonical DB

---

## Implementation Notes

### Key Constraints:
- Window: 2021-01-19 through 2026-01-19 (inclusive)
- Idempotent: UPSERT on (symbol, date) or (symbol, datetime)
- Atomic: Each symbol fetch produces one audit record
- Auditable: JSONL at runtime/audit/backfill_{job_id}.jsonl

### Quality Thresholds:
- MIN_DATA_QUALITY = 0.6
- NULL_THRESHOLD = 0.05 (5%)
- REJECT if any: negative price/volume, >5% nulls

### Provider Priority (from providers.yaml):
1. polygon (tier: primary, monthly_limit: 50000)
2. alpha_vantage (tier: primary, monthly_limit: 25000)
3. stooq (tier: secondary, monthly_limit: 100000)
4. yahoo (tier: fallback, monthly_limit: 200000)

---

## Commands to Execute

# Full 5-year backfill for all symbols
python tools/backfill_5y.py --start 2021-01-19 --end 2026-01-19 --symbols all

# Incremental sync for a single symbol
python tools/data_sync.py --symbol AAPL --start 2026-01-01 --end 2026-01-19

# Generate validation report
python tools/validation_report.py --start 2021-01-19 --end 2026-01-19

# Check backfill job status
python tools/backfill_5y.py --status --job_id <job_id>


