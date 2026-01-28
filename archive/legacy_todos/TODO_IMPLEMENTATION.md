# Institutional Market Data Governance & Ingestion - Implementation Tasks

## Phase 1: Provider Governance & Classification ✅ COMPLETE

### Task 1.1: Update data_router.py with Exact Provider Governance ✅
- [x] Add exact PROVIDER_CAPABILITIES matrix (yahoo, polygon, alpaca)
- [x] Add PROVIDER_PRIORITY = ["yahoo", "polygon", "alpaca"]
- [x] Add classify_symbol(symbol) function (exact implementation)
- [x] Add provider_entitled(provider) runtime check
- [x] Add entitlement_registry dict
- [x] Update select_provider to use exact logic
- [x] Ensure no Alpaca for >730 days history

### Task 1.2: Update ingest_history.py ✅
- [x] Use exact classify_symbol from data_router (already uses router.select_provider)
- [x] Add KILL_SWITCH check

## Phase 2: Live Trading Governance Gates ✅ COMPLETE

### Task 2.1: Update main.py ✅
- [x] Add check_1260_rows_requirement() function
- [x] Add governance_halt() function with exact log format
- [x] Add KILL_SWITCH check
- [x] Call governance check before initialize_system()
- [x] Emit exact governance log on halt

## Phase 3: Verification & Monitoring ✅ COMPLETE

### Task 3.1: Create verify_db_status.py ✅
- [x] Implement check_history_completeness()
- [x] Implement check_data_quality()
- [x] Implement check_provider_entitlements()
- [x] Implement run_full_verification()
- [x] Return structured status report

### Task 3.2: Update verify_backfill.py ✅
- [x] Fix table reference (trading_eligibility → symbol_governance)
- [x] Add quality score check
- [x] Add provider entitlement verification
- [x] Return exit code based on results

## Phase 4: Documentation ✅ COMPLETE

### Task 4.1: Create RUNBOOK.md ✅
- [x] Complete operational documentation
- [x] Quick start checklist
- [x] Emergency procedures
- [x] Troubleshooting guide

## Verification Results (Confirmed Working)

```bash
# Backfill verification - shows 2/2 active symbols compliant
python verify_backfill.py --json
# Result: {"passed": true, "total_symbols": 2, "compliant": 2}

# Comprehensive status check
python verify_db_status.py --json
# Result: All 5 checks executed, governance functioning correctly
```

## Files Modified

1. **data/collectors/data_router.py**
   - Added PROVIDER_CAPABILITIES, PROVIDER_PRIORITY
   - Added classify_symbol(), provider_entitled(), select_provider()
   - Added register_entitlement(), clear_unavailable_cache()

2. **main.py**
   - Added REQUIRED_HISTORY_ROWS, KILL_SWITCH_PATH
   - Added check_kill_switch(), governance_halt()
   - Added check_history_completeness(), check_1260_rows_requirement()
   - Updated start() to run governance checks before trading

3. **ingest_history.py**
   - Added KILL_SWITCH check and check_kill_switch() function

4. **verify_db_status.py** (NEW)
   - Comprehensive verification tool
   - Checks history, quality, governance, entitlements, audit

5. **verify_backfill.py** (UPDATED)
   - Fixed table reference
   - Added quality checks
   - Proper exit codes

6. **RUNBOOK.md** (NEW)
   - Complete operational documentation

## Next Steps for Production

1. Run ingestion: `python ingest_history.py --mode=deep_backfill`
2. Verify: `python verify_backfill.py && python verify_db_status.py`
3. Paper trade: `python main.py --mode=paper`
4. Live trade: `python main.py --mode=live`

## Implementation Complete ✅

All required governance rules, provider selection, and verification tools have been implemented and verified working.

