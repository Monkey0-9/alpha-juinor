# Institutional Market Data Governance & Ingestion Implementation Plan

## Information Gathered

### Current State Analysis

1. **ingest_history.py** - Main ingestion entry point
   - Uses `InstitutionalIngestionAgent` for batch processing
   - Has quality scoring (0.6 threshold)
   - Uses DataRouter for provider selection
   - Governance check at end (compares active symbols)

2. **data_router.py** - Provider routing
   - Has `_classify_ticker()` method
   - Has `select_provider()` but not exact implementation as specified
   - PROVIDER_CAPABILITIES exists but differs from spec
   - PROVIDER_PRIORITY is ["polygon", "yahoo", "alpaca"]

3. **main.py** - Live trading agent
   - Has `check_governance_gate()` for ACTIVE symbols
   - Has `load_252d_market_data()` for 252-bar window
   - No explicit 1260-row check before trading
   - No governance halt function

4. **governance_agent.py** - Symbol classification
   - Has `classify_symbol()` with state logic
   - ACTIVE: >=1260 rows AND quality >= 0.6
   - DEGRADED: 1000-1259 rows
   - QUARANTINED: < 1000 rows

5. **database/manager.py** - Database operations
   - Has `get_active_symbols()` method
   - Has `upsert_symbol_governance()` method
   - Has price history tables

6. **configs/universe.json** - 249+ symbols (stocks, crypto, FX, commodities)

7. **verify_backfill.py** - Verification script (references old table name)

## Plan: Comprehensive Implementation

### Step 1: Update data_router.py with Exact Provider Governance

**File:** `data/data_router.py`

Add/Update:
- Exact `PROVIDER_CAPABILITIES` matrix as specified
- Exact `PROVIDER_PRIORITY` = ["yahoo", "polygon", "alpaca"]
- `classify_symbol(symbol)` function (exact implementation)
- `select_provider(symbol, history_days)` with entitlement checks
- `provider_entitled(provider)` runtime check
- `entitlement_registry` for tracking entitlements

### Step 2: Update ingest_history.py for Strict Provider Selection

**File:** `ingest_history.py`

Changes:
- Import exact `classify_symbol` from data_router
- Use exact `select_provider` from DataRouter
- Ensure no Alpaca for multi-year history (max 730 days)
- Add governance audit logging

### Step 3: Add Governance Halt Function to main.py

**File:** `main.py`

Add:
- `check_1260_rows_requirement()` - Validates all ACTIVE symbols have >=1260 rows
- `governance_halt()` - Emits institutional governance log and exits
- Call check before `initialize_system()`
- Use exact log format:

```
[DATA_GOVERNANCE]
Missing historical data detected
Symbols affected: <N>
Required rows per symbol: 1260
Action required: Run ingest_history.py
System halted intentionally
```

### Step 4: Create Comprehensive Verification Script

**File:** `verify_db_status.py`

Implement:
- `check_history_completeness()` - Returns count of symbols with >=1260 rows
- `check_data_quality()` - Returns count of symbols with quality >= 0.6
- `check_provider_entitlements()` - Returns entitlement status
- `run_full_verification()` - Runs all checks and returns status

### Step 5: Update verify_backfill.py

**File:** `verify_backfill.py`

Changes:
- Fix table reference (trading_eligibility → symbol_governance)
- Add quality score check
- Add provider entitlement verification
- Return exit code based on results

### Step 6: Add Emergency Halt/Kill Switch Support

**Files:** `main.py`, `ingest_history.py`

Add:
- Check for `runtime/KILL_SWITCH` file
- Exit immediately if file exists
- Log halt message

### Step 7: Update Database Schema (if needed)

**File:** `database/schema.py`

Ensure:
- `symbol_governance` table exists with correct structure
- `data_quality` table exists with validation_flags
- `ingestion_audit` table exists

## Dependent Files to be Edited

1. `data/router.py` - Provider selection and classification
2. `ingest_history.py` - Ingestion entry point
3. `main.py` - Live trading with governance gates
4. `verify_backfill.py` - Verification script
5. `verify_db_status.py` - New comprehensive verification
6. `data/governance/governance_agent.py` - Symbol classification

## Followup Steps

1. Run `python ingest_history.py --mode=deep_backfill` to populate 5-year history
2. Run `python verify_backfill.py` to verify 1260 rows per symbol
3. Run `python verify_db_status.py` for comprehensive status
4. Only then: `python main.py --mode=paper` or `--mode=live`

## Key Metrics to Track

- Symbols with >= 1260 rows
- Symbols with quality >= 0.6
- Symbols with ACTIVE state
- Provider entitlement status
- Ingestion success/failure rates

