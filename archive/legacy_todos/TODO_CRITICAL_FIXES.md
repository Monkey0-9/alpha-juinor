# Critical System Fixes - TODO

## Issue 1: Database Constraint Failure
**Problem**: `final_decision` values don't match database constraints
- Database expects: `EXECUTE`, `HOLD`, `REJECT`, `ERROR`
- Code uses: `EXECUTE_BUY`, `EXECUTE_SELL`, `HOLD`, `REJECT`

## Issue 2: Database Locked
**Problem**: Multiple SQLite database lock errors
- WAL mode not properly configured
- Transaction handling issues

## Issue 3: Insufficient Buying Power
**Problem**: Orders exceed account buying power ($229,980.19)
- No pre-trade risk check for available capital

## Issue 4: Fractional Short Selling
**Problem**: `fractional orders cannot be sold short` for XLY
- No validation before short selling

---

## Fixes Status

### Step 1: Fix Decision Constants (meta_brain.py)
- [ ] Update DECISION_BUY to "EXECUTE_BUY" -> "EXECUTE" with side indicator
- [ ] Update DECISION_SELL to "EXECUTE_SELL" -> "EXECUTE" with side indicator
- [ ] Ensure schema.py decisions table accepts valid values

### Step 2: Fix Database Locking (sqlite_adapter.py)
- [ ] Improve connection handling with isolation_level=None for WAL
- [ ] Add proper timeout and retry logic
- [ ] Ensure WAL mode is properly enabled

### Step 3: Fix Buying Power Check (alpaca_handler.py)
- [ ] Add get_account() call to check buying power
- [ ] Validate order value against available capital
- [ ] Add proper error handling and logging

### Step 4: Fix Fractional Short Selling (alpaca_handler.py)
- [ ] Add validation for short selling (qty must be integer)
- [ ] Reject fractional short orders
- [ ] Log warning when order is blocked

### Step 5: Update contracts.py
- [ ] Add proper decision enum values matching database
- [ ] Add BUY/SELL action types

---

## Files to Modify:
1. `agents/meta_brain.py` - Decision constants
2. `database/adapters/sqlite_adapter.py` - Database connection
3. `execution/alpaca_handler.py` - Order validation
4. `contracts.py` - Decision enums
5. `database/schema.py` - Table constraints (if needed)

