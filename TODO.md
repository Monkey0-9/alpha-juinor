# Institutional Quant Trading System Fixes

## 1. Fix InstitutionalStrategy.generate_signals Return Type
- **File**: strategies/institutional_strategy.py
- **Function**: generate_signals
- **Problem**: Returns dict instead of DataFrame, causing .iloc[-1] to fail in main.py
- **Fix**: Aggregate alpha signals per ticker, return single-row DataFrame

## 2. Add Schema Enforcement in Data Router
- **File**: data/collectors/data_router.py
- **Function**: get_price_history
- **Problem**: No validation that returned data is DataFrame with required columns
- **Fix**: Add assertions for DataFrame type and column presence

## 3. Fix Dotenv Parsing and Environment Validation
- **File**: main.py
- **Function**: run_production_pipeline
- **Problem**: load_dotenv() may not parse correctly, no validation of env vars
- **Fix**: Validate critical env vars at startup, fail fast if missing

## 4. Add Graceful Failure in Live Engine
- **File**: engine/live_engine.py
- **Function**: run_once
- **Problem**: One asset failure crashes full rebalance
- **Fix**: Wrap per-asset logic in try-except, log errors, continue

## 5. Ensure Single-Row DataFrame Handling in Allocator
- **File**: portfolio/allocator.py
- **Function**: allocate
- **Problem**: Potential Series vs DataFrame issues in signal processing
- **Fix**: Add type checks and conversions for signals input
