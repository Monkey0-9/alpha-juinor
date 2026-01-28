# Institutional Trading System - Operational Runbook

## Quick Start Checklist

```bash
# 1. Set up environment
pip install -r requirements.txt

# 2. Run ingestion (Stage 1)
python ingest_history.py --mode=deep_backfill

# 3. Verify backfill
python verify_backfill.py
python verify_db_status.py

# 4. Paper trading (Stage 2)
python main.py --mode=paper

# 5. Live trading (after verification)
python main.py --mode=live
```

---

## System Architecture

### Two-Stage Execution Order (MANDATORY)

**STAGE 1 — BATCH INGESTION:**
```bash
python ingest_history.py --mode=deep_backfill
```
- Populates exactly 5 calendar years of daily OHLCV
- Stores adjusted_close, corporate actions, metadata
- Runs governance classification
- **MUST complete successfully before Stage 2**

**STAGE 2 — LIVE / RESEARCH:**
```bash
python main.py --mode=paper    # Paper trading
python main.py --mode=live     # Live trading
```
- Only runs after Stage 1 success
- Uses cached historical data (no multi-year fetches)

---

## Governance Rules

### ABSOLUTE NON-NEGOTIABLE PRINCIPLES

| Rule | Requirement | Action if Violated |
|------|-------------|-------------------|
| History Requirement | Every ACTIVE symbol must have exactly 1260 rows | HALT - emit governance log |
| Data Quality | quality_score >= 0.6 | Mark DEGRADED, block trading |
| Provider Selection | Use exact PROVIDER_CAPABILITIES matrix | Reject symbol, log decision |
| Entitlement Check | Verify provider_entitled() before use | Skip provider, continue |

### Symbol States

```
ACTIVE:       >= 1260 rows AND quality >= 0.6      → ALLOW TRADING
DEGRADED:     1000-1259 rows OR quality < 0.6      → BLOCK TRADING
QUARANTINED:  < 1000 rows OR critical failure      → BLOCK TRADING
```

---

## Provider Governance Matrix

```python
PROVIDER_CAPABILITIES = {
    "alpaca": {
        "stocks": True, "fx": False, "crypto": True,
        "commodities": False, "max_history_days": 730,
        "requires_entitlement": True
    },
    "yahoo": {
        "stocks": True, "fx": True, "crypto": True,
        "commodities": True, "max_history_days": 5000,
        "requires_entitlement": False
    },
    "polygon": {
        "stocks": True, "fx": True, "crypto": True,
        "commodities": False, "max_history_days": 5000,
        "requires_entitlement": True
    }
}

PROVIDER_PRIORITY = ["yahoo", "polygon", "alpaca"]
```

**FORBIDDEN BEHAVIORS:**
- ❌ Do NOT use Alpaca for multi-year history (>730 days)
- ❌ Do NOT fetch live prices during batch ingestion
- ❌ Do NOT retry on entitlement failures (400/401/403)
- ❌ Do NOT invent or delete symbols (read from universe.json only)

---

## Asset Classification

```python
def classify_symbol(symbol):
    if symbol.endswith("=X"):
        return "fx"
    if symbol.endswith("=F"):
        return "commodities"
    if "-USD" in symbol:
        return "crypto"
    return "stocks"
```

---

## Data Quality Score

```
score = 1.0 - (
    missing_dates_pct * 0.3 +
    duplicate_pct * 0.2 +
    zero_negative_flag * 0.2 +
    extreme_spike_flag * 0.3
)

Thresholds:
  >= 0.6  → ACCEPT
  < 0.6   → REJECT
  > 5% rejection rate → ALERT
```

---

## Operational Commands

### Ingestion

```bash
# Full 5-year backfill
python ingest_history.py --mode=deep_backfill

# Incremental daily update
python ingest_history.py --mode=incremental

# Verify database integrity
python ingest_history.py --mode=verify

# Custom date range
python ingest_history.py --start=2020-01-01 --end=2025-01-17

# Specific tickers only
python ingest_history.py --tickers=AAPL,MSFT,GOOGL
```

### Verification

```bash
# Comprehensive status check
python verify_db_status.py

# History completeness only
python verify_db_status.py --history

# Data quality only
python verify_db_status.py --quality

# Governance states only
python verify_db_status.py --governance

# JSON output (for scripts)
python verify_db_status.py --json

# Backfill verification
python verify_backfill.py

# Show failed symbols
python verify_backfill.py --symbols

# Strict mode (exit on failure)
python verify_backfill.py --strict
```

### Live Trading

```bash
# Paper trading mode
python main.py --mode=paper

# Live trading mode
python main.py --mode=live

# Single cycle (paper)
python main.py --run-once --mode=paper

# Specific tickers only
python main.py --tickers=AAPL,MSFT,GOOGL --mode=paper
```

### Daemon Operations

```bash
# 24/7 daemon with 30-min data refresh
python trading_daemon.py --data-refresh 30 --tick-interval 1.0

# Emergency halt
touch runtime/KILL_SWITCH

# Resume after halt
rm runtime/KILL_SWITCH
```

### Repair Tools

```bash
# Re-ingest specific date range
python repair_db.py --symbol=AAPL --from=2022-01-01 --to=2022-06-01

# Feature refresh
python feature_refresher.py

# ML alpha training
python scripts/train_ml_alpha.py
```

---

## Monitoring & Dashboard

### Exposed Panels

1. **Data Health**
   - Symbols: OK / Degraded / Quarantined
   - Average quality score
   - Missing history count

2. **Provider Health**
   - Entitlement status per provider
   - Failure rates
   - Latency metrics

3. **Ingestion Audit**
   - Last run_id
   - Symbols fetched
   - Rejects and failures

4. **Portfolio & Risk**
   - Exposure
   - VaR, CVaR

### Log Files

```
logs/live_trading.log
runtime/governance_halt.log
runtime/audit/audit.jsonl
runtime/raw/{run_id}/*.json.gz
```

---

## Emergency Procedures

### Missing History Detection

If system halts with:
```
[DATA_GOVERNANCE]
Missing historical data detected
Symbols affected: <N>
Required rows per symbol: 1260
Action required: Run ingest_history.py
System halted intentionally
```

**Recovery:**
```bash
# 1. Check which symbols are affected
python verify_backfill.py --symbols

# 2. Re-run ingestion
python ingest_history.py --mode=deep_backfill

# 3. Verify
python verify_backfill.py

# 4. Resume trading
python main.py --mode=paper
```

### Kill Switch Activation

```bash
# Emergency halt
touch runtime/KILL_SWITCH

# Check status
cat runtime/governance_halt.log

# Resume
rm runtime/KILL_SWITCH
python main.py --mode=paper
```

### Database Corruption

```bash
# Backup current state
cp runtime/institutional_trading.db runtime/backup_$(date +%Y%m%d).db

# Re-ingest all data
python ingest_history.py --mode=deep_backfill

# Verify
python verify_db_status.py
```

---

## Troubleshooting

### Common Issues

| Error | Cause | Solution |
|-------|-------|----------|
| "NO_VALID_PROVIDER" | No entitled provider for symbol | Check ALPACA_ENABLED, POLYGON_ENABLED env vars |
| "GUARD VIOLATION" | Multi-year fetch in live mode | Use DB queries only in live mode |
| "Insufficient history" | < 1260 rows | Run ingest_history.py |
| "Quality score < 0.6" | Data quality issues | Check provider logs, re-ingest |

### Check Provider Entitlements

```bash
# Check environment
env | grep -E "(ALPACA|POLYGON YAHOO)_ENABLED"

# Test provider connection
python -c "from data.providers.yahoo import YahooDataProvider; print(YahooDataProvider().test())"
```

### Check Database State

```bash
# Direct SQL query
sqlite3 runtime/institutional_trading.db "SELECT state, COUNT(*) FROM symbol_governance GROUP BY state"
```

---

## Files Reference

### Key Files

| File | Purpose |
|------|---------|
| `ingest_history.py` | Batch data ingestion |
| `main.py` | Live trading agent |
| `verify_db_status.py` | Comprehensive verification |
| `verify_backfill.py` | Backfill verification |
| `trading_daemon.py` | 24/7 daemon |
| `repair_db.py` | Database repair tool |
| `data/collectors/data_router.py` | Provider selection |
| `data/governance/governance_agent.py` | Symbol classification |
| `database/manager.py` | Database operations |
| `configs/universe.json` | Symbol universe |

### Database Tables

| Table | Purpose |
|-------|---------|
| `price_history` | Daily OHLCV data |
| `symbol_governance` | Symbol states |
| `data_quality` | Quality scores |
| `ingestion_audit` | Ingestion audit |
| `ingestion_audit_runs` | Run summaries |

---

## Success Criteria

Before running live trading:

- [ ] `python verify_backfill.py` shows 100% compliance
- [ ] `python verify_db_status.py` shows all checks passed
- [ ] No ERROR logs in `logs/live_trading.log`
- [ ] Kill switch file does not exist
- [ ] Governance log shows no halts

---

---

## Institutional Hardening & Monitoring

### Heartbeat Awareness
The system emits a heartbeat every 5 seconds to the console and `live.jsonl`.
`HEARTBEAT | uptime=360s | symbols=225 | cycles=10 | model_errors=0 | arima_fb=5`

- **model_errors**: Counter of ML prediction failures caught and bypassed.
- **arima_fb**: Counter of ARIMA non-convergence fallbacks to EWMA.
- **cycles**: Total number of successful decision loops completed.

### Structured Logging
All logs are dual-piped:
1. **Console**: Clean `Rich` formatting for human operators.
2. **live.jsonl**: Full structured audit trail for automated monitoring.
3. **errors.jsonl**: Dedicated error stream with stack traces for post-mortem.

### Safe Mode & Dynamic Risk
If the system detects >5 consecutive loop errors or critical data gaps, it enters **Safe Mode**.
- **Behavior**: Signal exposure is automatically scaled by **0.1x** (90% reduction).
- **Recovery**: Reset occurs automatically if valid signals resume without errors.

### Hardening Verification
Run the mandatory verification suite before any production promotion:

```bash
python -m pytest tests/test_feature_validation.py tests/test_ml_predict_safe.py tests/test_arima_fallback.py tests/test_heartbeat.py
```

---

## Support

