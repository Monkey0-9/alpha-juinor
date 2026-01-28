# Deployment Checklist

## Pre-Deployment Checks

### 1. Environment Setup

- [ ] Python 3.10+ installed
- [ ] Virtual environment created and activated
- [ ] All dependencies installed: `pip install -r requirements.txt`
- [ ] Runtime directories created: `runtime/`, `reports/`, `logs/`

### 2. Configuration Files

- [ ] `.env` file created with all required variables:
  ```bash
  # Data Providers
  POLYGON_API_KEY=your_key_here
  ALPHA_VANTAGE_API_KEY=your_key_here
  FRED_API_KEY=your_key_here

  # Broker
  ALPACA_API_KEY=your_key_here
  ALPACA_SECRET_KEY=your_secret_here
  ALPACA_BASE_URL=https://paper-api.alpaca.markets  # or live

  # Mode
  TRADING_MODE=paper  # or live
  ```

- [ ] `configs/providers.yaml` reviewed and updated with correct quotas
- [ ] `configs/golden_config.yaml` reviewed for universe and risk parameters

### 3. Database Setup

- [ ] Run migration tool to create audit database:
  ```bash
  python tools/migrate_audit_db.py --create
  ```
- [ ] Verify SQLite database created: `runtime/audit.db`
- [ ] Test database write access

### 4. Provider API Keys

- [ ] Polygon.io API key valid and tested
- [ ] Alpha Vantage API key valid and tested
- [ ] FRED API key valid and tested
- [ ] Yahoo Finance (no key required, but test connectivity)
- [ ] Stooq (no key required, but test connectivity)

### 5. Pre-Flight Tests

- [ ] Run unit tests:
  ```bash
  pytest tests/test_math_formulas.py -v
  ```

- [ ] Run integration test:
  ```bash
  python tests/integration/test_full_cycle_with_mock_providers.py
  ```

- [ ] Run load smoke test:
  ```bash
  python tests/load/test_worker_pool_performance.py
  ```

- [ ] Verify all tests pass

### 6. Monitoring Setup

- [ ] Metrics directory exists: `runtime/`
- [ ] Reports directory exists: `reports/`
- [ ] Logs directory exists: `logs/`
- [ ] Verify write permissions

### 7. Risk Limits Review

- [ ] Review `risk/engine.py` parameters:
  - `max_leverage` (default: 1.0)
  - `target_vol_limit` (default: 0.15)
  - `max_drawdown_limit` (default: 0.18)
  - `cvar_limit` (default: 0.06)

- [ ] Review `portfolio/allocator.py` parameters:
  - `max_position_pct` (default: 0.10)
  - `gamma` (Kelly fraction, default: 0.2)

### 8. Universe Configuration

- [ ] Review universe in `configs/golden_config.yaml`
- [ ] Verify all symbols are tradable
- [ ] Check for delisted or halted stocks
- [ ] Confirm universe size (default: 249)

---

## Deployment Steps

### 1. Initial Dry Run

```bash
# Run in paper mode with small universe
python main.py --mode paper
```

- [ ] Observe console output for errors
- [ ] Check `runtime/audit.log` for decisions
- [ ] Verify `runtime/audit.db` has records
- [ ] Review `reports/eod_summary_*.json`

### 2. Validate Decision Coverage

```bash
# Query audit database
sqlite3 runtime/audit.db "SELECT final_decision, COUNT(*) FROM decisions GROUP BY final_decision;"
```

- [ ] Verify total decisions = universe size
- [ ] Check decision distribution (EXECUTE/HOLD/REJECT/ERROR)
- [ ] Investigate any ERROR decisions

### 3. Provider Health Check

```bash
# Check provider usage
sqlite3 runtime/audit.db "SELECT json_extract(data_providers, '$') FROM decisions LIMIT 10;"
```

- [ ] Verify primary providers (Polygon, AlphaVantage) used
- [ ] Check Yahoo only used as fallback
- [ ] Review provider confidence scores in `runtime/provider_confidence.json`

### 4. Performance Validation

- [ ] Run load test to verify latency:
  ```bash
  python tests/load/test_worker_pool_performance.py
  ```
- [ ] Confirm median latency < 60s (goal: < 10s)
- [ ] Check throughput (decisions/sec)

### 5. Monitoring Setup

- [ ] Verify metrics file created: `runtime/metrics.prom`
- [ ] Check EOD report generated: `reports/eod_summary_*.json`
- [ ] Review logs for warnings/errors

---

## Production Deployment

### 1. Switch to Live Mode

⚠️ **CRITICAL**: Only after thorough paper trading validation!

- [ ] Update `.env`: `TRADING_MODE=live`
- [ ] Update Alpaca URL: `ALPACA_BASE_URL=https://api.alpaca.markets`
- [ ] Reduce universe size for initial live run (e.g., 10 symbols)
- [ ] Set conservative risk limits

### 2. Live Trading Checklist

- [ ] Capital allocated and available in broker account
- [ ] Risk limits appropriate for account size
- [ ] Emergency stop mechanism tested
- [ ] Manual override procedures documented
- [ ] Contact information for support/escalation

### 3. Monitoring During Live Trading

- [ ] Monitor console output in real-time
- [ ] Watch for ERROR decisions (investigate immediately)
- [ ] Check order fills and slippage
- [ ] Verify risk limits enforced
- [ ] Monitor drawdown and leverage

### 4. End-of-Day Procedures

- [ ] Review EOD summary report
- [ ] Check all orders filled or cancelled
- [ ] Verify portfolio reconciliation
- [ ] Archive audit logs
- [ ] Rotate database if needed:
  ```bash
  python tools/migrate_audit_db.py --rotate
  ```

---

## Rollback Procedure

### If Issues Detected

1. **Immediate Stop**
   ```bash
   # Kill running process
   Ctrl+C
   ```

2. **Flatten Positions** (if live)
   - Use broker interface to close all positions
   - Or run emergency liquidation script (if implemented)

3. **Investigate**
   - Check `runtime/audit.log` for ERROR decisions
   - Review `logs/` for exceptions
   - Query audit database for anomalies

4. **Rollback Code** (if code issue)
   ```bash
   git checkout <previous_commit>
   ```

5. **Restore Configuration**
   - Revert to last known good config
   - Check `.env` and `configs/` files

6. **Re-test**
   - Run integration and load tests
   - Verify in paper mode before resuming

---

## Health Checks (Daily)

- [ ] Check decision coverage: 100%?
- [ ] Review ERROR count: < 5%?
- [ ] Verify provider usage: Primary > 80%?
- [ ] Check data quality: Pass rate > 60%?
- [ ] Monitor conviction: Reasonable distribution?
- [ ] Review slippage: Within expectations?
- [ ] Check drawdown: Within limits?

---

## Escalation Contacts

- **System Issues**: [Your DevOps Contact]
- **Trading Issues**: [Your Trading Desk]
- **Risk Issues**: [Your Risk Manager]
- **Data Issues**: [Your Data Team]

---

## Notes

- Always test in paper mode first
- Start with small universe in live mode
- Gradually increase universe size
- Monitor closely for first week
- Review and adjust risk limits based on live performance
