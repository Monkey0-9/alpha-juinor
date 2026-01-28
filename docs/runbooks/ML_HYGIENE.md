# ML Hygiene & Governance - Operational Runbook

## Purpose

This runbook provides operational procedures for managing the ML Alpha system under the new governance framework. It covers emergency procedures, diagnostics, recovery, and safe re-enablement of ML capabilities.

---

## 1. Emergency Disable ML Alpha

**When to use:** Immediately upon detecting widespread ML failures, feature mismatches, or production incidents.

### Procedure

```bash
# 1. Edit golden config
vim configs/golden_config.yaml

# 2. Set ml_enabled to false
features:
  ml_enabled: false  # EMERGENCY GOVERNANCE

# 3. Restart trading system
pkill -f main.py
python main.py
```

**Expected behavior:**

- Heartbeat should show `ml_state=DISABLED_CONFIG`
- ML alpha returns neutral signals (0.0)
- No model loading attempts
- `model_errors` counter stops incrementing

---

## 2. Diagnosing ML Failures

### Check Governance State

```bash
# View structured error logs
tail -100 runtime/logs/errors.jsonl | jq 'select(.component=="ML_GOVERNANCE")'

# Look for DISABLED_BY_GOVERNANCE events
grep "DISABLED_BY_GOVERNANCE" runtime/logs/errors.jsonl | jq .
```

### Identify Feature Mismatches

```bash
# Check for feature validation errors
grep "FEATURE_MISMATCH" runtime/logs/errors.jsonl | jq '{symbol:.symbol, expected:.expected_count, got:.provided_count, missing:.missing_features}'
```

### Review Heartbeat Logs

```bash
# Check latest heartbeat
tail -20 runtime/logs/live.jsonl | grep HEARTBEAT

# Look for state transitions
grep "state=" runtime/logs/live.jsonl | tail -10
```

---

## 3. Feature Refresh Audit

Run the feature refresh audit to validate that all symbols can produce contract-compliant features.

```bash
# Audit all active symbols
python scripts/feature_audit_ml.py --all --out reports/feature_audit_$(date +%Y%m%d).json

# Audit specific symbols
python scripts/feature_audit_ml.py --symbols AAPL,MSFT,GOOGL --out reports/feature_audit_subset.json

# View summary
cat reports/feature_audit_*.json | jq '{total:.total_symbols, passed:.passed, failed:.failed, pass_rate:.pass_rate}'
```

**Interpreting results:**

- `pass_rate >= 0.95`: Good - proceed to retraining
- `pass_rate < 0.95`: Investigation required - check `missing_features_by_symbol`

---

## 4. Retraining Models

After confirming feature computation is stable:

```bash
# Retrain for specific symbols
python scripts/train_ml_alpha.py --symbols AAPL,MSFT,GOOGL

# Retrain all active symbols (SLOW - use with caution)
python scripts/train_ml_alpha.py

# Verify model metadata
cat models/ml_alpha/AAPL_v1_*/model_meta.json | jq '{features:.features, n_features:.n_features, trained_at:.trained_at}'
```

**Expected output:**

- `models/ml_alpha/{SYMBOL}_v1_{timestamp}/model.pkl`
- `models/ml_alpha/{SYMBOL}_v1_{timestamp}/model_meta.json`
- Metadata includes `features` array with 28 items

---

## 5. Validating Model Metadata

Before re-enabling ML, validate that all trained models have correct metadata:

```bash
# Create validation script if needed
python - <<'PYTHON'
import json
from pathlib import Path

models_dir = Path("models/ml_alpha")
for model_dir in models_dir.glob("*_v1_*"):
    if not model_dir.is_dir():
        continue
    meta_path = model_dir / "model_meta.json"
    if not meta_path.exists():
        print(f"MISSING: {model_dir.name}")
        continue
    with open(meta_path) as f:
        meta = json.load(f)
    if meta.get("n_features") != 28:
        print(f"BAD N_FEATURES: {model_dir.name} has {meta.get('n_features')}")
    elif not meta.get("features"):
        print(f"MISSING FEATURES LIST: {model_dir.name}")
    else:
        print(f"OK: {model_dir.name}")
PYTHON
```

---

## 6. Offline Backtest

Run offline backtest to validate model performance before live deployment:

```bash
# Run backtest with new models
python scripts/train_ml_offline.py --model-dir models/ml_alpha

# Or use backtester
python backtest/backtester.py --model models/ml_v1 --start 2025-01-01 --end 2026-01-23
```

**Acceptance criteria:**

- Sharpe ratio > 1.0
- Max drawdown < 20%
- No feature mismatch errors
- All symbols processed without exceptions

---

## 7. End-to-End Integration Test

Run E2E test to validate full trading loop:

```bash
# Run integration test with ML model
python test_integration_e2e.py --ml-model models/ml_v1 -v

# Check test output
cat test_output.txt | grep -E "(PASS|FAIL)"
```

---

## 8. Staging Rollout

Deploy to staging environment for 24-hour burn-in:

```bash
# 1. Deploy to staging
./deploy.sh staging

# 2. Enable ML on staging only
ssh staging-server "sed -i 's/ml_enabled: false/ml_enabled: true/' configs/golden_config.yaml"

# 3. Monitor for 24 hours
ssh staging-server "tail -f runtime/logs/live.jsonl" | grep -E "(HEARTBEAT|ML_GOVERNANCE|ERROR)"
```

**Monitoring checklist:**

- [ ] `model_errors` remains at 0 or low (< 5)
- [ ] `ml_state` stays as `ENABLED|OK` or `ENABLED|DEGRADED`
- [ ] No `DISABLED_BY_GOVERNANCE` events
- [ ] Sharpe ratio in line with backtest expectations

---

## 9. Production Re-Enable

**ONLY after staging validation passes:**

```bash
# 1. Edit production config
vim configs/golden_config.yaml

# 2. Set ml_enabled to true
features:
  ml_enabled: true

# 3. Graceful restart
pkill -SIGTERM -f main.py
sleep 5
python main.py &

# 4. Monitor logs
tail -f runtime/logs/live.jsonl | grep -E "(HEARTBEAT|ML_GOVERNANCE)"
```

### Auto-Disable Setup

Add monitoring alert (if not already configured):

```bash
# Example: Watch for governance disable events
while true; do
  if grep -q "DISABLED_BY_GOVERNANCE" runtime/logs/errors.jsonl; then
    # Auto-revert to disabled
    sed -i 's/ml_enabled: true/ml_enabled: false/' configs/golden_config.yaml
    # Send alert
    echo "ML_DISABLED|GOVERNANCE triggered - auto-reverted" | mail -s "CRITICAL: ML Governance" ops@example.com
    break
  fi
  sleep 60
done &
```

---

## 10. Rollback Procedure

If production issues arise after ML re-enable:

```bash
# IMMEDIATE ROLLBACK
sed -i 's/ml_enabled: true/ml_enabled: false/' configs/golden_config.yaml
pkill -SIGTERM -f main.py
python main.py &

# Capture logs for forensics
cp runtime/logs/errors.jsonl forensics/ml_incident_$(date +%Y%m%d_%H%M%S).jsonl

# Notify team
echo "ML rollback completed - investigate forensics/" | mail -s "ML Rollback" team@example.com
```

---

## 11. Recovery & Root Cause

After rollback, investigate:

1. **Check model_errors spike:**

   ```bash
   grep "model_errors=" runtime/logs/live.jsonl | tail -50
   ```

2. **Identify failing symbols:**

   ```bash
   grep "FEATURE_MISMATCH" runtime/logs/errors.jsonl | jq -r '.symbol' | sort | uniq -c
   ```

3. **Compare metadata vs runtime features:**

   ```bash
   python - <<'PYTHON'
   import json
   # Load model meta
   with open("models/ml_alpha/AAPL_v1_latest/model_meta.json") as f:
       meta = json.load(f)
   print("Model expects:", meta["features"])

   # Compare with contract
   from features.contract import load_feature_contract
   contract = load_feature_contract("ml_v1")
   print("Contract defines:", contract["features"])

   if meta["features"] == contract["features"]:
       print("✓ MATCH")
   else:
       print("✗ MISMATCH")
   PYTHON
   ```

---

## 12. Maintenance & Monitoring

### Daily Checks

```bash
# Check heartbeat state
tail -1 runtime/logs/live.jsonl | jq 'select(.message | contains("HEARTBEAT"))'

# Review model_errors trend
grep "model_errors=" runtime/logs/live.jsonl | tail -10
```

### Weekly Checks

```bash
# Re-run feature audit
python scripts/feature_audit_ml.py --all --out reports/weekly_audit_$(date +%Y%m%d).json

# Compare pass rate trend
jq -r '.pass_rate' reports/weekly_audit_*.json
```

---

## Contact & Escalation

- **L1 Support:** Disable ML via config, capture logs, escalate to L2
- **L2 Support:** Run diagnostics, feature audit, attempt retraining
- **L3 Support (ML Engineering):** Root cause analysis, contract updates, code fixes

**Emergency contacts:**

- ML Team Lead: <ml-lead@example.com>
- DevOps On-Call: <oncall@example.com>
- PagerDuty: <https://example.pagerduty.com/ml-governance>
