# OPS RECOVERY PLAYBOOK

## 🚨 CRITICAL INCIDENT RESPONSE

### 1. PROVIDER FAILURE (Rate > 10%)

**Symptom**: `PROVIDER_DOWN` alert.
**Action**:

1. Check `runtime/blocked_providers.json`.
2. Inspect `config/provider_capabilities.yaml` for backups.
3. If primary down, update `entitlement_router.block_provider` logic or manually edit blocklist.
4. Restart Ingestion.

### 2. DATA QUALITY DROP

**Symptom**: Average Quality < 0.6.
**Action**:

1. Check `output/ingestion_summary_*.csv` for specific symbols.
2. If "Empty DataFrame", check API status.
3. If "Spikes", check `validation_flags`.
4. Run `scripts/ingest_5y_batch.py` in dry-run mode to verify fix.

### 3. MODEL DRIFT (PSI > 0.2)

**Symptom**: `MODEL_DRIFT` alert.
**Action**:

1. Model is automatically moved to `SHADOW` status.
2. Data Science team to retrain model.
3. Run `ml/governor.py` analysis on training set vs live set.

### 4. MANUAL OVERRIDE (CRISIS RECOVERY)

When system locks due to `KILL_SWITCH` or `TAIL_RISK_EXCEED`:

1. **Req ID**: Generate request via CLI.
   `python scripts/request_override.py --strategy=STRAT_A --reason="False positive correlation spike"`
2. **Sign-off 1**: Senior Quant.
3. **Sign-off 2**: CRO / Head of Trading.
4. **Execution**: `governance/approval_workflow.py` grants access token.

**NEVER BYPASS WITHOUT 2 SIGN-OFFS.**
