# Disaster Recovery Checklist: "System Clean Slate"

**Trigger**: System crash, data corruption, or "Red Flag" execution error.
**Goal**: Restore safe trading capability within 15 minutes.

## Phase 1: Assess & Contain (Minutes 0-5)

1. [ ] **Stop the Agent**: `Ctrl+C` or `taskkill /IM python.exe /F`.
2. [ ] **Check Positions**: Log into Alpaca Dashboard.
    * Compare `Dashboard Positions` vs `Last Known Log State`.
    * *If mismatched*: Trust the **Broker** (Alpaca). The broker is truth.
3. [ ] **Cancel Open Orders**: Manually "Cancel All" in Alpaca to prevent "ghost" fills.

## Phase 2: Restore State (Minutes 5-10)

1. [ ] **Backup Logs**: Copy `runtime/logs/*.log` to `runtime/logs/crash_backup/`.
2. [ ] **Reset Cache**: Delete `runtime/cache/*` (force fresh data on restart).
3. [ ] **Safe Mode Restart**:

    ```bash
    python main.py --mode paper --run-once
    ```

    * This runs one cycle to verify connectivity and data feeds without looping.

## Phase 3: Resume Operations (Minutes 10-15)

1. [ ] **Verify Heartbeat**: Check `monitoring/dashboard.py`.
2. [ ] **Resume Loop**:

    ```bash
    python main.py --mode paper --duration 90 --log-level INFO
    ```

3. [ ] **Incident Report**: Document time/error in `runtime/logs/incident_log.txt`.
