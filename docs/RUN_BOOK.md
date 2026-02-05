# S-Class System Run Book

## Deployment Guide

### 1. Prerequisites

- Python 3.9+
- Dependencies: `pip install -r requirements.txt`
- Check `task.md` is "VERIFIED".

### 2. Start System (Live Paper Mode)

To launch the 30-day live paper trading run:

```bash
python run_validated_system.py --mode paper --duration 30
```

*Note: This process must remain running. Use `nohup` or `screen` on a server.*

---

## Operations & Monitoring

### Key Logs (`system.log`)

- **HEARTBEAT**: "Starting Cycle X/Y" (Active)
- **EXECUTION**: "EXECUTE TWAP" (Trading active)
- **RISK**: "CIRCUIT BREAKER TRIGGERED" (Critical Alert)

### State Persistence

The system saves state to `system_state.json` automatically:

- After every daily cycle.
- On graceful shutdown.
- Contains: Positions, Equity, High Water Mark.

---

## Recovery Procedures

### Scenario A: Process Crash / Restart

1. Simply restart the command: `python run_validated_system.py ...`
2. Logic: The system detects `system_state.json` and resumes positions/equity automatically.

### Scenario B: Circuit Breaker Triggered

**symptom**: Log shows "SYSTEM HALTED" and exits.

1. Analyze the drawdown in logs.
2. If false alarm or market recovered:
    - Edit `system_state.json`: Set `"breaker_triggered": false`.
3. Restart system.

### Scenario C: Corrupted State

1. Backup bad state: `mv system_state.json system_state.bak`
2. Restart system (Will start fresh with $100k equity).
