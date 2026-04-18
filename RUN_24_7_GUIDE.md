# 🔄 NEXUS INSTITUTIONAL 24/7 CONTINUOUS EXECUTION GUIDE

## ⚡ QUICK START: Run 24/7 Right Now

```bash
# Option 1: Windows - Run in console
python run_24_7.py --mode backtest --asset-class multi --venues 235

# Option 2: Linux/Mac - Run in console
python run_24_7.py --mode backtest --asset-class multi --venues 235

# Option 3: Windows - Setup Task Scheduler (auto-restart on system boot)
bash setup_windows_24_7.bat

# Option 4: Linux/Mac - Setup systemd (auto-restart on system boot)
bash setup_linux_24_7.sh
```

---

## 📋 WHAT IS 24/7 EXECUTION?

**Without 24/7 Setup:**
- Platform runs when you manually execute it
- Stops when you close the terminal
- No monitoring or recovery
- Manual log review

**With 24/7 Setup:**
- ✅ Runs continuously forever
- ✅ Auto-recovers from crashes
- ✅ Starts automatically on system reboot
- ✅ Logs all activity to files
- ✅ Tracks metrics (uptime, restarts, errors)
- ✅ Monitors process health
- ✅ Can run in background while you work

---

## 🚀 OPTION 1: RUN 24/7 IN CONSOLE (Manual)

**Use this to test 24/7 execution before automating:**

```bash
cd C:\mini-quant-fund
python run_24_7.py --mode backtest --asset-class multi --venues 235
```

**Expected Output:**
```
2026-04-17 10:30:45,123 - Nexus24x7 - INFO - Starting 24/7 execution monitor...
2026-04-17 10:30:45,125 - Nexus24x7 - INFO - Log file: logs/nexus_24_7_20260417_103045.log
2026-04-17 10:30:45,126 - Nexus24x7 - INFO - Metrics file: logs/metrics_24_7_20260417_103045.json
2026-04-17 10:30:46,200 - Nexus24x7 - INFO - STARTING NEXUS INSTITUTIONAL PLATFORM - CYCLE 1
...
```

**To Stop:**
- Press `Ctrl+C`

**Features:**
- Runs until you stop it
- Logs to: `logs/nexus_24_7_*.log`
- Metrics to: `logs/metrics_24_7_*.json`
- Automatic restart on crash (every 30 seconds)

---

## 🪟 OPTION 2: WINDOWS TASK SCHEDULER (Automated)

### Step 1: Open Command Prompt as Administrator

1. Press `Win + X` → Select "Command Prompt (Admin)" or "PowerShell (Admin)"
2. Navigate to project:
   ```cmd
   cd C:\mini-quant-fund
   ```

### Step 2: Run Setup Script

```cmd
setup_windows_24_7.bat
```

**This will:**
- ✅ Create startup batch file
- ✅ Register task with Windows Task Scheduler
- ✅ Enable auto-start on system reboot
- ✅ Show verification of task creation

### Step 3: Verify Installation

```cmd
REM View task details
schtasks /query /tn "NexusInstitutional24x7" /v

REM Check if running
tasklist | find "python"

REM View logs
type logs\nexus_24_7_*.log
```

### Troubleshooting Windows Setup

```cmd
# If setup fails, manually create the task:
schtasks /create /tn "NexusInstitutional24x7" /tr "c:\mini-quant-fund\run_24_7_startup.bat" /sc onstart /ru SYSTEM /rl highest

# Stop the running task
schtasks /end /tn "NexusInstitutional24x7"

# Delete the task
schtasks /delete /tn "NexusInstitutional24x7" /f

# View all tasks
schtasks /query

# Trigger task manually
schtasks /run /tn "NexusInstitutional24x7"
```

---

## 🐧 OPTION 3: LINUX SYSTEMD (Automated)

### Step 1: Make Script Executable

```bash
chmod +x setup_linux_24_7.sh
```

### Step 2: Run Setup Script (as root or with sudo)

```bash
# For system-wide service (recommended)
sudo bash setup_linux_24_7.sh

# OR for user-level service (no sudo needed)
bash setup_linux_24_7.sh
```

**This will:**
- ✅ Create systemd service file
- ✅ Enable auto-start on system boot
- ✅ Start the service immediately
- ✅ Show real-time logs

### Step 3: Verify Installation

```bash
# Check service status
systemctl status nexus-institutional-24x7
# OR for user-level
systemctl --user status nexus-institutional-24x7

# View real-time logs
journalctl -u nexus-institutional-24x7 -f

# View last 50 lines
journalctl -u nexus-institutional-24x7 -n 50
```

### Troubleshooting Linux Setup

```bash
# If service fails, check status
systemctl -e STATUS nexus-institutional-24x7

# View detailed logs
journalctl -u nexus-institutional-24x7 -n 100

# Restart service
systemctl restart nexus-institutional-24x7

# Stop service
systemctl stop nexus-institutional-24x7

# View service file
cat /etc/systemd/system/nexus-institutional-24x7.service

# Delete service (requires root)
sudo systemctl stop nexus-institutional-24x7
sudo systemctl disable nexus-institutional-24x7
sudo rm /etc/systemd/system/nexus-institutional-24x7.service
sudo systemctl daemon-reload
```

---

## 🎯 EXECUTION OPTIONS FOR 24/7 MONITOR

### By Mode

```bash
# Backtest mode (default) - Fast cycles
python run_24_7.py --mode backtest --asset-class multi --venues 235

# Paper trading mode - Live data, no orders
python run_24_7.py --mode paper --asset-class multi --venues 235

# Market making mode - Liquidity provision
python run_24_7.py --mode market-making --asset-class equities --venues 100

# Live trading mode (requires credentials)
python run_24_7.py --mode live --asset-class equities --venues 50
```

### By Asset Class

```bash
# Equities only
python run_24_7.py --asset-class equities

# Fixed income
python run_24_7.py --asset-class fixed-income

# Crypto
python run_24_7.py --asset-class crypto

# Derivatives
python run_24_7.py --asset-class derivatives

# Forex
python run_24_7.py --asset-class fx

# All asset classes
python run_24_7.py --asset-class multi
```

### With Duration (Useful for Testing)

```bash
# Run for 1 hour only
python run_24_7.py --duration 3600

# Run for 1 day
python run_24_7.py --duration 86400

# Run for 1 week
python run_24_7.py --duration 604800

# Run forever (default)
python run_24_7.py
```

---

## 📊 MONITORING 24/7 EXECUTION

### View Logs

```bash
# Windows: View latest log
type logs\nexus_24_7_*.log | tail -100

# Linux: Real-time log view
tail -f logs/nexus_24_7_*.log

# Windows: Count errors
findstr "ERROR" logs\nexus_24_7_*.log | find /c "ERROR"

# Linux: Count errors
grep "ERROR" logs/nexus_24_7_*.log | wc -l
```

### View Metrics

```bash
# Windows: View metrics JSON
type logs\metrics_24_7_*.json

# Linux: View with pretty printing
python -m json.tool logs/metrics_24_7_*.json

# Example metrics output:
# {
#   "start_time": "2026-04-17T10:30:45.123456",
#   "uptime_seconds": 3600,
#   "restarts": 2,
#   "errors": 0,
#   "execution_cycles": 12,
#   "last_status": "RUNNING",
#   "last_update": "2026-04-17T11:30:45.654321"
# }
```

### Check Process Status

```bash
# Windows: Is process running?
tasklist | find /i "python"

# Linux: Is process running?
ps aux | grep "run_24_7.py"

# See CPU/Memory usage
# Windows: Task Manager (Ctrl+Shift+Esc)
# Linux: top -p $(pgrep -f run_24_7.py)
```

---

## 🛠️ MAINTENANCE

### Daily Checks

```bash
# Check if still running
tasklist | find "python"  # Windows
ps aux | grep run_24_7.py  # Linux

# Review errors in log
grep "ERROR" logs/nexus_24_7_*.log | tail -10

# Check metrics
cat logs/metrics_24_7_*.json
```

### Weekly Maintenance

```bash
# Archive old logs (older than 7 days)
# Windows: ForFiles /S /D +7
# Linux: find logs -mtime +7 -exec gzip {} \;

# Restart service for fresh logs
schtasks /run /tn "NexusInstitutional24x7"  # Windows
systemctl restart nexus-institutional-24x7  # Linux

# Check disk space for logs
# Windows: dir logs
# Linux: du -sh logs/
```

---

## ⚠️ TROUBLESHOOTING 24/7 EXECUTION

### Problem: Service won't start

**Windows:**
```cmd
# Check error log
type logs\nexus_24_7_*.log

# Try running manually
python run_24_7.py

# Check Active Directory/Task Scheduler permissions
# - Ensure SYSTEM account has access to project folder
# - Check Event Viewer for errors
```

**Linux:**
```bash
# Check service status
systemctl -e STATUS nexus-institutional-24x7

# Check journalctl logs
journalctl -u nexus-institutional-24x7 -n 50

# Ensure Python virtual environment is activated in service
# Edit service file and verify ExecStart path
```

### Problem: Service keeps restarting

- Check log file for crashes: `grep "Process terminated" logs/*`
- Review error output: `tail -50 logs/nexus_24_7_*.log`
- Increase restart delay in code (currently 30 seconds)
- Run manual test: `python run_24_7.py --mode backtest`

### Problem: Disk filling up

- Logs stored in: `logs/nexus_24_7_*.log`
- Rotate logs manually: `gzip logs/nexus_24_7_*.log`
- Increase log retention or add cleanup script

### Problem: Resource usage

```bash
# Monitor CPU/Memory
# Windows: tasklist /v | find "python"
# Linux: top -p $(pgrep -f run_24_7.py)

# If high usage:
# 1. Reduce venues: --venues 50 instead of 235
# 2. Use paper mode instead of backtest
# 3. Add healthcheck delays in code
```

---

## 📈 EXPECTED BEHAVIOR

### First Run (Manual)
```
Starting 24/7 execution monitor...
Log file: logs/nexus_24_7_20260417_103045.log
Metrics file: logs/metrics_24_7_20260417_103045.json

STARTING NEXUS INSTITUTIONAL PLATFORM - CYCLE 1
Mode: backtest | Asset Classes: multi | Venues: 235
Process started with PID: 12345
[PLATFORM] Initializing Nexus Institutional Trading Platform...
...
```

### First Restart (After Crash)
```
Restarting platform... (restart #1)
Process terminated with code: 0
STARTING NEXUS INSTITUTIONAL PLATFORM - CYCLE 2
Process started with PID: 12346
...
```

### After 1 Hour
```
Uptime: 1:00:00, Cycles: 12, Restarts: 0
Total Runtime: 3600 seconds
```

---

## ✅ SUCCESS CHECKLIST

- [ ] Manual test works: `python run_24_7.py` runs without errors
- [ ] Logs created: `logs/nexus_24_7_*.log` contains output
- [ ] Metrics tracked: `logs/metrics_24_7_*.json` shows statistics
- [ ] Auto-restart works: Ctrl+C then check service restarted (Windows/Linux only)
- [ ] Task scheduled: Windows Task Scheduler or systemd shows service
- [ ] Auto-start enabled: Service starts on system reboot
- [ ] Monitoring active: Can view logs and metrics in real-time

---

## 🎓 NEXT STEPS

### Option A: Keep Manual Control
- Run `python run_24_7.py` whenever needed
- Monitor logs manually
- For testing/development

### Option B: Full Automation
- Run `setup_windows_24_7.bat` or `setup_linux_24_7.sh`
- Automatic 24/7 operation
- Recovery from crashes
- For production deployment

### Option C: Hybrid
- Use manual runs for testing
- Use automated setup for specific configurations
- Mix modes as needed

---

## 📞 SUPPORT

**For issues:**
1. Check `logs/nexus_24_7_*.log` for error messages
2. View `logs/metrics_24_7_*.json` for health stats
3. Manual test: `python run_24_7.py --mode backtest --venues 50`
4. Review INSTITUTIONAL_DEPLOYMENT_GUIDE.md

---

**Nexus Institutional v0.3.0 - 24/7 Continuous Trading Platform**
*Enterprise-Grade Continuous Execution with Auto-Recovery*

