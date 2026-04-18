# 🚀 NEXUS 24/7 - QUICK START CARD

## 3 WAYS TO RUN NEXUS 24/7

### 🔵 **OPTION A: Console Mode** (Test First)
```bash
cd C:\mini-quant-fund
python run_24_7.py --mode backtest --asset-class multi --venues 235
```
- Runs in current terminal
- Stops when you close terminal
- Good for testing
- Press `Ctrl+C` to stop

---

### 🟢 **OPTION B: Windows Auto-Start** (Permanent)
```cmd
# Open Command Prompt as ADMIN
cd C:\mini-quant-fund
setup_windows_24_7.bat
```
- Automatically starts on system reboot
- Runs 24/7 in background
- Restarts if it crashes
- Logs to: `logs/nexus_24_7_*.log`

**Verify Installation:**
```cmd
tasklist | find "python"
schtasks /query /tn "NexusInstitutional24x7" /v
```

---

### 🟡 **OPTION C: Linux Auto-Start** (Permanent)
```bash
# Open terminal
cd /home/user/mini-quant-fund
chmod +x setup_linux_24_7.sh
sudo bash setup_linux_24_7.sh
```
- Automatically starts on system reboot
- Runs 24/7 in background
- Restarts if it crashes
- Logs to: `logs/nexus_24_7_*.log`

**Verify Installation:**
```bash
systemctl status nexus-institutional-24x7
journalctl -u nexus-institutional-24x7 -f
```

---

## 📊 COMPARISON TABLE

| Feature | Console | Windows | Linux |
|---------|---------|---------|-------|
| Easy to start | ✅ Yes | ✅ Yes | ✅ Yes |
| Auto-restart | ❌ No | ✅ Yes | ✅ Yes |
| Auto-start on boot | ❌ No | ✅ Yes | ✅ Yes |
| Runs in background | ❌ No | ✅ Yes | ✅ Yes |
| Good for testing | ✅ Yes | ❌ No | ❌ No |
| Good for production | ❌ No | ✅ Yes | ✅ Yes |

---

## 🎯 WHICH OPTION SHOULD YOU CHOOSE?

**First Time Users:** Use **OPTION A** (Console)
- Test that everything works
- Watch logs in real-time
- Learn how it runs
- Then move to OPTION B or C

**Windows Users:** Use **OPTION B** (Window Scheduler)
- Set it up once
- Never touch it again
- Restarts automatically
- Perfect for production

**Linux/Mac Users:** Use **OPTION C** (systemd)
- Set it up once
- Never touch it again
- Restarts automatically
- Perfect for production

---

## 📋 EXECUTION OPTIONS

### Customize Your 24/7 Run

```bash
# Choose MODE:
--mode backtest          # Historical simulation (DEFAULT)
--mode paper             # Live data, no orders
--mode market-making     # Market making
--mode live              # Real trading (requires API keys)

# Choose ASSET CLASSES:
--asset-class equities       # Stocks only
--asset-class fixed-income   # Bonds and rates
--asset-class crypto         # Cryptocurrencies
--asset-class derivatives    # Options and futures
--asset-class fx             # Foreign exchange
--asset-class multi          # All asset classes (DEFAULT)

# Choose VENUES:
--venues 10              # 10 major venues (fastest)
--venues 50              # 50 venues (regional)
--venues 100             # 100 venues (global)
--venues 235             # 235+ venues (complete - DEFAULT)

# Set DURATION:
--duration 3600          # Run for 1 hour
--duration 86400         # Run for 1 day
--duration 604800        # Run for 1 week
# (omit for infinite)
```

### Example Configurations

```bash
# Equities only, 50 venues, backtest
python run_24_7.py --mode backtest --asset-class equities --venues 50

# Crypto market making, 100 venues
python run_24_7.py --mode market-making --asset-class crypto --venues 100

# Paper trading, all assets, all venues
python run_24_7.py --mode paper --asset-class multi --venues 235

# Test for 1 hour only
python run_24_7.py --duration 3600

# Production configuration (Windows Task Scheduler)
# Edit: setup_windows_24_7.bat line with run_24_7.py
# Then: setup_windows_24_7.bat
```

---

## 📊 MONITORING 24/7

### View Real-Time Logs

```bash
# Windows
type logs\nexus_24_7_*.log | tail -50

# Linux
tail -f logs/nexus_24_7_*.log
```

### View Metrics

```bash
# Windows
type logs\metrics_24_7_*.json

# Linux
cat logs/metrics_24_7_*.json | python -m json.tool
```

### Check if Running

```bash
# Windows
tasklist | find "python"

# Linux
ps aux | grep run_24_7.py
```

---

## 🛠️ COMMON COMMANDS

### Windows Commands

```cmd
# Verify task exists
schtasks /query /tn "NexusInstitutional24x7"

# Run task manually
schtasks /run /tn "NexusInstitutional24x7"

# Stop task
schtasks /end /tn "NexusInstitutional24x7"

# Delete task
schtasks /delete /tn "NexusInstitutional24x7" /f

# View logs
type logs\nexus_24_7_*.log

# Count lines in log
find /c ^ logs\nexus_24_7_*.log
```

### Linux Commands

```bash
# View service status
systemctl status nexus-institutional-24x7

# View live logs
journalctl -u nexus-institutional-24x7 -f

# View recent logs
journalctl -u nexus-institutional-24x7 -n 100

# Restart service
systemctl restart nexus-institutional-24x7

# Stop service
systemctl stop nexus-institutional-24x7

# View service file
cat /etc/systemd/system/nexus-institutional-24x7.service
```

---

## ✅ SUCCESS INDICATORS

You'll know it's working when:
- ✅ Console shows "STARTING NEXUS INSTITUTIONAL PLATFORM"
- ✅ Log file created in `logs/` directory
- ✅ Metrics file shows increasing uptime
- ✅ No ERROR messages in console
- ✅ Process stays alive (doesn't crash)

---

## ⚠️ TROUBLESHOOTING

| Problem | Solution |
|---------|----------|
| **Command not found** | Make sure you're in `C:\mini-quant-fund` directory |
| **Permission denied (Linux)** | Run `chmod +x setup_linux_24_7.sh` first |
| **Task won't start (Windows)** | Run Command Prompt AS ADMINISTRATOR first |
| **Process keeps crashing** | Check log file `tail -f logs/nexus_24_7_*.log` |
| **Port in use** | Change port in config if using live mode |
| **Not writing logs** | Ensure `logs/` directory exists and is writable |

---

## 📈 WHAT HAPPENS NOW

1. ✅ Platform monitors itself continuously
2. ✅ If it crashes, it restarts automatically
3. ✅ All activity logged to file
4. ✅ Metrics tracked every 60 seconds
5. ✅ You can close the console/terminal
6. ✅ Platform continues running 24/7

---

## 🎓 NEXT STEPS

1. **Choose your option** (A, B, or C above)
2. **Run the command** for your option
3. **Verify it works** using the verification commands
4. **View logs** to confirm success
5. **Let it run!** The platform is now on 24/7

---

## 📚 FULL DOCUMENTATION

For detailed information, see:
- `RUN_24_7_GUIDE.md` - Complete 24/7 setup guide
- `INSTITUTIONAL_DEPLOYMENT_GUIDE.md` - Production deployment guide
- `INSTITUTIONAL_COMPLETION_REPORT.md` - Full technical documentation

---

**Nexus Institutional v0.3.0 - Now Running 24/7** 🚀

