# 🎯 NEXUS LIVE TRADING - COMMAND REFERENCE

## Quick Command Reference (Copy & Paste)

### 🚀 BASIC COMMANDS

#### Windows
```batch
REM Setup (first time only)
setup_live_trading.bat

REM Start monitoring (default: 60-second updates, infinite)
start_live_monitor.bat
```

#### Mac/Linux
```bash
# Setup (first time only)
pip install feedparser requests textblob numpy pandas

# Start monitoring (default: 60-second updates, infinite)
python live_paper_trading.py --mode paper
```

---

### ⚙️ ADVANCED COMMANDS (All Platforms)

```bash
# Fast updates (every 30 seconds)
python live_paper_trading.py --interval 30

# Slow updates (every 120 seconds - less noisy)
python live_paper_trading.py --interval 120

# Run for exactly 1 hour
python live_paper_trading.py --duration 3600

# Run for 24 hours
python live_paper_trading.py --duration 86400

# Verbose logging (see everything)
python live_paper_trading.py --log-level DEBUG

# Quiet logging (errors only)
python live_paper_trading.py --log-level ERROR

# Combine options
python live_paper_trading.py --mode paper --interval 30 --duration 3600 --log-level INFO

# Choose execution mode (paper/backtest/live)
python live_paper_trading.py --mode paper        # Paper trading (recommended)
python live_paper_trading.py --mode backtest     # Historical data only
python live_paper_trading.py --mode live         # Real money (dangerous!)
```

---

### 📊 DASHBOARD & MONITORING

```bash
# Generate HTML dashboard
python src/nexus/institutional/live_dashboard.py

# Serve dashboard (requires Python http.server)
python -m http.server 8000
# Then visit: http://localhost:8000/live_dashboard.html

# Tail logs (Mac/Linux)
tail -f logs/*.log

# Watch logs (Windows PowerShell)
Get-Content logs/*.log -Tail 20 -Wait
```

---

### 🧪 TESTING & VALIDATION

```bash
# Run institutional system tests
python verify_institutional_system.py

# Test sentiment analysis
python -c "from textblob import TextBlob; print(TextBlob('Great earnings!').sentiment)"

# Test news fetching
python -c "import feedparser; print(len(feedparser.parse('https://feeds.bloomberg.com/markets/news.rss').entries))"

# Check Python environment
python sys.version
pip list
```

---

### 📝 HELPFUL ONE-LINERS

```bash
# See what Python packages are installed
pip list

# Check if feedparser is installed
pip show feedparser

# Check if TextBlob is installed
pip show textblob

# Get current directory
pwd  (Mac/Linux) or cd (Windows)

# List files in current directory
ls   (Mac/Linux) or dir (Windows)

# Find all Python files
find . -name "*.py"  (Mac/Linux) or where /r . *.py (Windows)

# Count lines in a file
wc -l live_monitor.py  (Mac/Linux) or findstr /c:"" live_monitor.py | find /v "" (Windows)

# Search for text in file
grep "NewsMonitor" live_monitor.py  (Mac/Linux) or findstr "NewsMonitor" live_monitor.py (Windows)
```

---

### 🎯 COMMON WORKFLOWS

#### Workflow 1: First Time Setup
```bash
# Step 1: Install dependencies
setup_live_trading.bat  (Windows) or pip install feedparser requests textblob

# Step 2: Start monitoring
python live_paper_trading.py --mode paper

# Step 3: Observe for 15+ minutes
# (Watch log output)

# Step 4: Stop with Ctrl+C
```

#### Workflow 2: Run Overnight Test
```bash
# Run for 8 hours (28,800 seconds)
python live_paper_trading.py --mode paper --duration 28800 --log-level INFO
```

#### Workflow 3: Collect Signals for Analysis
```bash
# Run for 1 week (604,800 seconds)
python live_paper_trading.py --mode paper --duration 604800 --log-level INFO

# Save output
python live_paper_trading.py --mode paper --duration 604800 > trading_signals.txt 2>&1
```

#### Workflow 4: High Frequency Testing
```bash
# Fast updates for testing
python live_paper_trading.py --interval 10 --duration 300 --log-level DEBUG
```

#### Workflow 5: Production Monitoring
```bash
# Stable, long-running configuration
python live_paper_trading.py --mode paper --interval 60 --log-level WARNING
```

---

## 📖 FILE LOCATIONS & PURPOSES

### Core Files
```
live_paper_trading.py          # Entry point - RUN THIS
src/nexus/institutional/
  ├─ live_monitor.py          # Core monitoring logic
  └─ live_dashboard.py        # Dashboard (optional)
```

### Configuration
```
config/production.yaml         # Trading parameters
config/development.yaml        # Dev parameters
```

### Logs & Output
```
logs/                          # System logs (auto-created)
live_dashboard.json            # Real-time dashboard data
live_dashboard.html            # HTML dashboard (optional)
```

### Documentation
```
LIVE_TRADING_SYSTEM_SUMMARY.md         # Overview (start here)
LIVE_MONITOR_QUICK_START.md            # Quick start guide
LIVE_PAPER_TRADING_GUIDE.md            # Comprehensive guide
LIVE_MONITORING_ARCHITECTURE.md        # Technical depth
```

---

## 🔍 PARAMETER REFERENCE

### --mode {paper|backtest|live}
```
paper      = Real prices, fake money (RECOMMENDED for learning)
backtest   = Historical data, fake money (testing only)
live       = Real prices, REAL MONEY (use ONLY after validation!)
```

### --interval N
```
Default: 60 seconds
Range: 10-600 seconds recommended
Smaller: More responsive, more data, more compute
Larger: Less noisy, less compute, slower reaction
```

### --duration N
```
Default: None (infinite, runs until Ctrl+C)
N = Seconds to run
3600 = 1 hour
86400 = 1 day
604800 = 1 week
```

### --log-level {DEBUG|INFO|WARNING|ERROR}
```
DEBUG   = See everything (useful for debugging)
INFO    = Normal operations (recommended)
WARNING = Important events only
ERROR   = Errors only (quiet mode)
```

---

## 🎓 TROUBLESHOOTING COMMANDS

```bash
# Check Python version (need 3.9+)
python --version

# Check if pip works
pip --version

# Install missing packages
pip install feedparser requests textblob

# Verify installations
pip show feedparser
pip show requests
pip show textblob

# Test feedparser
python -c "import feedparser; print('OK' if feedparser else 'FAILED')"

# Test requests
python -c "import requests; print('OK' if requests else 'FAILED')"

# Test TextBlob
python -c "from textblob import TextBlob; print(TextBlob('test').sentiment)"

# Check current directory
pwd  (or 'cd' on Windows to see)

# List Python files
ls *.py  (Mac/Linux) or dir *.py (Windows)

# See what's running (processes)
ps aux | grep python  (Mac/Linux) or tasklist | findstr python (Windows)

# Kill running process
pkill -f live_paper_trading  (Mac/Linux) or taskkill /f /im python.exe (Windows - caution!)
```

---

## 💡 PERFORMANCE TIPS

```bash
# If slow/high CPU: Increase interval
python live_paper_trading.py --interval 120  # 2-minute updates

# If too quiet: Decrease interval
python live_paper_trading.py --interval 30   # 30-second updates

# If verbose logs: Change level
python live_paper_trading.py --log-level WARNING  # Less output

# If want all details: Lower level
python live_paper_trading.py --log-level DEBUG    # Full debug output

# For analysis/data collection: Capture output
python live_paper_trading.py > output.txt 2>&1

# For quiet background runs
python live_paper_trading.py --log-level ERROR &  (Mac/Linux)
python live_paper_trading.py --log-level ERROR    (Windows - in separate command prompt)
```

---

## 📊 EXPECTED OUTPUT LINES

### If working correctly, you'll see:
```
✓ "Found 15 articles"
✓ "Retrieved data for 20 symbols"
✓ "Generated 8 trading alerts"
✓ "Bullish: NVDA, MSFT, GOOGL"
✓ "Bearish: SPY, QQQ, IWM"
✓ "Portfolio Value: $1,000,000.00"
```

### If something's wrong, you'll see:
```
✗ "No articles found" → Check internet
✗ "ModuleNotFoundError" → Run pip install
✗ "All sentiment scores are 0.0" → TextBlob issue
✗ "Error connecting to API" → API issue
✗ Crashes after <100 cycles → Memory leak
```

---

## 🎯 COMMAND COMBINATIONS FOR SPECIFIC USE CASES

### Case 1: First Time User
```bash
# Setup and test
setup_live_trading.bat
python live_paper_trading.py --mode paper --log-level INFO --duration 900
```

### Case 2: Overnight Testing
```bash
# Run 8 hours with moderate verbosity
python live_paper_trading.py --mode paper --duration 28800 --log-level INFO --interval 60
```

### Case 3: Debug Mode
```bash
# See everything for troubleshooting
python live_paper_trading.py --mode paper --log-level DEBUG --duration 300 --interval 30
```

### Case 4: Production Monitoring
```bash
# Stable, minimal logging, continuous
python live_paper_trading.py --mode paper --log-level WARNING --interval 60
```

### Case 5: Analysis & Data Collection
```bash
# Collect signals for a week with full logging
python live_paper_trading.py --mode paper --duration 604800 --log-level INFO > signals_week1.txt 2>&1
```

### Case 6: Quick Test
```bash
# 5-minute test run with 10-second updates
python live_paper_trading.py --mode paper --duration 300 --interval 10 --log-level INFO
```

---

## 🚨 IMPORTANT WARNINGS

```
⚠️  NEVER run --mode live without proper testing!
⚠️  Paper trading first (weeks 1-4)
⚠️  Validate strategy thoroughly before real money
⚠️  Start with $1K minimum if moving to real trading
⚠️  Understand the risks completely
⚠️  Have enough capital for margin requirements
```

---

## 📞 QUICK HELP

Find the right command above for your use case:

| Goal | Command |
|------|---------|
| First time setup | `setup_live_trading.bat` |
| Quick test | `python live_paper_trading.py` |
| Run 1 hour | `python live_paper_trading.py --duration 3600` |
| Fast updates | `python live_paper_trading.py --interval 30` |
| Verbose logging | `python live_paper_trading.py --log-level DEBUG` |
| Collect data 1 week | `python live_paper_trading.py --duration 604800 > signals.txt` |
| See instruction help | `python live_paper_trading.py --help` |

---

**Save this file. Bookmark this directory. Come back here when you need a command.** 📌

