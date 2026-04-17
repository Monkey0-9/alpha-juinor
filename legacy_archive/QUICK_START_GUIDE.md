# Quick Start Guide - MiniQuantFund v4.0.0

## Your Alpaca API Keys (Already Configured)

```
API Key: PKQIB6ZXGBD25RP2WTB5MZNZU5
Secret Key: 91adzg7PnervhHds2v8JqFZXBeLfNncyd5F1eazYf7XP
Base URL: https://paper-api.alpaca.markets (Paper Trading)
```

## Complete Setup & Running Instructions

### Step 1: Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install production dependencies
pip install -r requirements-production.txt

# Install additional packages for paper trading
pip install alpaca-trade-api pandas numpy asyncio aiohttp prometheus-client grafana-api
```

### Step 2: Database Setup

```bash
# Install PostgreSQL (if not installed)
# Windows: Download from https://www.postgresql.org/download/windows/
# Or use Docker: docker run --name postgres -e POSTGRES_PASSWORD=password -p 5432:5432 -d postgres

# Create database
createdb miniquantfund_paper

# Set database password in .env
# DATABASE_URL=postgresql://postgres:password@localhost:5432/miniquantfund_paper
```

### Step 3: Environment Setup

```bash
# Your .env file is already configured with Alpaca keys
# Just verify these variables are set:

export ALPACA_API_KEY=PKQIB6ZXGBD25RP2WTB5MZNZU5
export ALPACA_SECRET_KEY=91adzg7PnervhHds2v8JqFZXBeLfNncyd5F1eazYf7XP
export DB_PASSWORD=password
export REDIS_URL=redis://localhost:6379
```

### Step 4: Start Paper Trading System

```bash
# Method 1: Run paper trading system (Recommended for testing)
python run_paper_trading_system.py

# Method 2: Run original system
python run_complete_system.py

# Method 3: Run with specific configuration
python run_paper_trading_system.py --config config/paper_trading.json
```

### Step 5: Monitor System

```bash
# Check system status
curl http://localhost:8080/health

# View logs
tail -f logs/paper_trading.log

# Check metrics
curl http://localhost:9091/metrics
```

## What Happens When You Run

### 1. System Initialization
- Security infrastructure starts
- Market data feeds connect (Alpaca, Polygon, Yahoo)
- Broker integration connects (Alpaca paper trading)
- Risk management system activates
- Compliance system starts
- Monitoring system begins tracking

### 2. Paper Trading Begins
- System receives real market data
- Trading algorithms analyze market conditions
- Orders are placed in paper trading environment
- Risk limits enforced in real-time
- All trades tracked for compliance

### 3. Real-Time Monitoring
- Portfolio value tracked
- P&L calculated continuously
- Risk metrics monitored
- Alerts triggered for any issues
- Performance metrics recorded

## Expected Output

When you run `python run_paper_trading_system.py`, you should see:

```
MiniQuantFund v4.0.0 - Paper Trading System
============================================================
Starting paper trading system for testing...

2026-04-17 15:30:00 - MiniQuantFund-Paper - INFO - Initializing paper trading system
2026-04-17 15:30:01 - MiniQuantFund-Paper - INFO - Initializing security infrastructure
2026-04-17 15:30:02 - MiniQuantFund-Paper - INFO - Initializing monitoring system
2026-04-17 15:30:03 - MiniQuantFund-Paper - INFO - Initializing market data feeds
2026-04-17 15:30:04 - MiniQuantFund-Paper - INFO - Connected to Alpaca data feed
2026-04-17 15:30:05 - MiniQuantFund-Paper - INFO - Initializing paper trading broker integration
2026-04-17 15:30:06 - MiniQuantFund-Paper - INFO - Connected to Alpaca paper trading
2026-04-17 15:30:07 - MiniQuantFund-Paper - INFO - Paper trading system started
2026-04-17 15:30:08 - MiniQuantFund-Paper - INFO - Paper Account - Equity: $1,000,000.00, Buying Power: $2,000,000.00
```

## Troubleshooting

### Common Issues

1. **Database Connection Error**
   ```bash
   # Solution: Install PostgreSQL and create database
   createdb miniquantfund_paper
   ```

2. **Alpaca API Connection Error**
   ```bash
   # Solution: Check your API keys are valid
   # Visit https://paper-api.alpaca.markets/ to verify
   ```

3. **Missing Dependencies**
   ```bash
   # Solution: Install missing packages
   pip install alpaca-trade-api aiohttp prometheus-client
   ```

4. **Port Already in Use**
   ```bash
   # Solution: Kill existing process or change port
   netstat -ano | findstr :8080
   taskkill /PID <PID> /F
   ```

### Getting Help

1. **Check Logs**: `logs/paper_trading.log`
2. **System Status**: `curl http://localhost:8080/health`
3. **Documentation**: `docs/PAPER_TRADING_TESTING_GUIDE.md`

## Next Steps

1. **Run paper trading for 1-2 weeks** to validate system
2. **Monitor performance metrics** in logs
3. **Check risk management** is working correctly
4. **Review trading results** and optimize parameters
5. **Prepare for live trading** using transition guide

## Quick Commands Reference

```bash
# Start paper trading
python run_paper_trading_system.py

# Check system health
curl http://localhost:8080/health

# View real-time logs
tail -f logs/paper_trading.log

# Stop system (Ctrl+C or close terminal)

# Check trading results
python -c "
import asyncio
from run_paper_trading_system import PaperTradingSystem
system = PaperTradingSystem()
status = asyncio.run(system.get_system_status())
print(status)
"
```

Your system is ready to run! Start with paper trading to test everything thoroughly.
