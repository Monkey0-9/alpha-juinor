# ✅ ALPACA TRADING SETUP & EXECUTION GUIDE

## Overview

Your NEXUS trading system now has **full Alpaca integration** for real money trading.

**Two modes available:**
- **Paper Trading** (free, simulated): Test strategy with fake money
- **Live Trading** (real money): Execute real trades

---

## 🚀 QUICK START - 5 MINUTES

### Step 1: Install Alpaca SDK
```bash
pip install alpaca-trade-api
```

### Step 2: Setup Alpaca Credentials
```bash
python setup_alpaca.py
```

This will:
- Prompt you for Alpaca API Key & Secret Key
- Validate credentials
- Save to `.env.alpaca`
- Create `setup_alpaca_env.bat` for Windows

### Step 3: Start Trading
```bash
# Paper trading (free simulation)
python complete_trading_system.py --mode paper --broker alpaca

# Live trading (real money - use small amounts first)
python complete_trading_system.py --mode live --broker alpaca --capital 1000
```

---

## 📋 DETAILED SETUP

### Get Alpaca Account (Free)

1. Go to https://alpaca.markets/
2. Click "Sign Up"
3. Create account with email & password
4. Verify email
5. No credit card required for paper trading

### Get API Credentials

1. Login to Alpaca
2. Click "Dashboard" → "Integrations" → "API Keys"
3. Copy:
   - **API Key ID** (looks like: `PKXXXXXXXXXXXXXX`)
   - **Secret Key** (looks like: `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX`)
4. Keep these secret!

### Paper vs Live URLs

**Paper Trading** (free, simulated):
```
Base URL: https://paper-api.alpaca.markets
Used for: Testing strategies safely
```

**Live Trading** (real money):
```
Base URL: https://api.alpaca.markets
Used for: Making actual trades with real money
```

---

## 🔧 CONFIGURATION

### Option 1: Interactive Setup (Recommended)
```bash
python setup_alpaca.py
```

Walks you through:
- Paper vs Live choice
- API key entry
- Validation
- Auto-saves to files

### Option 2: Manual Environment Variables

**Linux/Mac:**
```bash
export APCA_API_KEY_ID="your_key_id_here"
export APCA_API_SECRET_KEY="your_secret_key_here"
export APCA_API_BASE_URL="https://paper-api.alpaca.markets"
python complete_trading_system.py --mode paper --broker alpaca
```

**Windows (PowerShell):**
```powershell
$env:APCA_API_KEY_ID = "your_key_id_here"
$env:APCA_API_SECRET_KEY = "your_secret_key_here"
$env:APCA_API_BASE_URL = "https://paper-api.alpaca.markets"
python complete_trading_system.py --mode paper --broker alpaca
```

**Windows (Command Prompt):**
```cmd
setx APCA_API_KEY_ID "your_key_id_here"
setx APCA_API_SECRET_KEY "your_secret_key_here"
setx APCA_API_BASE_URL "https://paper-api.alpaca.markets"
python complete_trading_system.py --mode paper --broker alpaca
```

### Option 3: .env File
Create `.env.alpaca` in project root:
```
APCA_API_KEY_ID=your_key_id_here
APCA_API_SECRET_KEY=your_secret_key_here
APCA_API_BASE_URL=https://paper-api.alpaca.markets
```

Then load it before running:
```bash
# Use the auto-generated batch file (Windows)
setup_alpaca_env.bat
```

---

## 📊 USAGE EXAMPLES

### Paper Trading (Safe Testing)
```bash
# Run for 1 hour against Alpaca paper API
python complete_trading_system.py \
    --mode paper \
    --broker alpaca \
    --duration 3600
```

What happens:
- ✓ Connects to Alpaca paper trading servers
- ✓ Gets real market prices from Alpaca
- ✓ Simulates order execution
- ✓ Tracks portfolio on your account
- ✓ No real money risked

### Live Trading ($1K - Start Small!)
```bash
# Start with $1,000 real money
python complete_trading_system.py \
    --mode live \
    --broker alpaca \
    --capital 1000
```

**⚠️ IMPORTANT - Risk Management:**
- Start with $1,000 (not your life savings!)
- Let it run 1-7 days first
- Monitor real P&L carefully
- Exit if losing more than 5%
- Scale up only after validation

### Extended Validation (1 week)
```bash
# Run for 7 days (604,800 seconds)
python complete_trading_system.py \
    --mode live \
    --broker alpaca \
    --capital 5000 \
    --duration 604800
```

---

## 🎯 WHAT HAPPENS WHEN TRADING

### When Using Paper Broker
```
News/Opportunities Found
        ↓
Order Created (BUY/SELL)
        ↓
Sent to Paper Account
        ↓
Simulated execution (our account)
        ↓
Portfolio updates locally
        ↓
P&L calculated
```

### When Using Alpaca Broker
```
News/Opportunities Found
        ↓
Order Created (BUY/SELL)
        ↓
Sent to Alpaca API
        ↓
Alpaca executes on real market
        ↓
Confirmation received
        ↓
Portfolio updates from Alpaca
        ↓
Real P&L tracked
```

---

## 📈 REAL TRADING OUTPUT

When paper trading with real Alpaca connection:

```
2026-04-18 14:30:45 - NexusTrading - INFO - ✓ Alpaca connected
Account: PA123456789 | Buying Power: $25,000.00

================================================================================
TRADING CYCLE #1
================================================================================
Found 3 trading opportunities
  ✓ Alpaca order submitted | BUY 50 AAPL | Status: pending
    (actual order on Alpaca servers)
  ✓ Alpaca order submitted | SELL 30 MSFT | Status: pending
    (actual order on Alpaca servers)
  ✓ Alpaca order submitted | BUY 100 SPY | Status: pending
    (actual order on Alpaca servers)

Portfolio Update:
  Cash: $24,750.00 (from Alpaca)
  Positions: 50 AAPL, -30 MSFT, 100 SPY
  Total Value: $25,000.00
```

---

## ✅ VALIDATION CHECKLIST

- [ ] Alpaca account created
- [ ] API credentials obtained
- [ ] Credentials validated via setup_alpaca.py
- [ ] Paper trading tested (1 hour)
- [ ] Orders executing successfully
- [ ] Portfolio updating from Alpaca
- [ ] Prices matching Alpaca levels
- [ ] Ready to move to live ($1K)

---

## ⚠️ IMPORTANT NOTES

### Paper Trading Safety
- No real money risked
- Executes against simulated market
- Good for testing system
- Shows realistic slippage/fills

### Live Trading Risks
- Real money can be lost
- Start with $1K maximum
- Strategy may not work live (70% failure rate common)
- Monitor actively first week
- Can lose entire capital if strategy flawed

### Common Issues

**"Alpaca not connected"**
- Missing API credentials
- Run: `python setup_alpaca.py`
- Check environment variables

**"Invalid credentials"**
- Copy/paste error in API key
- Check Alpaca dashboard
- Run setup again

**"Order rejected"**
- Insufficient buying power
- Symbol not supported
- Market closed (trading 9:30-16:00 ET)
- Check Alpaca documentation

**"Can't connect to Alpaca"**
- Network/firewall issue
- Alpaca API down
- Check https://status.alpaca.markets/
- Try with paper URL first

---

## 🎓 TIER PROGRESSION

**Your Current Status:** Tier 1 (Paper Trading)
- ✅ Beautiful architecture
- ✅ Live news monitoring
- ✅ Order execution engine
- ✅ Portfolio management
- ✅ Now connected to Alpaca

**Next: Tier 2 (Small Real Money)**
```
After paper trading validation:
1. Open Alpaca live account
2. Fund with $1,000
3. Run: python complete_trading_system.py --mode live --broker alpaca --capital 1000
4. Monitor for 1-7 days
5. Validate strategy performs as expected
6. Either scale or refactor based on results
```

**Decision Point:**
- If win rate > 55%: Scale to $5K-$10K
- If win rate < 45%: Analyze losses, adjust strategy
- If breakeven: Evaluate vs. market benchmarks

---

## 🔐 CREDENTIAL SECURITY

**NEVER:**
- Share API keys with anyone
- Put API keys in git/github
- Store in plain text in shared files
- Post in messages/chat

**DO:**
- Use environment variables
- Store in `.env.alpaca` (gitignored)
- Rotate keys periodically
- Use different keys for paper vs. live
- Limit permissions to trading only

---

## 📞 SUPPORT & RESOURCES

- **Alpaca Docs:** https://alpaca.markets/docs/
- **API Reference:** https://alpaca.markets/docs/api-references/
- **Status Page:** https://status.alpaca.markets/
- **Support:** support@alpaca.markets

---

## 🚀 YOU'RE NOW READY!

```bash
# Test paper trading first
python setup_alpaca.py                 # Setup credentials
python complete_trading_system.py --mode paper --broker alpaca --duration 3600

# After 1 week validation, try live
python complete_trading_system.py --mode live --broker alpaca --capital 1000

# Monitor and scale based on results
# 💰 Good luck trading!
```

Start small, validate thoroughly, scale if it works!
