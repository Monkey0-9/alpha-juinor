# 🦙 ALPACA PAPER TRADING SETUP GUIDE

## **Enable Real Trading with Alpaca Paper Trading**

---

## **📋 OVERVIEW**

Your HugeFunds platform now supports **real paper trading** with Alpaca Markets. This means:
- ✅ **Real order execution** (simulated with fake money)
- ✅ **Real market data** from Alpaca
- ✅ **Real position tracking**
- ✅ **Real portfolio management**
- ✅ **Real trading history**

**Paper trading lets you test strategies with $100,000 fake money before going live!**

---

## **🚀 QUICK START (5 Minutes)**

### **Step 1: Create Alpaca Account**

1. Go to: https://alpaca.markets
2. Click "Sign Up" (it's free)
3. Choose "Paper Trading" account
4. Complete registration

### **Step 2: Get API Keys**

1. Log in to your Alpaca dashboard
2. Go to: Paper Trading → API Keys
3. Click "Generate API Key"
4. Copy:
   - **API Key ID** (starts with PK...)
   - **API Secret Key**

**⚠️ KEEP THESE SECRET! Never share your API keys.**

### **Step 3: Configure HugeFunds**

#### **Option A: Environment Variables (Recommended)**

**Windows (PowerShell):**
```powershell
$env:ALPACA_API_KEY="your_api_key_here"
$env:ALPACA_API_SECRET="your_secret_key_here"
$env:ALPACA_PAPER_TRADING="true"
```

**Windows (CMD):**
```cmd
set ALPACA_API_KEY=your_api_key_here
set ALPACA_API_SECRET=your_secret_key_here
set ALPACA_PAPER_TRADING=true
```

**Make permanent (Windows):**
```powershell
[Environment]::SetEnvironmentVariable("ALPACA_API_KEY", "your_api_key_here", "User")
[Environment]::SetEnvironmentVariable("ALPACA_API_SECRET", "your_secret_key_here", "User")
[Environment]::SetEnvironmentVariable("ALPACA_PAPER_TRADING", "true", "User")
```

#### **Option B: .env File**

Create file `c:\mini-quant-fund\hugefunds\.env`:
```env
ALPACA_API_KEY=your_api_key_here
ALPACA_API_SECRET=your_secret_key_here
ALPACA_PAPER_TRADING=true
```

### **Step 4: Start Trading**

```bash
cd c:\mini-quant-fund\hugefunds
python start.py
```

**Expected output:**
```
[*] Initializing Alpaca Paper Trading...
[OK] Alpaca Paper Trading: CONNECTED
```

---

## **🧪 TEST ALPACA CONNECTION**

### **Test 1: Check Account Status**

```bash
curl http://localhost:8000/api/alpaca/account/status
```

**Expected response:**
```json
{
  "connected": true,
  "paper_trading": true,
  "status": "connected",
  "message": "Alpaca paper trading account active"
}
```

### **Test 2: Get Account Info**

```bash
curl http://localhost:8000/api/alpaca/account
```

**Expected response:**
```json
{
  "enabled": true,
  "status": "connected",
  "buying_power": 100000.00,
  "cash": 100000.00,
  "portfolio_value": 100000.00,
  "equity": 100000.00
}
```

### **Test 3: Check Market Clock**

```bash
curl http://localhost:8000/api/alpaca/clock
```

**Expected response:**
```json
{
  "enabled": true,
  "is_open": true,
  "message": "Market is OPEN"
}
```

---

## **💰 PLACE YOUR FIRST TRADE**

### **Buy Stock (Simple)**

```bash
curl -X POST "http://localhost:8000/api/alpaca/buy?symbol=AAPL&qty=10"
```

**Expected response:**
```json
{
  "action": "BUY",
  "symbol": "AAPL",
  "qty": 10,
  "result": {
    "success": true,
    "order_id": "...",
    "status": "filled",
    "message": "Order filled"
  }
}
```

### **Sell Stock (Simple)**

```bash
curl -X POST "http://localhost:8000/api/alpaca/sell?symbol=AAPL&qty=10"
```

### **Advanced Order (Limit Order)**

```bash
curl -X POST http://localhost:8000/api/alpaca/orders \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "TSLA",
    "qty": 5,
    "side": "buy",
    "order_type": "limit",
    "limit_price": 200.00,
    "time_in_force": "day"
  }'
```

---

## **📊 VIEW PORTFOLIO**

### **Get All Positions**

```bash
curl http://localhost:8000/api/alpaca/positions
```

**Shows:**
- Symbol, quantity, market value
- Unrealized P&L
- Current price
- Average entry price

### **Get Portfolio Summary**

```bash
curl http://localhost:8000/api/alpaca/portfolio/summary
```

**Shows:**
- Account value (buying power, cash, equity)
- Total positions value
- Total unrealized P&L
- Top gainers and losers

---

## **📜 VIEW ORDERS**

### **Get All Orders**

```bash
curl http://localhost:8000/api/alpaca/orders
```

### **Get Open Orders**

```bash
curl "http://localhost:8000/api/alpaca/orders?status=open"
```

### **Get Closed Orders**

```bash
curl "http://localhost:8000/api/alpaca/orders?status=closed"
```

---

## **🚨 EMERGENCY KILL SWITCH**

### **Close All Positions (Liquidate Everything)**

```bash
curl -X DELETE http://localhost:8000/api/alpaca/positions
```

**⚠️ WARNING: This sells ALL your positions immediately!**

### **Close Single Position**

```bash
curl -X DELETE http://localhost:8000/api/alpaca/positions/AAPL
```

---

## **🔧 API ENDPOINTS REFERENCE**

### **Account Management**
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/alpaca/account` | GET | Full account details |
| `/api/alpaca/account/status` | GET | Simple connection check |
| `/api/alpaca/initialize` | POST | Initialize connection |

### **Trading**
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/alpaca/orders` | GET | List all orders |
| `/api/alpaca/orders` | POST | Submit new order |
| `/api/alpaca/orders/{id}` | DELETE | Cancel order |
| `/api/alpaca/buy` | POST | Quick buy |
| `/api/alpaca/sell` | POST | Quick sell |

### **Portfolio**
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/alpaca/positions` | GET | List all positions |
| `/api/alpaca/positions/{symbol}` | GET | Get specific position |
| `/api/alpaca/positions/{symbol}` | DELETE | Close position |
| `/api/alpaca/positions` | DELETE | Close ALL positions |
| `/api/alpaca/portfolio/summary` | GET | Full portfolio view |

### **Market Data**
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/alpaca/clock` | GET | Market open/close status |
| `/api/alpaca/calendar` | GET | Trading days calendar |

---

## **⚙️ CONFIGURATION OPTIONS**

### **Environment Variables**

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ALPACA_API_KEY` | Yes | - | Your Alpaca API key |
| `ALPACA_API_SECRET` | Yes | - | Your Alpaca secret key |
| `ALPACA_PAPER_TRADING` | No | `true` | Always `true` for paper trading |

### **Order Types Supported**

- **market** - Execute at current market price
- **limit** - Execute at specified price or better
- **stop** - Trigger at stop price, then market order
- **stop_limit** - Trigger at stop price, then limit order

### **Time in Force**

- **day** - Valid for trading day only
- **gtc** - Good till cancelled
- **ioc** - Immediate or cancel
- **fok** - Fill or kill

---

## **🎯 TESTING WORKFLOWS**

### **Workflow 1: Buy and Hold**
```bash
# 1. Check account
curl http://localhost:8000/api/alpaca/account

# 2. Buy 10 shares of AAPL
curl -X POST "http://localhost:8000/api/alpaca/buy?symbol=AAPL&qty=10"

# 3. Check position
curl http://localhost:8000/api/alpaca/positions/AAPL

# 4. View portfolio
curl http://localhost:8000/api/alpaca/portfolio/summary
```

### **Workflow 2: Day Trading**
```bash
# 1. Buy at market open
curl -X POST "http://localhost:8000/api/alpaca/buy?symbol=TSLA&qty=5"

# 2. Check unrealized P&L
curl http://localhost:8000/api/alpaca/positions/TSLA

# 3. Sell when target reached
curl -X POST "http://localhost:8000/api/alpaca/sell?symbol=TSLA&qty=5"

# 4. View order history
curl "http://localhost:8000/api/alpaca/orders?status=closed"
```

### **Workflow 3: Emergency Exit**
```bash
# 1. Check all positions
curl http://localhost:8000/api/alpaca/positions

# 2. LIQUIDATE EVERYTHING
curl -X DELETE http://localhost:8000/api/alpaca/positions

# 3. Verify no positions remain
curl http://localhost:8000/api/alpaca/positions
```

---

## **🔒 SAFETY FEATURES**

### **Paper Trading Safeguards**

- ✅ **Fake money only** - No real funds at risk
- ✅ **$100,000 starting balance** - Standard paper trading amount
- ✅ **Same market data** - Real prices, real execution
- ✅ **Order limits** - Prevents accidental large orders
- ✅ **Kill switch** - Emergency liquidation endpoint

### **Best Practices**

1. **Always test with paper trading first**
2. **Start with small positions** (1-10 shares)
3. **Monitor your orders** in Alpaca dashboard
4. **Use limit orders** for better price control
5. **Set stop losses** to limit downside

---

## **📱 ALPACA DASHBOARD**

View your paper trading activity:

1. Go to: https://app.alpaca.markets/paper/dashboard
2. Log in with your Alpaca account
3. See:
   - Portfolio value
   - Open positions
   - Order history
   - Account activity

---

## **❌ TROUBLESHOOTING**

### **"Alpaca not configured" Error**

**Problem:** API keys not set

**Solution:**
```powershell
# Set environment variables
$env:ALPACA_API_KEY="PK..."
$env:ALPACA_API_SECRET="..."

# Or use .env file
```

### **"Market is closed" Error**

**Problem:** Trying to trade outside market hours

**Solution:**
- Check market clock: `curl http://localhost:8000/api/alpaca/clock`
- Alpaca paper trading follows regular market hours (9:30 AM - 4:00 PM ET)

### **"Insufficient buying power" Error**

**Problem:** Not enough cash for order

**Solution:**
- Check account: `curl http://localhost:8000/api/alpaca/account`
- Reduce order size
- Wait for previous orders to settle

### **"Invalid API credentials" Error**

**Problem:** Wrong API keys

**Solution:**
- Verify keys in Alpaca dashboard
- Check for extra spaces or characters
- Regenerate keys if needed

---

## **🎓 LEARNING RESOURCES**

### **Alpaca Documentation**
- API Docs: https://alpaca.markets/docs/
- Paper Trading: https://alpaca.markets/docs/trading/paper-trading/
- API Reference: https://alpaca.markets/docs/api-references/

### **Trading Basics**
- Market Orders vs Limit Orders
- Stop Loss and Take Profit
- Position Sizing
- Risk Management

---

## **✅ READY TO TRADE CHECKLIST**

- [ ] Alpaca account created
- [ ] API keys generated
- [ ] Environment variables set
- [ ] HugeFunds started successfully
- [ ] Alpaca connection verified
- [ ] Test order placed successfully
- [ ] Portfolio tracking working

**Once all checked, you're ready to trade with $100,000 fake money!** 🚀

---

## **🚀 NEXT STEPS**

1. **Connect your 14 strategies** to Alpaca for automated trading
2. **Set up risk management** rules in the system
3. **Test strategies** with paper trading
4. **Monitor performance** using the dashboard
5. **Go live** when ready (with real money account)

---

**Happy Trading! 📈**

*Remember: Paper trading is practice with fake money. Always test thoroughly before trading with real funds.*
