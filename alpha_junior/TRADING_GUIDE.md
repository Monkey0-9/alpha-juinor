# 🚀 Alpha Junior - Alpaca Paper Trading Guide

## High Return Trading System (Target: 50-60%)

---

## 📋 Setup Instructions

### Step 1: Get Alpaca API Keys
1. Go to https://alpaca.markets/
2. Sign up for a free paper trading account
3. Get your API Key ID and Secret Key
4. Go to dashboard → API Keys → Generate New Key

### Step 2: Configure API Keys
Edit `.env` file and add your keys:
```
ALPACA_API_KEY=PKUNNQ8INWN6B3TCUNWK
ALPACA_SECRET_KEY=your_actual_secret_key_here
```

### Step 3: Restart Server
Stop the current server (Ctrl+C), then:
```bash
python runner.py
```

You should see: **🚀 ALPACA PAPER TRADING: ENABLED**

---

## 📊 Trading API Endpoints

### Account Information
```bash
GET http://localhost:5000/api/trading/account
```
Returns: Balance, equity, buying power, portfolio value

### View Positions
```bash
GET http://localhost:5000/api/trading/positions
```
Returns: All open positions with P&L

### View Orders
```bash
GET http://localhost:5000/api/trading/orders
```
Returns: All orders (open, filled, cancelled)

### Place Order
```bash
POST http://localhost:5000/api/trading/order
Content-Type: application/json

{
  "symbol": "AAPL",
  "qty": 10,
  "side": "buy",
  "type": "market"
}
```

### Cancel Order
```bash
DELETE http://localhost:5000/api/trading/order/{order_id}
```

### Close Position
```bash
DELETE http://localhost:5000/api/trading/position/{symbol}
```

### Get Quote
```bash
GET http://localhost:5000/api/trading/quote/AAPL
```

### Get Historical Data
```bash
GET http://localhost:5000/api/trading/bars/AAPL?timeframe=1Day&limit=100
```

---

## 🎯 Automated Trading Strategy

### Execute Strategy
```bash
POST http://localhost:5000/api/trading/strategy/execute
Content-Type: application/json

{
  "strategy": "momentum",
  "symbols": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA"],
  "auto_trade": true
}
```

**Strategies:**
- `momentum` - Buy stocks with positive momentum, sell overbought
- Analyzes RSI and price momentum
- Targets high-volatility tech stocks for maximum returns

---

## 💰 High Return Targets (50-60%)

### Recommended Approach:

1. **Aggressive Tech Stocks** (30% target)
   - NVDA, TSLA, AMD, PLTR
   - High volatility = high returns
   - Use momentum strategy

2. **Growth Stocks** (20% target)
   - AAPL, MSFT, GOOGL, AMZN
   - Steady growth + momentum
   - Mix of long-term and swing trades

3. **Crypto-Related** (10% target)
   - COIN, MSTR, HOOD
   - High beta, crypto exposure

### Risk Management:
- Start with paper trading (fake money)
- Never risk more than 5% per trade
- Use stop-losses at -7%
- Take profits at +15%

---

## 🧪 Test Commands

### Check Account
```bash
curl http://localhost:5000/api/trading/account
```

### Buy 10 shares of AAPL
```bash
curl -X POST http://localhost:5000/api/trading/order \
  -H "Content-Type: application/json" \
  -d '{"symbol":"AAPL","qty":10,"side":"buy","type":"market"}'
```

### Run Strategy
```bash
curl -X POST http://localhost:5000/api/trading/strategy/execute \
  -H "Content-Type: application/json" \
  -d '{"strategy":"momentum","symbols":["NVDA","TSLA","AAPL"],"auto_trade":true}'
```

### View Portfolio Performance
```bash
curl http://localhost:5000/api/trading/portfolio/performance
```

---

## 📈 Sample Trading Session

1. **Check balance:** `$100,000` (paper money)
2. **Run strategy** on NVDA, TSLA, AAPL
3. **System buys** based on momentum signals
4. **Monitor positions** via `/api/trading/positions`
5. **Track P&L** in real-time
6. **Take profits** when target reached

---

## ⚠️ Important Notes

- **Paper Trading Only** - No real money at risk
- **High Returns = High Risk** - 50-60% target is aggressive
- **Past performance** not indicative of future results
- **Start small** - Test strategies before sizing up
- **Monitor regularly** - Check positions daily

---

## 🔗 Links

- Alpaca Dashboard: https://alpaca.markets/
- Paper Trading: https://app.alpaca.markets/paper/dashboard
- API Docs: https://alpaca.markets/docs/

---

**Ready to trade? Add your API keys and restart the server!** 🚀
