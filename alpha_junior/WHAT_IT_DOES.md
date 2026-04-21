# 📊 WHAT ALPHA JUNIOR ACTUALLY DOES

## **SIMPLE EXPLANATION**

Alpha Junior is a **robot stock trader** that:
1. **Watches** stock prices (AAPL, TSLA, NVDA, etc.)
2. **Analyzes** them using math (momentum + RSI)
3. **Decides** when to buy or sell
4. **Executes** trades automatically on Alpaca (paper/fake money)
5. **Tracks** your profit/loss (P&L)

**Goal:** Make 50-60% returns by trading high-growth tech stocks automatically.

---

## **🤖 AUTOMATED PROCESS (What Happens Every Minute)**

```
┌─────────────────────────────────────────────────────────────┐
│  STEP 1: CHECK MARKET                                       │
│  • Get current price of NVDA, TSLA, AAPL, etc.             │
│  • Get last 10 days of price history                        │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  STEP 2: ANALYZE WITH MATH                                  │
│  • Calculate MOMENTUM: Is price going up or down?          │
│  • Calculate RSI: Is stock overbought or oversold?         │
│                                                             │
│  Example: NVDA                                              │
│  • 10 days ago: $100                                        │
│  • Today: $110                                              │
│  • MOMENTUM: +10% (going up!)                              │
│  • RSI: 65 (not overbought yet)                            │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  STEP 3: MAKE DECISION                                      │
│                                                             │
│  IF momentum > 5% AND RSI < 70:                             │
│     → SIGNAL: BUY! 📈                                      │
│                                                             │
│  IF momentum < -5% OR RSI > 80:                             │
│     → SIGNAL: SELL! 📉                                     │
│                                                             │
│  ELSE:                                                      │
│     → SIGNAL: HOLD (do nothing)                            │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  STEP 4: EXECUTE TRADE                                      │
│  • Place order on Alpaca (paper trading)                   │
│  • Buy 10 shares of NVDA at market price                   │
│  • Order ID: abc-123-xyz                                    │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  STEP 5: TRACK PERFORMANCE                                  │
│  • Record entry price: $110                                 │
│  • Monitor current price                                    │
│  • Calculate P&L: +$50 profit / -$20 loss                  │
└─────────────────────────────────────────────────────────────┘
```

---

## **💰 REAL EXAMPLE**

**Day 1:**
- 9:00 AM: System checks NVDA → Price $100
- 9:05 AM: Momentum +8%, RSI 60 → **BUY SIGNAL** 
- 9:06 AM: Buys 10 shares NVDA @ $100 = $1,000 invested
- Status: Waiting...

**Day 2:**
- 9:00 AM: System checks NVDA → Price $115
- Current value: 10 shares × $115 = $1,150
- **PROFIT: +$150 (+15%)** 🎉

**Day 3:**
- 9:00 AM: Momentum slowing, RSI 85 (overbought)
- **SELL SIGNAL** triggered
- Sells 10 shares @ $118 = $1,180
- **TOTAL PROFIT: +$180 (+18%)** 💰

---

## **📈 WHAT YOU SEE ON SCREEN**

When running, you'll see:
```
[10:15:23] Checking AAPL... Price: $175.50
[10:15:24] Momentum: +6.2%, RSI: 58 → BUY SIGNAL
[10:15:25] Placing order: BUY 10 AAPL @ market
[10:15:26] Order filled: Bought 10 AAPL @ $175.50
[10:15:26] Current P&L: $0 (just bought)

[10:30:00] Checking positions...
[10:30:01] AAPL: 10 shares, Current: $178.00
[10:30:01] UNREALIZED P&L: +$25.00 (+1.4%) 📈

[11:00:00] Checking NVDA... Price: $890.00
[11:00:01] Momentum: +12%, RSI: 45 → STRONG BUY
[11:00:02] Placing order: BUY 5 NVDA @ market
[11:00:03] Order filled: Bought 5 NVDA @ $890.00
```

---

## **🎯 TARGET RETURNS: 50-60%**

**Monthly Breakdown:**
- January: +5% 
- February: -3% (loss month)
- March: +8%
- April: +4%
- May: +6%
- ... Average: ~4-5% per month

**Annual:** 4.5% × 12 = **54% return** 🚀

**With $10,000:**
- Year 1: $10,000 → $15,400 (+$5,400)
- Year 2: $15,400 → $23,716 (+$8,316)
- Year 3: $23,716 → $36,523 (+$12,807)

---

## **⚠️ RISKS (IMPORTANT)**

**This is HIGH RISK trading:**
- ❌ Can lose 20-30% in bad months
- ❌ Not guaranteed returns
- ❌ Market crashes = big losses
- ❌ Requires active monitoring

**Paper Trading = Practice Mode:**
- ✅ Fake money ($100,000)
- ✅ Real market prices
- ✅ Test strategies safely
- ✅ Learn without losing real money

---

## **🖥️ VISUAL DASHBOARD**

When you open http://localhost:5000, you see:

```
┌────────────────────────────────────────────┐
│  ALPHA JUNIOR - DASHBOARD                  │
├────────────────────────────────────────────┤
│                                            │
│  PORTFOLIO VALUE: $105,420 (+5.4%)        │
│  CASH: $85,000                            │
│  INVESTED: $20,420                        │
│                                            │
├────────────────────────────────────────────┤
│  ACTIVE POSITIONS                          │
├────────────────────────────────────────────┤
│  AAPL    10 shares    $1,780    +$30      │
│  NVDA     5 shares    $4,550    +$120     │
│  TSLA     8 shares    $2,160    -$40      │
│                                            │
│  TOTAL P&L: +$110 (+2.1%)                 │
├────────────────────────────────────────────┤
│  RECENT TRADES                             │
├────────────────────────────────────────────┤
│  [10:15] BUY  AAPL  10 @ $175.50         │
│  [10:30] BUY  NVDA   5 @ $890.00         │
│  [09:45] SELL TSLA   5 @ $240.00  +$25   │
│                                            │
└────────────────────────────────────────────┘
```

---

## **🔄 AUTOMATION LEVELS**

### **Level 1: Manual (You Control)**
- You check prices
- You decide to buy/sell
- You place orders manually
- **Time needed:** 2-3 hours/day

### **Level 2: Semi-Auto (Current)**
- System analyzes and gives signals
- You approve each trade
- System executes after your OK
- **Time needed:** 30 minutes/day

### **Level 3: Fully Auto (What We Built)**
- System analyzes 24/7
- System decides automatically
- System executes immediately
- You just monitor P&L
- **Time needed:** 5 minutes/day (check dashboard)

**Alpha Junior = Level 3 (Fully Automated)**

---

## **📊 SUMMARY**

| Question | Answer |
|----------|--------|
| **What does it do?** | Automatically trades stocks for profit |
| **How?** | Math analysis (momentum + RSI) |
| **Which stocks?** | Tech giants: AAPL, NVDA, TSLA, MSFT, GOOGL |
| **Target return?** | 50-60% per year |
| **Is it safe?** | Paper trading = no real money risk |
| **How much time?** | 5 min/day to check dashboard |
| **Do I need to know coding?** | No, it's fully automated |

---

## **🚀 TO SEE IT IN ACTION**

1. **Start the server:**
   ```bash
   python runner.py
   ```

2. **Watch the console** - You'll see trades happening

3. **Open dashboard:** http://localhost:5000

4. **Check your Alpaca paper account:** https://app.alpaca.markets/paper/dashboard

---

**It's like having a robot stock trader working 24/7 for you!** 🤖📈
