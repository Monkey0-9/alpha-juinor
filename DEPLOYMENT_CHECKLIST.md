# 📋 POST-FIX DEPLOYMENT CHECKLIST

## Pre-Deployment Verification ✓

- [x] All Python files compile without errors
- [x] Alpha signal generation optimized (3.5x stronger)
- [x] Risk parameters updated (TP 5%, SL -6%)
- [x] Position sizing improved (exponential weighting)
- [x] Strategy parameters tuned (8 strategies optimized)
- [x] Risk scaling logic improved
- [x] High conviction filter added
- [x] All changes documented

---

## Step 1: Paper Trading Mode (Days 1-3)

### Setup:
```bash
# Enable paper trading
set ALPACA_PAPER_TRADING=true

# Optional: Verify configuration
python -c "from nexus.utils.config import Config; print(f'TP: {Config.TAKE_PROFIT_THRESHOLD}, SL: {Config.STOP_LOSS_THRESHOLD}')"
# Should print: TP: 0.05, SL: -0.06
```

### Monitor These Metrics:
- [ ] Check logs for signal strength (should see > 0.30 values)
- [ ] Verify positions open and close correctly
- [ ] Count open positions (should be 20-30, not 50)
- [ ] Time positions are held (should be 2-4 days, not 5-7)
- [ ] Calculate win rate from 20 sample trades (should be > 50%)

### Log Locations:
- Main: `logs/nexus_24_7_*.log`
- Order details: `logs/nexus_orchestrator_*.log`
- Audit trail: `data/nexus_audit.db`

### Verify Signals in Logs:
```bash
# Search for signal values
findstr /C:"alpha=" logs/nexus_24_7_*.log | head -20
# Should show values like: alpha=0.45, alpha=0.32, alpha=0.28 (not 0.08-0.12)
```

---

## Step 2: Live Trading - Small Scale (Days 4-7)

### Position Sizing:
- [ ] Start with 10% of normal capital
- [ ] Increase to 25% after 50 profitable trades
- [ ] Increase to 50% after 100 profitable trades
- [ ] Full allocation after 1 week of positive returns

### Daily Checklist:
- [ ] Check P&L before market open
- [ ] Verify signals are > 0.30 average
- [ ] Monitor drawdown (should stay < -5% intraday)
- [ ] Ensure stop losses execute at -6%
- [ ] Verify take profits execute at 5%

---

## Step 3: Scaling Up (Week 2+)

### Performance Targets:

| Metric | Target | Check |
|--------|--------|-------|
| Daily Return | +0.06% to +0.10% | [ ] |
| Sharpe Ratio | 1.0 to 1.8 | [ ] |
| Win Rate | > 52% | [ ] |
| Max Drawdown | < -12% | [ ] |
| Avg Hold Time | 2-4 days | [ ] |
| Open Positions | 20-30 | [ ] |

### Scaling Rules:
- [ ] If any metric falls below target for 3 consecutive days: PAUSE
- [ ] If drawdown exceeds -15%: STOP and review
- [ ] If win rate drops below 48%: REVIEW strategy parameters

---

## Critical Validation Points

### 1. Signal Strength Verification
Check that signals are actually stronger:
```bash
# In Python
from nexus.core.alpha import AlphaEngine
import pandas as pd

engine = AlphaEngine()
data = pd.DataFrame({
    'close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
})
signal = engine.generate_signal(data)
print(f"Signal strength: {signal}")
# Should be > 0.30 for trending data
```

### 2. Position Sizing Verification
Check that position sizing is exponential:
```bash
from nexus.math.optimization import PortfolioOptimizer

optimizer = PortfolioOptimizer()
symbols = ["AAPL", "MSFT", "GOOG"]
signals = [0.2, 0.5, 0.8]
weights = optimizer.optimize_weights(symbols, signals)
print(weights)
# AAPL weight should be much smaller than GOOG
# Ratio should be roughly 1:8 (not 1:4)
```

### 3. Risk Parameter Verification
Check configuration took effect:
```bash
python -c "
from nexus.utils.config import Config
assert Config.TAKE_PROFIT_THRESHOLD == 0.05, 'TP not 5%'
assert Config.STOP_LOSS_THRESHOLD == -0.06, 'SL not -6%'
assert Config.MIN_HOLD_CYCLES == 2, 'Min hold not 2'
assert Config.MAX_OPEN_POSITIONS == 30, 'Max positions not 30'
print('✓ All config parameters correct')
"
```

---

## Troubleshooting Guide

### Problem: Signals Not Stronger
**Cause**: Volatility penalty still too aggressive  
**Fix**: Check nexus/core/alpha.py line ~20, volatility_penalty should divide by 3.5x
```python
volatility_penalty = 1.0 / (1.0 + volatility * 3.5)  # NOT 8.0
```

### Problem: Positions Holding Too Long
**Cause**: TAKE_PROFIT_THRESHOLD not updated  
**Fix**: Verify in nexus/utils/config.py it says 0.05, not 0.08
```bash
grep "TAKE_PROFIT" nexus/utils/config.py
# Should show: "0.05"
```

### Problem: Too Many Positions Open
**Cause**: High conviction filter not working  
**Fix**: Check nexus/core/engine.py for the filter:
```python
if abs(score) > 0.20:  # This line should exist
    top_targets[symbol] = score
```

### Problem: Sharpe Ratio Not Improving
**Cause**: Need more trade history  
**Fix**: Wait at least 2 weeks (need 100+ trades)
```bash
# Count trades from audit log
sqlite3 data/nexus_audit.db "SELECT COUNT(*) FROM trade_history WHERE status='filled';"
# Should be > 100 before evaluating
```

---

## Performance Tracking

### Create a Daily Log:
```csv
Date,Daily_Return,Sharpe,Win_Rate,Avg_Hold_Days,Open_Positions,Drawdown,Notes
2026-05-30,+0.02%,0.5,48%,3.2,28,-2.5%,Good signal strength
2026-05-31,+0.08%,0.8,52%,2.8,25,-1.2%,Profitable day
2026-06-01,+0.05%,0.9,51%,2.9,26,-0.8%,Consistent performance
```

### Weekly Review:
- [ ] Average daily return
- [ ] Sharpe ratio trend
- [ ] Win rate percentage
- [ ] Are signals > 0.30?
- [ ] Are positions held 2-4 days?
- [ ] Is drawdown controlled?

---

## When to Call for Help

🔴 **RED FLAGS** - Contact support if:
- Daily return < -0.05% for 3 consecutive days
- Drawdown exceeds -15%
- Win rate drops below 45%
- Open positions > 40
- Signals consistently < 0.20

🟡 **YELLOW FLAGS** - Review and decide:
- Sharpe ratio not improving after 1 week
- Average holding period > 5 days
- More than 35 positions open
- Win rate 48-50% (borderline)

🟢 **GREEN FLAGS** - Keep running:
- Daily return > +0.05%
- Sharpe ratio > 1.0
- Win rate > 52%
- Positions 20-30
- Hold times 2-4 days

---

## Files to Monitor

### Real-Time Logs:
- `logs/nexus_24_7_YYYYMMDD.log` - Main trading loop
- `logs/nexus_orchestrator_YYYYMMDD_HHMMSS.log` - Detailed actions

### Database:
- `data/nexus_audit.db` - Query for trade history
- `data/nexus_audit.db` - Governance decisions

### Query Examples:
```sql
-- Last 10 trades
SELECT * FROM trade_history ORDER BY timestamp DESC LIMIT 10;

-- Win rate
SELECT COUNT(*) as wins FROM trade_history WHERE side='sell' AND price > entry_price;

-- Average holding period
SELECT AVG(CAST((exit_time - entry_time) as FLOAT)) as avg_hold_minutes FROM trades;
```

---

## Success Criteria (Week 1)

✅ **Minimum Requirements**:
- Signal strength > 0.25 average
- Win rate > 50%
- Daily return > 0% average
- Max drawdown < -10%
- Positions held < 5 days

✅ **Good Performance**:
- Signal strength > 0.35 average
- Win rate > 52%
- Daily return > +0.05%
- Max drawdown < -8%
- Positions held < 4 days

✅ **Excellent Performance**:
- Signal strength > 0.40 average
- Win rate > 55%
- Daily return > +0.08%
- Max drawdown < -6%
- Positions held < 3 days
- Sharpe ratio > 1.2

---

## Next Review Date: May 30, 2026

**Prepared**: May 29, 2026  
**Status**: 🟢 **READY FOR PAPER TESTING**
