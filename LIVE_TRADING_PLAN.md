# 24/7 Live Trading Implementation Plan

## Objective
Create a robust 24/7 live trading system that:
1. Runs continuously without manual intervention
2. Makes per-second trading decisions (opportunity scanning)
3. Refreshes market data every 30-60 minutes (not per-second)
4. Shows real-time trading status per second in terminal

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    LIVE TRADING DAEMON                         │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              PER-SECOND DECISION LOOP                    │   │
│  │  - Heartbeat logging                                    │   │
│  │  - Real-time signal display                             │   │
│  │  - Position/P&L updates                                 │   │
│  │  - Risk metrics per second                              │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │                                    │
│                           ▼                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              META-BRAIN DECISION ENGINE                  │   │
│  │  - Aggregates agent signals                             │   │
│  │  - Makes BUY/SELL/HOLD decisions                        │   │
│  │  - Applies risk checks                                  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │                                    │
│                           ▼                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │           30-MINUTE DATA REFRESH LAYER                   │   │
│  │  - Fetches new market data                              │   │
│  │  - Updates cached features                              │   │
│  │  - Runs data quality checks                             │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │                                    │
│                           ▼                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              BROKER EXECUTION HANDLER                    │   │
│  │  - Alpaca (Paper/Live)                                  │   │
│  │  - Order management                                     │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Files to Create/Modify

### New Files
1. **`live_trading_daemon.py`** - Main 24/7 daemon entry point
2. **`orchestration/live_decision_loop.py`** - Per-second decision loop engine
3. **`orchestration/data_refresh_scheduler.py`** - 30-minute data refresh scheduler

### Modified Files
1. **`trading_daemon.py`** - Connect to real MetaBrain for live decisions
2. **`main.py`** - Add daemon mode support
3. **`configs/config_manager.py`** - Add live trading configuration

## Implementation Steps

### Step 1: Create Live Decision Loop Engine
- Per-second loop with precise timing
- Real-time status display
- Signal aggregation from MetaBrain
- Position and P&L tracking

### Step 2: Create Data Refresh Scheduler
- 30-60 minute data refresh interval
- Parallel data fetching
- Feature recomputation
- Data quality validation

### Step 3: Connect to MetaBrain
- Integrate with existing MetaBrain decision engine
- Use cached features for fast per-second decisions
- Real-time signal updates

### Step 4: Broker Integration
- Alpaca broker connection
- Order execution (paper/live modes)
- Position tracking

### Step 5: Terminal Output Dashboard
- Per-second heartbeat
- Live signal status per symbol
- Position and P&L display
- Risk metrics dashboard

## Timing Architecture

```
Second 0:  Decision loop tick     | Data refresh check (if 30min elapsed)
Second 1:  Decision loop tick     |
Second 2:  Decision loop tick     |
...
Second 1800: Decision loop tick   | REFRESH DATA (every 30 min)
Second 1801: Decision loop tick   | Recompute features
...
```

## Per-Second Terminal Output Format

```
================================================================================
[2024-01-15 14:30:01] 🔴 LIVE TRADING DAEMON - CYCLE #4532
================================================================================
📊 SYSTEM STATUS: RUNNING | UPTIME: 12h 30m 01s
💓 HEARTBEAT: ALIVE | LAST TRADE: 2024-01-15T14:29:58Z
📡 DATA STATUS: FRESH | LAST REFRESH: 2024-01-15T14:00:00Z | NEXT: 14:30:00

📈 POSITIONS (3 symbols):
┌─────────┬──────────┬──────────┬──────────┬──────────┬──────────┐
│ Symbol  │ Position │ Entry    │ Current  │ P&L      │ Conviction│
├─────────┼──────────┼──────────┼──────────┼──────────┼──────────┤
│ AAPL    │ +100     │ 185.50   │ 187.23   │ +$173.00 │ 0.85     │
│ MSFT    │ +50      │ 378.90   │ 380.12   │ +$61.00  │ 0.72     │
│ GOOGL   │ -25      │ 140.25   │ 139.87   │ +$9.50   │ 0.45     │
└─────────┴──────────┴──────────┴──────────┴──────────┴──────────┘

🎯 LIVE SIGNALS (All Symbols):
┌─────────┬──────────┬──────────┬──────────┬──────────┬──────────┐
│ Symbol  │ Signal   │ Mu_Hat   │ Sigma    │ Convict. │ Data Q   │
├─────────┼──────────┼──────────┼──────────┼──────────┼──────────┤
│ AAPL    │ BUY ⬆   │ 0.0125   │ 0.085    │ 0.85     │ 0.95     │
│ MSFT    │ HOLD ➡  │ 0.0032   │ 0.092    │ 0.72     │ 0.98     │
│ GOOGL   │ SELL ⬇  │ -0.0081  │ 0.078    │ 0.45     │ 0.94     │
│ NVDA    │ BUY ⬆   │ 0.0156   │ 0.112    │ 0.91     │ 0.96     │
│ META    │ BUY ⬆   │ 0.0098   │ 0.105    │ 0.88     │ 0.97     │
└─────────┴──────────┴──────────┴──────────┴──────────┴──────────┘

⚠️ RISK METRICS:
   Portfolio CVaR (95%): 2.3% | Leverage: 0.85x | VaR: 1.8%
   Max Position: 4.2% | Max Sector: 18.5% | Risk Regime: BULL

📋 DECISION SUMMARY (This Cycle):
   BUY Signals: 2 | SELL Signals: 1 | HOLD Signals: 2 | REJECT: 0
   Orders Generated: 3 | Orders Executed: 2 | Pending: 1

================================================================================

