# Live Trading Implementation Tasks

## Phase 1: Core Infrastructure
- [ ] Create `orchestration/live_decision_loop.py` - Per-second decision loop engine
- [ ] Create `orchestration/data_refresh_scheduler.py` - 30-minute data refresh scheduler
- [ ] Create `orchestration/live_portfolio_manager.py` - Position and P&L tracking
- [ ] Create `orchestration/live_signal_aggregator.py` - Real-time signal aggregation

## Phase 2: Live Trading Daemon
- [ ] Create `live_trading_daemon.py` - Main 24/7 daemon entry point
- [ ] Add graceful shutdown handling (SIGINT/SIGTERM)
- [ ] Add kill switch support
- [ ] Add health monitoring thread

## Phase 3: Integration with Existing Components
- [ ] Integrate with `MetaBrain` for decision making
- [ ] Integrate with `AlpacaExecutionHandler` for order execution
- [ ] Integrate with `DatabaseManager` for persistence
- [ ] Integrate with `DataRouter` for market data

## Phase 4: Terminal Dashboard
- [ ] Implement per-second heartbeat display
- [ ] Implement live position table
- [ ] Implement live signal display per symbol
- [ ] Implement risk metrics dashboard
- [ ] Implement decision summary display

## Phase 5: Testing & Validation
- [ ] Test per-second loop timing accuracy
- [ ] Test 30-minute data refresh
- [ ] Test broker connection (paper mode)
- [ ] Test graceful shutdown
- [ ] Test kill switch functionality

## Phase 6: Documentation
- [ ] Update README.md with live trading instructions
- [ ] Create run scripts for different modes
- [ ] Document configuration options

---

## Quick Start Commands

```bash
# Run in paper mode (simulated trading)
python live_trading_daemon.py --mode paper

# Run in live mode (real money)
python live_trading_daemon.py --mode live

# Run with 5-second decision loop (less verbose)
python live_trading_daemon.py --tick-interval 5

# Run with 15-minute data refresh (more frequent)
python live_trading_daemon.py --data-refresh 15

# Check daemon status
python live_trading_daemon.py --status

# Stop daemon
python live_trading_daemon.py --stop
```

