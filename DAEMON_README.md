# Trading Daemon Quick Start

## Start the 24/7 Daemon

```powershell
# Default: 5s trigger, 30min data refresh
python trading_daemon.py

# Custom intervals
python trading_daemon.py --trigger 5 --data-refresh 60

# Market hours only (9:30 AM - 4:00 PM ET, Mon-Fri)
python trading_daemon.py --market-hours-only
```

## Kill Switch

To immediately halt trading:
```powershell
New-Item -Path "runtime/KILL_SWITCH" -ItemType File
```

To resume:
```powershell
Remove-Item "runtime/KILL_SWITCH"
```

## Monitor Logs

```powershell
Get-Content logs/trading_daemon.log -Tail 50 -Wait
```

## Integration with Main Trading Logic

The daemon currently logs cycles. To integrate with your actual trading logic:

1. Open `trading_daemon.py`
2. Find the `_execute_trading_cycle` method
3. Uncomment and modify:
```python
from main import execute_trading_cycle
execute_trading_cycle(cycle_id, self.symbols)
```

## Health Monitoring

The daemon includes:
- **Heartbeat**: Monitors if cycles are executing
- **Error Tracking**: Auto-halts after 10 consecutive errors
- **Graceful Shutdown**: Ctrl+C or SIGTERM handled safely
