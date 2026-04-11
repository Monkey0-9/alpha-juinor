#!/usr/bin/env python3
"""
[CHK] Trading Execution Diagnostic
Checks why trades aren't executing and shows exactly what needs to be fixed.
"""

import os
import sys
from pathlib import Path

print("\n" + "=" * 80)
print("[CHK] TRADING EXECUTION DIAGNOSTIC")
print("=" * 80)

# Check 1: Environment Variables
print("\n1️⃣  ENVIRONMENT FLAGS:")
print(f"   EXECUTE_TRADES: {os.getenv('EXECUTE_TRADES', ' NOT SET')}")
print(f"   TRADING_MODE:   {os.getenv('TRADING_MODE', ' NOT SET')}")
print(f"   Paper Mode:     {os.getenv('PAPER_MODE', ' NOT SET')}")

# Check 2: Kill switches
print("\n2️⃣  KILL SWITCHES:")
kill_switch = Path("runtime/KILL_SWITCH")
local_kill_switch = Path("runtime/kill_switch_local.state")
print(
    f"   KILL_SWITCH exists:       {kill_switch.exists()} {' BLOCKING TRADES' if kill_switch.exists() else ' OK'}"
)
print(
    f"   kill_switch_local.state:  {local_kill_switch.exists()} {' BLOCKING TRADES' if local_kill_switch.exists() else ' OK'}"
)

# Check 3: Execution chain
print("\n3️⃣  EXECUTION LOGIC:")
execute_trades = os.getenv("EXECUTE_TRADES", "false").lower() == "true"
paper_mode = os.getenv("TRADING_MODE", "").lower() == "paper"

print(f"   EXECUTE_TRADES enabled: {execute_trades}")
print(f"   Paper mode active:      {paper_mode}")

if execute_trades or paper_mode:
    print(f"    TRADES WILL EXECUTE")
else:
    print(f"    TRADES BLOCKED!")
    print(f"\n      WHY: EXECUTE_TRADES=false AND not in paper mode")
    print(f"      FIX: Run one of these:")
    print(f"        • python start_trading.py --mode paper")
    print(f"        • export EXECUTE_TRADES=true && python main.py")
    print(f"        • export TRADING_MODE=paper && python main.py")

# Check 4: Broker setup
print("\n4️⃣  BROKER CONFIGURATION:")
alpaca_key = bool(os.getenv("ALPACA_API_KEY"))
alpaca_secret = bool(os.getenv("ALPACA_SECRET_KEY"))
print(f"   Alpaca API Key:    {f' Set' if alpaca_key else ' NOT SET'}")
print(f"   Alpaca Secret Key: {f' Set' if alpaca_secret else ' NOT SET'}")

if not (alpaca_key and alpaca_secret):
    print(f"\n      WARNING: Alpaca credentials missing!")
    print(f"      Orders will use MockBroker (paper trading only)")

# Check 5: Required modules
print("\n5️⃣  SYSTEM MODULES:")
try:
    from execution.parallel_strategy_engine import ParallelStrategyExecutor

    print(f"   ParallelStrategyExecutor:  Available")
except ImportError as e:
    print(f"   ParallelStrategyExecutor:  {e}")

try:
    from orchestration.live_decision_loop import LiveDecisionLoop

    print(f"   LiveDecisionLoop:          Available")
except ImportError as e:
    print(f"   LiveDecisionLoop:          {e}")

try:
    from brokers.mock_broker import MockBroker

    print(f"   MockBroker:                Available")
except ImportError as e:
    print(f"   MockBroker:                {e}")

# Check 6: Data directory
print("\n6️⃣  DATA & RUNTIME:")
runtime_dir = Path("runtime")
logs_dir = Path("logs")
print(
    f"   runtime/ directory: {f' Exists' if runtime_dir.exists() else ' Missing'}"
)
print(f"   logs/ directory:    {f' Exists' if logs_dir.exists() else ' Missing'}")

# Summary
print("\n" + "=" * 80)
print("[LST] SUMMARY & NEXT STEPS")
print("=" * 80)

if execute_trades or paper_mode:
    print("\n System is ready to execute trades!")
    print("\nStart trading with:")
    print("  python start_trading.py --mode paper    (Recommended: test first)")
    print("  python start_trading.py --mode live     (Real money: be careful!)")
else:
    print("\n Trades are BLOCKED due to missing EXECUTE_TRADES flag")
    print("\n🔧 QUICK FIX #1 - Run proper startup script:")
    print("  python start_trading.py --mode paper")
    print("\n🔧 QUICK FIX #2 - Set environment manually then run:")
    print("  set EXECUTE_TRADES=true")
    print("  python main.py")
    print("\n🔧 QUICK FIX #3 - Run in paper mode:")
    print("  set TRADING_MODE=paper")
    print("  python main.py")

print("\n" + "=" * 80)
