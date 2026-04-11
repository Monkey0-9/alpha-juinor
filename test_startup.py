#!/usr/bin/env python
"""Test startup script - simple version."""
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

print("=" * 80)
print("SYSTEM STARTUP TEST")
print("=" * 80)

try:
    print("\n[1] Importing InstitutionalLiveAgent...")
    from main import InstitutionalLiveAgent
    print("[OK] Import successful\n")

    print("[2] Creating agent instance...")
    agent = InstitutionalLiveAgent()
    print(f"[OK] Agent created with {len(agent.tickers)} tickers\n")

    print("[3] Agent cfg:")
    print(f"    - Execution mode: {agent.cfg['execution']['mode']}")
    print(f"    - Alpaca API Key exists: {bool(__import__('os').getenv('ALPACA_API_KEY'))}")
    print(f"    - Initial capital: ${agent.cfg['execution']['initial_capital']:,.0f}\n")

    print("[4] Starting agent.start()...")
    print("=" * 80)
    sys.stdout.flush()

    agent.start()

    print("\n[DONE] Agent.start() returned\n")

except KeyboardInterrupt:
    print("\n[INTERRUPTED]")
    sys.exit(0)
except Exception as e:
    print(f"\n[ERROR] {type(e).__name__}: {e}\n")
    import traceback
    traceback.print_exc()
    sys.exit(1)
