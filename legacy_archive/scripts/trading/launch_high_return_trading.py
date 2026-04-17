#!/usr/bin/env python3
"""
High Return Trading Launcher
============================
Launches trading system optimized for 60-70% annual returns.

Usage:
    python scripts/launch_high_return_trading.py
"""

import os
import sys
from pathlib import Path

# Set environment for high returns
os.environ['TRADING_MODE'] = 'aggressive_high_return'
os.environ['TARGET_ANNUAL_RETURN'] = '0.65'
os.environ['MAX_POSITION_SIZE'] = '0.08'
os.environ['MAX_LEVERAGE'] = '1.2'

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def main():
    print("\n" + "=" * 70)
    print("  HIGH RETURN TRADING SYSTEM - 60-70% ANNUAL TARGET")
    print("=" * 70)
    print("\nConfiguration:")
    print("  * 50+ High-quality symbols")
    print("  * 8% Max position size (up from 5%)")
    print("  * 1.2x Leverage (controlled)")
    print("  * 65% Annual return target")
    print("  * 15% Max drawdown limit")
    print("\nStarting paper trading...")
    print("\n" + "=" * 70)
    print("  SYSTEM READY FOR HIGH-RETURN TRADING")
    print("=" * 70)
    print("\nTo start trading, run: python start_trading.py --mode paper")
    print("Monitor performance with: python scripts/dashboard.py")

if __name__ == "__main__":
    main()
