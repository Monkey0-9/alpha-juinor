import os
import sys
from decimal import Decimal

import numpy as np
import pandas as pd

# Add project root to path
sys.path.append(os.getcwd())

from execution.trade_manager import ExitReason, get_trade_manager
from risk.portfolio_guardian import PortfolioGuardian


def test_portfolio_guardian_limits():
    """Verify that PortfolioGuardian enforces position limits."""
    print("--- Testing PortfolioGuardian Limits ---")
    guardian = PortfolioGuardian(max_ticker_weight=0.10)

    # Test 1: Within limit
    res1 = guardian.check_new_trade("AAPL", None, [], current_weight=0.05)
    print(f"Check AAPL 5% (Allowed 10%): {'PASS' if res1 else 'FAIL'}")

    # Test 2: Over limit
    res2 = guardian.check_new_trade("GOOG", None, ["AAPL"], current_weight=0.15)
    print(f"Check GOOG 15% (Allowed 10%): {'PASS' if not res2 else 'FAIL'}")

def test_trade_manager_exits():
    """Verify that TradeManager triggers exits based on price."""
    print("\n--- Testing TradeManager Exits ---")
    tm = get_trade_manager()

    symbol = "TEST"
    entry_price = Decimal("100.0")
    stop_loss = Decimal("95.0")
    take_profit = Decimal("105.0")

    trade_id = tm.open_trade(
        symbol=symbol,
        side="LONG",
        entry_price=entry_price,
        quantity=10,
        stop_loss=stop_loss,
        take_profit_1=take_profit
    )

    # Slide 1: Price at entry
    tm.update_prices({symbol: entry_price})
    exits = tm.check_exits()
    print(f"Price at {entry_price}: {len(exits)} exits triggered (Expected 0)")

    # Slide 2: Price hits stop loss
    tm.update_prices({symbol: stop_loss})
    exits = tm.check_exits()
    print(f"Price at {stop_loss}: {len(exits)} exits triggered (Expected 1)")
    if exits:
        print(f"Exit Reason: {exits[0].reason} (Expected {ExitReason.STOP_LOSS})")

if __name__ == "__main__":
    test_portfolio_guardian_limits()
    test_trade_manager_exits()
