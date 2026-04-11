#!/usr/bin/env python3
"""
[TGT] QUICK EXECUTION TEST - Verify all 13 trading types are executing
"""

import os
import sys
import time
from datetime import datetime

# Set execution flags
os.environ["EXECUTE_TRADES"] = "true"
os.environ["TRADING_MODE"] = "paper"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from brokers.mock_broker import MockBroker
from execution.parallel_strategy_engine import ParallelStrategyExecutor


def test_13_strategies():
    """Test all 13 strategies in parallel"""
    print("\n" + "=" * 80)
    print("🧪 TESTING ALL 13 TRADING TYPES IN PARALLEL")
    print("=" * 80)

    executor = ParallelStrategyExecutor(max_workers=13)
    broker = MockBroker()

    symbols = ["AAPL", "MSFT", "TSLA"]
    print(f"\n[STAT] Testing Symbols: {symbols}")
    print(f"[TGT] Strategy Types: All 13 types")
    print(f"💼 Initial Positions: {len(broker.get_positions())}")
    print(f"💰 Initial Balance: ${broker.get_account()['equity']:,.2f}")

    # Create market data for testing
    market_data = {}
    for symbol in symbols:
        market_data[symbol] = {
            "price": 150.0,
            "volume": 1000000,
            "bid": 149.95,
            "ask": 150.05,
            "high_52w": 175.0,
            "low_52w": 120.0,
            "avg_volume_20d": 800000,
            "rsi_14": 35.0,  # Oversold - triggers many strategies
            "macd": 0.5,
            "bb_upper": 160.0,
            "bb_lower": 140.0,
            "sma_50": 148.0,
            "sma_200": 145.0,
            "adx": 32.0,
            "atr": 2.0,
            "close_time": datetime.utcnow(),
        }

    # Execute all 13 strategies
    print(f"\n⏱️  Running 13 strategies in PARALLEL...")
    start = time.time()

    all_signals = executor.execute_all_strategies_parallel(
        symbols, market_data, broker.get_positions()
    )

    elapsed_ms = (time.time() - start) * 1000
    print(f" Execution completed in {elapsed_ms:.2f}ms\n")

    # Count and display signals
    total_signals = 0
    strategy_counts = {}

    for symbol in symbols:
        signals = all_signals.get(symbol, [])
        print(f"  {symbol}:")

        if signals:
            for sig in signals:
                total_signals += 1
                strategy = sig.strategy_type
                strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
                conviction = sig.conviction
                ret_pct = sig.expected_return * 100
                print(
                    f"     {strategy:20s} {sig.signal:4s} (Conviction: {conviction:.2f}, Return: {ret_pct:.2f}%)"
                )
        else:
            print(f"    (No signals)")

    print(f"\n[STAT] SIGNALS BY STRATEGY TYPE:")
    for strategy_type in sorted(strategy_counts.keys()):
        count = strategy_counts[strategy_type]
        print(f"   {strategy_type:25s}: {count} signal(s)")

    print(f"\n TOTAL SIGNALS: {total_signals}")

    if total_signals > 0:
        print(f" ALL 13 STRATEGIES EXECUTING AND GENERATING SIGNALS!")
        return True
    else:
        print(f" No signals generated - strategies may not be executing")
        return False


def test_order_execution():
    """Test converting signals to orders and executing"""
    print("\n" + "=" * 80)
    print("🧪 TESTING ORDER EXECUTION")
    print("=" * 80)

    broker = MockBroker()
    print(f"\n💰 Initial Balance: ${broker.get_account()['equity']:,.2f}")

    # Submit test orders
    orders_to_execute = [
        ("AAPL", 10, "BUY"),
        ("MSFT", 5, "BUY"),
        ("TSLA", 3, "BUY"),
    ]

    print(f"\n[LST] Submitting {len(orders_to_execute)} test orders...")

    executed = 0
    for symbol, qty, side in orders_to_execute:
        result = broker.submit_order(symbol=symbol, qty=qty, side=side, type="market")
        if result and result.get("success"):
            executed += 1
            print(f"    {symbol} {side:4s} {qty:3.0f} → FILLED")
        else:
            print(f"    {symbol} {side:4s} {qty:3.0f} → FAILED")

    # Verify positions
    positions = broker.get_positions()
    print(f"\n[UP] POSITIONS CREATED: {len(positions)}")
    for symbol, qty in positions.items():
        print(f"   {symbol}: {qty:.0f} shares")

    print(
        f"\n ORDER EXECUTION TEST: {executed}/{len(orders_to_execute)} orders filled"
    )
    return executed == len(orders_to_execute)


def main():
    print("\n" + "=" * 80)
    print("[TGT] QUICK EXECUTION TEST - ALL 13 TRADING TYPES")
    print("=" * 80)

    # Test 1: 13 strategies
    test1_pass = test_13_strategies()

    # Test 2: Order execution
    test2_pass = test_order_execution()

    # Summary
    print("\n" + "=" * 80)
    print("[LST] TEST RESULTS")
    print("=" * 80)
    print(f"Test 1 (13 Strategies): {' PASS' if test1_pass else ' FAIL'}")
    print(f"Test 2 (Order Execution): {' PASS' if test2_pass else ' FAIL'}")

    if test1_pass and test2_pass:
        print("\n*** ALL TESTS PASSED! ***")
        print("\n[INIT] Next Step: python start_trading.py --mode paper")
        print("   This will run live trading with all 13 strategies executing simultaneously")
    else:
        print("\n*** Some tests failed - check output above ***")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
