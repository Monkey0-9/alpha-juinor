#!/usr/bin/env python3
"""
Test script to verify trading execution is working.

This script:
1. Checks all prerequisites
2. Simulates a trading cycle
3. Verifies signals are being converted to orders
4. Confirms orders are being executed
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("TRADING_TEST")


def test_environment_setup():
    """Test 1: Environment setup"""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 1: Environment Setup")
    logger.info("=" * 70)

    checks = {
        "EXECUTE_TRADES enabled": os.environ.get("EXECUTE_TRADES", "").lower()
        == "true",
        "Runtime directory exists": Path("runtime").exists(),
        "Logs directory exists": Path("logs").exists(),
        "Database exists": Path("runtime/institutional_trading.db").exists(),
        "No kill switch blocking": not Path("runtime/KILL_SWITCH").exists(),
    }

    passed = 0
    for check, result in checks.items():
        status = " PASS" if result else " FAIL"
        logger.info(f"  {status}: {check}")
        if result:
            passed += 1

    logger.info(f"\n  Result: {passed}/{len(checks)} checks passed")
    return passed == len(checks)


def test_broker_compatibility():
    """Test 2: Broker compatibility"""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 2: Broker Compatibility")
    logger.info("=" * 70)

    try:
        from mini_quant_fund.brokers.alpaca_broker import AlpacaExecutionHandler
        from mini_quant_fund.brokers.mock_broker import MockBroker

        # Test MockBroker
        logger.info("  Testing MockBroker...")
        mock = MockBroker()

        has_submit_order = hasattr(mock, "submit_order")
        has_get_positions = hasattr(mock, "get_positions")

        logger.info(f"     MockBroker.submit_order: {has_submit_order}")
        logger.info(f"     MockBroker.get_positions: {has_get_positions}")

        if has_submit_order and has_get_positions:
            logger.info("   PASS: MockBroker compatible")
            return True
        else:
            logger.error("   FAIL: MockBroker missing methods")
            return False

    except Exception as e:
        logger.error(f"   FAIL: Broker test error: {e}")
        return False


def test_execution_methods():
    """Test 3: Execution methods exist"""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 3: Execution Methods")
    logger.info("=" * 70)

    try:
        from mini_quant_fund.orchestration.live_decision_loop import LiveDecisionLoop

        # Check methods exist
        methods = {
            "_process_signals_to_orders": hasattr(
                LiveDecisionLoop, "_process_signals_to_orders"
            ),
            "_execute_orders": hasattr(LiveDecisionLoop, "_execute_orders"),
            "_compute_signals": hasattr(LiveDecisionLoop, "_compute_signals"),
            "_run_decision_tick": hasattr(LiveDecisionLoop, "_run_decision_tick"),
        }

        all_present = True
        for method_name, present in methods.items():
            status = "" if present else ""
            logger.info(f"  {status} {method_name}: {present}")
            if not present:
                all_present = False

        if all_present:
            logger.info("   PASS: All execution methods present")
        else:
            logger.error("   FAIL: Missing execution methods")

        return all_present

    except Exception as e:
        logger.error(f"   FAIL: Method check error: {e}")
        return False


def test_signal_to_order_conversion():
    """Test 4: Signal to order conversion"""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 4: Signal-to-Order Conversion")
    logger.info("=" * 70)

    try:
        from mini_quant_fund.orchestration.live_decision_loop import LiveDecisionLoop, LiveSignal

        # Create test loop
        loop = LiveDecisionLoop(
            tick_interval=1.0,
            paper_mode=True,
            market_hours_only=False,
            symbols=["AAPL", "MSFT", "TSLA"],
        )

        # Create test signals
        test_signals = {
            "AAPL": LiveSignal(
                symbol="AAPL",
                signal="EXECUTE_BUY",
                mu_hat=0.05,
                sigma_hat=0.02,
                conviction=0.85,
                current_price=150.00,
            ),
            "MSFT": LiveSignal(
                symbol="MSFT",
                signal="EXECUTE_SELL",
                mu_hat=-0.02,
                sigma_hat=0.015,
                conviction=0.70,
                current_price=320.00,
            ),
            "TSLA": LiveSignal(
                symbol="TSLA", signal="HOLD", conviction=0.5, current_price=180.00
            ),
        }

        # Process signals to orders
        orders = loop._process_signals_to_orders(test_signals)

        logger.info(
            f"  Generated {len(orders)} orders from {len(test_signals)} signals"
        )

        # Verify orders
        expected_orders = 2  # AAPL BUY and MSFT SELL (TSLA is HOLD)

        if len(orders) == expected_orders:
            for order in orders:
                logger.info(
                    f"     {order['side']} {order['quantity']} {order['symbol']} @ ${order['price']:.2f}"
                )
            logger.info(f"   PASS: Correct number of orders generated")
            return True
        else:
            logger.error(
                f"   FAIL: Expected {expected_orders} orders, got {len(orders)}"
            )
            return False

    except Exception as e:
        logger.error(f"   FAIL: Signal conversion error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_order_execution():
    """Test 5: Order execution"""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 5: Order Execution")
    logger.info("=" * 70)

    try:
        from mini_quant_fund.orchestration.live_decision_loop import LiveDecisionLoop

        # Create test loop
        loop = LiveDecisionLoop(
            tick_interval=1.0,
            paper_mode=True,
            market_hours_only=False,
            symbols=["AAPL"],
        )

        # Create test orders
        test_orders = [
            {
                "symbol": "AAPL",
                "side": "BUY",
                "quantity": 10.0,
                "price": 150.00,
                "order_type": "MARKET",
                "conviction": 0.85,
            }
        ]

        # Execute orders
        result = loop._execute_orders(test_orders)

        logger.info(f"  Execution result:")
        logger.info(f"    Executed: {result['executed']}")
        logger.info(f"    Failed: {result['failed']}")
        logger.info(f"    Errors: {result['error_count']}")

        if result["executed"] > 0:
            logger.info(f"   PASS: Orders executed successfully")
            return True
        else:
            logger.error(f"   FAIL: No orders were executed")
            return False

    except Exception as e:
        logger.error(f"   FAIL: Order execution error: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    logger.info("\n")
    logger.info("╔════════════════════════════════════════════════════════════════╗")
    logger.info("║                                                                ║")
    logger.info("║              TRADING SYSTEM VERIFICATION TEST SUITE            ║")
    logger.info("║                                                                ║")
    logger.info("╚════════════════════════════════════════════════════════════════╝")

    # Run all tests
    results = {
        "1. Environment Setup": test_environment_setup(),
        "2. Broker Compatibility": test_broker_compatibility(),
        "3. Execution Methods": test_execution_methods(),
        "4. Signal-to-Order Conversion": test_signal_to_order_conversion(),
        "5. Order Execution": test_order_execution(),
    }

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, result in results.items():
        status = " PASS" if result else " FAIL"
        logger.info(f"  {status}: {test_name}")

    logger.info(f"\n  Overall: {passed}/{total} tests passed")

    if passed == total:
        logger.info("\n" + "🎉 " * 10)
        logger.info("ALL TESTS PASSED - TRADING SYSTEM IS OPERATIONAL! [INIT]")
        logger.info("🎉 " * 10)
        logger.info("\nNext step: Run 'python start_trading.py --mode paper'")
        return 0
    else:
        logger.error("\n Some tests failed. Check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
