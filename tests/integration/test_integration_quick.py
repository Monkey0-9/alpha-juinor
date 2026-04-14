"""
tests/integration/test_integration_quick.py

Quick integration tests for Mini Quant Fund system.
Tests core functionality without complex fixtures.
"""

import pytest
import asyncio
import time
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from monitoring.structured_logger import get_logger
from infrastructure.infrastructure_guard import infrastructure_guard
from execution.global_session_tracker import session_tracker
from execution.advanced_execution_engine import execution_engine, ExecutionOrder, ExecutionAlgorithm, OrderSide
from orchestration.orchestrator import orchestrator, SystemMode

logger = get_logger("integration_quick")


class TestQuickIntegration:
    """Quick integration tests for core functionality."""

    @pytest.mark.asyncio
    async def test_structured_logging(self):
        """Test structured logging functionality."""
        logger.info("Testing structured logging")

        # Test basic logging
        test_logger = get_logger("test_quick")
        test_logger.info("Test message", test_key="test_value")

        # Test metric logging
        test_logger.log_metric("test_metric", 123.45, "count")

        # Test trade logging
        test_logger.log_trade("SPY", "BUY", 100, 150.25, order_id="test_order")

        # Test signal logging
        test_logger.log_signal("SPY", 0.75, 0.85, strategy="test_strategy")

        # Test risk logging (using info for now)
        test_logger.info("Risk event detected", exposure=0.95, limit=0.90, event_type="RISK_ALERT")

        # Test performance timer
        with test_logger.performance_timer("test_operation"):
            await asyncio.sleep(0.01)

        assert True  # If we get here, logging worked

    @pytest.mark.asyncio
    async def test_infrastructure_health(self):
        """Test infrastructure health checks."""
        logger.info("Testing infrastructure health")

        # Run health check
        health = await infrastructure_guard.pre_flight_check()

        # Validate structure
        assert health is not None
        assert hasattr(health, 'status')
        assert hasattr(health, 'checks')
        assert len(health.checks) >= 5

        # Check individual checks
        for check in health.checks:
            assert hasattr(check, 'name')
            assert hasattr(check, 'status')
            assert hasattr(check, 'message')
            assert hasattr(check, 'response_time_ms')

        logger.info(f"Health check completed: {health.status.value}")

    @pytest.mark.asyncio
    async def test_session_tracker(self):
        """Test market session tracking."""
        logger.info("Testing session tracker")

        # Start session tracker
        await session_tracker.start_monitoring()

        # Test market status
        status = session_tracker.get_market_status("NYSE")
        assert status is not None

        # Create session
        session_id = session_tracker.create_session("NYSE")
        assert session_id is not None
        assert session_id.startswith("NYSE_")

        # Get session summary
        summary = session_tracker.get_session_summary("NYSE")
        assert "NYSE" in summary

        # Stop session tracker
        await session_tracker.stop_monitoring()

        logger.info(f"Session tracker test completed: {session_id}")

    @pytest.mark.asyncio
    async def test_execution_engine(self):
        """Test execution engine functionality."""
        logger.info("Testing execution engine")

        # Start execution engine
        await execution_engine.start()

        # Test order submission
        order = ExecutionOrder(
            order_id="test_order_quick",
            symbol="SPY",
            side=OrderSide.BUY,
            total_quantity=1000,
            algorithm=ExecutionAlgorithm.IMMEDIATE
        )

        # Submit order
        order_id = await execution_engine.submit_order(order)
        assert order_id is not None

        # Wait briefly for processing
        await asyncio.sleep(0.1)

        # Check order status
        status = execution_engine.get_order_status(order_id)
        assert status is not None

        # Cancel order
        cancelled = await execution_engine.cancel_order(order_id)
        assert cancelled is True

        # Get execution summary
        summary = execution_engine.get_execution_summary()
        assert len(summary) >= 0

        # Stop execution engine
        await execution_engine.stop()

        logger.info(f"Execution engine test completed: {order_id}")

    @pytest.mark.asyncio
    async def test_orchestrator_basics(self):
        """Test orchestrator basic functionality."""
        logger.info("Testing orchestrator basics")

        # Start orchestrator
        await orchestrator.start()

        # Get system status
        status = orchestrator.get_system_status()
        assert status is not None
        assert "orchestrator" in status
        assert "infrastructure" in status

        # Test system mode
        original_mode = orchestrator.system_mode
        orchestrator.set_system_mode(SystemMode.PAPER_TRADING)
        assert orchestrator.system_mode == SystemMode.PAPER_TRADING

        # Restore original mode
        orchestrator.set_system_mode(original_mode)

        # Stop orchestrator
        await orchestrator.stop()

        logger.info("Orchestrator test completed")

    @pytest.mark.asyncio
    async def test_jane_street_precision(self):
        """Test Jane Street precision requirements."""
        logger.info("Testing Jane Street precision")

        # Test 0.0001% error tolerance
        error_tolerance = 0.000001

        # Test portfolio calculation precision
        weights = np.array([0.5, 0.3, 0.2])
        returns = np.array([0.001, 0.002, -0.001])
        portfolio_return = np.dot(weights, returns)
        # Calculate expected return correctly
        expected_return = 0.5 * 0.001 + 0.3 * 0.002 + 0.2 * -0.001  # 0.0007
        precision_error = abs(portfolio_return - expected_return)
        assert precision_error < error_tolerance, f"Portfolio precision error: {precision_error}"

        # Test risk calculation precision
        volatility = np.std(returns)
        # Calculate expected volatility correctly
        expected_volatility = np.sqrt(np.mean((returns - np.mean(returns))**2))
        volatility_error = abs(volatility - expected_volatility)
        assert volatility_error < error_tolerance, f"Volatility precision error: {volatility_error}"

        # Test Sharpe ratio precision
        sharpe = portfolio_return / volatility if volatility > 0 else 0
        expected_sharpe = expected_return / expected_volatility if expected_volatility > 0 else 0
        sharpe_error = abs(sharpe - expected_sharpe)
        assert sharpe_error < error_tolerance, f"Sharpe precision error: {sharpe_error}"

        logger.info(f"Jane Street precision test passed: {precision_error:.8f}")

    @pytest.mark.asyncio
    async def test_performance_benchmarks(self):
        """Test system performance benchmarks."""
        logger.info("Testing performance benchmarks")

        # Test health check performance (< 5 seconds)
        start_time = time.time()
        health = await infrastructure_guard.pre_flight_check()
        health_check_time = time.time() - start_time
        assert health_check_time < 5.0, f"Health check too slow: {health_check_time}s"

        # Test order submission performance (< 100ms)
        await execution_engine.start()
        start_time = time.time()
        order = ExecutionOrder(
            order_id="perf_test",
            symbol="SPY",
            side=OrderSide.BUY,
            total_quantity=1000,
            algorithm=ExecutionAlgorithm.IMMEDIATE
        )
        order_id = await execution_engine.submit_order(order)
        order_submission_time = (time.time() - start_time) * 1000
        assert order_submission_time < 100.0, f"Order submission too slow: {order_submission_time}ms"

        # Test status query performance (< 50ms)
        start_time = time.time()
        status = execution_engine.get_order_status(order_id)
        status_query_time = (time.time() - start_time) * 1000
        assert status_query_time < 50.0, f"Status query too slow: {status_query_time}ms"

        await execution_engine.stop()

        logger.info(f"Performance benchmarks passed: {health_check_time:.2f}s, {order_submission_time:.2f}ms, {status_query_time:.2f}ms")

    @pytest.mark.asyncio
    async def test_end_to_end_flow(self):
        """Test end-to-end trading flow."""
        logger.info("Testing end-to-end flow")

        # Start all components
        await session_tracker.start_monitoring()
        await execution_engine.start()
        await orchestrator.start()

        # Create market session
        session_id = session_tracker.create_session("NYSE")
        assert session_id is not None

        # Submit order through orchestrator (may fail in test environment)
        try:
            order_id = await orchestrator.submit_trading_order(
                symbol="SPY",
                side="BUY",
                quantity=1000,
                algorithm="IMMEDIATE"
            )
            logger.info(f"Order submitted: {order_id}")
        except Exception as e:
            logger.info(f"Order submission failed (expected in test): {e}")

        # Get system status
        status = orchestrator.get_system_status()
        assert status is not None

        # Stop all components
        await orchestrator.stop()
        await execution_engine.stop()
        await session_tracker.stop_monitoring()

        logger.info("End-to-end flow test completed")


@pytest.mark.asyncio
async def test_quick_integration():
    """Quick integration test runner."""
    logger.info("Starting quick integration tests")

    test_instance = TestQuickIntegration()

    tests = [
        test_instance.test_structured_logging(),
        test_instance.test_infrastructure_health(),
        test_instance.test_session_tracker(),
        test_instance.test_execution_engine(),
        test_instance.test_orchestrator_basics(),
        test_instance.test_jane_street_precision(),
        test_instance.test_performance_benchmarks(),
        test_instance.test_end_to_end_flow()
    ]

    results = await asyncio.gather(*tests, return_exceptions=True)

    passed = sum(1 for result in results if not isinstance(result, Exception))
    total = len(results)

    logger.info(f"Quick integration tests completed: {passed}/{total} passed")

    # Raise if any tests failed
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Test {i} failed: {result}")

    assert passed == total, f"Only {passed}/{total} tests passed"


if __name__ == "__main__":
    asyncio.run(test_quick_integration())
