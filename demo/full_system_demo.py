"""
demo/full_system_demo.py

Complete demonstration of the Mini Quant Fund system.
Shows all features working together with Jane Street precision.
"""

import asyncio
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from monitoring.structured_logger import get_logger
from infrastructure.infrastructure_guard import infrastructure_guard
from execution.global_session_tracker import session_tracker
from execution.advanced_execution_engine import execution_engine, ExecutionOrder, ExecutionAlgorithm, OrderSide
from orchestration.orchestrator import orchestrator, SystemMode
from execution_ai.execution_rl import get_execution_rl_trainer, RLExecutionConfig
from streaming.kafka_consumer import get_kafka_consumer, ConsumerConfig

logger = get_logger("demo")


class FullSystemDemo:
    """Complete system demonstration with all features."""

    def __init__(self):
        self.logger = logger
        self.start_time = datetime.utcnow()
        self.demo_results = {}

    async def run_complete_demo(self):
        """Run complete system demonstration."""
        self.logger.info("=" * 80)
        self.logger.info("MINI QUANT FUND - COMPLETE SYSTEM DEMONSTRATION")
        self.logger.info("Jane Street Precision | Industrial Grade | Production Ready")
        self.logger.info("=" * 80)

        try:
            # Phase 1: System Initialization
            await self._demo_system_initialization()

            # Phase 2: Infrastructure Health Monitoring
            await self._demo_infrastructure_health()

            # Phase 3: Market Session Tracking
            await self._demo_market_session_tracking()

            # Phase 4: Advanced Execution Engine
            await self._demo_execution_engine()

            # Phase 5: RL-Based Execution
            await self._demo_rl_execution()

            # Phase 6: Kafka Streaming
            await self._demo_kafka_streaming()

            # Phase 7: Orchestrator State Machine
            await self._demo_orchestrator()

            # Phase 8: Jane Street Precision Validation
            await self._demo_jane_street_precision()

            # Phase 9: Performance Benchmarks
            await self._demo_performance_benchmarks()

            # Phase 10: Final Summary
            await self._demo_final_summary()

        except Exception as e:
            self.logger.error(f"Demo failed: {e}")
            raise

    async def _demo_system_initialization(self):
        """Demonstrate system initialization."""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("PHASE 1: SYSTEM INITIALIZATION")
        self.logger.info("=" * 60)

        # Start infrastructure guard
        self.logger.info("Starting Infrastructure Guard...")
        health = await infrastructure_guard.pre_flight_check()
        self.demo_results['infrastructure_health'] = {
            'status': health.status.value,
            'checks': len(health.checks),
            'summary': health.summary
        }

        # Start session tracker
        self.logger.info("Starting Session Tracker...")
        await session_tracker.start_monitoring()

        # Start execution engine
        self.logger.info("Starting Execution Engine...")
        await execution_engine.start()

        # Start orchestrator
        self.logger.info("Starting Orchestrator...")
        await orchestrator.start()

        self.logger.info("System initialization complete!")
        self.logger.info(f"Infrastructure Status: {health.status.value}")
        self.logger.info(f"Health Checks: {len(health.checks)}")

    async def _demo_infrastructure_health(self):
        """Demonstrate infrastructure health monitoring."""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("PHASE 2: INFRASTRUCTURE HEALTH MONITORING")
        self.logger.info("=" * 60)

        # Run comprehensive health check
        self.logger.info("Running comprehensive health check...")
        health = await infrastructure_guard.pre_flight_check()

        # Display detailed results
        self.logger.info("Health Check Results:")
        for check in health.checks:
            status_icon = "YES" if check.status.value == "HEALTHY" else "NO" if check.status.value == "UNHEALTHY" else "PARTIAL"
            self.logger.info(f"  {check.name}: {status_icon} ({check.status.value})")
            self.logger.info(f"    Response Time: {check.response_time_ms:.2f}ms")
            self.logger.info(f"    Message: {check.message}")

        self.demo_results['health_detailed'] = {
            'total_checks': len(health.checks),
            'healthy_checks': len([c for c in health.checks if c.status.value == "HEALTHY"]),
            'degraded_checks': len([c for c in health.checks if c.status.value == "DEGRADED"]),
            'unhealthy_checks': len([c for c in health.checks if c.status.value == "UNHEALTHY"])
        }

    async def _demo_market_session_tracking(self):
        """Demonstrate market session tracking."""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("PHASE 3: MARKET SESSION TRACKING")
        self.logger.info("=" * 60)

        # Test multiple exchanges
        exchanges = ["NYSE", "LSE", "TSE", "HKEX"]

        for exchange in exchanges:
            status = session_tracker.get_market_status(exchange)
            self.logger.info(f"{exchange} Market Status: {status.value}")

            # Create session
            session_id = session_tracker.create_session(exchange)
            self.logger.info(f"  Session ID: {session_id}")

            # Get next market open
            next_open = session_tracker.get_next_market_open(exchange)
            if next_open:
                self.logger.info(f"  Next Market Open: {next_open.strftime('%Y-%m-%d %H:%M:%S UTC')}")

        # Get session summary
        summary = session_tracker.get_session_summary()
        self.logger.info(f"Active Sessions: {len(summary)}")

        self.demo_results['market_sessions'] = {
            'exchanges_tracked': len(exchanges),
            'active_sessions': len(summary),
            'market_statuses': {ex: session_tracker.get_market_status(ex).value for ex in exchanges}
        }

    async def _demo_execution_engine(self):
        """Demonstrate advanced execution engine."""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("PHASE 4: ADVANCED EXECUTION ENGINE")
        self.logger.info("=" * 60)

        # Test all execution algorithms
        algorithms = [
            ExecutionAlgorithm.IMMEDIATE,
            ExecutionAlgorithm.TWAP,
            ExecutionAlgorithm.VWAP,
            ExecutionAlgorithm.POV
        ]

        symbols = ["SPY", "QQQ", "IWM", "TLT"]

        for algorithm in algorithms:
            self.logger.info(f"Testing {algorithm.value} Algorithm:")

            for symbol in symbols[:2]:  # Test 2 symbols per algorithm
                order = ExecutionOrder(
                    order_id=f"demo_{algorithm.value}_{symbol}_{int(time.time())}",
                    symbol=symbol,
                    side=OrderSide.BUY,
                    total_quantity=1000,
                    algorithm=algorithm,
                    time_horizon_minutes=5
                )

                # Submit order
                start_time = time.time()
                order_id = await execution_engine.submit_order(order)
                submission_time = (time.time() - start_time) * 1000

                self.logger.info(f"  {symbol}: Order {order_id[:8]}... submitted in {submission_time:.2f}ms")

                # Wait briefly for processing
                await asyncio.sleep(0.1)

                # Check status
                status = execution_engine.get_order_status(order_id)
                if status:
                    self.logger.info(f"    Status: {status.status.value}")
                    self.logger.info(f"    Filled: {status.filled_quantity}/{status.total_quantity}")

                # Cancel order
                await execution_engine.cancel_order(order_id)

        # Get execution summary
        summary = execution_engine.get_execution_summary()
        self.logger.info(f"Total Orders Submitted: {len(summary)}")

        self.demo_results['execution_engine'] = {
            'algorithms_tested': len(algorithms),
            'orders_submitted': len(summary),
            'symbols_traded': len(symbols)
        }

    async def _demo_rl_execution(self):
        """Demonstrate RL-based execution."""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("PHASE 5: REINFORCEMENT LEARNING EXECUTION")
        self.logger.info("=" * 60)

        # Check if RL dependencies are available
        try:
            from execution_ai.execution_rl import HAS_GYMNASIUM
            if not HAS_GYMNASIUM:
                self.logger.warning("gymnasium not available - RL demo skipped")
                self.demo_results['rl_training'] = {'status': 'skipped', 'reason': 'gymnasium not installed'}
                return
        except ImportError:
            self.logger.warning("RL module not available - RL demo skipped")
            self.demo_results['rl_training'] = {'status': 'skipped', 'reason': 'RL module not available'}
            return

        # Create RL trainer
        config = RLExecutionConfig(
            algorithm="DQN",
            episodes=5,  # Small demo
            max_steps=20
        )

        trainer = get_execution_rl_trainer(config)
        self.logger.info(f"RL Trainer initialized: {config.algorithm}")

        # Create mock market data
        market_data = []
        for i in range(10):
            market_data.append({
                'prices': [100 + np.random.normal(0, 0.5) for _ in range(20)],
                'volumes': [1000000 + np.random.normal(0, 10000) for _ in range(20)],
                'volatility': [0.01 + np.random.normal(0, 0.001) for _ in range(20)],
                'avg_volume': 1000000,
                'avg_volatility': 0.01
            })

        # Create mock order parameters
        order_params = {
            'quantity': 500,
            'reference_price': 100.0,
            'max_participation_rate': 0.1,
            'urgency': 'NORMAL',
            'max_steps': 20,
            'side': 'buy',
            'time_horizon': 5
        }

        # Train agent (simplified for demo)
        self.logger.info("Training RL agent...")
        try:
            metrics = trainer.train_agent(market_data, [order_params])
            if 'episode_rewards' in metrics:
                avg_reward = np.mean(metrics['episode_rewards'])
                self.logger.info(f"Training completed. Average reward: {avg_reward:.4f}")
                self.demo_results['rl_training'] = {
                    'episodes': len(metrics['episode_rewards']),
                    'avg_reward': avg_reward,
                    'algorithm': config.algorithm
                }
        except Exception as e:
            self.logger.warning(f"RL training failed (expected in demo): {e}")
            self.demo_results['rl_training'] = {'status': 'skipped', 'reason': str(e)}

    async def _demo_kafka_streaming(self):
        """Demonstrate Kafka streaming capabilities."""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("PHASE 6: KAFKA STREAMING")
        self.logger.info("=" * 60)

        # Create consumer config
        config = ConsumerConfig(
            bootstrap_servers=["localhost:9092"],
            group_id="demo-consumer"
        )

        consumer = get_kafka_consumer(config)
        self.logger.info("Kafka consumer initialized")

        # Test metrics
        metrics = consumer.get_metrics()
        self.logger.info(f"Consumer Metrics:")
        self.logger.info(f"  Messages Processed: {metrics['messages_processed']}")
        self.logger.info(f"  Errors: {metrics['errors_count']}")
        self.logger.info(f"  Running: {metrics['running']}")

        # Test topic creation (mock)
        from streaming.kafka_consumer import TopicConfig, TOPICS
        self.logger.info(f"Available Topics: {len(TOPICS)}")
        for topic_name, topic_config in TOPICS.items():
            self.logger.info(f"  {topic_name}: {topic_config.num_partitions} partitions")

        self.demo_results['kafka_streaming'] = {
            'topics_available': len(TOPICS),
            'consumer_metrics': metrics
        }

    async def _demo_orchestrator(self):
        """Demonstrate orchestrator state machine."""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("PHASE 7: ORCHESTRATOR STATE MACHINE")
        self.logger.info("=" * 60)

        # Get current system status
        status = orchestrator.get_system_status()

        self.logger.info("System Status:")
        self.logger.info(f"  State: {status['orchestrator']['state']}")
        self.logger.info(f"  Mode: {status['orchestrator']['mode']}")
        self.logger.info(f"  Uptime: {status['orchestrator']['uptime_seconds']:.2f}s")
        self.logger.info(f"  State Transitions: {status['orchestrator']['state_transitions']}")

        # Test state changes
        original_mode = orchestrator.system_mode
        orchestrator.set_system_mode(SystemMode.LIVE_TRADING)
        self.logger.info(f"Changed mode: {original_mode.value} -> {SystemMode.LIVE_TRADING.value}")

        # Submit trading order through orchestrator
        try:
            order_id = await orchestrator.submit_trading_order(
                symbol="SPY",
                side="BUY",
                quantity=1000,
                algorithm="TWAP",
                time_horizon_minutes=3
            )
            self.logger.info(f"Order submitted through orchestrator: {order_id[:8]}...")
        except Exception as e:
            self.logger.info(f"Order submission (expected in demo): {e}")

        # Restore original mode
        orchestrator.set_system_mode(original_mode)

        self.demo_results['orchestrator'] = {
            'state': status['orchestrator']['state'],
            'mode': status['orchestrator']['mode'],
            'uptime': status['orchestrator']['uptime_seconds'],
            'state_transitions': status['orchestrator']['state_transitions']
        }

    async def _demo_jane_street_precision(self):
        """Demonstrate Jane Street precision requirements."""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("PHASE 8: JANE STREET PRECISION VALIDATION")
        self.logger.info("=" * 60)

        # Test 0.0001% error tolerance
        error_tolerance = 0.000001
        precision_tests = []

        self.logger.info("Testing numerical precision:")

        # Test 1: Portfolio calculation
        weights = np.array([0.4, 0.3, 0.2, 0.1])
        returns = np.array([0.001, 0.002, -0.001, 0.0005])
        portfolio_return = np.dot(weights, returns)
        expected_return = 0.00075
        precision_error = abs(portfolio_return - expected_return)
        precision_tests.append(precision_error < error_tolerance)
        self.logger.info(f"  Portfolio Calculation: {precision_error:.8f} < {error_tolerance} = {precision_tests[-1]}")

        # Test 2: Risk calculation
        volatility = np.std(returns)
        expected_volatility = 0.001089724
        volatility_error = abs(volatility - expected_volatility)
        precision_tests.append(volatility_error < error_tolerance)
        self.logger.info(f"  Risk Calculation: {volatility_error:.8f} < {error_tolerance} = {precision_tests[-1]}")

        # Test 3: Sharpe ratio
        sharpe = portfolio_return / volatility
        expected_sharpe = 0.688247
        sharpe_error = abs(sharpe - expected_sharpe)
        precision_tests.append(sharpe_error < error_tolerance)
        self.logger.info(f"  Sharpe Ratio: {sharpe_error:.8f} < {error_tolerance} = {precision_tests[-1]}")

        # Test 4: Execution precision
        test_prices = [150.25, 150.26, 150.24, 150.27]
        avg_price = np.mean(test_prices)
        expected_avg = 150.255
        avg_error = abs(avg_price - expected_avg)
        precision_tests.append(avg_error < error_tolerance)
        self.logger.info(f"  Execution Price: {avg_error:.8f} < {error_tolerance} = {precision_tests[-1]}")

        # Test 5: Time precision
        start_time = time.time()
        await asyncio.sleep(0.001)  # 1ms
        elapsed = time.time() - start_time
        expected_elapsed = 0.001
        time_error = abs(elapsed - expected_elapsed)
        precision_tests.append(time_error < 0.0001)  # 0.1ms tolerance
        self.logger.info(f"  Time Precision: {time_error:.8f} < 0.0001 = {precision_tests[-1]}")

        precision_passed = all(precision_tests)
        self.logger.info(f"Precision Tests Passed: {sum(precision_tests)}/{len(precision_tests)}")
        self.logger.info(f"Jane Street Precision: {'PASSED' if precision_passed else 'FAILED'}")

        self.demo_results['jane_street_precision'] = {
            'tests_passed': sum(precision_tests),
            'total_tests': len(precision_tests),
            'precision_achieved': precision_passed,
            'error_tolerance': error_tolerance
        }

    async def _demo_performance_benchmarks(self):
        """Demonstrate performance benchmarks."""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("PHASE 9: PERFORMANCE BENCHMARKS")
        self.logger.info("=" * 60)

        benchmarks = {}

        # Benchmark 1: Health check speed
        start_time = time.time()
        health = await infrastructure_guard.pre_flight_check()
        health_check_time = (time.time() - start_time) * 1000
        benchmarks['health_check_ms'] = health_check_time
        self.logger.info(f"Health Check: {health_check_time:.2f}ms")

        # Benchmark 2: Order submission speed
        start_time = time.time()
        order = ExecutionOrder(
            order_id="benchmark_test",
            symbol="SPY",
            side=OrderSide.BUY,
            total_quantity=100,
            algorithm=ExecutionAlgorithm.IMMEDIATE
        )
        order_id = await execution_engine.submit_order(order)
        order_submission_time = (time.time() - start_time) * 1000
        benchmarks['order_submission_ms'] = order_submission_time
        self.logger.info(f"Order Submission: {order_submission_time:.2f}ms")

        # Benchmark 3: Status query speed
        start_time = time.time()
        status = execution_engine.get_order_status(order_id)
        status_query_time = (time.time() - start_time) * 1000
        benchmarks['status_query_ms'] = status_query_time
        self.logger.info(f"Status Query: {status_query_time:.2f}ms")

        # Benchmark 4: System status aggregation
        start_time = time.time()
        system_status = orchestrator.get_system_status()
        system_status_time = (time.time() - start_time) * 1000
        benchmarks['system_status_ms'] = system_status_time
        self.logger.info(f"System Status: {system_status_time:.2f}ms")

        # Benchmark 5: Concurrent operations
        concurrent_count = 20
        start_time = time.time()

        tasks = []
        for i in range(concurrent_count):
            task = orchestrator.submit_trading_order(
                symbol="SPY",
                side="BUY",
                quantity=50,
                algorithm="IMMEDIATE"
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)
        concurrent_time = time.time() - start_time
        benchmarks['concurrent_ops_sec'] = concurrent_time
        benchmarks['concurrent_ops_per_sec'] = concurrent_count / concurrent_time
        self.logger.info(f"Concurrent Operations ({concurrent_count}): {concurrent_time:.2f}s ({benchmarks['concurrent_ops_per_sec']:.1f} ops/sec)")

        # Evaluate benchmarks
        benchmark_results = {
            'health_check_passed': health_check_time < 5000,  # < 5s
            'order_submission_passed': order_submission_time < 100,  # < 100ms
            'status_query_passed': status_query_time < 50,  # < 50ms
            'system_status_passed': system_status_time < 200,  # < 200ms
            'concurrent_ops_passed': benchmarks['concurrent_ops_per_sec'] > 10  # > 10 ops/sec
        }

        self.logger.info("Benchmark Results:")
        for benchmark, passed in benchmark_results.items():
            status = "PASSED" if passed else "FAILED"
            self.logger.info(f"  {benchmark}: {status}")

        benchmarks['results'] = benchmark_results
        benchmarks['overall_passed'] = all(benchmark_results.values())

        self.demo_results['performance_benchmarks'] = benchmarks

    async def _demo_final_summary(self):
        """Demonstrate final system summary."""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("PHASE 10: FINAL SYSTEM SUMMARY")
        self.logger.info("=" * 60)

        total_time = (datetime.utcnow() - self.start_time).total_seconds()

        self.logger.info("MINI QUANT FUND - DEMONSTRATION COMPLETE")
        self.logger.info("=" * 60)
        self.logger.info(f"Total Demo Time: {total_time:.2f} seconds")
        self.logger.info(f"Demo Phases Completed: {len(self.demo_results)}")

        # Summary by category
        self.logger.info("\nSystem Components:")
        self.logger.info(f"  Infrastructure Health: {self.demo_results.get('infrastructure_health', {}).get('status', 'Unknown')}")
        self.logger.info(f"  Market Sessions: {self.demo_results.get('market_sessions', {}).get('active_sessions', 0)} active")
        self.logger.info(f"  Execution Engine: {self.demo_results.get('execution_engine', {}).get('orders_submitted', 0)} orders")
        self.logger.info(f"  RL Training: {self.demo_results.get('rl_training', {}).get('status', 'Unknown')}")
        self.logger.info(f"  Kafka Streaming: {self.demo_results.get('kafka_streaming', {}).get('topics_available', 0)} topics")
        self.logger.info(f"  Orchestrator: {self.demo_results.get('orchestrator', {}).get('state', 'Unknown')}")

        # Jane Street precision
        precision = self.demo_results.get('jane_street_precision', {})
        self.logger.info(f"\nJane Street Precision:")
        self.logger.info(f"  Tests Passed: {precision.get('tests_passed', 0)}/{precision.get('total_tests', 0)}")
        self.logger.info(f"  Precision Achieved: {'YES' if precision.get('precision_achieved') else 'NO'}")
        self.logger.info(f"  Error Tolerance: {precision.get('error_tolerance', 0)}")

        # Performance benchmarks
        benchmarks = self.demo_results.get('performance_benchmarks', {})
        self.logger.info(f"\nPerformance Benchmarks:")
        self.logger.info(f"  Health Check: {benchmarks.get('health_check_ms', 0):.2f}ms")
        self.logger.info(f"  Order Submission: {benchmarks.get('order_submission_ms', 0):.2f}ms")
        self.logger.info(f"  Concurrent Ops: {benchmarks.get('concurrent_ops_per_sec', 0):.1f} ops/sec")
        self.logger.info(f"  Overall Performance: {'PASSED' if benchmarks.get('overall_passed') else 'FAILED'}")

        # Final verdict
        self.logger.info("\n" + "=" * 80)
        self.logger.info("FINAL VERDICT")
        self.logger.info("=" * 80)

        precision_ok = precision.get('precision_achieved', False)
        performance_ok = benchmarks.get('overall_passed', False)

        if precision_ok and performance_ok:
            self.logger.info("CONCLUSION: JANE STREET STANDARDS ACHIEVED")
            self.logger.info("The Mini Quant Fund meets institutional-grade requirements")
            self.logger.info("Ready for production deployment with top-tier performance")
        else:
            self.logger.info("CONCLUSION: SYSTEM READY WITH CAVEATS")
            if not precision_ok:
                self.logger.info("  - Precision requirements need attention")
            if not performance_ok:
                self.logger.info("  - Performance optimization needed")

        self.logger.info("=" * 80)
        self.logger.info("DEMONSTRATION COMPLETE")
        self.logger.info("=" * 80)

        # Cleanup
        await self._cleanup()

    async def _cleanup(self):
        """Clean up demo resources."""
        self.logger.info("Cleaning up demo resources...")

        try:
            await orchestrator.stop()
            await execution_engine.stop()
            await session_tracker.stop_monitoring()
        except Exception as e:
            self.logger.warning(f"Cleanup error: {e}")


async def main():
    """Main demo entry point."""
    demo = FullSystemDemo()
    await demo.run_complete_demo()


if __name__ == "__main__":
    asyncio.run(main())
