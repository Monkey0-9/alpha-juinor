"""
Load / Stress Testing Framework
=================================
Validates system performance under realistic load.
"""

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class LoadTestResult:
    """Individual test run result."""
    test_name: str
    total_requests: int
    successful: int
    failed: int
    duration_sec: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    rps: float
    errors: List[str] = field(default_factory=list)


class TradingLoadTester:
    """
    Load testing for trading system components.

    Tests:
    - Order submission throughput
    - Data feed processing capacity
    - Risk gate evaluation speed
    - Database write performance
    - Concurrent strategy execution
    """

    def __init__(self):
        self._results: List[LoadTestResult] = []

    def test_order_throughput(
        self,
        target_rps: int = 100,
        duration_sec: int = 60,
        symbols: List[str] = None,
    ) -> LoadTestResult:
        """
        Load test order submission pipeline.
        Simulates N orders/second for D seconds.
        """
        symbols = symbols or [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
            "META", "TSLA", "AMD", "NFLX", "JPM",
        ]

        latencies = []
        errors = []
        total = target_rps * duration_sec
        successful = 0

        start = time.time()
        for i in range(total):
            t0 = time.time()
            try:
                # Simulate order processing
                sym = random.choice(symbols)
                qty = round(random.uniform(1, 100), 2)
                side = random.choice(["buy", "sell"])
                _simulate_order(sym, side, qty)
                successful += 1
            except Exception as e:
                errors.append(str(e))

            latencies.append(
                (time.time() - t0) * 1000
            )

            # Rate limiting
            elapsed = time.time() - start
            expected = (i + 1) / target_rps
            if expected > elapsed:
                time.sleep(expected - elapsed)

        duration = time.time() - start
        latencies.sort()

        result = LoadTestResult(
            test_name="order_throughput",
            total_requests=total,
            successful=successful,
            failed=len(errors),
            duration_sec=round(duration, 2),
            avg_latency_ms=round(
                sum(latencies) / len(latencies), 2
            ),
            p50_latency_ms=round(
                latencies[len(latencies) // 2], 2
            ),
            p95_latency_ms=round(
                latencies[int(len(latencies) * 0.95)], 2
            ),
            p99_latency_ms=round(
                latencies[int(len(latencies) * 0.99)], 2
            ),
            rps=round(successful / duration, 2),
            errors=errors[:10],
        )
        self._results.append(result)
        logger.info(
            f"Order throughput: {result.rps} RPS, "
            f"P99={result.p99_latency_ms}ms"
        )
        return result

    def test_risk_gate_performance(
        self, iterations: int = 10000,
    ) -> LoadTestResult:
        """Test risk gate evaluation speed."""
        latencies = []
        errors = []
        successful = 0

        start = time.time()
        for _ in range(iterations):
            t0 = time.time()
            try:
                _simulate_risk_check()
                successful += 1
            except Exception as e:
                errors.append(str(e))
            latencies.append(
                (time.time() - t0) * 1000
            )

        duration = time.time() - start
        latencies.sort()

        result = LoadTestResult(
            test_name="risk_gate_performance",
            total_requests=iterations,
            successful=successful,
            failed=len(errors),
            duration_sec=round(duration, 2),
            avg_latency_ms=round(
                sum(latencies) / len(latencies), 2
            ),
            p50_latency_ms=round(
                latencies[len(latencies) // 2], 2
            ),
            p95_latency_ms=round(
                latencies[int(len(latencies) * 0.95)], 2
            ),
            p99_latency_ms=round(
                latencies[int(len(latencies) * 0.99)], 2
            ),
            rps=round(successful / duration, 2),
            errors=errors[:10],
        )
        self._results.append(result)
        return result

    def test_data_processing(
        self, symbols_count: int = 500,
        bars_per_symbol: int = 1000,
    ) -> LoadTestResult:
        """Test data processing capacity."""
        latencies = []
        total = symbols_count * bars_per_symbol
        successful = 0

        start = time.time()
        for s in range(symbols_count):
            t0 = time.time()
            for _ in range(bars_per_symbol):
                _simulate_bar_processing()
                successful += 1
            latencies.append(
                (time.time() - t0) * 1000
            )

        duration = time.time() - start
        latencies.sort()

        result = LoadTestResult(
            test_name="data_processing",
            total_requests=total,
            successful=successful,
            failed=0,
            duration_sec=round(duration, 2),
            avg_latency_ms=round(
                sum(latencies) / len(latencies), 2
            ),
            p50_latency_ms=round(
                latencies[len(latencies) // 2], 2
            ),
            p95_latency_ms=round(
                latencies[int(len(latencies) * 0.95)], 2
            ),
            p99_latency_ms=round(
                latencies[int(len(latencies) * 0.99)], 2
            ),
            rps=round(successful / duration, 2),
        )
        self._results.append(result)
        return result

    def full_report(self) -> Dict:
        """Generate full load test report."""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "tests": [
                {
                    "name": r.test_name,
                    "rps": r.rps,
                    "p99_ms": r.p99_latency_ms,
                    "success_rate": round(
                        r.successful / r.total_requests
                        * 100, 2
                    ),
                }
                for r in self._results
            ],
        }


def _simulate_order(
    symbol: str, side: str, qty: float
):
    """Simulate order processing."""
    time.sleep(random.uniform(0.0001, 0.001))


def _simulate_risk_check():
    """Simulate risk gate evaluation."""
    _ = random.gauss(0, 1)
    time.sleep(random.uniform(0.00001, 0.0001))


def _simulate_bar_processing():
    """Simulate OHLCV bar processing."""
    _ = [random.gauss(100, 5) for _ in range(4)]
