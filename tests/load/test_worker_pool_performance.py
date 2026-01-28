"""
Load Test: Worker Pool Performance
Measures decision generation latency for large universe.
"""

import time
import sys
import os
import statistics
from unittest.mock import Mock
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from orchestration.cycle_orchestrator import CycleOrchestrator


def generate_mock_universe(size: int = 249):
    """Generate mock universe of symbols"""
    # Use real-ish symbols
    base_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX']
    symbols = []
    for i in range(size):
        if i < len(base_symbols):
            symbols.append(base_symbols[i])
        else:
            symbols.append(f"SYM{i:03d}")
    return symbols


def mock_data_provider(symbol: str, **kwargs):
    """Fast mock data provider"""
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'Open': 100.0,
        'High': 101.0,
        'Low': 99.0,
        'Close': 100.0,
        'Volume': 1000000
    }, index=dates)
    data.attrs['provider'] = 'mock'
    return data


def test_worker_pool_performance(universe_size: int = 249, max_workers: int = 50):
    """
    Test decision generation performance.
    Target: median < 60s for 249 symbols with 50 workers
    Goal: median < 10s with optimization
    """

    print(f"\n{'='*60}")
    print(f"LOAD TEST: {universe_size} symbols with {max_workers} workers")
    print(f"{'='*60}\n")

    # Generate universe
    test_universe = generate_mock_universe(universe_size)

    # Create orchestrator
    orchestrator = CycleOrchestrator(mode="test")
    orchestrator.universe_manager.get_active_universe = Mock(return_value=test_universe)
    orchestrator.data_router.get_price_history = mock_data_provider

    # Measure latency
    start_time = time.time()
    results = orchestrator.run_cycle()
    total_duration = time.time() - start_time

    # Calculate per-symbol latency
    per_symbol_latency = total_duration / len(results) if results else 0

    # Metrics
    decision_counts = {}
    for d in results:
        decision_counts[d.final_decision.value] = decision_counts.get(d.final_decision.value, 0) + 1

    # Report
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Universe Size:        {len(test_universe)}")
    print(f"Decisions Generated:  {len(results)}")
    print(f"Total Duration:       {total_duration:.2f}s")
    print(f"Per-Symbol Latency:   {per_symbol_latency:.3f}s")
    print(f"Throughput:           {len(results)/total_duration:.1f} decisions/sec")
    print(f"\nDecision Breakdown:")
    for decision, count in sorted(decision_counts.items()):
        print(f"  {decision:10s}: {count:4d} ({count/len(results)*100:.1f}%)")

    # Performance Assessment
    print(f"\n{'='*60}")
    print(f"PERFORMANCE ASSESSMENT")
    print(f"{'='*60}")

    if total_duration < 10:
        print(f"✅ EXCELLENT: {total_duration:.1f}s < 10s goal")
    elif total_duration < 60:
        print(f"✅ PASS: {total_duration:.1f}s < 60s target")
    else:
        print(f"❌ FAIL: {total_duration:.1f}s > 60s target")

    # Coverage check
    if len(results) == len(test_universe):
        print(f"✅ 100% Decision Coverage")
    else:
        print(f"❌ Incomplete Coverage: {len(results)}/{len(test_universe)}")

    # Return metrics for programmatic use
    return {
        'universe_size': len(test_universe),
        'decisions_generated': len(results),
        'total_duration_sec': total_duration,
        'per_symbol_latency_sec': per_symbol_latency,
        'throughput_per_sec': len(results)/total_duration if total_duration > 0 else 0,
        'decision_counts': decision_counts,
        'pass': total_duration < 60 and len(results) == len(test_universe)
    }


if __name__ == "__main__":
    # Run load test
    metrics = test_worker_pool_performance(universe_size=249, max_workers=50)

    if metrics['pass']:
        print("\n✅ Load test PASSED!")
        sys.exit(0)
    else:
        print("\n❌ Load test FAILED!")
        sys.exit(1)
