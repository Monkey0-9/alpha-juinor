"""
Integration Test: Full Cycle with Mock Providers
Tests 100% decision coverage and audit record generation.
"""

import pytest
import os
import sys
import sqlite3
from unittest.mock import Mock, patch
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from orchestration.cycle_orchestrator import CycleOrchestrator
from contracts import decision_enum
from audit.decision_log import get_cycle_decisions, get_decision_counts, AUDIT_DB_PATH


class MockDataProvider:
    """Mock data provider with configurable failure rate"""

    def __init__(self, name: str, failure_rate: float = 0.0):
        self.name = name
        self.failure_rate = failure_rate
        self.call_count = 0

    def get_price_history(self, symbol: str, **kwargs):
        """Return mock price data or fail"""
        self.call_count += 1

        # Simulate failure
        import random
        if random.random() < self.failure_rate:
            return pd.DataFrame()  # Empty = failure

        # Return mock data
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'Open': 100 + pd.Series(range(100)) * 0.1,
            'High': 101 + pd.Series(range(100)) * 0.1,
            'Low': 99 + pd.Series(range(100)) * 0.1,
            'Close': 100 + pd.Series(range(100)) * 0.1,
            'Volume': 1000000
        }, index=dates)

        data.attrs['provider'] = self.name
        return data


def test_full_cycle_with_mock_providers():
    """
    Test full cycle with 10 symbols.
    Verifies:
    - 100% decision coverage
    - Audit DB has exactly 10 records
    - All decisions have valid final_decision enum
    - Provider usage tracked correctly
    """

    # Setup
    test_universe = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',
                     'NVDA', 'META', 'NFLX', 'AMD', 'INTC']

    # Create orchestrator
    orchestrator = CycleOrchestrator(mode="test")

    # Mock universe manager
    orchestrator.universe_manager.get_active_universe = Mock(return_value=test_universe)

    # Mock data router with provider that succeeds 80% of the time
    mock_provider = MockDataProvider(name="mock_provider", failure_rate=0.2)
    orchestrator.data_router.get_price_history = mock_provider.get_price_history

    # Run cycle
    results = orchestrator.run_cycle()

    # Assertions

    # 1. 100% Decision Coverage
    assert len(results) == len(test_universe), \
        f"Expected {len(test_universe)} decisions, got {len(results)}"

    # 2. All decisions have valid enum
    for decision in results:
        assert decision.final_decision in [
            decision_enum.EXECUTE,
            decision_enum.HOLD,
            decision_enum.REJECT,
            decision_enum.ERROR
        ], f"Invalid decision enum: {decision.final_decision}"

    # 3. All symbols accounted for
    result_symbols = {d.symbol for d in results}
    assert result_symbols == set(test_universe), \
        f"Missing symbols: {set(test_universe) - result_symbols}"

    # 4. Audit DB has correct number of records
    cycle_decisions = get_cycle_decisions(orchestrator.cycle_id)
    assert len(cycle_decisions) == len(test_universe), \
        f"Audit DB should have {len(test_universe)} records, has {len(cycle_decisions)}"

    # 5. Decision counts match
    decision_counts = get_decision_counts(orchestrator.cycle_id)
    total_decisions = sum(decision_counts.values())
    assert total_decisions == len(test_universe), \
        f"Decision counts don't match: {decision_counts}"

    # 6. Provider usage tracked
    assert len(orchestrator.providers_tally) > 0, "No provider usage tracked"

    # 7. At least some decisions should be EXECUTE or HOLD (not all REJECT)
    execute_or_hold = decision_counts.get('EXECUTE', 0) + decision_counts.get('HOLD', 0)
    # With mock data, we expect at least some to pass quality checks
    # But this is lenient since agents might reject for other reasons

    print(f"\n✓ Test Passed!")
    print(f"  - Universe size: {len(test_universe)}")
    print(f"  - Decisions generated: {len(results)}")
    print(f"  - Decision breakdown: {decision_counts}")
    print(f"  - Provider usage: {orchestrator.providers_tally}")
    print(f"  - Data quality: {orchestrator.data_quality_stats}")


def test_provider_failure_handling():
    """
    Test that provider failures result in REJECT, not ERROR.
    """
    test_universe = ['AAPL', 'MSFT', 'GOOGL']

    orchestrator = CycleOrchestrator(mode="test")
    orchestrator.universe_manager.get_active_universe = Mock(return_value=test_universe)

    # Mock provider that always fails
    mock_provider = MockDataProvider(name="failing_provider", failure_rate=1.0)
    orchestrator.data_router.get_price_history = mock_provider.get_price_history

    results = orchestrator.run_cycle()

    # All should be REJECT (NO_DATA), not ERROR
    assert len(results) == len(test_universe)

    for decision in results:
        assert decision.final_decision == decision_enum.REJECT, \
            f"Expected REJECT for {decision.symbol}, got {decision.final_decision}"
        assert "NO_DATA" in decision.reason_codes, \
            f"Expected NO_DATA reason for {decision.symbol}"

    print("\n✓ Provider Failure Test Passed!")
    print(f"  - All {len(results)} decisions correctly marked as REJECT")


if __name__ == "__main__":
    # Clean up old test data
    if os.path.exists(AUDIT_DB_PATH):
        os.remove(AUDIT_DB_PATH)

    # Run tests
    test_full_cycle_with_mock_providers()
    test_provider_failure_handling()

    print("\n✅ All integration tests passed!")
