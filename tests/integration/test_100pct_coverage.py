"""
Integration test for 100% decision coverage with SystemHalt enforcement.
Tests that every symbol in universe gets exactly one decision.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestration.cycle_orchestrator import CycleOrchestrator
from audit.decision_log import SystemHalt
import sqlite3
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_100pct_coverage():
    """Test that 100% of universe gets decisions"""

    logger.info("=" * 80)
    logger.info("TEST: 100% Decision Coverage with SystemHalt")
    logger.info("=" * 80)

    try:
        # Create orchestrator
        orch = CycleOrchestrator(mode="paper")

        # Get universe size
        universe = orch.universe_manager.get_active_universe()
        universe_size = len(universe)
        logger.info(f"Universe size: {universe_size}")

        # Run cycle
        results = orch.run_cycle()

        # Verify coverage
        assert len(results) == universe_size, f"Coverage mismatch: {len(results)} != {universe_size}"
        logger.info(f"✓ Coverage test PASSED: {len(results)}/{universe_size}")

        # Verify audit DB
        conn = sqlite3.connect('runtime/audit.db')
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM decisions WHERE cycle_id = ?", (orch.cycle_id,))
        db_count = cursor.fetchone()[0]
        conn.close()

        assert db_count == universe_size, f"Audit DB mismatch: {db_count} != {universe_size}"
        logger.info(f"✓ Audit DB test PASSED: {db_count} records")

        # Verify all decisions have valid enum
        from contracts import decision_enum
        valid_decisions = {decision_enum.EXECUTE, decision_enum.HOLD, decision_enum.REJECT, decision_enum.ERROR}
        for d in results:
            assert d.final_decision in valid_decisions, f"Invalid decision: {d.final_decision}"
        logger.info(f"✓ Decision enum test PASSED")

        logger.info("=" * 80)
        logger.info("ALL TESTS PASSED ✓")
        logger.info("=" * 80)

        return True

    except SystemHalt as e:
        logger.error(f"✗ SystemHalt raised (expected behavior): {e}")
        return False
    except Exception as e:
        logger.error(f"✗ Test FAILED: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = test_100pct_coverage()
    sys.exit(0 if success else 1)
