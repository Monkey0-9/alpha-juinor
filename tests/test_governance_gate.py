
import unittest
from unittest.mock import MagicMock, patch
from execution.gates import ExecutionGatekeeper
from database.manager import DatabaseManager
from database.schema import SymbolGovernanceRecord

class TestGovernanceGate(unittest.TestCase):
    def setUp(self):
        self.gate = ExecutionGatekeeper()
        self.db = DatabaseManager()

    def test_block_quarantined_symbol(self):
        """Verify that a QUARANTINED symbol is blocked."""
        symbol = "DEBUG_QUARANTINED"
        self.db.upsert_symbol_governance(SymbolGovernanceRecord(
            symbol=symbol,
            history_rows=100,
            state="QUARANTINED",
            last_checked_ts="2026-01-21T00:00:00"
        ))

        is_ok, reason, scaled_qty = self.gate.validate_execution(
            symbol=symbol, qty=100, side="BUY", price=150.0,
            adv_30d=1000000, volatility=0.02
        )

        self.assertFalse(is_ok)
        self.assertEqual(reason, "REJECTED_GOVERNANCE_QUARANTINED")
        self.assertEqual(scaled_qty, 0.0)

    def test_allow_active_symbol(self):
        """Verify that an ACTIVE symbol is allowed if other gates pass."""
        symbol = "DEBUG_ACTIVE"
        self.db.upsert_symbol_governance(SymbolGovernanceRecord(
            symbol=symbol,
            history_rows=1300,
            state="ACTIVE",
            last_checked_ts="2026-01-21T00:00:00"
        ))

        with patch('risk.market_impact_models.TransactionCostModel.estimate_total_cost') as mock_estimate:
            mock_estimate.return_value = {'cost_bps': 5.0} # Below 20bps limit

            is_ok, reason, scaled_qty = self.gate.validate_execution(
                symbol=symbol, qty=100, side="BUY", price=150.0,
                adv_30d=1000000, volatility=0.02
            )

            self.assertTrue(is_ok)
            self.assertEqual(reason, "OK")
            self.assertEqual(scaled_qty, 100)

if __name__ == "__main__":
    unittest.main()
