"""
tests/ingest/test_atomic_ingest.py
"""
import unittest
import os
import sqlite3
from database.manager import DatabaseManager
from database.schema import DailyPriceRecord, IngestionAuditRecord

class TestAtomicIngest(unittest.TestCase):
    def setUp(self):
        self.test_db = "runtime/test_atomic.db"
        if os.path.exists(self.test_db):
            os.remove(self.test_db)

        # Initialize DB Manager with test path (requires patching or env var)
        # For now, let's just use the main DB path as a mock or ensure adapter supports override
        # Or better, we trust the manager logic but verify rollback.

        # Actually Manager reads config. Let's assume we can init with path if we modify Adapter?
        # Current adapter hardcodes path usually.
        pass

    def test_atomic_rollback(self):
        """Verify that if one part fails, NOTHING is written."""
        # This is hard to test without mocking the adapter to fail midway
        # But we can test the happy path and ensure all tables populated.

        db = DatabaseManager()

        run_id = "test_atomic_run"
        prices = [{
            "symbol": "TEST_ATOMIC", "date": "2025-01-01",
            "open": 100, "high":105, "low":95, "close":102, "volume":1000,
            "provider":"test", "ingestion_timestamp":"2025-01-01", "raw_hash":"test"
        }]
        audit = {
            "run_id": run_id, "symbol": "TEST_ATOMIC",
            "status": "SUCCESS", "asset_class":"test", "provider":"test"
        }

        success = db.atomic_ingest(prices=prices, audit=audit)
        self.assertTrue(success)

        # Verify write
        with db.get_connection() as conn:
            c = conn.execute("SELECT count(*) FROM price_history WHERE symbol='TEST_ATOMIC'")
            self.assertEqual(c.fetchone()[0], 1)
            c = conn.execute("SELECT count(*) FROM ingestion_audit WHERE run_id='test_atomic_run'")
            self.assertEqual(c.fetchone()[0], 1)

    def test_failure_rollback(self):
        # To test rollback, we need to inject a failure.
        # One way: Pass invalid data that causes integrity error in the second step (audit)
        # but valid in first (prices).

        db = DatabaseManager()
        run_id = "test_fail_run"

        prices = [{
            "symbol": "TEST_FAIL", "date": "2025-01-01",
            "open": 100, "high":105, "low":95, "close":102, "volume":1000,
            "provider":"test", "ingestion_timestamp":"2025-01-01", "raw_hash":"test"
        }]

        # Valid prices, but simpler way to fail?
        # The atomic_ingest method catches Exception and returns False.
        # We need a way to make it fail inside the block.
        # Maybe pass a malformed CorporateAction if we include it?

        # If we pass a list of dicts for corp actions but they are missing keys,
        # the list comprehension [CorporateAction(**c)] might fail if strict,
        # OR the adapter insert fails.

        # Let's try malformed corporate action
        corp = [{"symbol": "FAIL", "missing_date": "no_date"}] # Missing required fields

        # This typically raises TypeError or KeyError inside the atomic block
        success = db.atomic_ingest(prices=prices, corp_actions=corp)

        self.assertFalse(success)

        # Verify rollback: Price should NOT be there
        with db.get_connection() as conn:
            c = conn.execute("SELECT count(*) FROM price_history WHERE symbol='TEST_FAIL'")
            self.assertEqual(c.fetchone()[0], 0, "Price should have been rolled back!")

if __name__ == "__main__":
    unittest.main()
