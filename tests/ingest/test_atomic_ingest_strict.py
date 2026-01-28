"""
tests/ingest/test_atomic_ingest_strict.py
Requirement: Verify atomic ingestion mechanics.
"""
import unittest
import pandas as pd
from unittest.mock import MagicMock, patch
from scripts.ingest_5y_batch import BatchIngestionAgent

class TestAtomicIngestStrict(unittest.TestCase):
    def setUp(self):
        self.agent = BatchIngestionAgent("test_run", "test_univ")
        self.agent.db = MagicMock()

    def test_atomic_write_call_structure(self):
        # Mock inputs
        df = pd.DataFrame({
            "Open": [100.0], "High": [105.0], "Low": [95.0], "Close": [102.0], "Volume": [1000]
        }, index=pd.to_datetime(["2021-01-01"]))
        actions = []
        score = 0.95
        flags = {"duplicates": 0}
        spikes = pd.Series([0], index=df.index)
        vol_spikes = pd.Series([0], index=df.index)

        # Call
        self.agent._atomic_write("AAPL", df, actions, score, flags, (spikes, vol_spikes), "yahoo", "SUCCESS", "ts")

        # Verify db.atomic_ingest called with correct kwargs
        self.agent.db.atomic_ingest.assert_called_once()
        args, kwargs = self.agent.db.atomic_ingest.call_args

        self.assertIn("prices", kwargs)
        self.assertIn("quality", kwargs)
        self.assertIsNone(kwargs["audit"]) # Decoupled audit check

    def test_audit_log_separate(self):
        # Mock atomic_ingest success
        self.agent.db.atomic_ingest.return_value = True

        df = pd.DataFrame() # Empty handled elsewhere, assuming valid df here
        df = pd.DataFrame({"Open": [1]}, index=pd.to_datetime(["2020-01-01"]))
        spikes = pd.Series([0], index=df.index)

        self.agent._atomic_write("AAPL", df, [], 1.0, {}, (spikes, spikes), "yahoo", "SUCCESS", "ts")

        # Verify log_ingestion_audit called separately
        self.agent.db.log_ingestion_audit.assert_called_once()

if __name__ == "__main__":
    unittest.main()
