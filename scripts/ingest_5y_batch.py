"""
Backward-compatible import path for batch ingestion agent.

The implementation was moved to scripts/data/ingest_5y_batch.py.
"""

from scripts.data.ingest_5y_batch import BatchIngestionAgent

__all__ = ["BatchIngestionAgent"]
