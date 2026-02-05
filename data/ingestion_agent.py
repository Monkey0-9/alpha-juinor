"""
Compatibility shim for renamed ingestion agent.
Maintains backward compatibility for existing imports.
"""
from data.ingestion.ingest_process import DataIngestionAgent as InstitutionalIngestionAgent
