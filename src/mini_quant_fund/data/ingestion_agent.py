"""
Compatibility shim for renamed ingestion agent.
Maintains backward compatibility for existing imports.
"""
from mini_quant_fund.data.ingestion.ingest_process import DataIngestionAgent as InstitutionalIngestionAgent
