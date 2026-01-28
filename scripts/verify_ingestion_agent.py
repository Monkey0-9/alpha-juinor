"""
Verification Script for Institutional Ingestion Agent.
Validates symbol classification, routing, archiving, and quality scoring.
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

# Add root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from data.ingestion_agent import InstitutionalIngestionAgent
from data.collectors.data_router import DataRouter
from database.manager import DatabaseManager

def test_symbol_classification():
    print("\n[TEST] Symbol Classification...")
    router = DataRouter()

    test_cases = {
        "AAPL": "stocks",
        "BTC-USD": "crypto",
        "EURUSD=X": "fx",
        "GC=F": "commodities"
    }

    for symbol, expected in test_cases.items():
        actual = router._classify_ticker(symbol)
        print(f"  {symbol} -> {actual} (Expected: {expected})")
        assert actual == expected, f"Classification failed for {symbol}"
    print("  SUCCESS")

def test_quality_scoring():
    print("\n[TEST] Data Quality Scoring...")
    import pandas as pd
    import numpy as np

    agent = InstitutionalIngestionAgent(tickers=["AAPL"])

    # Perfect data
    dates = pd.date_range(end=datetime.now(), periods=1260)
    df_perfect = pd.DataFrame({
        "Open": [100.0]*1260, "High": [101.0]*1260, "Low": [99.0]*1260,
        "Close": [100.0]*1260, "Volume": [1000000]*1260
    }, index=dates)

    perfect_res = agent.calculate_quality_score(df_perfect)
    print(f"  Perfect Data Score: {perfect_res['score']}")
    assert perfect_res['score'] == 1.0

    # Data with zero prices
    df_zero = df_perfect.copy()
    df_zero.iloc[100, 3] = 0.0 # Close at index 100
    zero_res = agent.calculate_quality_score(df_zero)
    print(f"  Zero Price Score: {zero_res['score']}")
    assert zero_res['score'] < 1.0
    assert zero_res['flags']['zero_negative_flag'] is True

    print("  SUCCESS")

def test_ingestion_audit():
    print("\n[TEST] Ingestion Audit Persistence...")
    agent = InstitutionalIngestionAgent(tickers=["TEST_AUDIT"])
    # Mocking a small run
    agent.stats["processed"] = 1
    agent.stats["successful"] = 1
    agent.stats["quality_scores"] = [0.95]
    agent.stats["end_time"] = datetime.utcnow().isoformat()

    summary = agent.finalize_run()
    print(f"  Finalize Summary: {summary}")

    db = DatabaseManager()
    runs = db.get_ingestion_audit(run_id=agent.run_id)
    # Note: run_full_universe wasn't called, so ingestion_audit table won't have rows yet,
    # but ingestion_audit_runs should have our summary.

    conn = db.get_connection()
    row = conn.execute("SELECT * FROM ingestion_audit_runs WHERE run_id = ?", (agent.run_id,)).fetchone()
    print(f"  DB Run Record: {dict(row) if row else 'NONE'}")
    assert row is not None
    assert row['successful'] == 1
    print("  SUCCESS")

if __name__ == "__main__":
    try:
        test_symbol_classification()
        test_quality_scoring()
        test_ingestion_audit()
        print("\n[VERIFICATION COMPLETE] All institutional logic verified.")
    except Exception as e:
        print(f"\n[VERIFICATION FAILED] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
