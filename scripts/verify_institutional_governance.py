import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.manager import DatabaseManager
from database.schema import DailyPriceRecord, DataQualityRecord, TradingEligibilityRecord
from data.governance.governance_agent import SymbolGovernor
from main import InstitutionalLiveAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VERIFY_GOV")

def run_verification():
    db = DatabaseManager()
    gov = SymbolGovernor(db)

    test_symbols = ["TEST_ACTIVE", "TEST_DEGRADED", "TEST_QUARANTINED"]

    # Clean up old test data
    print("Cleaning up old test data...")
    print("Cleaning up old test data...")
    for sym in test_symbols:
        with db.get_connection() as conn:
            conn.execute("DELETE FROM price_history WHERE symbol = ?", (sym,))
            conn.execute("DELETE FROM data_quality WHERE symbol = ?", (sym,))
            conn.execute("DELETE FROM trading_eligibility WHERE symbol = ?", (sym,))
            # conn.commit() is handled by context manager if adapter supports it,
            # or auto-commit for sqlite adapter transaction mode.
            # But the context manager I added earlier yielded the connection.
            # Let's check if the context manager commits.
            # PostgresAdapter.get_connection() commits.
            # SQLiteAdapter doesn't expose get_connection context manager yet?
            # Wait, `DatabaseManager.get_connection` implementation:
            # if hasattr(self.adapter, "_get_connection"): return connection_context(self.adapter._get_connection)
            # And `connection_context` doesn't commit explicitly.
            # So for SQLite I might need manual commit if not in transaction.
            # But the original code called `conn = self._get_connection()` which returned the SAME connection object (thread local).
            # So `db._get_connection().commit()` worked.
            # With `with db.get_connection() as conn:`, `conn` is the connection.
            conn.commit()

    # 1. Prepare Mock Data
    print("Setting up mock data for symbols...")

    # TEST_ACTIVE: 1300 rows, 1.0 quality -> state ACTIVE
    db.upsert_daily_prices_batch([
        DailyPriceRecord(
            symbol="TEST_ACTIVE", date=(datetime.utcnow() - timedelta(days=i)).strftime('%Y-%m-%d'),
            open=100.0, high=105.0, low=95.0, close=101.0, volume=1000, adjusted_close=101.0,
            provider="mock", raw_hash="abc", ingestion_timestamp=datetime.utcnow().isoformat()
        ) for i in range(1300)
    ])
    db.log_data_quality(DataQualityRecord(
        symbol="TEST_ACTIVE", run_id="verify_run", quality_score=1.0,
        recorded_at=datetime.utcnow().isoformat()
    ))

    # TEST_DEGRADED: 1100 rows, 1.0 quality -> state DEGRADED
    db.upsert_daily_prices_batch([
        DailyPriceRecord(
            symbol="TEST_DEGRADED", date=(datetime.utcnow() - timedelta(days=i)).strftime('%Y-%m-%d'),
            open=100.0, high=105.0, low=95.0, close=101.0, volume=1000, adjusted_close=101.0,
            provider="mock", raw_hash="abc", ingestion_timestamp=datetime.utcnow().isoformat()
        ) for i in range(1100)
    ])
    db.log_data_quality(DataQualityRecord(
        symbol="TEST_DEGRADED", run_id="verify_run", quality_score=1.0,
        recorded_at=datetime.utcnow().isoformat()
    ))

    # TEST_QUARANTINED: 10 rows, 1.0 quality -> state QUARANTINED (rows < 1000)
    db.upsert_daily_prices_batch([
        DailyPriceRecord(
            symbol="TEST_QUARANTINED", date=(datetime.utcnow() - timedelta(days=i)).strftime('%Y-%m-%d'),
            open=100.0, high=105.0, low=95.0, close=101.0, volume=1000, adjusted_close=101.0,
            provider="mock", raw_hash="abc", ingestion_timestamp=datetime.utcnow().isoformat()
        ) for i in range(10)
    ])
    db.log_data_quality(DataQualityRecord(
        symbol="TEST_QUARANTINED", run_id="verify_run", quality_score=1.0,
        recorded_at=datetime.utcnow().isoformat()
    ))

    # 2. Run Classification
    print("Running governance classification sweep...")
    gov.classify_all(test_symbols)

    # 3. Verify Database State
    print("Verifying database state...")
    for sym in test_symbols:
        with db.get_connection() as conn:
            row = conn.execute("SELECT state, history_rows, data_quality FROM trading_eligibility WHERE symbol = ?", (sym,)).fetchone()
        if not row:
            print(f"FAIL: No entry found in trading_eligibility for {sym}")
            sys.exit(1)

        print(f"DB Check: symbol={sym} state={row['state']} rows={row['history_rows']} quality={row['data_quality']}")

        if sym == "TEST_ACTIVE" and row['state'] != "ACTIVE":
            print(f"FAIL: {sym} state is {row['state']}, expected ACTIVE")
            sys.exit(1)
        elif sym == "TEST_DEGRADED" and row['state'] != "DEGRADED":
            print(f"FAIL: {sym} state is {row['state']}, expected DEGRADED")
            sys.exit(1)
        elif sym == "TEST_QUARANTINED" and row['state'] != "QUARANTINED":
            print(f"FAIL: {sym} state is {row['state']}, expected QUARANTINED")
            sys.exit(1)

    # 4. Verify Live System Gate
    print("Verifying Live System Gate...")
    agent = InstitutionalLiveAgent(tickers=test_symbols)
    agent.check_governance_gate()

    print(f"Agent Tickers: {agent.tickers}")
    if len(agent.tickers) != 1 or "TEST_ACTIVE" not in agent.tickers:
        print(f"FAIL: Agent tickers mismatch. Got {agent.tickers}")
        sys.exit(1)

    # 5. Verify 252-day Market Data Loading
    print("Verifying Market Data Loading...")
    agent.initialize_system()
    if "TEST_ACTIVE" not in agent.market_data:
        print("FAIL: TEST_ACTIVE not in market_data")
        sys.exit(1)

    data = agent.market_data["TEST_ACTIVE"]
    print(f"TEST_ACTIVE metrics: last_price={data['last_price']} vol={data['volatility']:.4f} rows={data['row_count']}")

    if data["row_count"] != 252:
        print(f"FAIL: Row count mismatch. Got {data['row_count']}")
        sys.exit(1)

    print("VERIFICATION SUCCESSFUL")

if __name__ == "__main__":
    run_verification()
