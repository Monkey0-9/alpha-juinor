"""
Institutional Ingestion Entry Point.
Usage: python ingest_history.py --mode=[incremental|verify|deep_backfill]
"""
import argparse
import sys
import logging
import time
import os
from datetime import datetime, timedelta
from data.ingestion_agent import InstitutionalIngestionAgent
from data.universe_manager import UnifiedUniverseManager
from database.manager import DatabaseManager

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
logger = logging.getLogger("INGEST_HISTORY")

# Kill switch path
KILL_SWITCH_PATH = "runtime/KILL_SWITCH"


def check_kill_switch() -> bool:
    """Check for binary Kill Switch file."""
    if os.path.exists(KILL_SWITCH_PATH):
        logger.critical(f"[KILL_SWITCH] Activated: '{KILL_SWITCH_PATH}' found. Ingestion HALTED.")
        return True
    return False

def main():
    parser = argparse.ArgumentParser(description="Institutional History Ingestion")
    parser.add_argument("--mode", choices=["incremental", "verify", "deep_backfill"], default="deep_backfill")
    parser.add_argument("--start", type=str, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, help="End date YYYY-MM-DD")
    parser.add_argument("--tickers", type=str, help="Comma-separated tickers (optional override)")

    args = parser.parse_args()

    logger.info(f"[INGESTION] Starting Stage 1 Ingestion in mode: {args.mode}")

    # Check Kill Switch FIRST - before any other operations
    if check_kill_switch():
        logger.critical("[INGESTION] Kill switch active. Aborting ingestion.")
        sys.exit(1)

    # Initialize Components
    db = DatabaseManager()
    run_id = f"BATCH_{datetime.utcnow().strftime('%Y%m%d%H%M')}"
    agent = InstitutionalIngestionAgent(run_id=run_id)
    # Inject DB (hack or clean? Agent creates its own DB manager usually)
    # Inspecting agent: it creates its own.

    # DETERMINE UNIVERSE
    if args.tickers:
        tickers = [t.strip() for t in args.tickers.split(",")]
    else:
        # Load from universe configuration (Golden Source)
        universe_mgr = UnifiedUniverseManager()
        tickers = universe_mgr.get_active_universe()

    logger.info(f"[INGESTION] Target Universe: {len(tickers)} symbols")

    # MODE HANDLING
    if args.mode == "deep_backfill":
        logger.info("[INGESTION] Running DEEP BACKFILL (5-Year MANDATE)")
        # Ingestion agent handles the 6-year lookback internally to guarantee 1260 days
        agent.run_full_universe(tickers)

    elif args.mode == "verify":
        logger.info("[INGESTION] Verifying DB integrity...")
        # Verification logic could be added here
        pass

    elif args.mode == "incremental":
        logger.info("[INGESTION] Running INCREMENTAL update")
        # Reuse pipeline with shorter window?
        # For now, safe default is pipeline which handles updates
        agent.run_pipeline(tickers)

    # FINAL VERIFICATION
    logger.info("[INGESTION] Pipeline finished. Verifying compliance...")

    from data.governance.governance_agent import SymbolGovernor
    gov = SymbolGovernor(db)
    gov.classify_all()

    # AUDIT
    conn = db.get_connection()
    cursor = conn.execute("SELECT COUNT(*) FROM symbol_governance WHERE state='ACTIVE'")
    active_count = cursor.fetchone()[0]

    logger.info(f"[INGESTION] Governance Check: {active_count} symbols are ACTIVE (1260+ rows)")

    if active_count < len(tickers) * 0.9:
         logger.warning(f"[INGESTION] WARNING: Only {active_count}/{len(tickers)} symbols are ACTIVE.")
         sys.exit(1) # Signal failure to pipeline

    sys.exit(0)

if __name__ == "__main__":
    main()
