
import logging
import sys
import os

# Add project root to path
sys.path.insert(0, os.getcwd())

from data.ingestion.ingest_process import DataIngestionAgent
from data.universe_manager import UnifiedUniverseManager

def verify_live():
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Starting LIVE Verification of Data Ingestion (5 Tickers)...")

    # Pick 5 core tickers
    test_tickers = ["AAPL", "SPY", "MSFT", "GOOGL", "GLD"]

    agent = DataIngestionAgent(test_tickers)
    summary = agent.run_full_universe(test_tickers) # This calls finalize_run which returns summary

    # Check results in DB?
    # summary returned
    if summary['successful'] == len(test_tickers):
        logger.info("VERIFICATION PASSED: All 5 tickers ingested successfully.")
        sys.exit(0)
    else:
        logger.error(f"VERIFICATION FAILED: {summary['successful']}/{len(test_tickers)} successful.")
        sys.exit(1)

if __name__ == "__main__":
    verify_live()
