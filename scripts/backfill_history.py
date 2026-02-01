
import sys
import os
import logging

# Add project root
sys.path.insert(0, os.getcwd())

from data.ingestion.ingest_process import DataIngestionAgent
from data.universe_manager import UnifiedUniverseManager

def run_backfill():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    logger.info("Starting Full Historical Backfill...")

    mgr = UnifiedUniverseManager()
    tickers = mgr.get_active_universe()

    logger.info(f"Loaded {len(tickers)} tickers from Universe Manager.")

    agent = DataIngestionAgent(tickers)
    summary = agent.run_full_universe(tickers)

    logger.info(f"Backfill Complete. Summary: {summary}")

if __name__ == "__main__":
    run_backfill()
