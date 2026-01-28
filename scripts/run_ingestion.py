#!/usr/bin/env python3
"""
Institutional Data Ingestion Runner.
Use this script to fetch 5-year historical data for the system universe.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Add root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from data.universe_manager import UnifiedUniverseManager
from data.ingestion_agent import InstitutionalIngestionAgent

def main():
    parser = argparse.ArgumentParser(description="Institutional Market Data Ingestion Runner")
    parser.add_argument("--tickers", type=str, help="Comma-separated list of tickers (optional)")
    parser.add_argument("--full", action="store_true", help="Ingest full universe from config")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )

    universe_mgr = UnifiedUniverseManager()

    if args.full:
        tickers = universe_mgr.get_active_universe()
    elif args.tickers:
        tickers = args.tickers.split(",")
    else:
        print("Error: Specify --tickers or --full")
        sys.exit(1)

    print(f"Starting Institutional Ingestion for {len(tickers)} symbols...")
    agent = InstitutionalIngestionAgent(tickers=tickers)
    agent.run_full_universe(tickers)

    print("\n[COMPLETE] Institutional Ingestion Run Finished.")

if __name__ == "__main__":
    main()
