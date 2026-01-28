#!/usr/bin/env python3
"""
Institutional DB Repair & Backfill Tool

Usage:
python repair_db.py --symbol=AAPL --from=2020-01-01 --to=2021-01-01
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ingest_history import InstitutionalIngestionAgent

def main():
    parser = argparse.ArgumentParser(description="Institutional DB Repair & Backfill Tool")
    parser.add_argument("--symbol", type=str, required=True, help="Symbol to repair")
    parser.add_argument("--from", dest="from_date", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--to", dest="to_date", type=str, required=True, help="End date (YYYY-MM-DD)")
    args = parser.parse_args()

    # Calculate required days
    start = datetime.strptime(args.from_date, "%Y-%m-%d")
    end = datetime.strptime(args.to_date, "%Y-%m-%d") if args.to_date else datetime.utcnow()
    days = (datetime.utcnow() - start).days

    print(f"[*] Initiating repair for {args.symbol} | Start: {args.from_date} | Days back: {days}")

    agent = InstitutionalIngestionAgent(run_id=f"repair_{args.symbol}_{datetime.utcnow().strftime('%Y%m%d')}")
    result = agent.process_symbol(args.symbol, required_history_days=days)

    print(f"[+] Repair Finished | Symbol: {args.symbol} | Status: {result.get('status')} | Quality: {result.get('quality_score', 'N/A')}")

if __name__ == "__main__":
    main()
