
import sqlite3
import time
import sys
import os
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingest_history import InstitutionalIngestionAgent

DB = "runtime/institutional_trading.db"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BACKFILL")

def get_missing_symbols(min_rows=1260):
    if os.path.exists("required_backfill.txt"):
        with open("required_backfill.txt", "r") as f:
            syms = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(syms)} symbols from required_backfill.txt")
        return syms

    try:
        con = sqlite3.connect(DB)
        cur = con.cursor()
        # Fallback: Get active symbols that have insufficient history
        cur.execute("""
            SELECT t.symbol, COUNT(ph.date) as c
            FROM trading_eligibility t
            LEFT JOIN price_history ph ON t.symbol = ph.symbol
            WHERE t.status='ACTIVE'
            GROUP BY t.symbol
            HAVING c < ?
        """, (min_rows,))
        return [r[0] for r in cur.fetchall()]
    except Exception as e:
        logger.error(f"Database error: {e}")
        return []

def backfill_symbol(agent, symbol):
    logger.info(f"Backfilling {symbol}...")
    try:
        # Request 5 years (approx 1825 days)
        result = agent.process_symbol(symbol, required_history_days=1825)
        logger.info(f"Result for {symbol}: {result}")
    except Exception as e:
        logger.error(f"Failed to backfill {symbol}: {e}")

if __name__ == "__main__":
    agent = InstitutionalIngestionAgent()
    missing = get_missing_symbols()
    logger.info(f"Found {len(missing)} active symbols with insufficient history.")

    for s in missing:
        backfill_symbol(agent, s)
        time.sleep(1)  # politeness / rate-limit
