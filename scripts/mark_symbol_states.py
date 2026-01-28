"""
Symbol State Classification Script

Automatically classifies symbols as ACTIVE, DEGRADED, or QUARANTINED
based on data quality and completeness.

States:
- ACTIVE: ≥1260 rows AND quality ≥0.6 → eligible for trading
- DEGRADED: <1260 rows → needs more history
- QUARANTINED: quality <0.6 → data quality issues

Run after ingestion or on-demand to update trading_eligibility table.
"""

import sqlite3
import sys
import os
from datetime import datetime
from pathlib import Path

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.quality import compute_data_quality
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
MIN_ROWS = 1260  # ~5 years of trading days
MIN_QUALITY = 0.6
DB_PATH = "runtime/institutional_trading.db"


def ensure_table_exists(conn):
    """Create trading_eligibility table if it doesn't exist."""
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS trading_eligibility (
            symbol TEXT PRIMARY KEY,
            state TEXT NOT NULL,
            reason TEXT,
            data_quality_score REAL,
            row_count INTEGER,
            last_checked TEXT NOT NULL
        )
    """)
    conn.commit()
    logger.info("Ensured trading_eligibility table exists")


def get_all_symbols(conn):
    """Get all unique symbols from price_history."""
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT symbol FROM price_history ORDER BY symbol")
    symbols = [row[0] for row in cursor.fetchall()]
    logger.info(f"Found {len(symbols)} unique symbols in price_history")
    return symbols


def classify_symbol(conn, symbol):
    """
    Classify a single symbol based on data quality and completeness.

    Returns:
        Tuple of (state, reason, quality_score, row_count)
    """
    cursor = conn.cursor()

    # Get price history for this symbol
    query = """
        SELECT date, close, volume
        FROM price_history
        WHERE symbol = ?
        ORDER BY date ASC
    """
    df = pd.read_sql_query(query, conn, params=(symbol,))

    row_count = len(df)

    if row_count == 0:
        return "QUARANTINED", "NO_DATA", 0.0, 0

    # Compute quality score
    quality_score, quality_reasons = compute_data_quality(df)

    # Classification logic
    if row_count >= MIN_ROWS and quality_score >= MIN_QUALITY:
        state = "ACTIVE"
        reason = "OK"
    elif row_count < MIN_ROWS:
        state = "DEGRADED"
        reason = f"INSUFFICIENT_HISTORY:rows={row_count}<{MIN_ROWS}"
    else:
        state = "QUARANTINED"
        reason = f"LOW_QUALITY:score={quality_score:.2f}:reasons={','.join(quality_reasons)}"

    return state, reason, quality_score, row_count


def update_trading_eligibility(conn):
    """Update trading_eligibility table for all symbols."""
    ensure_table_exists(conn)

    symbols = get_all_symbols(conn)

    if not symbols:
        logger.warning("No symbols found in price_history table")
        return

    cursor = conn.cursor()
    now = datetime.utcnow().isoformat()

    stats = {"ACTIVE": 0, "DEGRADED": 0, "QUARANTINED": 0}

    for symbol in symbols:
        state, reason, quality_score, row_count = classify_symbol(conn, symbol)

        # Update database
        cursor.execute("""
            INSERT OR REPLACE INTO trading_eligibility
            (symbol, state, reason, data_quality_score, row_count, last_checked)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (symbol, state, reason, quality_score, row_count, now))

        stats[state] += 1

        # Log details for non-ACTIVE symbols
        if state != "ACTIVE":
            logger.info(f"{symbol}: {state} - {reason}")

    conn.commit()

    # Print summary
    logger.info("=" * 60)
    logger.info("Symbol Classification Summary:")
    logger.info(f"  ACTIVE:      {stats['ACTIVE']:4d} symbols (ready for trading)")
    logger.info(f"  DEGRADED:    {stats['DEGRADED']:4d} symbols (insufficient history)")
    logger.info(f"  QUARANTINED: {stats['QUARANTINED']:4d} symbols (quality issues)")
    logger.info(f"  TOTAL:       {sum(stats.values()):4d} symbols")
    logger.info("=" * 60)

    return stats


def main():
    """Main execution."""
    if not os.path.exists(DB_PATH):
        logger.error(f"Database not found: {DB_PATH}")
        logger.error("Please run ingestion first to populate price_history")
        sys.exit(1)

    logger.info(f"Connecting to database: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)

    try:
        stats = update_trading_eligibility(conn)

        # Exit with error if no ACTIVE symbols
        if stats and stats.get("ACTIVE", 0) == 0:
            logger.error("No ACTIVE symbols found! Check data ingestion and quality.")
            sys.exit(2)

        logger.info("Trading eligibility updated successfully")

    except Exception as e:
        logger.error(f"Error updating trading eligibility: {e}", exc_info=True)
        sys.exit(1)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
