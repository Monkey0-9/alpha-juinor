"""
scripts/migrate_pnl_attribution.py

Adds pnl_attribution_daily table.
"""
import sys
import os
import sqlite3
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MIGRATE_PNL")

def run_migration():
    db_path = "runtime/institutional_trading.db"
    if not os.path.exists(db_path):
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        logger.info("Creating pnl_attribution_daily table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pnl_attribution_daily (
                date TEXT NOT NULL,
                symbol TEXT NOT NULL,
                alpha_bps REAL,
                beta REAL,
                market_contribution REAL,
                residual_noise REAL,
                r_squared REAL,
                correlation REAL,
                treynor_ratio REAL,
                information_ratio REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY(date, symbol)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pnl_attr_date ON pnl_attribution_daily(date)")
        conn.commit()
    except Exception as e:
        logger.error(f"Migration failed: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    run_migration()
