"""
scripts/migrate_alpha_decay.py

Adds alpha_decay_metrics table.
"""
import sys
import os
import sqlite3
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MIGRATE_DECAY")

def run_migration():
    db_path = "runtime/institutional_trading.db"
    if not os.path.exists(db_path):
        logger.warning(f"Database not found: {db_path}")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        logger.info("Creating alpha_decay_metrics table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alpha_decay_metrics (
                strategy_id TEXT NOT NULL,
                date TEXT NOT NULL,
                rolling_ic_30d REAL,
                rolling_ic_60d REAL,
                rolling_ic_90d REAL,
                decay_score REAL,
                capacity_utilization REAL,
                status TEXT,
                recommendation TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY(strategy_id, date)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_decay_strategy ON alpha_decay_metrics(strategy_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_decay_date ON alpha_decay_metrics(date)")
        conn.commit()
        logger.info("Migration complete.")
    except Exception as e:
        logger.error(f"Migration failed: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    run_migration()
