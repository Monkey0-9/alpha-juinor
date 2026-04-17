"""
scripts/migrate_strict_batch.py

Adds spike_flag and volume_spike_flag to price_history.
"""
import sys
import os
import sqlite3
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MIGRATE_STRICT")

def run_migration():
    db_path = "runtime/institutional_trading.db"
    if not os.path.exists(db_path):
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Check if columns exist
        cursor.execute("PRAGMA table_info(price_history)")
        columns = [info[1] for info in cursor.fetchall()]

        if "spike_flag" not in columns:
            logger.info("Adding spike_flag column...")
            cursor.execute("ALTER TABLE price_history ADD COLUMN spike_flag INTEGER DEFAULT 0")

        if "volume_spike_flag" not in columns:
            logger.info("Adding volume_spike_flag column...")
            cursor.execute("ALTER TABLE price_history ADD COLUMN volume_spike_flag INTEGER DEFAULT 0")

        conn.commit()
    except Exception as e:
        logger.error(f"Migration failed: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    run_migration()
