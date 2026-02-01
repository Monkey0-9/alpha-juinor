
import sqlite3
import logging
from pathlib import Path

# DB Path
DB_PATH = "runtime/institutional_trading.db"

def migrate():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("migration")

    path = Path(DB_PATH)
    if not path.exists():
        logger.info("Database does not exist. No migration needed.")
        return

    conn = sqlite3.connect(str(path))
    cursor = conn.cursor()

    try:
        # Check if vwap exists
        cursor.execute("PRAGMA table_info(price_history)")
        columns = [row[1] for row in cursor.fetchall()]

        if "vwap" not in columns:
            logger.info("Adding 'vwap' column...")
            cursor.execute("ALTER TABLE price_history ADD COLUMN vwap REAL")

        if "trade_count" not in columns:
            logger.info("Adding 'trade_count' column...")
            cursor.execute("ALTER TABLE price_history ADD COLUMN trade_count INTEGER")

        conn.commit()
        logger.info("Migration successful.")

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    migrate()
