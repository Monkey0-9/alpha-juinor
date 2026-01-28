"""
scripts/migrate_week3.py

Migration script to align SQLite DB with Week 3 Production Schema.
- Adds raw_row_json to price_history
- Adds row_count, data_quality_score to ingestion_audit
"""

import sys
import os
import sqlite3
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.manager import DatabaseManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MIGRATE")

def run_migration():
    db_path = "runtime/institutional_trading.db"

    if not os.path.exists(db_path):
        logger.info("No DB found, schema will be created fresh on first run.")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 1. price_history: Add raw_row_json
    try:
        cursor.execute("SELECT raw_row_json FROM price_history LIMIT 1")
    except sqlite3.OperationalError:
        logger.info("Adding raw_row_json to price_history...")
        cursor.execute("ALTER TABLE price_history ADD COLUMN raw_row_json TEXT")

    # 2. ingestion_audit: Add row_count, data_quality_score
    try:
        cursor.execute("SELECT row_count FROM ingestion_audit LIMIT 1")
    except sqlite3.OperationalError:
        logger.info("Adding row_count to ingestion_audit...")
        cursor.execute("ALTER TABLE ingestion_audit ADD COLUMN row_count INTEGER DEFAULT 0")

    try:
        cursor.execute("SELECT data_quality_score FROM ingestion_audit LIMIT 1")
    except sqlite3.OperationalError:
        logger.info("Adding data_quality_score to ingestion_audit...")
        cursor.execute("ALTER TABLE ingestion_audit ADD COLUMN data_quality_score REAL DEFAULT 0.0")

    conn.commit()
    conn.close()
    logger.info("Migration Week 3 complete.")

if __name__ == "__main__":
    run_migration()
