"""
scripts/migrate_week4.py

Section J: Decision Records & Manual Overrides.
"""
import sys
import os
import sqlite3
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MIGRATE_W4")

def run_migration():
    db_path = "runtime/institutional_trading.db"
    if not os.path.exists(db_path):
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 1. decision_records
    try:
        cursor.execute("SELECT id FROM decision_records LIMIT 1")
    except sqlite3.OperationalError:
        logger.info("Creating decision_records table...")
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS decision_records (
              id TEXT PRIMARY KEY,
              run_id TEXT,
              timestamp TIMESTAMP,
              symbol TEXT,
              final_decision TEXT,
              reason_codes JSON,
              allocations JSON,
              duels JSON,
              model_versions JSON,
              data_quality_score REAL,
              execution_id TEXT
            );
        ''')

    # 2. manual_overrides
    try:
        cursor.execute("SELECT id FROM manual_overrides LIMIT 1")
    except sqlite3.OperationalError:
        logger.info("Creating manual_overrides table...")
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS manual_overrides (
              id TEXT PRIMARY KEY,
              run_id TEXT,
              strategy_id TEXT,
              requested_by TEXT,
              justification TEXT,
              signoffs JSON,
              status TEXT,
              created_at TIMESTAMP
            );
        ''')

    conn.commit()
    conn.close()
    logger.info("Week 4 Migration Complete.")

if __name__ == "__main__":
    run_migration()
