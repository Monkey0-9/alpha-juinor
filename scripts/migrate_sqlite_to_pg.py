
import os
import sys
import sqlite3
import logging
import argparse
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.manager import DatabaseManager, get_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DB_MIGRATION")

def migrate(sqlite_path: str, pg_url: str):
    """
    Migrate data from SQLite to Postgres.
    """
    if not os.path.exists(sqlite_path):
        logger.error(f"SQLite DB not found at {sqlite_path}")
        return

    logger.info(f"Connecting to SQLite: {sqlite_path}")
    sqlite_conn = sqlite3.connect(sqlite_path)

    logger.info(f"Connecting to Postgres...")
    pg_engine = create_engine(pg_url)

    # Tables to migrate
    tables = [
        'price_history', 'price_history_intraday', 'symbol_governance',
        'features', 'ingestion_audit', 'data_quality', 'provider_metrics'
    ]

    for table in tables:
        try:
            logger.info(f"Migrating table: {table}...")
            df = pd.read_sql(f"SELECT * FROM {table}", sqlite_conn)
            if df.empty:
                logger.info(f"Skipping empty table: {table}")
                continue

            # Write to Postgres
            # using chunksize for larger tables
            df.to_sql(table, pg_engine, if_exists='append', index=False, chunksize=1000)
            logger.info(f"Migrated {len(df)} rows to {table}.")

        except Exception as e:
            logger.error(f"Failed to migrate table {table}: {e}")

    logger.info("Migration complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate SQLite to Postgres")
    parser.add_argument("--sqlite", default="runtime/institutional_trading.db", help="Path to SQLite DB")
    parser.add_argument("--pg_url", required=True, help="Postgres Connection URL")

    args = parser.parse_args()
    migrate(args.sqlite, args.pg_url)
