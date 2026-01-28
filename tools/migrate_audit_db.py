"""
Audit Database Migration Tool.
Creates SQLite database, migrates JSONL records, and manages rotation.
"""

import sqlite3
import json
import os
import argparse
from datetime import datetime
import shutil

AUDIT_DIR = 'runtime'
AUDIT_DB_PATH = os.path.join(AUDIT_DIR, 'audit.db')
AUDIT_JSONL_PATH = os.path.join(AUDIT_DIR, 'audit.log')
SCHEMA_PATH = 'audit/decision_schema.sql'


def create_database():
    """Create SQLite database with schema"""
    print(f"Creating database at {AUDIT_DB_PATH}...")

    conn = sqlite3.connect(AUDIT_DB_PATH)
    cursor = conn.cursor()

    # Read schema
    if os.path.exists(SCHEMA_PATH):
        with open(SCHEMA_PATH, 'r') as f:
            schema = f.read()
        cursor.executescript(schema)
        print("✓ Schema loaded from file")
    else:
        # Fallback inline schema
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cycle_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                data_providers TEXT NOT NULL,
                alphas TEXT NOT NULL,
                sigmas TEXT NOT NULL,
                conviction REAL NOT NULL,
                conviction_zscore REAL NOT NULL,
                risk_checks TEXT NOT NULL,
                pm_override TEXT NOT NULL,
                final_decision TEXT NOT NULL,
                reason_codes TEXT NOT NULL,
                order_data TEXT,
                raw_traceback TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_cycle_id ON decisions(cycle_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_symbol ON decisions(symbol)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_final_decision ON decisions(final_decision)')
        print("✓ Schema created inline")

    conn.commit()
    conn.close()
    print(f"✓ Database created successfully")


def migrate_jsonl():
    """Migrate existing JSONL records to SQLite"""
    if not os.path.exists(AUDIT_JSONL_PATH):
        print(f"No JSONL file found at {AUDIT_JSONL_PATH}, skipping migration")
        return

    print(f"Migrating records from {AUDIT_JSONL_PATH}...")

    conn = sqlite3.connect(AUDIT_DB_PATH)
    cursor = conn.cursor()

    migrated = 0
    errors = 0

    with open(AUDIT_JSONL_PATH, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                record = json.loads(line.strip())

                # Extract fields (handle both old and new formats)
                cursor.execute('''
                    INSERT INTO decisions (
                        cycle_id, symbol, timestamp,
                        data_providers, alphas, sigmas,
                        conviction, conviction_zscore,
                        risk_checks, pm_override,
                        final_decision, reason_codes,
                        order_data, raw_traceback
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    record.get('cycle_id', 'UNKNOWN'),
                    record.get('symbol', 'UNKNOWN'),
                    record.get('timestamp', datetime.now().isoformat()),
                    json.dumps(record.get('data_providers', {})),
                    json.dumps(record.get('alphas', {})),
                    json.dumps(record.get('sigmas', {})),
                    record.get('conviction', 0.0),
                    record.get('conviction_zscore', 0.0),
                    json.dumps(record.get('risk_checks', [])),
                    record.get('pm_override', 'UNKNOWN'),
                    record.get('final_decision', 'ERROR'),
                    json.dumps(record.get('reason_codes', [])),
                    json.dumps(record.get('order')) if record.get('order') else None,
                    record.get('raw_traceback')
                ))

                migrated += 1

                if migrated % 100 == 0:
                    print(f"  Migrated {migrated} records...")

            except Exception as e:
                errors += 1
                print(f"  Error on line {line_num}: {e}")

    conn.commit()
    conn.close()

    print(f"✓ Migration complete: {migrated} records migrated, {errors} errors")


def rotate_logs(keep_days: int = 30):
    """Rotate old audit logs"""
    print(f"Rotating logs (keeping last {keep_days} days)...")

    # Archive current JSONL
    if os.path.exists(AUDIT_JSONL_PATH):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_path = f"{AUDIT_JSONL_PATH}.{timestamp}"
        shutil.copy2(AUDIT_JSONL_PATH, archive_path)
        print(f"✓ Archived JSONL to {archive_path}")

    # TODO: Implement cleanup of old archives based on keep_days
    print(f"✓ Rotation complete")


def main():
    parser = argparse.ArgumentParser(description="Audit Database Migration Tool")
    parser.add_argument('--create', action='store_true', help="Create database")
    parser.add_argument('--migrate', action='store_true', help="Migrate JSONL to SQLite")
    parser.add_argument('--rotate', action='store_true', help="Rotate logs")
    parser.add_argument('--all', action='store_true', help="Run all operations")
    parser.add_argument('--keep-days', type=int, default=30, help="Days to keep in rotation")

    args = parser.parse_args()

    if args.all or args.create:
        create_database()

    if args.all or args.migrate:
        migrate_jsonl()

    if args.all or args.rotate:
        rotate_logs(args.keep_days)

    if not any([args.create, args.migrate, args.rotate, args.all]):
        parser.print_help()


if __name__ == "__main__":
    main()
