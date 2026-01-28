#!/usr/bin/env python3
"""
PostgreSQL Database Status Verification Script.

This script verifies that PostgreSQL with TimescaleDB is properly configured
and accessible.

Usage:
    python scripts/verify_db_status_pg.py
"""

import os
import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from database.adapters.postgres_manager import PostgresManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def verify_postgres_status() -> bool:
    """Verify PostgreSQL database status."""
    print("=" * 60)
    print("PostgreSQL Database Status Verification")
    print("=" * 60)

    # Check environment variables
    print("\n[1/5] Checking Environment Variables...")
    required_vars = [
        ("POSTGRES_HOST", os.getenv("POSTGRES_HOST", "localhost")),
        ("POSTGRES_PORT", os.getenv("POSTGRES_PORT", "5432")),
        ("POSTGRES_DB", os.getenv("POSTGRES_DB", "mini_quant")),
        ("POSTGRES_USER", os.getenv("POSTGRES_USER", "mini_quant")),
    ]

    for name, value in required_vars:
        status = "✓" if value else "✗"
        print(f"  {status} {name}: {value}")

    # Check if password is set (masked)
    password = os.getenv("POSTGRES_PASSWORD", "")
    print(f"  {'✓' if password else '✗'} POSTGRES_PASSWORD: {'***' if password else 'NOT SET'}")

    # Initialize PostgreSQL manager
    print("\n[2/5] Initializing PostgreSQL Manager...")
    try:
        pg_manager = PostgresManager()
        print("  ✓ PostgreSQL Manager initialized")
    except Exception as e:
        print(f"  ✗ Failed to initialize: {e}")
        return False

    # Run health check
    print("\n[3/5] Running Health Check...")
    try:
        health = pg_manager.health_check()
        status = health.get("status", "unknown")
        if status == "healthy":
            print(f"  ✓ Database is healthy")
            print(f"    Engine: {health.get('engine', 'unknown')}")
            print(f"    Schema Version: {health.get('schema_version', 'unknown')}")
        else:
            print(f"  ✗ Database status: {status}")
            if "error" in health:
                print(f"    Error: {health['error']}")
            return False
    except Exception as e:
        print(f"  ✗ Health check failed: {e}")
        return False

    # Check table counts
    print("\n[4/5] Checking Table Counts...")
    table_counts = health.get("table_counts", {})
    for table, count in table_counts.items():
        print(f"  - {table}: {count} rows")

    # Test basic operations
    print("\n[5/5] Testing Basic Operations...")
    try:
        # Test get_active_symbols
        active_symbols = pg_manager.get_active_symbols()
        print(f"  ✓ Active symbols query: {len(active_symbols)} symbols")

        # Test symbol coverage
        coverage = pg_manager.get_symbol_coverage("2024-01-01", "2024-12-31")
        print(f"  ✓ Symbol coverage query: {coverage.get('symbols_with_data', 0)} symbols with data")

    except Exception as e:
        print(f"  ✗ Basic operations test failed: {e}")
        return False

    print("\n" + "=" * 60)
    print("PostgreSQL Verification: PASSED")
    print("=" * 60)

    return True


def main():
    """Main entry point."""
    success = verify_postgres_status()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
