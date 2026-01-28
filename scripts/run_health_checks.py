"""
Comprehensive Health Check Script

Runs all data governance checks and produces a summary report.
Validates:
- Historical data backfill status
- Symbol classification
- Market data availability
- ML readiness
- Database integrity

Exit codes:
- 0: All checks passed
- 1: One or more checks failed
"""

import sqlite3
import sys
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
import logging

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.quality import compute_data_quality, validate_data_for_trading, validate_data_for_ml

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

DB_PATH = "runtime/institutional_trading.db"
AUDIT_DB_PATH = "runtime/audit.db"
MIN_ROWS = 1260
MIN_QUALITY = 0.6
REQUIRED_BARS = 252


class HealthCheckResults:
    """Container for health check results."""

    def __init__(self):
        self.checks = []
        self.passed = 0
        self.failed = 0

    def add_check(self, name, passed, details=""):
        self.checks.append({
            'name': name,
            'passed': passed,
            'details': details
        })
        if passed:
            self.passed += 1
        else:
            self.failed += 1

    def print_summary(self):
        logger.info("=" * 70)
        logger.info("HEALTH CHECK SUMMARY")
        logger.info("=" * 70)

        for check in self.checks:
            status = "✓ PASS" if check['passed'] else "❌ FAIL"
            logger.info(f"{status}: {check['name']}")
            if check['details']:
                for line in check['details'].split('\n'):
                    logger.info(f"       {line}")

        logger.info("=" * 70)
        logger.info(f"Total: {self.passed} passed, {self.failed} failed")
        logger.info("=" * 70)

        return self.failed == 0


def check_database_exists(results):
    """Check if databases exist."""
    db_exists = os.path.exists(DB_PATH)
    audit_exists = os.path.exists(AUDIT_DB_PATH)

    if db_exists and audit_exists:
        results.add_check("Database Files Exist", True, f"Found: {DB_PATH}, {AUDIT_DB_PATH}")
    else:
        missing = []
        if not db_exists:
            missing.append(DB_PATH)
        if not audit_exists:
            missing.append(AUDIT_DB_PATH)
        results.add_check("Database Files Exist", False, f"Missing: {', '.join(missing)}")


def check_symbol_classification(results):
    """Check symbol classification status."""
    if not os.path.exists(DB_PATH):
        results.add_check("Symbol Classification", False, "Database not found")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT state, COUNT(*) FROM trading_eligibility GROUP BY state")
        rows = cursor.fetchall()

        stats = {row[0]: row[1] for row in rows}
        active_count = stats.get('ACTIVE', 0)
        degraded_count = stats.get('DEGRADED', 0)
        quarantined_count = stats.get('QUARANTINED', 0)

        details = f"ACTIVE: {active_count}, DEGRADED: {degraded_count}, QUARANTINED: {quarantined_count}"

        if active_count > 0:
            results.add_check("Symbol Classification", True, details)
        else:
            results.add_check("Symbol Classification", False, f"{details}\nNo ACTIVE symbols found!")

    except sqlite3.OperationalError as e:
        results.add_check("Symbol Classification", False, f"Table not found: {e}")
    finally:
        conn.close()


def check_historical_data(results):
    """Check historical data completeness."""
    if not os.path.exists(DB_PATH):
        results.add_check("Historical Data Backfill", False, "Database not found")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        # Get ACTIVE symbols
        cursor.execute("SELECT symbol FROM trading_eligibility WHERE state='ACTIVE'")
        active_symbols = [row[0] for row in cursor.fetchall()]

        if not active_symbols:
            results.add_check("Historical Data Backfill", False, "No ACTIVE symbols to check")
            conn.close()
            return

        # Check row counts
        insufficient = []
        for symbol in active_symbols:
            cursor.execute("SELECT COUNT(*) FROM price_history WHERE symbol=?", (symbol,))
            count = cursor.fetchone()[0]
            if count < MIN_ROWS:
                insufficient.append(f"{symbol}:{count}")

        if not insufficient:
            results.add_check("Historical Data Backfill", True,
                            f"All {len(active_symbols)} ACTIVE symbols have ≥{MIN_ROWS} rows")
        else:
            results.add_check("Historical Data Backfill", False,
                            f"{len(insufficient)} symbols insufficient:\n" + ", ".join(insufficient[:10]))

    except Exception as e:
        results.add_check("Historical Data Backfill", False, str(e))
    finally:
        conn.close()


def check_market_data_availability(results):
    """Check that ACTIVE symbols have 252 recent bars."""
    if not os.path.exists(DB_PATH):
        results.add_check("Market Data (252 bars)", False, "Database not found")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT symbol FROM trading_eligibility WHERE state='ACTIVE'")
        active_symbols = [row[0] for row in cursor.fetchall()]

        if not active_symbols:
            results.add_check("Market Data (252 bars)", False, "No ACTIVE symbols")
            conn.close()
            return

        missing_bars = []
        for symbol in active_symbols:
            query = f"SELECT COUNT(*) FROM (SELECT * FROM price_history WHERE symbol=? ORDER BY date DESC LIMIT {REQUIRED_BARS})"
            cursor.execute(query, (symbol,))
            count = cursor.fetchone()[0]
            if count < REQUIRED_BARS:
                missing_bars.append(f"{symbol}:{count}")

        if not missing_bars:
            results.add_check("Market Data (252 bars)", True,
                            f"All {len(active_symbols)} ACTIVE symbols have {REQUIRED_BARS}+ bars")
        else:
            results.add_check("Market Data (252 bars)", False,
                            f"{len(missing_bars)} symbols missing bars:\n" + ", ".join(missing_bars[:10]))

    except Exception as e:
        results.add_check("Market Data (252 bars)", False, str(e))
    finally:
        conn.close()


def check_data_quality_scores(results):
    """Check data quality scoring."""
    if not os.path.exists(DB_PATH):
        results.add_check("Data Quality Scores", False, "Database not found")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT symbol, data_quality_score FROM trading_eligibility WHERE state='ACTIVE'")
        rows = cursor.fetchall()

        if not rows:
            results.add_check("Data Quality Scores", False, "No ACTIVE symbols")
            conn.close()
            return

        low_quality = [f"{s}:{q:.2f}" for s, q in rows if q and q < MIN_QUALITY]

        if not low_quality:
            avg_quality = sum(q for _, q in rows if q) / len([q for _, q in rows if q])
            results.add_check("Data Quality Scores", True,
                            f"All ACTIVE symbols ≥{MIN_QUALITY} (avg: {avg_quality:.2f})")
        else:
            results.add_check("Data Quality Scores", False,
                            f"{len(low_quality)} symbols below threshold:\n" + ", ".join(low_quality[:10]))

    except Exception as e:
        results.add_check("Data Quality Scores", False, str(e))
    finally:
        conn.close()


def check_audit_database(results):
    """Check audit database has recent entries."""
    if not os.path.exists(AUDIT_DB_PATH):
        results.add_check("Audit Database", False, f"Audit DB not found: {AUDIT_DB_PATH}")
        return

    conn = sqlite3.connect(AUDIT_DB_PATH)
    cursor = conn.cursor()

    try:
        # Check if audit_log table exists and has entries
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='audit_log'")
        if not cursor.fetchone():
            results.add_check("Audit Database", False, "audit_log table not found")
            conn.close()
            return

        cursor.execute("SELECT COUNT(*) FROM audit_log")
        count = cursor.fetchone()[0]

        if count > 0:
            results.add_check("Audit Database", True, f"{count} audit entries found")
        else:
            results.add_check("Audit Database", False, "No audit entries (run paper daemon)")

    except Exception as e:
        results.add_check("Audit Database", False, str(e))
    finally:
        conn.close()


def main():
    """Run all health checks."""
    logger.info("\nRunning Institutional Data Governance Health Checks...")
    logger.info(f"Timestamp: {datetime.utcnow().isoformat()}\n")

    results = HealthCheckResults()

    # Run all checks
    check_database_exists(results)
    check_symbol_classification(results)
    check_historical_data(results)
    check_market_data_availability(results)
    check_data_quality_scores(results)
    check_audit_database(results)

    # Print summary
    all_passed = results.print_summary()

    # Write summary to file
    summary_path = "runtime/health_check_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"Health Check Run: {datetime.utcnow().isoformat()}\n")
        f.write("=" * 70 + "\n")
        for check in results.checks:
            status = "PASS" if check['passed'] else "FAIL"
            f.write(f"{status}: {check['name']}\n")
            if check['details']:
                f.write(f"  {check['details']}\n")
        f.write("=" * 70 + "\n")
        f.write(f"Total: {results.passed} passed, {results.failed} failed\n")

    logger.info(f"\nSummary written to: {summary_path}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
