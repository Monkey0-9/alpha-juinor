#!/usr/bin/env python3
"""
Institutional Backfill Verification Tool.

Verifies that all symbols have the required 5-year (1260 trading days) of history.

Usage:
    python verify_backfill.py              # Run verification
    python verify_backfill.py --symbols    # Show symbols with < 1260 rows
    python verify_backfill.py --json       # JSON output
    python verify_backfill.py --strict     # Exit with error if any symbol fails
"""

import argparse
import json
import logging
import sqlite3
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s'
)
logger = logging.getLogger("VERIFY_BACKFILL")

# Constants
REQUIRED_ROWS = 1260
DEFAULT_DB = "runtime/institutional_trading.db"


def verify_1260_rows(db_path: str = DEFAULT_DB, show_failed: bool = False) -> Dict[str, Any]:
    """
    Verify that all active symbols have >= 1260 rows of history.

    Args:
        db_path: Path to SQLite database
        show_failed: If True, print failed symbols

    Returns:
        Dict with verification results
    """
    result = {
        'timestamp': datetime.utcnow().isoformat(),
        'required_rows': REQUIRED_ROWS,
        'passed': False,
        'total_symbols': 0,
        'compliant': 0,
        'failed': 0,
        'failed_symbols': [],
        'details': []
    }

    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row

        # Get active symbols from governance table (only ACTIVE state)
        cursor = conn.execute("SELECT symbol FROM symbol_governance WHERE state = 'ACTIVE'")
        active_symbols = [r[0] for r in cursor.fetchall()]

        result['total_symbols'] = len(active_symbols)

        if not active_symbols:
            logger.warning("No ACTIVE symbols found in governance table")
            logger.info("Run ingest_history.py first to populate data")
            return result

        # Check row counts for active symbols
        placeholders = ','.join(['?'] * len(active_symbols))
        query = f"""
            SELECT symbol, COUNT(*) as row_count
            FROM price_history
            WHERE symbol IN ({placeholders})
            GROUP BY symbol
        """

        cursor = conn.execute(query, active_symbols)
        results = cursor.fetchall()

        count_map = {r['symbol']: r['row_count'] for r in results}

        compliant = 0
        failed_symbols = []
        failed_details = []

        for symbol in active_symbols:
            count = count_map.get(symbol, 0)
            if count >= REQUIRED_ROWS:
                compliant += 1
            else:
                failed_symbols.append(symbol)
                failed_details.append({
                    'symbol': symbol,
                    'actual': count,
                    'required': REQUIRED_ROWS,
                    'deficit': REQUIRED_ROWS - count
                })

        result['compliant'] = compliant
        result['failed'] = len(failed_symbols)
        result['failed_symbols'] = failed_symbols
        result['details'] = failed_details
        result['passed'] = len(failed_symbols) == 0

        conn.close()

        # Log results
        logger.info(f"Verification: {compliant}/{len(active_symbols)} symbols have >= {REQUIRED_ROWS} rows")

        if show_failed and failed_symbols:
            logger.warning("Symbols with insufficient history:")
            for detail in failed_details:
                logger.warning(f"  {detail['symbol']}: {detail['actual']} rows (need {detail['required']})")

    except Exception as e:
        logger.error(f"Verification failed: {e}")
        result['error'] = str(e)
        result['passed'] = False

    return result


def verify_data_quality(db_path: str = DEFAULT_DB, min_score: float = 0.6) -> Dict[str, Any]:
    """
    Verify that all active symbols have quality score >= min_score.

    Args:
        db_path: Path to SQLite database
        min_score: Minimum acceptable quality score

    Returns:
        Dict with verification results
    """
    result = {
        'timestamp': datetime.utcnow().isoformat(),
        'min_score': min_score,
        'passed': False,
        'total_symbols': 0,
        'compliant': 0,
        'failed': 0,
        'failed_symbols': [],
        'avg_score': 0.0
    }

    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row

        # Get latest quality score per active symbol
        cursor = conn.execute("""
            SELECT sg.symbol, MAX(dq.quality_score) as latest_score
            FROM symbol_governance sg
            LEFT JOIN data_quality dq ON sg.symbol = dq.symbol
            WHERE sg.state = 'ACTIVE'
            GROUP BY sg.symbol
        """)
        results = cursor.fetchall()

        total = len(results)
        compliant = 0
        failed_symbols = []
        score_sum = 0.0

        for row in results:
            score = row['latest_score'] or 0.0
            score_sum += score
            if score >= min_score:
                compliant += 1
            else:
                failed_symbols.append({
                    'symbol': row['symbol'],
                    'score': score
                })

        result['total_symbols'] = total
        result['compliant'] = compliant
        result['failed'] = len(failed_symbols)
        result['failed_symbols'] = [s['symbol'] for s in failed_symbols]
        result['failed_details'] = failed_symbols
        result['avg_score'] = score_sum / total if total > 0 else 0.0
        result['passed'] = len(failed_symbols) == 0

        conn.close()

        avg_pct = result['avg_score'] * 100
        logger.info(f"Quality: avg={avg_pct:.1f}%, {compliant}/{total} symbols >= {min_score}")

    except Exception as e:
        logger.error(f"Quality verification failed: {e}")
        result['error'] = str(e)
        result['passed'] = False

    return result


def print_summary(results: Dict[str, Any]) -> None:
    """Print human-readable summary."""
    print("\n" + "=" * 60)
    print("BACKFILL VERIFICATION SUMMARY")
    print("=" * 60)

    passed = results.get('passed', False)

    print(f"\nRequired rows per symbol: {results.get('required_rows', REQUIRED_ROWS)}")
    print(f"Compliant symbols: {results.get('compliant', 0)}/{results.get('total_symbols', 0)}")
    print(f"Failed symbols: {results.get('failed', 0)}")

    if results.get('failed_symbols'):
        print(f"\nFailed symbols (first 10):")
        for sym in results['failed_symbols'][:10]:
            print(f"  - {sym}")
        if len(results['failed_symbols']) > 10:
            print(f"  ... and {len(results['failed_symbols']) - 10} more")

    print("\n" + "=" * 60)
    if passed:
        print("STATUS: ALL SYMBOLS HAVE REQUIRED HISTORY")
    else:
        print("STATUS: VERIFICATION FAILED")
        print("Action required: Run ingest_history.py")
    print("=" * 60 + "\n")


def print_governance_halt(failed_symbols: List[str]) -> None:
    """Print institutional governance halt message."""
    print("\n[DATA_GOVERNANCE]")
    print("Missing historical data detected")
    print(f"Symbols affected: {len(failed_symbols)}")
    print(f"Required rows per symbol: {REQUIRED_ROWS}")
    print("Action required: Run ingest_history.py")
    print("System halted intentionally\n")


def main():
    parser = argparse.ArgumentParser(
        description="Institutional Backfill Verification"
    )
    parser.add_argument("--symbols", action="store_true",
                        help="Show symbols with insufficient history")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--quality", action="store_true", help="Also check data quality")
    parser.add_argument("--strict", action="store_true",
                        help="Exit with error if any symbol fails")
    parser.add_argument("--db", type=str, default=DEFAULT_DB, help="Database path")

    args = parser.parse_args()

    # Run history verification
    history_result = verify_1260_rows(db_path=args.db, show_failed=args.symbols)

    # Optionally run quality verification
    quality_result = None
    if args.quality:
        quality_result = verify_data_quality(db_path=args.db)

    # Combined results
    if quality_result:
        history_result['quality_check'] = quality_result
        all_passed = history_result['passed'] and quality_result['passed']
    else:
        all_passed = history_result['passed']

    history_result['all_checks_passed'] = all_passed

    if args.json:
        print(json.dumps(history_result, indent=2, default=str))
    else:
        print_summary(history_result)

        if not history_result['passed']:
            print_governance_halt(history_result.get('failed_symbols', []))

    # Exit with appropriate code
    if args.strict:
        sys.exit(0 if all_passed else 1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()

