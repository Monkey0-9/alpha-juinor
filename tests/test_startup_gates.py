#!/usr/bin/env python3
"""
Quick startup test to verify governance gates pass.
Since all 225 symbols now have >= 1260 rows, this should NOT halt.
"""

import sys
import os

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database.manager import DatabaseManager
from data.governance.governance_agent import SymbolGovernor

def test_governance_gates():
    """Test the governance gate sequence"""
    print("=" * 80)
    print("GOVERNANCE GATE TEST")
    print("=" * 80)

    # 1. Initialize database
    print("\n[1/4] Initializing database...")
    db = DatabaseManager()
    print("✓ Database initialized")

    # 2. Run symbol classification
    print("\n[2/4] Running symbol classification...")
    governor = SymbolGovernor()
    governor.classify_all()
    print("✓ Symbol classification complete")

    # 3. Get active symbols
    print("\n[3/4] Fetching ACTIVE symbols...")
    active_symbols = db.get_active_symbols()
    print(f"✓ Found {len(active_symbols)} ACTIVE symbols")

    if not active_symbols:
        print("❌ FAIL: No ACTIVE symbols found")
        return False

    # 4. Check 1260-row requirement
    print(f"\n[4/4] Checking 1260-row requirement for {len(active_symbols)} symbols...")

    conn = db.get_connection()
    placeholders = ','.join(['?'] * len(active_symbols))
    query = f"""
        SELECT symbol, COUNT(*) as row_count
        FROM price_history
        WHERE symbol IN ({placeholders})
        GROUP BY symbol
    """

    cursor = conn.execute(query, active_symbols)
    results = cursor.fetchall()
    count_map = {row['symbol']: row['row_count'] for row in results}

    missing = []
    for symbol in active_symbols:
        count = count_map.get(symbol, 0)
        if count < 1260:
            missing.append({'symbol': symbol, 'actual': count, 'required': 1260})

    if missing:
        print(f"❌ FAIL: {len(missing)} symbols missing required history")
        for m in missing[:5]:
            print(f"   - {m['symbol']}: {m['actual']}/1260 rows")
        return False

    print(f"✓ All {len(active_symbols)} symbols have >= 1260 rows")

    # Summary
    print("\n" + "=" * 80)
    print("GOVERNANCE GATE TEST: PASSED ✅")
    print("=" * 80)
    print(f"Active symbols: {len(active_symbols)}")
    print(f"Total rows: {sum(count_map.values()):,}")
    print(f"Average rows per symbol: {sum(count_map.values()) // len(count_map)}")
    print("\n✅ System is ready to start!")
    return True

if __name__ == "__main__":
    try:
        success = test_governance_gates()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
