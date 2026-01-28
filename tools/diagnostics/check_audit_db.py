#!/usr/bin/env python3
"""
Check execution decisions audit database
"""
import sqlite3
from datetime import datetime, timedelta

db_path = "audit/execution_decisions.db"

try:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    print(f"\n{'='*70}")
    print(f"EXECUTION DECISIONS AUDIT")
    print(f"{'='*70}")

    # Check if table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    print(f"\nTables found: {', '.join(tables)}")

    if not tables:
        print("\nNo tables found in audit database")
    else:
        # Assume main table is first one or named execution_decisions
        table_name = 'execution_decisions' if 'execution_decisions' in tables else tables[0]

        # Get total count
        cursor.execute(f"SELECT COUNT(*) as count FROM {table_name}")
        total = cursor.fetchone()[0]
        print(f"\nTotal decisions logged: {total}")

        # Group by decision type
        print(f"\n{'='*70}")
        print(f"DECISIONS BY TYPE")
        print(f"{'='*70}")
        cursor.execute(f"""
            SELECT decision, COUNT(*) as count
            FROM {table_name}
            GROUP BY decision
            ORDER BY count DESC
        """)
        for row in cursor.fetchall():
            print(f"  {row['decision']}: {row['count']}")

        # Recent decisions
        print(f"\n{'='*70}")
        print(f"RECENT DECISIONS (Last 20)")
        print(f"{'='*70}")
        cursor.execute(f"""
            SELECT * FROM {table_name}
            ORDER BY timestamp DESC
            LIMIT 20
        """)

        for i, row in enumerate(cursor.fetchall(), 1):
            print(f"\n{i}. {row['symbol']} - {row['decision']}")
            print(f"   Time: {row['timestamp'][:19] if 'timestamp' in row.keys() else 'N/A'}")
            if 'reason_codes' in row.keys():
                print(f"   Reasons: {row['reason_codes']}")
            if 'target_weight' in row.keys():
                print(f"   Target Weight: {row['target_weight']:.4f}")
            if 'notional_usd' in row.keys():
                print(f"   Notional: ${row['notional_usd']:.2f}")

        # Decisions in last hour
        print(f"\n{'='*70}")
        print(f"DECISIONS IN LAST HOUR")
        print(f"{'='*70}")
        one_hour_ago = (datetime.utcnow() - timedelta(hours=1)).isoformat()
        cursor.execute(f"""
            SELECT decision, COUNT(*) as count
            FROM {table_name}
            WHERE timestamp > ?
            GROUP BY decision
        """, (one_hour_ago,))

        decisions_last_hour = cursor.fetchall()
        if not decisions_last_hour:
            print("  No decisions in last hour")
        else:
            for row in decisions_last_hour:
                print(f"  {row['decision']}: {row['count']}")

    print(f"\n{'='*70}\n")

except sqlite3.OperationalError as e:
    print(f"\nDatabase error: {e}")
    print("The audit database may not have been initialized yet")
except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
finally:
    if 'conn' in locals():
        conn.close()
