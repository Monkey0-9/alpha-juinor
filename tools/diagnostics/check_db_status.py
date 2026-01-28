#!/usr/bin/env python
"""Quick DB status check script."""
import sqlite3

DB = "runtime/institutional_trading.db"

def main():
    con = sqlite3.connect(DB)
    cur = con.cursor()

    # 1. Distinct symbols count
    result = cur.execute('SELECT COUNT(DISTINCT symbol) FROM price_history').fetchone()
    print("=" * 60)
    print(f"Distinct symbols in price_history: {result[0]}")
    print("=" * 60)

    # 2. Row counts per symbol (sorted ascending)
    print("\nSymbol row counts (sorted ascending, top 30):")
    print("-" * 60)
    results = cur.execute('''
        SELECT symbol, COUNT(*) as rows, MIN(date) as first, MAX(date) as last
        FROM price_history GROUP BY symbol ORDER BY rows ASC LIMIT 30
    ''').fetchall()
    for row in results:
        print(f"{row[0]}: {row[1]} rows | {row[2]} to {row[3]}")

    # 3. Active symbols count (using symbol_governance table)
    result = cur.execute("SELECT COUNT(*) FROM symbol_governance WHERE state='ACTIVE'").fetchone()
    print()
    print("=" * 60)
    print(f"Active symbols in symbol_governance: {result[0]}")
    print("=" * 60)

    # 4. Symbols with < 1260 rows (need backfill)
    print("\nSymbols with < 1260 rows (need backfill):")
    print("-" * 60)
    results = cur.execute('''
        SELECT symbol, COUNT(*) as rows
        FROM price_history GROUP BY symbol HAVING rows < 1260
    ''').fetchall()
    print(f"Count needing backfill: {len(results)}")
    for row in results:
        print(f"  {row[0]}: {row[1]} rows")

    # 5. Full summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total = cur.execute('SELECT COUNT(*) FROM price_history').fetchone()[0]
    print(f"Total rows in price_history: {total}")
    avg = cur.execute('SELECT AVG(cnt) FROM (SELECT COUNT(*) as cnt FROM price_history GROUP BY symbol)').fetchone()[0]
    print(f"Average rows per symbol: {avg:.1f}")

    con.close()
    print("\nDone.")

if __name__ == "__main__":
    main()

