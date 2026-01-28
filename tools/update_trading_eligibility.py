import sqlite3
import os

db_path = r"C:\mini-quant-fund\runtime\institutional_trading.db"

def update_eligibility():
    if not os.path.exists(db_path):
        print(f"Database not found: {db_path}")
        return

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Ensure table exists
    cur.execute("""
        CREATE TABLE IF NOT EXISTS trading_eligibility (
            symbol TEXT PRIMARY KEY,
            state TEXT,
            history_rows INTEGER,
            data_quality REAL,
            reason TEXT,
            last_checked TEXT
        )
    """)

    # Get all symbols from price_history
    cur.execute("""
        SELECT symbol, COUNT(*) as rows, AVG(quality_score) as dq
        FROM price_history
        LEFT JOIN data_quality USING(symbol)
        GROUP BY symbol
    """)
    results = cur.fetchall()

    from datetime import datetime
    now = datetime.now().isoformat()

    for symbol, rows, dq in results:
        dq = dq if dq is not None else 0.0

        # ACTIVE if rows >= 1260 AND data_quality >= 0.6.
        # DEGRADED if rows >= 1000 but <1260 OR 0.5 <= data_quality < 0.6.
        # QUARANTINED if rows < 1000 OR data_quality < 0.5.

        state = 'QUARANTINED'
        reason = 'INSUFFICIENT_DATA_OR_QUALITY'

        if rows >= 1260 and dq >= 0.6:
            state = 'ACTIVE'
            reason = 'MEETS_INSTITUTIONAL_GATE'
        elif (1000 <= rows < 1260) or (0.5 <= dq < 0.6):
            state = 'DEGRADED'
            reason = 'SUB_OPTIMAL_HISTORY_OR_QUALITY'

        cur.execute("""
            INSERT OR REPLACE INTO trading_eligibility (symbol, state, history_rows, data_quality, reason, last_checked)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (symbol, state, rows, dq, reason, now))

    conn.commit()
    print(f"Updated {len(results)} symbols in trading_eligibility.")
    conn.close()

if __name__ == "__main__":
    update_eligibility()
