import sqlite3
import json
import pandas as pd

def verify():
    conn = sqlite3.connect('runtime/institutional_trading.db')

    # Load universe
    try:
        with open("configs/universe.json", "r") as f:
            universe = json.load(f)
        tickers = universe.get("active_tickers", [])
    except:
        tickers = []

    print(f"[*] Universe size: {len(tickers)} symbols")

    # Check each ticker
    missing = []
    insufficient = []
    for ticker in tickers:
        try:
            query = "SELECT COUNT(*) FROM price_history WHERE symbol = ?"
            count = conn.execute(query, (ticker,)).fetchone()[0]
            if count == 0:
                missing.append(ticker)
            elif count < 1260:
                insufficient.append((ticker, count))
        except:
            missing.append(ticker)

    print("\n--- GOVERNANCE CHECK: DATA AVAILABILITY ---")
    print(f"[!] Missing completely: {len(missing)} symbols")
    if missing:
        print(f"    Sample missing: {missing[:10]}")
    print(f"[!] Insufficient data (<1260): {len(insufficient)} symbols")
    if insufficient:
        print(f"    Sample insufficient: {insufficient[:10]}")

    if not missing and not insufficient:
        print("[OK] All symbols in universe have 1260+ rows.")

    # Check for entitlement failures
    try:
        audit_query = "SELECT provider, COUNT(*) as count FROM ingestion_audit WHERE reason_code='ENTITLEMENT_FAILURE' GROUP BY provider"
        df_audit = pd.read_sql_query(audit_query, conn)
        if not df_audit.empty:
            print("\n--- ENTITLEMENT FAILURES ---")
            print(df_audit)
    except:
        pass

    conn.close()

if __name__ == "__main__":
    verify()
