
import sqlite3
import pandas as pd

DB="runtime/institutional_trading.db"

def check_market_data():
    con=sqlite3.connect(DB)
    try:
        syms = [r[0] for r in con.execute("SELECT symbol FROM trading_eligibility WHERE state='ACTIVE'").fetchall()]
        print(f"Checking {len(syms)} active symbols...")
        for s in syms:
            df = pd.read_sql_query("SELECT date FROM price_history WHERE symbol=? ORDER BY date DESC LIMIT 252", con, params=(s,))
            print(f"{s}: {len(df)} rows")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        con.close()

if __name__ == "__main__":
    check_market_data()
