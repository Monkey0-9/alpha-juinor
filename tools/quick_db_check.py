import sqlite3
import pandas as pd
import os

def check_db_comprehensive():
    conn = sqlite3.connect("runtime/institutional_trading.db")

    # 1. Get all symbols in price_history
    q_counts = "SELECT symbol, COUNT(*) as rows FROM price_history GROUP BY symbol"
    df_counts = pd.read_sql_query(q_counts, conn)

    # 2. Get latest quality scores
    q_quality = """
    SELECT symbol, quality_score, provider
    FROM data_quality
    WHERE (symbol, recorded_at) IN (
        SELECT symbol, MAX(recorded_at)
        FROM data_quality
        GROUP BY symbol
    )
    """
    df_quality = pd.read_sql_query(q_quality, conn)

    # Merge
    df_status = pd.merge(df_counts, df_quality, on="symbol", how="left")

    # Fill missing values
    df_status['quality_score'] = df_status['quality_score'].fillna(0.0)
    df_status['provider'] = df_status['provider'].fillna('unknown')

    # 3. Save to CSV
    os.makedirs("runtime", exist_ok=True)
    df_status.to_csv("runtime/backfill_status.csv", index=False)
    print(f"Status report saved to runtime/backfill_status.csv. Symbols: {len(df_status)}")

    # Summary Statistics
    ready = df_status[(df_status['rows'] >= 1260) & (df_status['quality_score'] >= 0.6)]
    print(f"\n--- BACKFILL SUMMARY ---")
    print(f"Total symbols found: {len(df_status)}")
    print(f"Symbols meeting Institutional Gate (>=1260 rows, >=0.6 quality): {len(ready)}")
    print(f"Percent Ready: {len(ready)/len(df_status):.1%}" if len(df_status) > 0 else "N/A")

    print("\nBottom 10 by row count:")
    print(df_status.sort_values("rows").head(10).to_string(index=False))

    conn.close()

if __name__ == "__main__":
    check_db_comprehensive()
