"""
tools/recompute_data_quality.py

Recompute data quality scores for ALL symbols.
"""
import sys
import os
import sqlite3
import pandas as pd
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json

# Add project root
sys.path.append(os.getcwd())

from data.collectors.data_router import DataRouter
from database.manager import DatabaseManager
from database.schema import SymbolGovernanceRecord

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RECOMPUTE_QUALITY")

BATCH_SIZE = 50
MAX_WORKERS = 10

def validate_dataframe(df):
    """
    Minimal validation logic consistent with DataRouter.
    """
    if df is None or df.empty:
        return 0.0, ["EMPTY"]

    reason_codes = []

    # 1. Missing Dates
    missing_dates_pct = df.isnull().sum().sum() / max(1, (len(df) * len(df.columns)))

    # 2. Duplicates
    duplicate_pct = df.index.duplicated().sum() / max(1, len(df))

    # 3. Bad Prices
    zero_negative_flag = 0.0
    cols = [c for c in ["Open", "High", "Low", "Close"] if c in df.columns]
    if cols and (df[cols] <= 0).any().any():
        zero_negative_flag = 1.0

    # 4. Spikes
    extreme_spike_flag = 0.0
    if "Volume" in df.columns:
        vol = df["Volume"]
        if vol.std() > 0:
             if (vol > vol.mean() + 6 * vol.std()).any():
                 extreme_spike_flag = 1.0

    penalty = (missing_dates_pct * 0.3) + (duplicate_pct * 0.2) + (zero_negative_flag * 0.2) + (extreme_spike_flag * 0.05)
    score = max(0.0, 1.0 - penalty)

    if score < 1.0:
        if missing_dates_pct > 0: reason_codes.append("MISSING_DATA")
        if duplicate_pct > 0: reason_codes.append("DUPLICATES")
        if zero_negative_flag > 0: reason_codes.append("BAD_PRICES")
        if extreme_spike_flag > 0: reason_codes.append("VOL_SPIKE")

    return score, reason_codes

def process_symbol(symbol, db):
    try:
        # Load data from DB directly to be fast
        with db.get_connection() as conn:
            # Load last 1260 rows if possible, or all
            df = pd.read_sql(f"SELECT * FROM price_history WHERE symbol='{symbol}' ORDER BY date DESC LIMIT 1500", conn)

        if df.empty:
            return symbol, 0.0, 0, ["NO_DATA"]

        # Set index
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)

        score, reasons = validate_dataframe(df)
        return symbol, score, len(df), reasons

    except Exception as e:
        logger.error(f"Error processing {symbol}: {e}")
        return symbol, 0.0, 0, [str(e)]

def main():
    db = DatabaseManager()

    # Get all symbols
    with db.get_connection() as conn:
        cursor = conn.execute("SELECT DISTINCT symbol FROM price_history")
        symbols = [r[0] for r in cursor.fetchall()]

    logger.info(f"Recomputing quality for {len(symbols)} symbols...")

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_symbol, sym, db): sym for sym in symbols}

        comp_count = 0
        for future in as_completed(futures):
            results.append(future.result())
            comp_count += 1
            if comp_count % 100 == 0:
                print(f"Processed {comp_count}/{len(symbols)}", end="\r")

    print(f"\nCompleted recompute. Updating database...")

    # Bulk update
    gov_updates = []
    quality_logs = []
    ts = datetime.utcnow().isoformat()

    for sym, score, rows, reasons in results:
        state = "ACTIVE" if rows >= 1260 and score >= 0.75 else "QUARANTINED"
        reason = "Recomputed" if state == "ACTIVE" else f"Low Quality: {score:.2f} or Rows: {rows}"

        gov_updates.append((rows, score, state, reason, ts, sym))
        quality_logs.append((sym, score, json.dumps(reasons), "recompute", ts))

    with db.get_connection() as conn:
        conn.executemany("""
            UPDATE symbol_governance
            SET history_rows=?, data_quality=?, state=?, reason=?, last_checked_ts=?
            WHERE symbol=?
        """, gov_updates)

        # Insert into data_quality
        conn.executemany("""
            INSERT INTO data_quality (symbol, quality_score, validation_flags, provider, recorded_at)
            VALUES (?, ?, ?, ?, ?)
        """, quality_logs)

    logger.info("Database updated successfully.")

    # Re-run Update Universe
    import tools.update_universe
    tools.update_universe.main()

if __name__ == "__main__":
    main()
