"""
tools/repair_ingest.py

Repair script to re-ingest history for failing symbols.
"""
import logging
import sys
import os
import pandas as pd
from datetime import datetime
import time

# Add project root to path
sys.path.append(os.getcwd())

from data.collectors.data_router import DataRouter
from database.manager import DatabaseManager
from database.schema import SymbolGovernanceRecord, DailyPriceRecord

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("REPAIR_INGEST")

def repair_symbol(router: DataRouter, db: DatabaseManager, symbol: str):
    logger.info(f"Attempting repair for {symbol}...")

    # Try to fetch 10 years of history
    start_date = "2015-01-01"

    try:
        # Use DataRouter with long history allowed
        # It will try best provider (likely Yahoo for this fallback)
        df = router.get_price_history(
            ticker=symbol,
            start_date=start_date,
            allow_long_history=True
        )

        if df is None or df.empty:
            logger.error(f"Failed to fetch data for {symbol} (Empty result)")
            return False

        # Check if we got enough data
        if len(df) < 1260:
            logger.warning(f"Fetched data for {symbol} but insufficient history ({len(df)} rows)")

        # Validate Quality
        quality = router._validate_data_quality(df)
        score = quality['score']
        reasons = quality['reason_codes']

        logger.info(f"Fetched {len(df)} rows for {symbol}. Quality Score: {score:.2f}")

        # Convert DF to DailyPriceRecord objects
        records = []
        ingest_ts = datetime.utcnow().isoformat()

        # Ensure index is DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
             df.index = pd.to_datetime(df.index)

        # Standardize columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
             if col not in df.columns:
                 df[col] = 0.0

        for date, row in df.iterrows():
             try:
                 record = DailyPriceRecord(
                     symbol=symbol,
                     date=date.strftime('%Y-%m-%d'),
                     open=float(row.get('Open', 0)),
                     high=float(row.get('High', 0)),
                     low=float(row.get('Low', 0)),
                     close=float(row.get('Close', 0)),
                     volume=float(row.get('Volume', 0)),
                     vwap=float(row.get('VWAP', 0)) if 'VWAP' in row else None,
                     trade_count=int(row.get('TradeCount', 0)) if 'TradeCount' in row else None,
                     adjusted_close=float(row.get('Adj Close', row.get('Close', 0))),
                     provider="repair_fallback",
                     raw_hash="repair_hash",
                     validation_flags={},
                     ingestion_timestamp=ingest_ts
                 )
                 records.append(record)
             except Exception as e:
                 # Skip bad row
                 pass

        if not records:
             logger.error(f"No valid records created for {symbol}")
             return False

        # Persist
        db.upsert_price_history(records)

        # Update Governance/Confidence
        state = "ACTIVE" if len(df) >= 1260 and score >= 0.6 else "QUARANTINED"
        reason = "Repaired" if state == "ACTIVE" else f"Repaired but insufficient: {len(df)} rows"

        gov_record = SymbolGovernanceRecord(
            symbol=symbol,
            history_rows=len(df),
            data_quality=score,
            state=state,
            reason=reason,
            last_checked_ts=datetime.utcnow().isoformat(),
            metadata={"source": "repair_ingest", "quality_reasons": reasons}
        )

        try:
            db.upsert_symbol_governance(gov_record)
        except AttributeError:
             pass

        # Also write to data_quality table
        try:
            with db.get_connection() as conn:
                conn.execute(
                    "INSERT INTO data_quality (symbol, quality_score, failure_reasons, provider, recorded_at) VALUES (?, ?, ?, ?, ?)",
                    (symbol, score, str(reasons), "repair_router", datetime.utcnow().isoformat())
                )
        except Exception as e:
            pass

        return True

    except Exception as e:
        logger.error(f"Crash during repair of {symbol}: {e}")
        return False

def main():
    if len(sys.argv) < 2:
        try:
            df_low = pd.read_csv("low_quality.csv")
            symbols = df_low['symbol'].tolist()
        except Exception:
            print("Usage: python tools/repair_ingest.py <symbol> ...")
            return
    else:
        symbols = sys.argv[1:]

    print(f"Starting repair for {len(symbols)} symbols...")

    router = DataRouter()
    db = DatabaseManager()

    success_count = 0
    for i, sym in enumerate(symbols):
        print(f"[{i+1}/{len(symbols)}] Repairing {sym}...")
        if repair_symbol(router, db, sym):
            success_count += 1
        time.sleep(0.5)

    print(f"Repair complete. Success: {success_count}/{len(symbols)}")

if __name__ == "__main__":
    main()
