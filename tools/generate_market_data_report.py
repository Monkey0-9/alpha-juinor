import pandas as pd
import sqlite3
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MarketDataCheck")

DB_PATH = "runtime/institutional_trading.db"
OUTPUT_FILE = "runtime/market_data_check.csv"
MIN_BARS = 252

def check_market_data():
    conn = sqlite3.connect(DB_PATH)

    # Get ACTIVE symbols
    logger.info("Fetching ACTIVE symbols...")
    active_symbols_df = pd.read_sql("SELECT symbol FROM trading_eligibility WHERE state='ACTIVE'", conn)
    active_symbols = active_symbols_df['symbol'].tolist()

    logger.info(f"Found {len(active_symbols)} ACTIVE symbols.")

    results = []

    for symbol in active_symbols:
        # Check row count in price_history for this symbol
        # We want to check RECENT data, but simply checking count is a good proxy if we assume ingestion is recent
        # Better: check max date

        query = f"SELECT COUNT(*) as count, MAX(date) as last_date FROM price_history WHERE symbol='{symbol}'"
        stats = pd.read_sql(query, conn).iloc[0]

        count = stats['count']
        last_date = stats['last_date']

        has_252 = count >= MIN_BARS

        results.append({
            "symbol": symbol,
            "rows": count,
            "last_date": last_date,
            "has_252_bars": has_252,
            "state": "ACTIVE"
        })

    conn.close()

    # Create DataFrame and save
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_FILE, index=False)
    logger.info(f"Report saved to {OUTPUT_FILE}")
    print(df)

if __name__ == "__main__":
    check_market_data()
