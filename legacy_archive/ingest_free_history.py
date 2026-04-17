
import os
import sys
import json
import logging
import hashlib
import pandas as pd
from datetime import datetime, timedelta
from typing import List

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger("FREE_INGEST")

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mini_quant_fund.data.collectors.data_router import DataRouter
from mini_quant_fund.database.manager import DatabaseManager
from mini_quant_fund.database.schema import DailyPriceRecord, SymbolGovernanceRecord

def ingest_free_history(symbols: List[str] = None, years: int = 5):
    """
    Ingest historical data for symbols using FREE sources (Yahoo/Stooq).
    """
    db = DatabaseManager()
    router = DataRouter(enable_cache=True)
    
    if not symbols:
        # Load from universe
        with open('configs/universe.json', 'r') as f:
            universe = json.load(f)
        symbols = universe.get('symbols', [])[:100] # Cap at 100
        
    logger.info(f"🚀 Starting Free History Ingestion for {len(symbols)} symbols ({years} years)")
    
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365 * years)).strftime('%Y-%m-%d')
    
    success_count = 0
    
    for symbol in symbols:
        try:
            logger.info(f"⏳ Fetching {symbol}...")
            # We use allow_long_history=True and force Yahoo/Stooq via the router
            df = router.get_price_history(
                symbol, 
                start_date=start_date, 
                end_date=end_date, 
                allow_long_history=True
            )
            
            if not df.empty:
                # Convert to DailyPriceRecord
                records = []
                for idx, row in df.iterrows():
                    date_str = idx.strftime('%Y-%m-%d')
                    # Generate a unique hash for provenance
                    raw_hash = hashlib.md5(f"{symbol}{date_str}".encode()).hexdigest()
                    
                    records.append(DailyPriceRecord(
                        symbol=symbol,
                        date=date_str,
                        open=float(row['Open']),
                        high=float(row['High']),
                        low=float(row['Low']),
                        close=float(row['Close']),
                        adjusted_close=float(row.get('Adjusted_Close', row['Close'])),
                        volume=int(row['Volume']),
                        vwap=float(row['Close']), # Approximation
                        trade_count=0, # Unavailable in free data
                        provider='yahoo',
                        raw_hash=raw_hash,
                        ingestion_timestamp=datetime.now().isoformat()
                    ))
                
                # Save to database
                db.upsert_daily_prices_batch(records)
                
                # Activate symbol in governance
                # Note: Table has history_rows, data_quality, state, reason, last_checked_ts
                gov_record = SymbolGovernanceRecord(
                    symbol=symbol,
                    history_rows=len(df),
                    data_quality=0.95, # High confidence in Yahoo
                    state='ACTIVE',
                    reason='Free history ingested via Yahoo',
                    last_checked_ts=datetime.now().isoformat()
                )
                db.upsert_symbol_governance(gov_record)
                
                logger.info(f"✅ Ingested {len(df)} bars for {symbol}")
                success_count += 1
            else:
                logger.warning(f"❌ No data found for {symbol}")
                
        except Exception as e:
            logger.error(f"💥 Failed to ingest {symbol}: {e}")
            
    logger.info(f"🏁 Ingestion Complete: {success_count}/{len(symbols)} symbols updated.")
    print(f"\n✨ DATABASE WARMED UP: {success_count} symbols now have full institutional lookback.")

if __name__ == "__main__":
    # To save time in this demo, we'll just do a subset if no args
    subset = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "JNJ", 
              "WMT", "PG", "MA", "UNH", "HD", "DIS", "BAC", "ADBE", "CRM", "PYPL"]
    ingest_free_history(symbols=subset, years=5)
