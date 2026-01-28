"""
scripts/daily_attribution.py

Phase 2: P&L Decomposition Job.
Runs nightly to update pnl_attribution_daily table.
"""

import sys
import os
import argparse
import logging
import pandas as pd
from datetime import datetime, timedelta

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.manager import DatabaseManager
from risk.pnl_decomposition import PnLDecomposer, DecompositionResult
from database.schema import DecompositionRecord

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger("DAILY_ATTRIBUTION")

class DailyAttributionJob:
    def __init__(self, benchmark_symbol: str = "SPY", lookback_days: int = 90):
        self.db = DatabaseManager()
        self.decomposer = PnLDecomposer()
        self.benchmark = benchmark_symbol
        self.lookback = lookback_days

    def run(self, specific_date: str = None):
        """
        Run attribution for all active symbols.
        specific_date: "YYYY-MM-DD" or None (today)
        """
        target_date = specific_date or datetime.utcnow().strftime("%Y-%m-%d")
        logger.info(f"Running attribution for {target_date} (Lookback: {self.lookback}d)")

        # 1. Get Universe
        # Using db manager to get active symbols
        symbols = self.db.get_active_symbols()
        if not symbols:
            logger.warning("No active symbols found.")
            return

        # 2. Fetch Benchmark Data
        benchmark_prices = self._get_returns(self.benchmark, target_date, self.lookback)
        if benchmark_prices.empty:
            logger.error(f"Benchmark {self.benchmark} has no data. Aborting.")
            return

        benchmark_returns = benchmark_prices['close'].pct_change().dropna()

        # 3. Iterate Symbols
        results = []
        for symbol in symbols:
            if symbol == self.benchmark: continue

            try:
                asset_prices = self._get_returns(symbol, target_date, self.lookback)
                if asset_prices.empty:
                    continue

                asset_returns = asset_prices['close'].pct_change().dropna()

                # Decompose
                res = self.decomposer.decompose(symbol, asset_returns, benchmark_returns)

                if res.valid:
                    rec = DecompositionRecord(
                        date=target_date,
                        symbol=symbol,
                        alpha_bps=res.alpha_bps,
                        beta=res.beta,
                        market_contribution=res.beta * (benchmark_returns.iloc[-1] if not benchmark_returns.empty else 0), # Daily contrib
                        residual_noise=res.residual_vol,
                        r_squared=res.r_squared,
                        correlation=res.correlation,
                        treynor_ratio=res.treynor,
                        information_ratio=res.information_ratio
                    )
                    results.append(rec)
                    logger.info(f"[{symbol}] Beta: {res.beta:.2f}, Alpha: {res.alpha_bps:.1f} bps, R2: {res.r_squared:.2f}")
                else:
                    logger.debug(f"[{symbol}] Invalid decomposition")

            except Exception as e:
                logger.error(f"[{symbol}] Validation failed: {e}")

        # 4. Storage
        if results:
            self._store_results(results)
            logger.info(f"Stored {len(results)} attribution records.")

    def _get_returns(self, symbol, end_date, window):
        # Calculate start date
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        start_dt = end_dt - timedelta(days=window * 2) # Buffer for trading days

        prices_df = self.db.get_daily_prices(symbol, start_dt.strftime("%Y-%m-%d"), end_date)
        if prices_df.empty: return pd.DataFrame()

        # Ensure sorted
        prices_df.sort_values('date', inplace=True)
        return prices_df

    def _store_results(self, records):
        # Need to implement this method in manager or use adapter
        # Adapter likely not exposed directly on manager object if not defined.
        # But wait, manager.adapter IS accessible?
        # Or I can use execute_sql (if exposed) or add method to manager.
        # I SHOULD add clean method to manager to maintain abstraction.
        # For now, if I can't edit manager easily, I'll access adapter._cursor if possible or use raw sql.
        # Creating a helper or assuming manager.upsert_pnl_attribution exists (I should add it).

        # I will check manager content from view_file result.
        pass # Will be implemented by calling manager update

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", help="YYYY-MM-DD")
    parser.add_argument("--benchmark", default="SPY")
    args = parser.parse_args()

    job = DailyAttributionJob(args.benchmark)

    # We need to add persist method to manager.
    # Monkey patch for now or rely on next step to add it to manager
    def upsert_pnl_attribution_batch(self, records):
        with self.get_connection() as conn:
            data = [(r.date, r.symbol, r.alpha_bps, r.beta, r.market_contribution, r.residual_noise,
                     r.r_squared, r.correlation, r.treynor_ratio, r.information_ratio) for r in records]
            conn.executemany("""
                INSERT OR REPLACE INTO pnl_attribution_daily
                (date, symbol, alpha_bps, beta, market_contribution, residual_noise, r_squared, correlation, treynor_ratio, information_ratio)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, data)
            conn.commit()

    import types
    job.db.upsert_pnl_attribution_batch = types.MethodType(upsert_pnl_attribution_batch, job.db)
    job._store_results = job.db.upsert_pnl_attribution_batch

    job.run(args.date)
