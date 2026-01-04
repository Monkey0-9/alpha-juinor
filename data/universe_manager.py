# data/universe_manager.py
import os
import time
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class UnifiedUniverseManager:
    """
    Institutional Universe Discovery & Filtering Engine.
    Exchanges: NYSE, NASDAQ, AMEX.
    Filters: Liquidity (ADV), Market Cap, Price, Tradability.
    """
    
    def __init__(self, provider, cache_dir: str = "data/cache"):
        self.provider = provider
        self.cache_dir = cache_dir
        self.cache_path = os.path.join(cache_dir, "universe_manifest.parquet")
        os.makedirs(cache_dir, exist_ok=True)

    def discover_and_filter(self, config: Dict, force_refresh: bool = False) -> List[str]:
        """Main entry point for daily universe selection."""
        # 1. Load Cache if valid
        if not force_refresh and os.path.exists(self.cache_path):
            mtime = os.path.getmtime(self.cache_path)
            if (time.time() - mtime) < 86400: # 24 hours
                logger.info("Loading universe from cache...")
                df = pd.read_parquet(self.cache_path)
                return self._get_tickers_from_df(df)

        logger.info("Refreshing full-market universe discovery...")
        
        # 2. Fetch Assets from Broker (Alpaca / Exchange Listings)
        # Pseudocode for provider.get_assets()
        raw_assets = self.provider.get_all_assets() # Needs implementation in AlpacaDataProvider
        
        # 3. Initial Structural Filter
        valid_exchanges = config.get('exchanges', ['NYSE', 'NASDAQ', 'AMEX'])
        universe = [
            a for a in raw_assets 
            if a['exchange'] in valid_exchanges 
            and a['tradable'] 
            and a['status'] == 'active'
        ]
        
        # 4. Data Enrichment (ADV, Market Cap, Last Price)
        # This would optimally be a batch call
        enriched_df = self.provider.enrich_universe_data([a['symbol'] for a in universe])
        
        # 5. Apply Hard Filters
        f_cfg = config.get('filters', {})
        
        mask = (
            (enriched_df['avg_dollar_volume_30d'] >= f_cfg.get('min_dollar_volume_30d', 100000)) &
            (enriched_df['market_cap'] >= f_cfg.get('min_market_cap', 50000000)) &
            (enriched_df['listed_only'] == True)
        )
        
        if not f_cfg.get('allow_penny_stocks', False):
            mask &= (enriched_df['last_price'] >= f_cfg.get('min_price', 1.00))
            
        final_universe = enriched_df[mask].copy()
        
        # 6. Apply Bucketing for Metadata
        final_universe['bucket'] = self._assign_buckets(final_universe, config.get('thresholds', {}))
        
        # 7. Final Cache
        final_universe.to_parquet(self.cache_path)
        logger.info(f"Universe discovery complete. {len(final_universe)} assets passed filters.")
        
        return self._get_tickers_from_df(final_universe)

    def _assign_buckets(self, df, thresholds):
        l_cap = thresholds.get('large_cap', 10e9)
        m_cap = thresholds.get('mid_cap', 2e9)
        
        buckets = []
        for val in df['market_cap']:
            if val >= l_cap: buckets.append('Large')
            elif val >= m_cap: buckets.append('Mid')
            else: buckets.append('Small')
        return buckets

    def _get_tickers_from_df(self, df):
        return df['symbol'].tolist()
