
import logging
import json
import pandas as pd
from typing import List
from pathlib import Path

logger = logging.getLogger(__name__)

class UnifiedUniverseManager:
    """
    Manages the 'Investable Universe'.
    Institutional Feature: Auto-Rotation based on Liquidity & Tradability.
    """
    def __init__(self, config_path: str = "configs/universe.json"):
        if not isinstance(config_path, (str, Path)):
             raise TypeError(f"config_path must be a string or Path, got {type(config_path)}")
        
        self.config_path = Path(config_path)
        self.cache_path = Path("data/cache/universe_discovery.parquet")
        # Default core universe (Defensive + Growth + Crypto)
        self.core_tickers = [
            "SPY", "QQQ", "IWM", "TLT", "GLD",  # Core Macro
            "AAPL", "MSFT", "NVDA", "GOOGL",    # Mega Cap
            "BTC-USD", "ETH-USD"                # Crypto
        ]
        
    def get_active_universe(self) -> List[str]:
        """Returns the current active ticker list."""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    data = json.load(f)
                    return data.get("active_tickers", self.core_tickers)
            except Exception as e:
                logger.error(f"Failed to load universe config: {e}")
                return self.core_tickers
        return self.core_tickers

    def rotate_universe(self, provider, max_size: int = 25) -> List[str]:
        """
        Dynamic Universe Rotation.
        1. Validates Core Tickers.
        2. Scans for top liquidity candidates (Mock implementation for now or expanded list).
        3. Caps at max_size.
        """
        logger.info("[ROTATION] ROTATING UNIVERSE (Institutional Quality Check)...")
        
        candidates = self.core_tickers.copy()
        
        # In a real system, we would fetch a screener result here.
        # For this implementation, we validate the existing candidates + potentially some watch list
        
        validated = []
        stats = []
        
        # Check Tradability & Liquidity
        for tk in candidates:
            try:
                # Blacklist Filter (Institutional Requirement)
                # Filter out Preferreds, Warrants, Rights, and other non-tradables
                blacklist_suffixes = ('.PRI', '.PR', '.WS', '.RT', '.P', '.W')
                if tk.upper().endswith(blacklist_suffixes) or 'WARRANT' in tk.upper():
                     logger.warning(f"Universe Rotation: Dropping {tk} (Blacklisted Instrument Type)")
                     continue

                # Quick check: Get yesterday's data
                quote = provider.get_latest_price(tk)
                if quote is None or quote <= 0:
                    logger.warning(f"Universe Rotation: Dropping {tk} (No Quote)")
                    continue
                    
                # Price Filter (Penny stock filter)
                if "-USD" not in tk and quote < 5.0:
                    logger.warning(f"Universe Rotation: Dropping {tk} (Price ${quote} < $5)")
                    continue
                    
                validated.append(tk)
            except Exception as e:
                logger.warning(f"Universe Rotation: Error checking {tk}: {e}")
                
        # Limit Size (Sort by priority - Core first, then others)
        final_list = validated[:max_size]
        
        # Save new state
        try:
            with open(self.config_path, 'w') as f:
                json.dump({"active_tickers": final_list, "last_updated": str(pd.Timestamp.now())}, f, indent=4)
            logger.info(f"Universe Rotation Complete. Active Size: {len(final_list)}")
        except Exception as e:
            logger.error(f"Failed to save universe: {e}")
            
        return final_list

    def discover_and_filter(self, universe_config: dict, force_refresh: bool = False) -> List[str]:
        """
        Retrieves the active universe.
        Merges config tickers with persistent managed tickers.
        Applies blacklist.
        """
        # 1. Start with managed universe
        active = self.get_active_universe()
        
        # 2. Merge with config if provided
        config_tickers = universe_config.get("tickers", [])
        if config_tickers:
             # Merge and dedup
             active = list(set(active + config_tickers))
             
        # 3. Apply Blacklist Hardening
        final_list = []
        blacklist_suffixes = ('.PRI', '.PR', '.WS', '.RT', '.P', '.W')
        for tk in active:
             if tk.upper().endswith(blacklist_suffixes) or 'WARRANT' in tk.upper():
                  continue
             final_list.append(tk)
             
        # 4. Limit size if too large (Safety)
        if len(final_list) > 500:
             logger.warning(f"Universe truncated from {len(final_list)} to 500.")
             final_list = final_list[:500]
             
        # Hardening: Save Metadata Cache for main.py usage
        try:
             # Create basic metadata dataframe
             meta_df = pd.DataFrame({"symbol": final_list})
             # Add default columns expected by Allocator/RiskManager
             meta_df["market_cap"] = 1e9 # Default to mid-cap to avoid bucket issues
             meta_df["sector"] = "Unknown"
             
             self.cache_path.parent.mkdir(parents=True, exist_ok=True)
             meta_df.to_parquet(self.cache_path)
        except Exception as e:
             logger.warning(f"Failed to save universe metadata cache: {e}")
             
        return final_list
