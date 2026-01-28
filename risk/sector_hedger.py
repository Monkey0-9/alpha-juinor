import pandas as pd
from typing import Dict, List, Any
from risk.sector_mapping import get_sector, get_hedge_etf
import logging

# risk/sector_hedger.py

logger = logging.getLogger("SECTOR_HEDGER")

class SectorHedger:
    """
    Calculates sector exposure and generates neutralizing ETF trades.
    Target: Zero net exposure per sector.
    """

    def __init__(self, sector_limit: float = 0.15):
        self.sector_limit = sector_limit

    def calculate_hedge_overlay(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Input: Existing strategy weights.
        Output: Offsetting ETF weights to neutralize sector exposure.
        """
        sector_exposure = {}
        for symbol, weight in weights.items():
            sector = get_sector(symbol)
            sector_exposure[sector] = sector_exposure.get(sector, 0.0) + weight

        hedges = {}
        for sector, exposure in sector_exposure.items():
            if sector == "Unknown": continue

            # If net exposure in a sector is significant, neutralize it
            # HedgeWeight = -Exposure
            hedge_etf = get_hedge_etf(sector)

            if abs(exposure) > 0.001: # Threshold for tiny crumbs
                # Invert the exposure to create a hedge
                hedge_weight = -exposure
                hedges[hedge_etf] = hedges.get(hedge_etf, 0.0) + hedge_weight

                action = "NEUTRALIZING"
                logger.info(f"[HEDGE] Sector={sector} Exposure={exposure:.2%} ETF={hedge_etf} Action={action} Weight={hedge_weight:.2%}")

        return hedges
