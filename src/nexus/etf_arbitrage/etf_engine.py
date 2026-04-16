from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class ArbitrageOpportunity:
    etf_ticker: str
    premium_discount: float
    expected_profit: float
    action: str # "create" or "redeem"

class ETFArbitrageEngine:
    """Institutional ETF Arbitrage Engine with Transaction Cost Analysis"""
    
    def __init__(self, threshold_bps: float = 10.0):
        self.threshold = threshold_bps / 10000
        
    def detect_arbitrage(self, etf_price: float, nav: float, tca_cost: float) -> Optional[ArbitrageOpportunity]:
        """
        Detect premium/discount to NAV net of transaction costs.
        """
        gross_diff = (etf_price - nav) / nav
        # Net difference after considering basket execution costs (slippage + fees)
        net_diff = abs(gross_diff) - tca_cost
        
        if net_diff > self.threshold:
            action = "create" if gross_diff > 0 else "redeem"
            return ArbitrageOpportunity(
                etf_ticker="ETF_V3",
                premium_discount=gross_diff,
                expected_profit=(net_diff * 1000000), # Net of TCA
                action=action
            )
        return None
