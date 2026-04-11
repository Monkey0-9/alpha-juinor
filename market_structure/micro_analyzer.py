
import logging
import pandas as pd
from typing import Dict, List, Any

logger = logging.getLogger("MICROSTRUCTURE")

class MicrostructureAnalyzer:
    """
    Analyzes L2 Order Book depth and trade prints.
    Bridges the gap to 'Institutional Microstructure' needs.
    """
    def __init__(self):
        logger.info("MicrostructureAnalyzer initialized")

    def detect_toxic_flow(self, trades: pd.DataFrame) -> float:
        """
        Estimate PIN (Probability of Informed Trading).
        High toxicity triggers immediate order cancellation.
        """
        if trades.empty:
            return 0.0
        
        # Simple volume imbalance proxy
        buy_vol = trades[trades['side'] == 'buy']['volume'].sum()
        sell_vol = trades[trades['side'] == 'sell']['volume'].sum()
        total_vol = buy_vol + sell_vol
        
        imbalance = abs(buy_vol - sell_vol) / total_vol if total_vol > 0 else 0
        return imbalance

    def analyze_liquidity_depth(self, order_book: Dict[str, List[Any]]) -> float:
        """
        Calculate depth at best bid/ask.
        Prevents filling against 'thin' books that cause massive slippage.
        """
        bids = order_book.get('bids', [])
        asks = order_book.get('asks', [])
        
        if not bids or not asks:
            return 0.0
            
        top_bid_qty = bids[0][1]
        top_ask_qty = asks[0][1]
        
        return (top_bid_qty + top_ask_qty) / 2

def get_micro_analyzer() -> MicrostructureAnalyzer:
    return MicrostructureAnalyzer()
