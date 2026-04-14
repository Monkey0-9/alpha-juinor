from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ComplianceReport:
    trade_id: str
    symbol: str
    is_best_ex: bool
    nbbo_price: float
    exec_price: float
    deviation_bps: float

class BestExecutionMonitor:
    """Monitor compliance with best execution (Scaffold)"""
    
    def analyze_execution(self, trade_id: str, symbol: str, exec_price: float, nbbo_price: float) -> ComplianceReport:
        """
        Compare execution price to NBBO.
        """
        deviation = (exec_price - nbbo_price) / nbbo_price * 10000
        is_best = abs(deviation) < 5.0 # 5 bps tolerance
        
        return ComplianceReport(
            trade_id=trade_id,
            symbol=symbol,
            is_best_ex=is_best,
            nbbo_price=nbbo_price,
            exec_price=exec_price,
            deviation_bps=deviation
        )
