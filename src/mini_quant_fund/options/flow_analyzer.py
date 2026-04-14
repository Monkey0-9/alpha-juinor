from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class FlowSignal:
    ticker: str
    option_type: str
    strike: float
    expiration: str
    volume: int
    open_interest: int
    sentiment: str
    signal_strength: float

class OptionsFlowAnalyzer:
    """Detect unusual options activity"""
    
    def detect_unusual_flow(self, market_data: List[Dict]) -> List[FlowSignal]:
        """
        - High volume vs OI
        - Whale detection (large orders)
        """
        signals = []
        for data in market_data:
            vol_oi_ratio = data["volume"] / max(1, data["open_interest"])
            
            if vol_oi_ratio > 2.0 and data["volume"] > 500:
                signals.append(FlowSignal(
                    ticker=data["ticker"],
                    option_type=data["option_type"],
                    strike=data["strike"],
                    expiration=data["expiration"],
                    volume=data["volume"],
                    open_interest=data["open_interest"],
                    sentiment="bullish" if data["option_type"] == "call" else "bearish",
                    signal_strength=min(1.0, vol_oi_ratio / 10.0)
                ))
        return signals
