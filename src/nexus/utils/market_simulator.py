import numpy as np
import pandas as pd
from typing import Dict, List

class MicrostructureSimulator:
    """Simulates real L2 Limit Order Book (LOB) dynamics for institutional testing"""
    
    def __init__(self, symbol: str, mid_price: float = 150.0):
        self.symbol = symbol
        self.mid_price = mid_price
        
    def generate_order_book(self, depth: int = 10) -> Dict:
        """Generate a realistic LOB with spread and imbalance"""
        spread = 0.01 + np.random.gamma(1, 0.01)
        best_bid = self.mid_price - (spread / 2)
        best_ask = self.mid_price + (spread / 2)
        
        bids = []
        asks = []
        for i in range(depth):
            # Volume follows a power-law distribution in real books
            bid_vol = int(1000 * (1 / (i + 1)**0.8))
            ask_vol = int(1000 * (1 / (i + 1)**0.8))
            
            bids.append({"price": round(best_bid - i*0.01, 2), "size": bid_vol})
            asks.append({"price": round(best_ask + i*0.01, 2), "size": ask_vol})
            
        # Update mid price based on random walk (Geometric Brownian Motion)
        drift = 0.0001
        volatility = 0.02
        self.mid_price *= np.exp((drift - 0.5 * volatility**2) + volatility * np.random.normal())
            
        return {
            "symbol": self.symbol,
            "bids": bids,
            "asks": asks,
            "mid": self.mid_price,
            "imbalance": (bids[0]["size"] - asks[0]["size"]) / (bids[0]["size"] + asks[0]["size"])
        }
