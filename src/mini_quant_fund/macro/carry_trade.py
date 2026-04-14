from typing import List, Dict
import pandas as pd

class CarryTradeEngine:
    """FX and Fixed Income Carry Trade Strategy"""
    
    def identify_carry_opportunities(self, interest_rates: Dict[str, float]) -> List[Dict]:
        """
        Identify pairs with high interest rate differential.
        """
        sorted_rates = sorted(interest_rates.items(), key=lambda x: x[1], reverse=True)
        high_yield = sorted_rates[0]
        low_yield = sorted_rates[-1]
        
        return [{
            "pair": f"{high_yield[0]}/{low_yield[0]}",
            "carry_bps": (high_yield[1] - low_yield[1]) * 10000,
            "action": "long_high_short_low"
        }]
