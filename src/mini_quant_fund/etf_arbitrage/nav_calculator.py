from typing import Dict, List

class NAVCalculator:
    """Real-time NAV computation (Scaffold)"""
    
    def calculate_nav(self, basket: Dict[str, float], prices: Dict[str, float], cash: float = 0.0) -> float:
        """
        Calculate Net Asset Value.
        basket: {symbol: units}
        prices: {symbol: price}
        """
        nav = cash
        for symbol, units in basket.items():
            nav += units * prices.get(symbol, 0.0)
        return nav
