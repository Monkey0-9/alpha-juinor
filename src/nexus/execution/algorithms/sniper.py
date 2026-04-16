class SniperAlgorithm:
    """Aggressive liquidity taker when target price hit"""
    
    def __init__(self, target_price: float):
        self.target_price = target_price
        
    def should_snipe(self, current_price: float, side: str) -> bool:
        if side == "buy":
            return current_price <= self.target_price
        else:
            return current_price >= self.target_price
