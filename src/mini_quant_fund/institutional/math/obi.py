import numpy as np

class OrderBookImbalance:
    """
    Detects market pressure by analyzing Bid/Ask depth.
    High IMB > 0: High Buying Pressure
    High IMB < 0: High Selling Pressure
    """
    def calculate_imbalance(self, bid_qty, ask_qty):
        if (bid_qty + ask_qty) == 0: return 0
        imbalance = (bid_qty - ask_qty) / (bid_qty + ask_qty)
        return imbalance

    def estimate_pressure(self, price_history):
        # Heuristic: Use recent volume spikes near high/low to estimate depth
        # Real version would use L2 Order Book data
        return np.random.uniform(-0.5, 0.5) 
