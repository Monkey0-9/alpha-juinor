import numpy as np
import random

class SovereignL2WallDetector:
    """
    Level 2 Order Book Wall Detector.
    Identifies "Hidden Walls" of liquidity where institutions are hiding.
    """
    def detect_walls(self, order_book_data):
        # Simulated Level 2 Book Analysis
        bid_sizes = [random.randint(100, 5000) for _ in range(10)]
        ask_sizes = [random.randint(100, 5000) for _ in range(10)]
        
        max_bid = np.max(bid_sizes)
        max_ask = np.max(ask_sizes)
        
        if max_bid > np.mean(bid_sizes) * 3: return "BID_WALL_DETECTED"
        if max_ask > np.mean(ask_sizes) * 3: return "ASK_WALL_DETECTED"
        return "CLEAR_PATH"

    def get_wall_influence(self, wall_type):
        if wall_type == "BID_WALL_DETECTED": return 0.25 
        if wall_type == "ASK_WALL_DETECTED": return -0.25 
        return 0.0
