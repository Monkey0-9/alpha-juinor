import numpy as np

class VPINDetector:
    """
    Volume-synchronized Probability of Informed Trading (VPIN).
    Detects order flow toxicity to avoid being "picked off" by informed whales.
    """
    def calculate_toxicity(self, volumes, prices):
        if len(volumes) < 10: return 0.5
        
        # Calculate volume buckets
        avg_vol = np.mean(volumes)
        v_bucket = avg_vol * 0.5 # 50% of avg volume per bucket
        
        # Estimate buy/sell imbalance (approximate using price changes)
        price_diffs = np.diff(prices)
        imbalance = np.sum(np.abs(price_diffs * volumes[1:])) / np.sum(volumes)
        
        # Toxicity score (0 to 1)
        # Higher score means higher probability of informed trading (toxic)
        toxicity = min(1.0, imbalance / (avg_vol * 0.01)) 
        return toxicity
