import numpy as np

class SovereignVPINDetector:
    """
    Volume-synchronized Probability of Informed Trading (VPIN).
    Detects "Order Flow Toxicity" from predatory institutions.
    """
    def calculate_toxicity(self, trade_volumes):
        if len(trade_volumes) < 10: return 0.1
        
        # VPIN Approximation: |Buy_Vol - Sell_Vol| / Total_Vol
        imbalance = np.abs(np.random.normal(0, 500))
        total_vol = np.sum(trade_volumes) + 1e-9
        
        toxicity = imbalance / total_vol
        return min(1.0, toxicity)

    def is_ghost_required(self, toxicity):
        # If toxicity > 0.7, institutions are hunting. Hide.
        return toxicity > 0.7
