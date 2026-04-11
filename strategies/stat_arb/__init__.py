"""
Statistical Arbitrage Package
"""

# Expose key classes
from .engine import StatArbEngine


class KalmanPairsTrader:
    """
    Kalman Filter-based Pairs Trading strategy.
    Uses Kalman filtering to dynamically estimate hedge ratios
    and detect mean-reverting spreads.
    """

    def __init__(self, initial_state: dict = None):
        """Initialize Kalman filter parameters."""
        self.state = initial_state or {}
        self.filter_initialized = False

    def estimate_hedge_ratio(self, prices_pair: dict):
        """Estimate dynamically changing hedge ratio using Kalman filter."""
        return 1.0  # Placeholder

    def generate_signal(self, spread_z_score: float) -> str:
        """Generate trading signal based on spread z-score."""
        if spread_z_score > 2.0:
            return "SELL"
        elif spread_z_score < -2.0:
            return "BUY"
        return "HOLD"
