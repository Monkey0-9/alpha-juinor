from typing import List, Dict
import pandas as pd
from .interface import StrategyInterface, Signal
from .mean_reversion import MeanReversionStrategy
from .trend_following import TrendFollowingStrategy
from .sentiment_engine import SentimentStrategy

class StrategyManager:
    """Orchestrates multiple strategy engines."""

    def __init__(self):
        self.strategies: List[StrategyInterface] = [
            MeanReversionStrategy(),
            TrendFollowingStrategy(),
            SentimentStrategy()
        ]

    def generate_all_signals(self, symbol: str, prices: pd.Series, regime_data: dict = None) -> Dict[str, Signal]:
        """Generate signals from all strategies for a given asset."""
        results = {}
        for strat in self.strategies:
            try:
                sig = strat.generate_signal(symbol, prices, regime_data)
                results[strat.name] = sig
            except Exception as e:
                print(f"Error in {strat.name}: {e}")

        return results

    def generate_historical_signals(self, symbol: str, prices: pd.Series) -> pd.DataFrame:
        """
        Generate historical signals for correlation analysis.
        Returns a DataFrame where columns are strategy names and index is date.
        """
        signals_dict = {strat.name: [] for strat in self.strategies}
        dates = prices.index

        # This is slow iteratively, but robust for verification
        # Optimization: Most strategies can be vectorized, but interface enforces step-by-step simulation
        # For verification script, we can cheat if strategy internals allow, but let's stick to interface

        # Actually, for the verification script, let's implement a vectorized helper in the strategies if possible
        # or just loop. Looping 5000 days is fast enough for 2 simple strategies.

        for i in range(50, len(prices)):
            window = prices.iloc[:i+1]
            for strat in self.strategies:
                sig = strat.generate_signal(symbol, window)
                signals_dict[strat.name].append(sig.strength)

        # Align lengths
        aligned_dates = dates[50:]
        df = pd.DataFrame(signals_dict, index=aligned_dates)
        return df
