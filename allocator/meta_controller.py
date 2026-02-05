from typing import Dict, Optional
import pandas as pd
import sys
import os

# Add parent directory to path to import enhanced_strategy
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from enhanced_strategy import RegimeDetector, MarketRegime
from strategy_factory.interface import Signal

class MetaController:
    """
    The Brain of the Multi-Strategy Platform.
    Allocates capital to strategies based on Market Regime.
    """

    def __init__(self):
        self.regime_detector = RegimeDetector()

        # Strategy Names
        self.MR_STRAT = "MeanRestoration_RSI"
        self.TF_STRAT = "TrendFollowing_12M"
        self.SENT_STRAT = "Sentiment_NLP"

        # Allocation Logic Table
        self.ALLOCATION_TABLE = {
            'BULL':   {self.TF_STRAT: 0.60, self.MR_STRAT: 0.25, self.SENT_STRAT: 0.15},
            'NORMAL': {self.TF_STRAT: 0.25, self.MR_STRAT: 0.60, self.SENT_STRAT: 0.15},
            'BEAR':   {self.TF_STRAT: 0.50, self.MR_STRAT: 0.30, self.SENT_STRAT: 0.20},
            'CRISIS': {self.TF_STRAT: 0.00, self.MR_STRAT: 0.00, self.SENT_STRAT: 0.00}
        }

    def get_allocation_weights(self, benchmark_prices: pd.Series) -> Dict[str, float]:
        """
        Determine target weights for each strategy based on current regime.
        """
        regime_data = self.regime_detector.detect(benchmark_prices)
        regime_name = regime_data.regime

        weights = self.ALLOCATION_TABLE.get(regime_name, {self.TF_STRAT: 0.5, self.MR_STRAT: 0.5})

        # Apply Risk Multiplier (Volatility Targeting at Portfolio Level)
        # e.g., if Crisis, risk_mult is 0.3 -> Scale down everything
        # Actually, Allocation Table implementation for Crisis is 0/0, but let's adhere to regime multiplier too

        # If we use the allocation table, probabilities sum to 1.0 (except Crisis).
        # We can then scale by risk_multiplier if we want absolute position sizing.
        # For now, let's return relative weights between strategies.

        return weights, regime_data

    def generate_portfolio_signals(
        self,
        strategies_signals: Dict[str, Signal],
        benchmark_prices: pd.Series
    ) -> Dict[str, float]:
        """
        Combine strategy signals into final portfolio weights using regime logic.

        Args:
            strategies_signals: Dict {strat_name: Signal}
            benchmark_prices: Series for regime detection

        Returns:
            Dict {symbol: net_weight}
        """
        # 1. Get Regime & Allocation
        strat_weights, regime_data = self.get_allocation_weights(benchmark_prices)

        # 2. Combine Signals
        # Net Signal = Sum(Strat_Weight * Strat_Signal_Strength)
        # Assuming single symbol context here for simplicity or loop outside

        # We need to know which symbol we are trading.
        # The Signal object has the symbol.
        # Assuming all signals are for the same symbol for now.

        net_signal = 0.0
        symbol = None

        for strat_name, signal in strategies_signals.items():
            if symbol is None:
                symbol = signal.symbol
            elif symbol != signal.symbol:
                # Mismatch - in a real system handle this, for now skip or error
                continue

            weight = strat_weights.get(strat_name, 0.0)
            net_signal += weight * signal.strength

        # 3. Apply Global Risk Scaling (Regime risk multiplier output from detector)
        if regime_data:
            net_signal *= regime_data.risk_multiplier

        return {symbol: net_signal}, regime_data
