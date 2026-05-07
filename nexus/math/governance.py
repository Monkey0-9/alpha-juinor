import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

class LatticeVoter:
    """
    Consensus mechanism for multi-model signal aggregation.
    """
    def aggregate_signals(self, model_votes: Dict[str, float]) -> float:
        """
        Computes weighted consensus from disparate model outputs.
        """
        if not model_votes:
            return 0.0
        # Simple weighted average for institutional stability
        return float(sum(model_votes.values()) / len(model_votes))

class StrategySwitcher:
    """
    Dynamic strategy selection based on market regime.
    """
    def select_strategy(self, regime: str) -> str:
        """
        Maps market regime to the optimal execution strategy.
        """
        mapping = {
            "BULL": "Trend Following",
            "BEAR": "Short Bias / Volatility Arb",
            "SIDEWAYS": "Mean Reversion",
            "TURBULENT": "Risk-Off / Liquidity Focus"
        }
        selected = mapping.get(regime, "Mean Reversion")
        logger.info(f"Market regime {regime} detected. Switching to {selected} strategy.")
        return selected
