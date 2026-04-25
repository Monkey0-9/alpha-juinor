import logging

logger = logging.getLogger("StrategySwitcher")

class SovereignStrategySwitcher:
    """
    Dynamically switches strategies based on Market Regime.
    BULL -> Momentum Pursuit
    BEAR -> Protective Hedging
    TURBULENT -> Volatility Arbitrage
    SIDEWAYS -> Mean Reversion
    """
    def get_optimal_strategy(self, regime):
        strategies = {
            "BULL": "Momentum Pursuit",
            "BEAR": "Protective Hedging",
            "TURBULENT": "Volatility Arbitrage",
            "SIDEWAYS": "Mean Reversion",
            "UNKNOWN": "Cash Preservation"
        }
        selected = strategies.get(regime, "Cash Preservation")
        logger.info(f"[STRATEGY] Switching to: {selected}")
        return selected
