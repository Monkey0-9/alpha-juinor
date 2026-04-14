# strategies/factory.py
from typing import Dict, Any, List
from .base import BaseStrategy
from .institutional_strategy import InstitutionalStrategy

class StrategyFactory:
    """
    Factory for creating trading strategies based on configuration.
    """
    
    _STRATEGIES = {
        "institutional": InstitutionalStrategy,
        # Add more strategy types here (arbitrage, market_making, etc.)
    }
    
    @classmethod
    def create_strategy(cls, config: Dict[str, Any]) -> BaseStrategy:
        strategy_type = config.get("type", "institutional")
        if strategy_type not in cls._STRATEGIES:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
            
        strategy_class = cls._STRATEGIES[strategy_type]
        return strategy_class(config)
