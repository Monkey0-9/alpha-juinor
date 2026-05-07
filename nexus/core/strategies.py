import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class BaseStrategy:
    """Base class for all strategy modules."""
    name: str = "Base"

    def score(self, symbol: str, alpha: float, history: pd.DataFrame, regime: str) -> float:
        if history.empty:
            return 0.0
        return float(alpha)

class MomentumStrategy(BaseStrategy):
    name = "Momentum"

    def score(self, symbol: str, alpha: float, history: pd.DataFrame, regime: str) -> float:
        if history.empty or "close" not in history.columns:
            return 0.0
        prices = history["close"].astype(float)
        short = prices.rolling(20, min_periods=1).mean()
        long = prices.rolling(50, min_periods=1).mean()
        momentum = (short.iloc[-1] - long.iloc[-1]) / max(long.iloc[-1], 1)
        volatility = prices.pct_change().std() if len(prices) > 1 else 0.0
        score = alpha * 0.65 + momentum * 0.25 - volatility * 0.10
        logger.debug(f"MomentumScore {symbol}: alpha={alpha:.4f}, mom={momentum:.4f}, vol={volatility:.4f}")
        return float(np.tanh(score * 10))

class MeanReversionStrategy(BaseStrategy):
    name = "MeanReversion"

    def score(self, symbol: str, alpha: float, history: pd.DataFrame, regime: str) -> float:
        if history.empty or "close" not in history.columns:
            return 0.0
        prices = history["close"].astype(float)
        sma = prices.rolling(34, min_periods=1).mean()
        deviation = (prices.iloc[-1] - sma.iloc[-1]) / max(sma.iloc[-1], 1)
        signal = -deviation * 0.7 + alpha * 0.3
        if regime == "SIDEWAYS":
            signal *= 1.3
        return float(np.tanh(signal * 8))

class VolatilityArbitrageStrategy(BaseStrategy):
    name = "VolatilityArbitrage"

    def score(self, symbol: str, alpha: float, history: pd.DataFrame, regime: str) -> float:
        if history.empty or "close" not in history.columns:
            return 0.0
        returns = history["close"].pct_change().dropna()
        vol = float(returns.std()) if not returns.empty else 0.0
        mean_return = float(returns.mean()) if not returns.empty else 0.0
        score = alpha * 0.4 + mean_return * 0.4 - vol * 0.2
        if regime == "TURBULENT":
            score *= 0.85
        return float(np.tanh(score * 12))

class MacroOverlayStrategy(BaseStrategy):
    name = "MacroOverlay"

    def score(self, symbol: str, alpha: float, history: pd.DataFrame, regime: str) -> float:
        if history.empty or "close" not in history.columns:
            return 0.0
        returns = history["close"].pct_change().dropna()
        momentum = float(returns.tail(10).mean()) if len(returns) >= 1 else 0.0
        score = alpha * 0.3 + momentum * 0.5
        if regime == "BEAR":
            score -= 0.15
        return float(np.tanh(score * 8))

class StrategyFactory:
    @staticmethod
    def all_strategies() -> list[BaseStrategy]:
        return [
            MomentumStrategy(),
            MeanReversionStrategy(),
            VolatilityArbitrageStrategy(),
            MacroOverlayStrategy(),
        ]

    @staticmethod
    def get_strategy(name: str) -> Optional[BaseStrategy]:
        for strategy in StrategyFactory.all_strategies():
            if strategy.name == name:
                return strategy
        return None
