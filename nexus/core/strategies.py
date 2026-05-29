import logging
import numpy as np
import pandas as pd
from typing import Optional

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
        
        # FIXED: Use shorter lookback windows for faster response
        short = prices.rolling(10, min_periods=1).mean()  # Was 20
        long = prices.rolling(30, min_periods=1).mean()   # Was 50
        momentum = (short.iloc[-1] - long.iloc[-1]) / max(long.iloc[-1], 1)
        
        # Lower volatility penalty to keep signals strong
        volatility = prices.pct_change().std() if len(prices) > 1 else 0.0
        
        score = alpha * 0.60 + momentum * 0.30 - volatility * 0.10
        return float(np.tanh(score * 8))


class MeanReversionStrategy(BaseStrategy):
    name = "MeanReversion"

    def score(self, symbol: str, alpha: float, history: pd.DataFrame, regime: str) -> float:
        if history.empty or "close" not in history.columns:
            return 0.0
        prices = history["close"].astype(float)
        
        # FIXED: Shorter lookback for faster reversion detection
        sma = prices.rolling(20, min_periods=1).mean()  # Was 34
        deviation = (prices.iloc[-1] - sma.iloc[-1]) / max(sma.iloc[-1], 1)
        
        # Stronger mean reversion signal
        signal = -deviation * 0.75 + alpha * 0.25
        
        # Boost in sideways markets where reversion works
        if regime == "SIDEWAYS":
            signal *= 1.5  # Was 1.3
        elif regime == "TURBULENT":
            signal *= 0.7  # Be cautious in turbulent
            
        return float(np.tanh(signal * 7))


class VolatilityArbitrageStrategy(BaseStrategy):
    name = "VolatilityArbitrage"

    def score(self, symbol: str, alpha: float, history: pd.DataFrame, regime: str) -> float:
        if history.empty or "close" not in history.columns:
            return 0.0
        returns = history["close"].pct_change().dropna()
        vol = returns.std() if not returns.empty else 0.0
        mean_return = returns.mean() if not returns.empty else 0.0
        
        # FIXED: Less aggressive volatility penalty
        score = alpha * 0.50 + mean_return * 0.40 - vol * 0.10
        
        # Adjust for regime
        if regime == "TURBULENT":
            score *= 0.75  # More conservative
        elif regime == "BULL":
            score *= 1.1  # Favor in bull markets
            
        return float(np.tanh(score * 10))


class MacroOverlayStrategy(BaseStrategy):
    name = "MacroOverlay"

    def score(self, symbol: str, alpha: float, history: pd.DataFrame, regime: str) -> float:
        if history.empty or "close" not in history.columns:
            return 0.0
        returns = history["close"].pct_change().dropna()
        momentum = returns.tail(10).mean() if len(returns) >= 1 else 0.0
        score = alpha * 0.3 + momentum * 0.5
        if regime == "BEAR":
            score -= 0.15
        return float(np.tanh(score * 8))


class RSISetupStrategy(BaseStrategy):
    name = "RSI_Tactical"

    def score(self, symbol: str, alpha: float, history: pd.DataFrame, regime: str) -> float:
        if history.empty or "close" not in history.columns or len(history) < 14:
            return 0.0
        delta = history["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        last_rsi = rsi.iloc[-1]
        
        # FIXED: Better thresholds and alpha weighting
        if last_rsi < 35:  # Was 30 - more sensitive
            return 0.75  # Oversold, higher conviction
        if last_rsi > 65:  # Was 70
            return -0.75  # Overbought
        
        # Normal range: combine with alpha
        return alpha * 0.6


class BreakoutStrategy(BaseStrategy):
    name = "InstitutionalBreakout"

    def score(self, symbol: str, alpha: float, history: pd.DataFrame, regime: str) -> float:
        if history.empty or "close" not in history.columns or len(history) < 20:
            return 0.0
        high_20 = history["high"].rolling(20).max().iloc[-2]
        low_20 = history["low"].rolling(20).min().iloc[-2]
        current = history["close"].iloc[-1]

        if current > high_20:
            return 0.9
        if current < low_20:
            return -0.9
        return alpha * 0.2


class BollingerBandStrategy(BaseStrategy):
    name = "Bollinger_Reversion"

    def score(self, symbol: str, alpha: float, history: pd.DataFrame, regime: str) -> float:
        if history.empty or len(history) < 20:
            return 0.0
        prices = history["close"].astype(float)
        ma = prices.rolling(20).mean()
        std = prices.rolling(20).std()
        upper = ma + (std * 2)
        lower = ma - (std * 2)
        current = prices.iloc[-1]

        # FIXED: Stronger signals at extremes
        if current > upper.iloc[-1]:
            return -0.80  # Was -0.85
        if current < lower.iloc[-1]:
            return 0.80  # Was 0.85
        
        # Middle zone: blend with alpha
        return alpha * 0.4


class MACDStrategy(BaseStrategy):
    name = "MACD_Institutional"

    def score(self, symbol: str, alpha: float, history: pd.DataFrame, regime: str) -> float:
        if history.empty or len(history) < 26:
            return 0.0
        prices = history["close"].astype(float)
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()

        # FIXED: Stronger signals on crossovers
        if macd.iloc[-1] > signal.iloc[-1] and macd.iloc[-2] <= signal.iloc[-2]:
            return 0.80  # Was 0.75 - bullish crossover
        if macd.iloc[-1] < signal.iloc[-1] and macd.iloc[-2] >= signal.iloc[-2]:
            return -0.80  # Was -0.75 - bearish crossover
        
        # Trend following: if MACD > signal, favor longs
        if macd.iloc[-1] > signal.iloc[-1]:
            return alpha * 0.5
        else:
            return alpha * 0.3


class VolumeTrendStrategy(BaseStrategy):
    name = "VolumePriceTrend"

    def score(self, symbol: str, alpha: float, history: pd.DataFrame, regime: str) -> float:
        if history.empty or "volume" not in history.columns:
            return 0.0
        vpt = (history["volume"] * (history["close"].pct_change())).cumsum()
        if len(vpt) < 2:
            return 0.0
        if vpt.iloc[-1] > vpt.iloc[-2]:
            return 0.6 + (alpha * 0.4)
        return -0.4 + (alpha * 0.4)


class StochasticStrategy(BaseStrategy):
    name = "Stochastic_Momentum"

    def score(self, symbol: str, alpha: float, history: pd.DataFrame, regime: str) -> float:
        if history.empty or len(history) < 14:
            return 0.0
        low_14 = history["low"].rolling(14).min()
        high_14 = history["high"].rolling(14).max()
        k = 100 * (history["close"] - low_14) / (high_14 - low_14)
        d = k.rolling(3).mean()

        if k.iloc[-1] < 20 and k.iloc[-1] > d.iloc[-1]:
            return 0.8
        if k.iloc[-1] > 80 and k.iloc[-1] < d.iloc[-1]:
            return -0.8
        return alpha * 0.3


class EMACrossoverStrategy(BaseStrategy):
    name = "EMA_Trend_Follower"

    def score(self, symbol: str, alpha: float, history: pd.DataFrame, regime: str) -> float:
        if history.empty or len(history) < 50:
            return 0.0
        ema8 = history["close"].ewm(span=8).mean()
        ema21 = history["close"].ewm(span=21).mean()
        ema50 = history["close"].ewm(span=50).mean()

        if ema8.iloc[-1] > ema21.iloc[-1] > ema50.iloc[-1]:
            return 0.95
        if ema8.iloc[-1] < ema21.iloc[-1] < ema50.iloc[-1]:
            return -0.95
        return alpha * 0.1


class SuperTrendStrategy(BaseStrategy):
    name = "SuperTrend_Alpha"

    def score(self, symbol: str, alpha: float, history: pd.DataFrame, regime: str) -> float:
        if history.empty or len(history) < 10:
            return 0.0
        atr = (history["high"] - history["low"]).rolling(10).mean()
        hl2 = (history["high"] + history["low"]) / 2
        upper = hl2 + (3 * atr)
        lower = hl2 - (3 * atr)

        if history["close"].iloc[-1] > upper.iloc[-1]:
            return 0.9
        if history["close"].iloc[-1] < lower.iloc[-1]:
            return -0.9
        return alpha * 0.3


class FisherTransformStrategy(BaseStrategy):
    name = "Fisher_Cycle"

    def score(self, symbol: str, alpha: float, history: pd.DataFrame, regime: str) -> float:
        if history.empty or len(history) < 10:
            return 0.0
        max_h = history["high"].rolling(10).max()
        min_l = history["low"].rolling(10).min()
        val = 0.66 * ((history["close"] - min_l) / (max_h - min_l) - 0.5)
        fisher = 0.5 * np.log((1 + val) / (1 - val + 1e-6))

        if fisher.iloc[-1] > 1.5:
            return -0.7
        if fisher.iloc[-1] < -1.5:
            return 0.7
        return alpha * 0.4


class HurstExponentStrategy(BaseStrategy):
    name = "Hurst_Persistance"

    def score(self, symbol: str, alpha: float, history: pd.DataFrame, regime: str) -> float:
        if history.empty or len(history) < 50:
            return 0.0
        lags = range(2, 20)
        tau = [np.sqrt(np.std(np.subtract(history["close"][lag:].to_numpy(),
                                         history["close"][:-lag].to_numpy())))
               for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        hurst = poly[0] * 2.0

        if hurst > 0.65:
            return alpha * 1.5
        if hurst < 0.35:
            return -alpha * 0.5
        return alpha


# Module-level singletons — avoids re-creating 14 objects per call
_ALL_STRATEGIES: list[BaseStrategy] = [
    MomentumStrategy(),
    MeanReversionStrategy(),
    VolatilityArbitrageStrategy(),
    MacroOverlayStrategy(),
    RSISetupStrategy(),
    BreakoutStrategy(),
    BollingerBandStrategy(),
    MACDStrategy(),
    VolumeTrendStrategy(),
    StochasticStrategy(),
    EMACrossoverStrategy(),
    SuperTrendStrategy(),
    FisherTransformStrategy(),
    HurstExponentStrategy(),
]


class StrategyFactory:
    @staticmethod
    def all_strategies() -> list[BaseStrategy]:
        return _ALL_STRATEGIES

    @staticmethod
    def get_strategy(name: str) -> Optional[BaseStrategy]:
        for strategy in _ALL_STRATEGIES:
            if strategy.name == name:
                return strategy
        return None
