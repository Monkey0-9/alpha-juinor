"""
nexus/core/strategies.py — Superhuman Strategy Scoring Engine

Upgrades from 14 equal-weighted strategies to an adaptive ensemble with:
  - Confidence-gated scoring (low-confidence → 50% weight)
  - Dynamic RSI/Stochastic levels (rolling percentile, not fixed 30/70)
  - AdaptiveEnsembleStrategy: meta-strategy tracking per-regime hit-rates
  - OrderFlowPressureStrategy: institutional order flow detection
  - RegimePersistenceStrategy: exploits regime autocorrelation
  - All strategies return float scores; confidence tracked externally
"""
import logging
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple
from collections import deque

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Base Strategy                                                        #
# ------------------------------------------------------------------ #

class BaseStrategy:
    """Base class for all strategy modules."""
    name: str = "Base"

    def score(self, symbol: str, alpha: float, history: pd.DataFrame, regime: str) -> float:
        if history.empty:
            return 0.0
        return float(alpha)

    def score_with_confidence(
        self, symbol: str, alpha: float, history: pd.DataFrame, regime: str
    ) -> Tuple[float, float]:
        """Returns (score, confidence) where confidence ∈ [0, 1]."""
        s = self.score(symbol, alpha, history, regime)
        return s, 0.6  # base confidence


# ------------------------------------------------------------------ #
# Core 14 Strategies — upgraded with confidence gating                #
# ------------------------------------------------------------------ #

class MomentumStrategy(BaseStrategy):
    name = "Momentum"

    def score(self, symbol: str, alpha: float, history: pd.DataFrame, regime: str) -> float:
        if history.empty or "close" not in history.columns:
            return 0.0
        prices = history["close"].astype(float)
        short = prices.rolling(20, min_periods=1).mean()
        long  = prices.rolling(50, min_periods=1).mean()
        momentum   = (short.iloc[-1] - long.iloc[-1]) / max(long.iloc[-1], 1)
        volatility = prices.pct_change().std() if len(prices) > 1 else 0.0

        # Regime boost: momentum strategies shine in BULL
        regime_mult = 1.25 if regime == "BULL" else (0.70 if regime == "BEAR" else 1.0)
        score = (alpha * 0.55 + momentum * 0.35 - volatility * 0.10) * regime_mult
        return float(np.tanh(score * 10))

    def score_with_confidence(
        self, symbol: str, alpha: float, history: pd.DataFrame, regime: str
    ) -> Tuple[float, float]:
        if len(history) < 50:
            return 0.0, 0.3
        s = self.score(symbol, alpha, history, regime)
        # Confidence: how clearly separated are short/long EMAs?
        prices = history["close"].astype(float)
        sep = abs(
            float(
                prices.rolling(20, min_periods=1).mean().iloc[-1]
                - prices.rolling(50, min_periods=1).mean().iloc[-1]
            )
        )
        conf = float(np.tanh(sep / max(float(prices.iloc[-1]) * 0.005, 1e-6)))
        return s, max(0.3, min(1.0, conf))


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
            signal *= 1.35
        elif regime == "BULL":
            signal *= 0.75  # mean reversion weaker in trending markets
        return float(np.tanh(signal * 8))

    def score_with_confidence(
        self, symbol: str, alpha: float, history: pd.DataFrame, regime: str
    ) -> Tuple[float, float]:
        s = self.score(symbol, alpha, history, regime)
        # Confidence higher in SIDEWAYS regime
        conf = 0.80 if regime == "SIDEWAYS" else 0.45
        return s, conf


class VolatilityArbitrageStrategy(BaseStrategy):
    name = "VolatilityArbitrage"

    def score(self, symbol: str, alpha: float, history: pd.DataFrame, regime: str) -> float:
        if history.empty or "close" not in history.columns:
            return 0.0
        returns = history["close"].pct_change().dropna()
        vol = returns.std() if not returns.empty else 0.0
        mean_return = returns.mean() if not returns.empty else 0.0
        score = alpha * 0.4 + mean_return * 0.4 - vol * 0.2
        if regime == "TURBULENT":
            score *= 0.80
        return float(np.tanh(score * 12))


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
        elif regime == "BULL":
            score += 0.10
        return float(np.tanh(score * 8))


class RSISetupStrategy(BaseStrategy):
    name = "RSI_Tactical"

    def score(self, symbol: str, alpha: float, history: pd.DataFrame, regime: str) -> float:
        if history.empty or "close" not in history.columns or len(history) < 14:
            return 0.0
        delta = history["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs   = gain / (loss + 1e-9)
        rsi  = 100 - (100 / (1 + rs))
        last_rsi = float(rsi.iloc[-1])

        # Dynamic levels: use rolling 5th/95th percentile instead of fixed 30/70
        rsi_roll = rsi.dropna()
        os_level = float(rsi_roll.quantile(0.12)) if len(rsi_roll) >= 14 else 30.0
        ob_level = float(rsi_roll.quantile(0.88)) if len(rsi_roll) >= 14 else 70.0

        if last_rsi < os_level:
            conf_boost = (os_level - last_rsi) / max(os_level, 1.0)
            return float(min(0.95, 0.75 + conf_boost))
        if last_rsi > ob_level:
            conf_boost = (last_rsi - ob_level) / max(100 - ob_level, 1.0)
            return float(max(-0.95, -0.75 - conf_boost))
        return float(alpha * 0.5)

    def score_with_confidence(
        self, symbol: str, alpha: float, history: pd.DataFrame, regime: str
    ) -> Tuple[float, float]:
        s = self.score(symbol, alpha, history, regime)
        # High confidence when RSI is extreme
        rsi_extreme = abs(s) > 0.70
        conf = 0.85 if rsi_extreme else 0.50
        return s, conf


class BreakoutStrategy(BaseStrategy):
    name = "InstitutionalBreakout"

    def score(self, symbol: str, alpha: float, history: pd.DataFrame, regime: str) -> float:
        if history.empty or "close" not in history.columns or len(history) < 20:
            return 0.0
        high_20 = history["high"].rolling(20).max().iloc[-2]
        low_20  = history["low"].rolling(20).min().iloc[-2]
        current = history["close"].iloc[-1]
        if current > high_20:
            # Volume confirmation: breakout with rising volume is stronger
            vol_conf = 1.0
            if "volume" in history.columns and len(history) >= 5:
                avg_vol = float(history["volume"].tail(20).mean())
                last_vol = float(history["volume"].iloc[-1])
                if avg_vol > 0:
                    vol_conf = min(1.35, 1.0 + (last_vol / avg_vol - 1.0) * 0.5)
            return float(min(0.95, 0.85 * vol_conf))
        if current < low_20:
            return -0.90
        return float(alpha * 0.2)


class BollingerBandStrategy(BaseStrategy):
    name = "Bollinger_Reversion"

    def score(self, symbol: str, alpha: float, history: pd.DataFrame, regime: str) -> float:
        if history.empty or len(history) < 20:
            return 0.0
        prices = history["close"].astype(float)
        ma  = prices.rolling(20).mean()
        std = prices.rolling(20).std()
        upper = ma + (std * 2)
        lower = ma - (std * 2)
        current = prices.iloc[-1]

        # %B oscillator for magnitude of signal
        band_range = float(upper.iloc[-1] - lower.iloc[-1])
        if band_range < 1e-6:
            return 0.0
        pct_b = (current - float(lower.iloc[-1])) / band_range

        if pct_b > 1.0:
            return float(np.tanh(-(pct_b - 1.0) * 5 - 0.5))
        if pct_b < 0.0:
            return float(np.tanh(abs(pct_b) * 5 + 0.5))
        return float(alpha * 0.3)


class MACDStrategy(BaseStrategy):
    name = "MACD_Institutional"

    def score(self, symbol: str, alpha: float, history: pd.DataFrame, regime: str) -> float:
        if history.empty or len(history) < 26:
            return 0.0
        prices = history["close"].astype(float)
        exp1   = prices.ewm(span=12, adjust=False).mean()
        exp2   = prices.ewm(span=26, adjust=False).mean()
        macd   = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        hist   = macd - signal

        # Use histogram slope for conviction
        hist_slope = float(hist.diff().iloc[-1]) if len(hist) >= 2 else 0.0

        if macd.iloc[-1] > signal.iloc[-1] and macd.iloc[-2] <= signal.iloc[-2]:
            return float(min(0.90, 0.70 + abs(hist_slope) * 50))
        if macd.iloc[-1] < signal.iloc[-1] and macd.iloc[-2] >= signal.iloc[-2]:
            return float(max(-0.90, -0.70 - abs(hist_slope) * 50))
        return float(alpha * 0.4)


class VolumeTrendStrategy(BaseStrategy):
    name = "VolumePriceTrend"

    def score(self, symbol: str, alpha: float, history: pd.DataFrame, regime: str) -> float:
        if history.empty or "volume" not in history.columns:
            return 0.0
        vpt = (history["volume"] * (history["close"].pct_change())).cumsum()
        if len(vpt) < 2:
            return 0.0
        vpt_ma = vpt.rolling(10, min_periods=1).mean()
        vpt_signal = float(vpt.iloc[-1] - vpt_ma.iloc[-1])
        normalized = float(np.tanh(vpt_signal / max(abs(vpt.std()), 1e-6) * 2))
        return float(normalized * 0.7 + alpha * 0.3)


class StochasticStrategy(BaseStrategy):
    name = "Stochastic_Momentum"

    def score(self, symbol: str, alpha: float, history: pd.DataFrame, regime: str) -> float:
        if history.empty or len(history) < 14:
            return 0.0
        low_14  = history["low"].rolling(14).min()
        high_14 = history["high"].rolling(14).max()
        denom   = (high_14 - low_14).replace(0, np.nan)
        k = 100 * (history["close"] - low_14) / denom
        d = k.rolling(3).mean()

        if k.isna().iloc[-1] or d.isna().iloc[-1]:
            return float(alpha * 0.3)

        # Dynamic levels based on rolling percentile
        k_vals = k.dropna()
        os_level = float(k_vals.quantile(0.15)) if len(k_vals) >= 14 else 20.0
        ob_level = float(k_vals.quantile(0.85)) if len(k_vals) >= 14 else 80.0

        if k.iloc[-1] < os_level and k.iloc[-1] > d.iloc[-1]:
            return 0.82
        if k.iloc[-1] > ob_level and k.iloc[-1] < d.iloc[-1]:
            return -0.82
        return float(alpha * 0.3)


class EMACrossoverStrategy(BaseStrategy):
    name = "EMA_Trend_Follower"

    def score(self, symbol: str, alpha: float, history: pd.DataFrame, regime: str) -> float:
        if history.empty or len(history) < 50:
            return 0.0
        ema8  = history["close"].ewm(span=8).mean()
        ema21 = history["close"].ewm(span=21).mean()
        ema50 = history["close"].ewm(span=50).mean()

        if ema8.iloc[-1] > ema21.iloc[-1] > ema50.iloc[-1]:
            separation = (float(ema8.iloc[-1]) - float(ema50.iloc[-1])) / max(float(ema50.iloc[-1]), 1e-6)
            return float(min(0.98, 0.80 + separation * 3.0))
        if ema8.iloc[-1] < ema21.iloc[-1] < ema50.iloc[-1]:
            separation = (float(ema50.iloc[-1]) - float(ema8.iloc[-1])) / max(float(ema50.iloc[-1]), 1e-6)
            return float(max(-0.98, -0.80 - separation * 3.0))
        return float(alpha * 0.1)

    def score_with_confidence(
        self, symbol: str, alpha: float, history: pd.DataFrame, regime: str
    ) -> Tuple[float, float]:
        s = self.score(symbol, alpha, history, regime)
        # Very high confidence when all 3 EMAs are aligned
        conf = 0.90 if abs(s) > 0.75 else 0.45
        return s, conf


class SuperTrendStrategy(BaseStrategy):
    name = "SuperTrend_Alpha"

    def score(self, symbol: str, alpha: float, history: pd.DataFrame, regime: str) -> float:
        if history.empty or len(history) < 10:
            return 0.0
        atr   = (history["high"] - history["low"]).rolling(10).mean()
        hl2   = (history["high"] + history["low"]) / 2
        upper = hl2 + (3 * atr)
        lower = hl2 - (3 * atr)

        close = history["close"].iloc[-1]
        if close > float(upper.iloc[-1]):
            return 0.90
        if close < float(lower.iloc[-1]):
            return -0.90
        return float(alpha * 0.3)


class FisherTransformStrategy(BaseStrategy):
    name = "Fisher_Cycle"

    def score(self, symbol: str, alpha: float, history: pd.DataFrame, regime: str) -> float:
        if history.empty or len(history) < 10:
            return 0.0
        max_h  = history["high"].rolling(10).max()
        min_l  = history["low"].rolling(10).min()
        denom  = (max_h - min_l).replace(0, np.nan)
        val    = 0.66 * ((history["close"] - min_l) / denom - 0.5)
        val    = val.clip(-0.999, 0.999)
        fisher = 0.5 * np.log((1 + val) / (1 - val + 1e-6))

        if fisher.iloc[-1] > 1.5:
            return -0.72
        if fisher.iloc[-1] < -1.5:
            return 0.72
        return float(alpha * 0.4)


class HurstExponentStrategy(BaseStrategy):
    name = "Hurst_Persistance"

    def score(self, symbol: str, alpha: float, history: pd.DataFrame, regime: str) -> float:
        if history.empty or len(history) < 50:
            return 0.0
        lags = range(2, 20)
        tau = [
            np.sqrt(np.std(np.subtract(
                history["close"][lag:].to_numpy(),
                history["close"][:-lag].to_numpy()
            )))
            for lag in lags
        ]
        valid = [(np.log(lag), np.log(t)) for lag, t in zip(lags, tau) if t > 0]
        if len(valid) < 2:
            return float(alpha)
        log_lags, log_tau = zip(*valid)
        poly  = np.polyfit(log_lags, log_tau, 1)
        hurst = poly[0] * 2.0

        if hurst > 0.65:     # Trending: amplify momentum alpha
            return float(np.tanh(alpha * 1.6))
        if hurst < 0.35:     # Mean-reverting: flip sign for mean-reversion play
            return float(np.tanh(-alpha * 0.6))
        return float(alpha)  # Random walk: no edge → neutral


# ------------------------------------------------------------------ #
# NEW: Adaptive Ensemble Strategy                                      #
# ------------------------------------------------------------------ #

class AdaptiveEnsembleStrategy(BaseStrategy):
    """
    Meta-strategy that dynamically weights the base 14 strategies
    based on their trailing directional accuracy per market regime.

    Maintains a per-regime, per-strategy hit-rate window.
    High hit-rate strategies get boosted; low hit-rate strategies get penalized.
    """
    name = "Adaptive_Ensemble"

    def __init__(self, window: int = 20) -> None:
        super().__init__()
        self.window = window
        # regime -> strategy_name -> deque of (predicted_sign, realized_sign)
        self._regime_history: Dict[str, Dict[str, deque[int]]] = {}

    def record_outcome(
        self,
        regime: str,
        strategy_name: str,
        predicted_sign: int,
        realized_sign: int,
    ) -> None:
        """Record whether a strategy's prediction was correct in this regime."""
        if regime not in self._regime_history:
            self._regime_history[regime] = {}
        if strategy_name not in self._regime_history[regime]:
            self._regime_history[regime][strategy_name] = deque(maxlen=self.window)
        self._regime_history[regime][strategy_name].append(
            1 if predicted_sign == realized_sign else 0
        )

    def get_strategy_weight(self, regime: str, strategy_name: str) -> float:
        """Returns hit-rate-based weight for a strategy in the given regime."""
        history = self._regime_history.get(regime, {}).get(strategy_name, deque())
        if len(history) < 3:
            return 1.0  # neutral prior
        hit_rate = sum(history) / len(history)
        # Map hit_rate to weight: 0.5 (chance) → 0.5 weight; 1.0 → 2.0 weight
        return float(max(0.1, (hit_rate - 0.5) * 3.0 + 1.0))

    def score(self, symbol: str, alpha: float, history: pd.DataFrame, regime: str) -> float:
        """Score = IC-weighted average of all base strategy scores."""
        strategies = _BASE_STRATEGIES
        total_weight = 0.0
        weighted_score = 0.0

        for strat in strategies:
            w = self.get_strategy_weight(regime, strat.name)
            s = strat.score(symbol, alpha, history, regime)
            weighted_score += w * s
            total_weight   += w

        if total_weight < 1e-9:
            return float(alpha)
        return float(np.tanh(weighted_score / total_weight))


# ------------------------------------------------------------------ #
# NEW: Order Flow Pressure Strategy                                    #
# ------------------------------------------------------------------ #

class OrderFlowPressureStrategy(BaseStrategy):
    """
    Institutional order flow detection via volume delta and price velocity.

    Logic:
    - Cumulative Delta: tracks net buying vs. selling pressure
    - Price velocity: speed and acceleration of recent price moves
    - When delta and velocity agree → strong institutional signal
    """
    name = "OrderFlow_Pressure"

    def score(self, symbol: str, alpha: float, history: pd.DataFrame, regime: str) -> float:
        if history.empty or "volume" not in history.columns or len(history) < 10:
            return 0.0

        close   = history["close"].astype(float)
        volume  = history["volume"].astype(float)
        returns = close.pct_change().dropna()

        # Volume delta: positive when price up, negative when price down
        cum_delta = (volume * np.sign(returns.reindex(close.index).fillna(0.0))).cumsum()
        delta_ma  = cum_delta.rolling(10, min_periods=1).mean()
        delta_signal = float(np.tanh(
            (float(cum_delta.iloc[-1]) - float(delta_ma.iloc[-1]))
            / max(float(cum_delta.abs().mean()), 1e-6)
        ))

        # Price velocity
        price_velocity   = float(close.diff().tail(5).mean())
        price_accel      = float(close.diff().diff().tail(3).mean())
        velocity_signal  = float(np.tanh((price_velocity + 0.3 * price_accel)
                                         / max(float(close.std()), 1e-6) * 10))

        # Agreement multiplier
        agreement = 1.0 if (delta_signal * velocity_signal) > 0 else 0.5
        combined = (delta_signal * 0.55 + velocity_signal * 0.45) * agreement

        return float(np.tanh(combined * 1.2 + alpha * 0.3))


# ------------------------------------------------------------------ #
# NEW: Regime Persistence Strategy                                     #
# ------------------------------------------------------------------ #

class RegimePersistenceStrategy(BaseStrategy):
    """
    Exploits regime autocorrelation.

    Markets stay in regime far longer than naive models assume.
    In BULL regime: amplify momentum signals.
    In BEAR regime: amplify short signals with protective bias.
    In TURBULENT: use mean-reversion with wide stops.
    """
    name = "Regime_Persistence"

    def score(self, symbol: str, alpha: float, history: pd.DataFrame, regime: str) -> float:
        if history.empty or "close" not in history.columns:
            return 0.0

        close   = history["close"].astype(float)
        returns = close.pct_change().dropna()
        if returns.empty:
            return 0.0

        recent_return = float(returns.tail(5).mean())

        if regime == "BULL":
            # Amplify positive alpha; protect against short signals
            score = alpha * 1.3 + recent_return * 5.0
            return float(np.tanh(score))

        elif regime == "BEAR":
            # Penalize long positions; amplify downside alpha
            score = alpha * 0.6 - recent_return * 4.0
            return float(np.tanh(score))

        elif regime == "TURBULENT":
            # High-frequency mean reversion in turbulent markets
            sma  = float(close.rolling(10, min_periods=1).mean().iloc[-1])
            dev  = (float(close.iloc[-1]) - sma) / max(sma, 1.0)
            return float(np.tanh(-dev * 6.0))

        else:  # SIDEWAYS
            sma = float(close.rolling(20, min_periods=1).mean().iloc[-1])
            dev = (float(close.iloc[-1]) - sma) / max(sma, 1.0)
            score = -dev * 0.6 + alpha * 0.4
            return float(np.tanh(score * 5))


# ------------------------------------------------------------------ #
# Strategy Registry                                                    #
# ------------------------------------------------------------------ #

# Base 14 strategies (singletons to avoid re-creating per call)
_BASE_STRATEGIES: List[BaseStrategy] = [
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

# Adaptive meta-strategy singleton
_ADAPTIVE_ENSEMBLE = AdaptiveEnsembleStrategy(window=20)

# Extended strategy set including new superhuman strategies
_ALL_STRATEGIES: List[BaseStrategy] = _BASE_STRATEGIES + [
    OrderFlowPressureStrategy(),
    RegimePersistenceStrategy(),
    _ADAPTIVE_ENSEMBLE,
]


class StrategyFactory:
    @staticmethod
    def all_strategies() -> List[BaseStrategy]:
        return _ALL_STRATEGIES

    @staticmethod
    def base_strategies() -> List[BaseStrategy]:
        return _BASE_STRATEGIES

    @staticmethod
    def get_adaptive_ensemble() -> AdaptiveEnsembleStrategy:
        return _ADAPTIVE_ENSEMBLE

    @staticmethod
    def get_strategy(name: str) -> Optional[BaseStrategy]:
        for strategy in _ALL_STRATEGIES:
            if strategy.name == name:
                return strategy
        return None
