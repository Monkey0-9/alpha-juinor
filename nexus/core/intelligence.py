import logging
from typing import Dict, List, Any
import numpy as np
import pandas as pd
from nexus.core.strategies import StrategyFactory
from nexus.math.indicators import RegimeDetector
from nexus.math.governance import LatticeVoter, StrategySwitcher
from nexus.math.risk import RiskEngine

logger = logging.getLogger(__name__)


class MarketBrain:
    """Adaptive market intelligence engine for Nexus."""

    def __init__(self) -> None:
        self.regime_detector = RegimeDetector()
        self.strategy_switcher = StrategySwitcher()
        self.signal_voter = LatticeVoter()
        self.risk_engine = RiskEngine()

    def analyze_market(
        self,
        benchmark_data: pd.DataFrame,
        positions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Analyze market conditions and select optimal strategy."""
        logger.info("Analyzing market conditions...")
        regime = self.regime_detector.detect(benchmark_data)
        strategy = self.strategy_switcher.select_strategy(regime)
        sentiment = self.calculate_global_sentiment(positions)
        macro_profile = self.assess_macro(benchmark_data)
        event_profile = self.detect_event_drivers(
            benchmark_data, positions
        )

        # Compute real ensemble agreement across strategies
        strategies = StrategyFactory.all_strategies()
        strategy_names = [s.name for s in strategies]
        agreement = self._compute_strategy_agreement(
            benchmark_data, regime
        )

        if sentiment > 0.5:
            market_forecast = "BULLISH_CONTINUATION"
            sentiment_label = "Bullish Bias"
        elif sentiment < 0.5:
            market_forecast = "BEARISH_REVERSION"
            sentiment_label = "Bearish Bias"
        else:
            market_forecast = "NEUTRAL"
            sentiment_label = "Neutral"

        logger.info(
            f"Regime: {regime} | Strategy: {strategy} | "
            f"Agreement: {agreement:.0%} | Forecast: {market_forecast}"
        )

        return {
            "regime": regime,
            "selected_strategy": strategy,
            "market_sentiment": sentiment,
            "market_sentiment_label": sentiment_label,
            "macro_profile": macro_profile,
            "event_profile": event_profile,
            "strategy_agreement": agreement,
            "market_forecast": market_forecast,
            "strategy_universe": strategy_names,
        }

    def _compute_strategy_agreement(
        self,
        benchmark_data: pd.DataFrame,
        regime: str,
    ) -> float:
        """Measure agreement ratio across all 14 strategies.

        Returns the fraction of strategies that agree on direction
        (all positive or all negative). A value of 1.0 means perfect
        consensus; 0.5 means an even split.
        """
        if benchmark_data.empty:
            return 0.5
        strategies = StrategyFactory.all_strategies()
        scores = []
        for strat in strategies:
            s = strat.score("SPY", 0.0, benchmark_data, regime)
            scores.append(s)
        if not scores:
            return 0.5
        positive = sum(1 for s in scores if s > 0)
        negative = sum(1 for s in scores if s < 0)
        total = len(scores)
        return max(positive, negative) / total if total else 0.5

    def calculate_global_sentiment(
        self, positions: List[Dict[str, Any]]
    ) -> float:
        if not positions:
            return 0.5
        pnl_values = [
            float(p.get("unrealized_plpc", 0.0)) for p in positions
        ]
        avg_pnl = float(np.mean(pnl_values))
        sentiment = 0.5 + np.clip(avg_pnl * 10.0, -0.4, 0.4)
        return float(sentiment)

    def assess_macro(
        self, benchmark_data: pd.DataFrame
    ) -> Dict[str, float]:
        if benchmark_data.empty or "close" not in benchmark_data.columns:
            return {
                "momentum": 0.0,
                "volatility": 0.0,
                "trend_strength": 0.0,
            }
        returns = benchmark_data["close"].pct_change().dropna()
        if returns.empty:
            return {
                "momentum": 0.0,
                "volatility": 0.0,
                "trend_strength": 0.0,
            }
        momentum = float(returns.tail(10).mean())
        volatility = float(returns.std())
        lookback = max(0, len(benchmark_data) - 21)
        trend_strength = float(
            (benchmark_data["close"].iloc[-1]
             / benchmark_data["close"].iloc[lookback])
            - 1
        )
        return {
            "momentum": momentum,
            "volatility": volatility,
            "trend_strength": trend_strength,
        }

    def detect_event_drivers(
        self,
        benchmark_data: pd.DataFrame,
        positions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        event_risk = "low"
        if positions and any(
            abs(float(p.get("unrealized_plpc", 0.0))) > 0.05
            for p in positions
        ):
            event_risk = "medium"
        if len(positions) > 5:
            pnl_std = np.std([
                float(p.get("unrealized_plpc", 0.0))
                for p in positions
            ])
            if pnl_std > 0.04:
                event_risk = "high"

        large_move = 0.0
        if len(benchmark_data) > 5:
            large_move = float(
                benchmark_data["close"]
                .pct_change()
                .tail(5)
                .abs()
                .max()
            )
        return {
            "event_risk": event_risk,
            "active_position_count": len(positions),
            "large_momentum_move": large_move,
        }

    def aggregate_signals(
        self, symbol: str, signals: Dict[str, float]
    ) -> float:
        if not signals:
            return 0.0
        return self.signal_voter.aggregate_signals(signals)

    def score_asset(
        self,
        symbol: str,
        alpha: float,
        history: pd.DataFrame,
        regime: str,
    ) -> Dict[str, Any]:
        strategy_scores = {}
        for strategy in StrategyFactory.all_strategies():
            score = strategy.score(symbol, alpha, history, regime)
            strategy_scores[strategy.name] = float(score)
        consensus = self.aggregate_signals(symbol, strategy_scores)

        # Deterministic alpha drift using exponential decay
        predicted_alpha_drift = alpha * np.exp(-0.01)
        weighted_consensus = float(
            np.tanh(consensus * 1.1 + predicted_alpha_drift * 0.2)
        )

        return {
            "symbol": symbol,
            "alpha": float(alpha),
            "predicted_alpha_drift": float(predicted_alpha_drift),
            "regime": regime,
            "strategy_scores": strategy_scores,
            "consensus": consensus,
            "weighted_consensus": weighted_consensus,
        }

    def build_portfolio_signals(
        self,
        raw_signals: Dict[str, float],
        historical_data: Dict[str, pd.DataFrame],
        regime: str,
    ) -> Dict[str, float]:
        portfolio_scores = {}
        for symbol, alpha in raw_signals.items():
            history = historical_data.get(symbol, pd.DataFrame())
            asset_score = self.score_asset(
                symbol, alpha, history, regime
            )
            portfolio_scores[symbol] = asset_score[
                "weighted_consensus"
            ]
        return portfolio_scores

    def classify_strategy(
        self,
        symbol: str,
        alpha: float,
        history: pd.DataFrame,
        regime: str,
        macro_profile: Dict[str, float],
    ) -> str:
        """Classify trade strategy based on signals and conditions."""
        if history.empty or "close" not in history.columns:
            return "Neutral"
        returns = history["close"].pct_change().dropna()
        volatility = float(returns.std()) if not returns.empty else 0.0
        momentum = macro_profile.get("momentum", 0.0)
        trend_strength = macro_profile.get("trend_strength", 0.0)

        if abs(alpha) > 0.6 and volatility > 0.03:
            return "Event-Driven"
        elif momentum > 0.005 and trend_strength > 0.05:
            return "Macro"
        elif abs(alpha) > 0.4 and regime == "SIDEWAYS":
            return "Quant"
        elif volatility < 0.02 and abs(alpha) > 0.3:
            return "Multi-Strategy"
        else:
            return "Neutral"
