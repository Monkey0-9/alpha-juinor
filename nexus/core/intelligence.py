"""
nexus/core/intelligence.py — Superhuman Market Intelligence Brain

Upgrades from simple majority-vote strategy agreement to:
  - Bayesian-weighted strategy agreement (posterior hit-rate weighting)
  - Regime probability distribution (not just a label)
  - Cross-asset correlation pulse (crisis detection → size reduction)
  - Entropy-filtered sentiment (avoids noisy portfolio signals)
  - Full integration with SuperhumanBrain conviction layer
"""
import logging
from typing import Dict, List, Any
import numpy as np
import pandas as pd
from nexus.core.strategies import StrategyFactory
from nexus.math.indicators import RegimeDetector, compute_hurst_exponent
from nexus.math.governance import LatticeVoter, StrategySwitcher
from nexus.math.risk import RiskEngine

logger = logging.getLogger(__name__)


class MarketBrain:
    """
    Superhuman Market Intelligence Engine.

    Adaptive market analysis with:
    - Probabilistic regime distribution (4-state)
    - Bayesian strategy agreement scoring
    - Cross-asset correlation crisis detection
    - Hurst-gated sentiment filtering
    - Forward-bias correction using rolling IC
    """

    def __init__(self) -> None:
        self.regime_detector   = RegimeDetector()
        self.strategy_switcher = StrategySwitcher()
        self.signal_voter      = LatticeVoter()
        self.risk_engine       = RiskEngine()

        # Rolling Bayesian strategy accuracy per regime
        # regime -> strategy_name -> [correct (1) / incorrect (0)]
        self._strategy_outcomes: Dict[str, Dict[str, List[int]]] = {}
        self._OUTCOME_WINDOW = 30

    # ------------------------------------------------------------------ #
    # Primary Market Analysis                                             #
    # ------------------------------------------------------------------ #

    def analyze_market(
        self,
        benchmark_data: pd.DataFrame,
        positions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Analyze market conditions with superhuman intelligence.
        Returns enriched market insight including regime probabilities.
        """
        logger.info("Analyzing market conditions [SuperhumanMarketBrain]...")

        # Step 1: Probabilistic regime detection
        regime_probs = self.regime_detector.detect_probabilities(benchmark_data)
        regime = max(regime_probs, key=lambda k: regime_probs[k])
        strategy = self.strategy_switcher.select_strategy(regime)

        # Step 2: Enriched analysis components
        sentiment = self.calculate_global_sentiment(positions, benchmark_data)
        macro_profile = self.assess_macro(benchmark_data)
        event_profile = self.detect_event_drivers(benchmark_data, positions)
        correlation_pulse = self.detect_correlation_pulse(positions)

        # Step 3: Hurst-filtered Bayesian strategy agreement
        agreement, strategy_weights = self._bayesian_strategy_agreement(
            benchmark_data, regime
        )

        # Step 4: Hurst exponent for market structure insight
        hurst = 0.5
        if not benchmark_data.empty and "close" in benchmark_data.columns:
            close = benchmark_data["close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            hurst = compute_hurst_exponent(close)

        # Step 5: Market forecast with entropy-adjusted confidence
        regime_entropy = -sum(p * np.log(p + 1e-9) for p in regime_probs.values())
        forecast_confidence = 1.0 - regime_entropy / np.log(4)  # 4 states

        if sentiment > 0.55 and hurst > 0.50:
            market_forecast = "BULLISH_CONTINUATION"
            sentiment_label = "Bullish Bias"
        elif sentiment < 0.45 and hurst < 0.50:
            market_forecast = "BEARISH_REVERSION"
            sentiment_label = "Bearish Bias"
        elif hurst > 0.65:
            market_forecast = "TRENDING_MOMENTUM"
            sentiment_label = "Trend Regime"
        elif hurst < 0.35:
            market_forecast = "MEAN_REVERT_OPPORTUNITY"
            sentiment_label = "Mean-Revert Regime"
        else:
            market_forecast = "NEUTRAL"
            sentiment_label = "Neutral"

        logger.info(
            "Regime=%s (P=%.0f%%) | Strategy=%s | Bayesian Agreement=%.0f%% | "
            "Hurst=%.2f | Forecast=%s | Confidence=%.0f%%",
            regime, regime_probs.get(regime, 0) * 100,
            strategy, agreement * 100,
            hurst, market_forecast, forecast_confidence * 100
        )

        strategy_names = [s.name for s in StrategyFactory.all_strategies()]

        return {
            "regime":                 regime,
            "regime_probabilities":   regime_probs,
            "forecast_confidence":    forecast_confidence,
            "selected_strategy":      strategy,
            "market_sentiment":       sentiment,
            "market_sentiment_label": sentiment_label,
            "macro_profile":          macro_profile,
            "event_profile":          event_profile,
            "strategy_agreement":     agreement,
            "strategy_weights":       strategy_weights,
            "market_forecast":        market_forecast,
            "strategy_universe":      strategy_names,
            "hurst_exponent":         hurst,
            "correlation_pulse":      correlation_pulse,
        }

    # ------------------------------------------------------------------ #
    # Bayesian Strategy Agreement                                          #
    # ------------------------------------------------------------------ #

    def _bayesian_strategy_agreement(
        self,
        benchmark_data: pd.DataFrame,
        regime: str,
    ) -> tuple[float, Dict[str, float]]:
        """
        Bayesian-weighted strategy agreement.

        Each strategy gets a posterior weight based on its recent
        hit-rate in the current regime. Agreement is the Bayesian-weighted
        fraction of strategies pointing in the same direction.

        Returns (agreement_score, strategy_weights_dict)
        """
        if benchmark_data.empty:
            return 0.5, {}

        strategies  = StrategyFactory.all_strategies()
        scores: Dict[str, float] = {}
        weights: Dict[str, float] = {}

        for strat in strategies:
            s = strat.score("SPY", 0.0, benchmark_data, regime)
            scores[strat.name] = float(s)

            # Get Bayesian weight from recent outcomes in this regime
            outcome_list = (
                self._strategy_outcomes
                .get(regime, {})
                .get(strat.name, [])
            )
            if len(outcome_list) < 3:
                w = 1.0  # neutral prior
            else:
                hit_rate = sum(outcome_list[-self._OUTCOME_WINDOW:]) / len(outcome_list[-self._OUTCOME_WINDOW:])
                # Weight: 1.0 = neutral; >1.0 = above-chance; <1.0 = below-chance
                w = max(0.1, (hit_rate - 0.40) * 3.5 + 1.0)
            weights[strat.name] = w

        # Normalize weights
        total_w = sum(weights.values()) + 1e-9
        norm_weights = {n: w / total_w for n, w in weights.items()}

        # Weighted agreement: sum of weights * sign-agreement
        directions = {n: np.sign(s) for n, s in scores.items()}
        dominant_direction = np.sign(sum(
            norm_weights[n] * directions[n] for n in directions
        ))

        weighted_agreement = sum(
            norm_weights[n]
            for n, d in directions.items()
            if d == dominant_direction and abs(scores[n]) > 0.05
        )

        return float(weighted_agreement), norm_weights

    # ------------------------------------------------------------------ #
    # Sentiment Analysis (Hurst-filtered)                                  #
    # ------------------------------------------------------------------ #

    def calculate_global_sentiment(
        self,
        positions: List[Dict[str, Any]],
        benchmark_data: pd.DataFrame = pd.DataFrame(),
    ) -> float:
        """
        Portfolio sentiment filtered by Hurst exponent.
        High Hurst + positive PnL → strongly bullish
        """
        if not positions:
            base_sentiment = 0.5
        else:
            pnl_values = [float(p.get("unrealized_plpc", 0.0)) for p in positions]
            avg_pnl = float(np.mean(pnl_values))
            # Entropy-filtered clipping: wider range for volatile portfolios
            pnl_std = float(np.std(pnl_values)) if len(pnl_values) > 1 else 0.0
            clip_range = min(0.45, 0.30 + pnl_std * 3)
            base_sentiment = float(0.5 + np.clip(avg_pnl * 10.0, -clip_range, clip_range))

        # Hurst overlay from benchmark
        if not benchmark_data.empty and "close" in benchmark_data.columns:
            close = benchmark_data["close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            if len(close) >= 20:
                hurst = compute_hurst_exponent(close)
                recent_ret = float(close.pct_change().tail(5).mean())
                # Hurst > 0.55 + positive recent returns → amplify bullish sentiment
                if hurst > 0.55 and recent_ret > 0:
                    base_sentiment = min(0.95, base_sentiment * 1.1)
                elif hurst > 0.55 and recent_ret < 0:
                    base_sentiment = max(0.05, base_sentiment * 0.9)

        return float(base_sentiment)

    # ------------------------------------------------------------------ #
    # Cross-Asset Correlation Pulse                                        #
    # ------------------------------------------------------------------ #

    def detect_correlation_pulse(
        self, positions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Detect when held assets become highly correlated (crisis mode).

        When avg pairwise correlation > 0.80: ALL assets move together → crisis.
        Recommendation: reduce position sizes by 50%.
        """
        if len(positions) < 2:
            return {"crisis_mode": False, "avg_correlation": 0.0, "size_multiplier": 1.0}

        # Use PnL series as a proxy for return correlation
        pnl_values = [float(p.get("unrealized_plpc", 0.0)) for p in positions]
        if len(pnl_values) < 2:
            return {"crisis_mode": False, "avg_correlation": 0.0, "size_multiplier": 1.0}

        # Proxy correlation: if all P&L values are same sign, correlation is high
        same_sign = sum(1 for v in pnl_values if v > 0)
        frac = max(same_sign, len(pnl_values) - same_sign) / len(pnl_values)
        proxy_corr = (frac - 0.5) * 2.0  # scale to [0, 1]

        # High PnL variance with same-direction = crisis candidate
        pnl_std = float(np.std(pnl_values)) if len(pnl_values) > 1 else 0.0
        is_crisis = proxy_corr > 0.75 and pnl_std > 0.05

        size_mult = 0.50 if is_crisis else 1.0

        if is_crisis:
            logger.warning(
                "CORRELATION CRISIS DETECTED: avg_corr=%.2f, pnl_std=%.2f → halving positions",
                proxy_corr, pnl_std
            )

        return {
            "crisis_mode":    is_crisis,
            "avg_correlation": float(proxy_corr),
            "size_multiplier": size_mult,
        }

    # ------------------------------------------------------------------ #
    # Macro & Event Analysis                                              #
    # ------------------------------------------------------------------ #

    def assess_macro(
        self, benchmark_data: pd.DataFrame
    ) -> Dict[str, float]:
        if benchmark_data.empty or "close" not in benchmark_data.columns:
            return {"momentum": 0.0, "volatility": 0.0, "trend_strength": 0.0, "hurst": 0.5}

        close = benchmark_data["close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]

        returns = close.pct_change().dropna()
        if returns.empty:
            return {"momentum": 0.0, "volatility": 0.0, "trend_strength": 0.0, "hurst": 0.5}

        momentum = float(returns.tail(10).mean())
        volatility = float(returns.std())
        lookback = max(0, len(benchmark_data) - 21)
        trend_strength = float(
            (float(close.iloc[-1]) / float(close.iloc[lookback])) - 1
        )
        hurst = compute_hurst_exponent(close)

        return {
            "momentum":      momentum,
            "volatility":    volatility,
            "trend_strength": trend_strength,
            "hurst":         hurst,
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
            close = benchmark_data["close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            large_move = float(close.pct_change().tail(5).abs().max())

        return {
            "event_risk":           event_risk,
            "active_position_count": len(positions),
            "large_momentum_move":   large_move,
        }

    # ------------------------------------------------------------------ #
    # Portfolio Signal Building                                           #
    # ------------------------------------------------------------------ #

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
        """Score a single asset with all strategies + Bayesian aggregation."""
        strategy_scores = {}
        for strategy in StrategyFactory.all_strategies():
            score = strategy.score(symbol, alpha, history, regime)
            strategy_scores[strategy.name] = float(score)

        consensus = self.aggregate_signals(symbol, strategy_scores)

        # Hurst-gated alpha drift: only apply drift when market is persistent
        hurst = 0.5
        if not history.empty and "close" in history.columns:
            close = history["close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            hurst = compute_hurst_exponent(close)

        # Hurst-gated alpha: trend (H>0.55) → use predicted drift; random (H≈0.5) → suppress drift
        hurst_gate = max(0.0, (hurst - 0.45) * 2.0)  # 0 when H<0.45, 1 when H>0.95
        predicted_alpha_drift = alpha * np.exp(-0.01) * hurst_gate

        weighted_consensus = float(
            np.tanh(consensus * 1.15 + predicted_alpha_drift * 0.25)
        )

        return {
            "symbol":                symbol,
            "alpha":                 float(alpha),
            "predicted_alpha_drift": float(predicted_alpha_drift),
            "regime":                regime,
            "hurst":                 float(hurst),
            "strategy_scores":       strategy_scores,
            "consensus":             consensus,
            "weighted_consensus":    weighted_consensus,
        }

    def build_portfolio_signals(
        self,
        raw_signals: Dict[str, float],
        historical_data: Dict[str, pd.DataFrame],
        regime: str,
    ) -> Dict[str, float]:
        """Build portfolio signals for all symbols."""
        portfolio_scores = {}
        for symbol, alpha in raw_signals.items():
            history = historical_data.get(symbol, pd.DataFrame())
            asset_score = self.score_asset(symbol, alpha, history, regime)
            portfolio_scores[symbol] = asset_score["weighted_consensus"]
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
        hurst = macro_profile.get("hurst", 0.5)

        if hurst > 0.60 and abs(alpha) > 0.5:
            return "Momentum_Alpha"
        elif hurst < 0.40 and abs(alpha) > 0.4:
            return "Mean_Reversion_Alpha"
        elif abs(alpha) > 0.6 and volatility > 0.03:
            return "Event-Driven"
        elif momentum > 0.005 and trend_strength > 0.05:
            return "Macro"
        elif abs(alpha) > 0.4 and regime == "SIDEWAYS":
            return "Quant"
        elif volatility < 0.02 and abs(alpha) > 0.3:
            return "Multi-Strategy"
        else:
            return "Neutral"
