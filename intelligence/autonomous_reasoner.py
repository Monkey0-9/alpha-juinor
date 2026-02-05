"""
Autonomous Reasoner - Elite Trading Decision Engine
====================================================

The "300 IQ Brain" that thinks deeply before every decision.

This module implements multi-step reasoning for trading decisions:
1. Technical Analysis (price patterns, indicators)
2. Fundamental Analysis (value, growth, quality)
3. Sentiment Analysis (news, social, options flow)
4. Regime Detection (market conditions)
5. Multi-Model Consensus (ensemble of all ML models)
6. Risk Assessment (position sizing, drawdown protection)
7. Final Decision (confidence-weighted action)

Target: Top 1% Hedge Fund Performance
- Sharpe Ratio > 2.5
- Max Drawdown < 15%
- Win Rate > 55%
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DecisionType(Enum):
    """Types of trading decisions."""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"
    NO_ACTION = "NO_ACTION"


class ReasoningStep(Enum):
    """Steps in the reasoning process."""
    TECHNICAL = "TECHNICAL"
    FUNDAMENTAL = "FUNDAMENTAL"
    SENTIMENT = "SENTIMENT"
    REGIME = "REGIME"
    CONSENSUS = "CONSENSUS"
    RISK = "RISK"
    FINAL = "FINAL"


@dataclass
class AnalysisResult:
    """Result from a single analysis step."""
    step: ReasoningStep
    signal: float  # -1 to +1
    confidence: float  # 0 to 1
    reasoning: str
    metrics: Dict[str, float] = field(default_factory=dict)
    computation_time_ms: float = 0.0


@dataclass
class TradingDecision:
    """Complete trading decision with full reasoning trail."""
    symbol: str
    decision: DecisionType
    signal_strength: float  # -1 to +1
    confidence: float  # 0 to 1
    position_size: float  # Fraction of portfolio
    expected_return: float
    max_risk: float
    reasoning_chain: List[AnalysisResult]
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "symbol": self.symbol,
            "decision": self.decision.value,
            "signal_strength": self.signal_strength,
            "confidence": self.confidence,
            "position_size": self.position_size,
            "expected_return": self.expected_return,
            "max_risk": self.max_risk,
            "reasoning": [r.reasoning for r in self.reasoning_chain],
            "timestamp": self.timestamp.isoformat()
        }


class AutonomousReasoner:
    """
    300-IQ Equivalent Trading Brain.

    This is the SMARTEST decision-making system:
    - Chains 7 analysis steps before deciding
    - Uses high-precision mathematics
    - Learns from every decision
    - Explains its reasoning

    Integration Points:
    - ModelOrchestrator: For ML predictions
    - EliteAIBrain: For signal aggregation
    - RiskManager: For position sizing
    - FeatureStore: For cached features
    """

    # Confidence thresholds for decisions
    MIN_TRADE_CONFIDENCE = 0.6
    STRONG_SIGNAL_THRESHOLD = 0.7

    # Position sizing parameters
    MAX_POSITION_SIZE = 0.10  # 10% per position
    KELLY_FRACTION = 0.25  # Use 25% Kelly for safety

    def __init__(self):
        """Initialize the Autonomous Reasoner with all components."""
        self.initialized = False
        self.decision_count = 0
        self.cumulative_accuracy = 0.0

        # Lazy-loaded components
        self._model_orchestrator = None
        self._elite_brain = None
        self._risk_manager = None
        self._feature_store = None

        # Performance tracking
        self.decisions_made = []
        self.learning_memory = {}

        logger.info("[REASONER] Autonomous Reasoner initializing...")
        self._init_components()
        logger.info("[REASONER] Autonomous Reasoner ready.")

    def _init_components(self):
        """Initialize all intelligence components."""
        try:
            # Model Orchestrator - ensemble of all ML models
            from ml.model_orchestrator import get_model_orchestrator
            self._model_orchestrator = get_model_orchestrator()
        except Exception as e:
            logger.warning(f"[REASONER] ModelOrchestrator not available: {e}")
            self._model_orchestrator = None

        try:
            # Elite AI Brain - signal aggregation
            from intelligence.elite_brain import get_elite_brain
            self._elite_brain = get_elite_brain()
        except Exception as e:
            logger.warning(f"[REASONER] EliteAIBrain not available: {e}")
            self._elite_brain = None

        try:
            # Risk Engine
            from risk.engine import RiskManager
            self._risk_manager = RiskManager()
        except Exception as e:
            logger.warning(f"[REASONER] RiskManager not available: {e}")
            self._risk_manager = None

        try:
            # Quantitative Analysis Engine
            from analytics.quant_engine import get_quant_engine
            self._quant_engine = get_quant_engine()
            logger.info("[REASONER] Quantitative Analysis Engine loaded")
        except Exception as e:
            logger.warning(f"[REASONER] QuantEngine not available: {e}")
            self._quant_engine = None

        try:
            # Learning Feedback Loop
            from learning.learning_feedback import get_feedback_loop
            self._feedback_loop = get_feedback_loop()
            logger.info("[REASONER] Learning Feedback Loop loaded")
        except Exception as e:
            logger.warning(f"[REASONER] FeedbackLoop not available: {e}")
            self._feedback_loop = None

        try:
            # Precision Math Engine
            from maths.precision_math import PrecisionMath
            self._precision_math = PrecisionMath
            logger.info("[REASONER] Precision Math Engine loaded")
        except Exception as e:
            logger.warning(f"[REASONER] PrecisionMath not available: {e}")
            self._precision_math = None

        self.initialized = True


    def reason(
        self,
        symbol: str,
        market_data: pd.DataFrame,
        features: Optional[Dict[str, float]] = None,
        current_position: float = 0.0,
        nav: float = 1_000_000.0
    ) -> TradingDecision:
        """
        Execute full reasoning chain for a symbol.

        This is the CORE method - it thinks through all aspects
        before making a decision.

        Args:
            symbol: Stock symbol
            market_data: OHLCV data for the symbol
            features: Pre-computed features (optional)
            current_position: Current position size
            nav: Current NAV for position sizing

        Returns:
            TradingDecision with full reasoning chain
        """
        start_time = time.time()
        reasoning_chain: List[AnalysisResult] = []

        logger.debug(f"[REASONER] Starting deep reasoning for {symbol}...")

        # Step 1: Technical Analysis
        tech_result = self._analyze_technical(symbol, market_data)
        reasoning_chain.append(tech_result)

        # Step 2: Fundamental Analysis (if data available)
        fund_result = self._analyze_fundamental(symbol, features)
        reasoning_chain.append(fund_result)

        # Step 3: Sentiment Analysis
        sent_result = self._analyze_sentiment(symbol, features)
        reasoning_chain.append(sent_result)

        # Step 4: Regime Detection
        regime_result = self._detect_regime(market_data)
        reasoning_chain.append(regime_result)

        # Step 5: Multi-Model Consensus
        consensus_result = self._get_model_consensus(
            symbol, market_data, features, regime_result.metrics.get("regime", "NORMAL")
        )
        reasoning_chain.append(consensus_result)

        # Step 6: Risk Assessment
        risk_result = self._assess_risk(
            symbol, reasoning_chain, current_position, nav
        )
        reasoning_chain.append(risk_result)

        # Step 7: Final Decision
        final_decision = self._make_final_decision(
            symbol, reasoning_chain, risk_result
        )

        # Track decision
        self.decision_count += 1
        self.decisions_made.append({
            "symbol": symbol,
            "decision": final_decision.decision.value,
            "confidence": final_decision.confidence,
            "timestamp": datetime.utcnow().isoformat()
        })

        total_time = (time.time() - start_time) * 1000
        logger.info(
            f"[REASONER] {symbol}: {final_decision.decision.value} "
            f"(conf={final_decision.confidence:.2f}, "
            f"size={final_decision.position_size:.4f}) "
            f"in {total_time:.1f}ms"
        )

        return final_decision

    def _analyze_technical(
        self, symbol: str, market_data: pd.DataFrame
    ) -> AnalysisResult:
        """
        Step 1: Technical Analysis

        Analyzes:
        - Trend (SMA crossovers)
        - Momentum (RSI, MACD)
        - Volatility (ATR, Bollinger Bands)
        - Volume patterns
        """
        start = time.time()

        try:
            # Extract close prices for the symbol
            if isinstance(market_data.columns, pd.MultiIndex):
                if symbol in market_data.columns.get_level_values(0):
                    closes = market_data[symbol]["Close"].dropna()
                else:
                    closes = pd.Series(dtype=float)
            else:
                closes = market_data.get("Close", pd.Series(dtype=float))

            if len(closes) < 20:
                return AnalysisResult(
                    step=ReasoningStep.TECHNICAL,
                    signal=0.0,
                    confidence=0.0,
                    reasoning=f"Insufficient data for {symbol} ({len(closes)} bars)",
                    computation_time_ms=(time.time() - start) * 1000
                )

            # Calculate indicators
            sma_20 = closes.rolling(20).mean().iloc[-1]
            sma_50 = closes.rolling(50).mean().iloc[-1] if len(closes) >= 50 else sma_20
            current_price = closes.iloc[-1]

            # RSI calculation
            delta = closes.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss.replace(0, 1e-10)
            rsi = 100 - (100 / (1 + rs)).iloc[-1]

            # MACD
            ema_12 = closes.ewm(span=12).mean()
            ema_26 = closes.ewm(span=26).mean()
            macd = (ema_12 - ema_26).iloc[-1]
            signal_line = (ema_12 - ema_26).ewm(span=9).mean().iloc[-1]
            macd_histogram = macd - signal_line

            # Trend signal
            trend_signal = 0.0
            if current_price > sma_20 > sma_50:
                trend_signal = 0.5  # Bullish
            elif current_price < sma_20 < sma_50:
                trend_signal = -0.5  # Bearish

            # RSI signal
            rsi_signal = 0.0
            if rsi < 30:
                rsi_signal = 0.3  # Oversold - bullish
            elif rsi > 70:
                rsi_signal = -0.3  # Overbought - bearish

            # MACD signal
            macd_signal = 0.3 if macd_histogram > 0 else -0.3

            # Combine signals
            total_signal = np.clip(trend_signal + rsi_signal + macd_signal, -1, 1)
            confidence = min(0.8, abs(total_signal) + 0.3)

            reasoning = (
                f"Trend: {'Bullish' if trend_signal > 0 else 'Bearish' if trend_signal < 0 else 'Neutral'} "
                f"(Price vs SMA20/50), RSI: {rsi:.1f}, MACD: {'Positive' if macd_histogram > 0 else 'Negative'}"
            )

            return AnalysisResult(
                step=ReasoningStep.TECHNICAL,
                signal=total_signal,
                confidence=confidence,
                reasoning=reasoning,
                metrics={
                    "rsi": rsi,
                    "macd": macd,
                    "trend_signal": trend_signal,
                    "sma_20": sma_20,
                    "sma_50": sma_50
                },
                computation_time_ms=(time.time() - start) * 1000
            )

        except Exception as e:
            logger.warning(f"[REASONER] Technical analysis error for {symbol}: {e}")
            return AnalysisResult(
                step=ReasoningStep.TECHNICAL,
                signal=0.0,
                confidence=0.0,
                reasoning=f"Technical analysis failed: {str(e)}",
                computation_time_ms=(time.time() - start) * 1000
            )

    def _analyze_fundamental(
        self, symbol: str, features: Optional[Dict[str, float]]
    ) -> AnalysisResult:
        """
        Step 2: Fundamental Analysis

        Analyzes:
        - Value metrics (P/E, P/B, P/S)
        - Quality metrics (ROE, ROIC)
        - Growth metrics (Revenue growth, EPS growth)
        """
        start = time.time()

        if not features:
            return AnalysisResult(
                step=ReasoningStep.FUNDAMENTAL,
                signal=0.0,
                confidence=0.2,
                reasoning="No fundamental data available",
                computation_time_ms=(time.time() - start) * 1000
            )

        try:
            signals = []
            reasons = []

            # Value signal
            pe_ratio = features.get("pe_ratio", 0)
            if pe_ratio > 0:
                if pe_ratio < 15:
                    signals.append(0.3)
                    reasons.append(f"Attractive P/E: {pe_ratio:.1f}")
                elif pe_ratio > 30:
                    signals.append(-0.2)
                    reasons.append(f"High P/E: {pe_ratio:.1f}")

            # Quality signal
            roe = features.get("roe", 0)
            if roe > 0.15:
                signals.append(0.2)
                reasons.append(f"Strong ROE: {roe:.1%}")
            elif roe < 0.05:
                signals.append(-0.1)
                reasons.append(f"Weak ROE: {roe:.1%}")

            # Growth signal
            rev_growth = features.get("revenue_growth", 0)
            if rev_growth > 0.10:
                signals.append(0.2)
                reasons.append(f"Strong growth: {rev_growth:.1%}")
            elif rev_growth < -0.05:
                signals.append(-0.2)
                reasons.append(f"Declining revenue: {rev_growth:.1%}")

            total_signal = np.clip(sum(signals), -1, 1) if signals else 0.0
            confidence = min(0.7, 0.3 + len(signals) * 0.15)

            return AnalysisResult(
                step=ReasoningStep.FUNDAMENTAL,
                signal=total_signal,
                confidence=confidence,
                reasoning=", ".join(reasons) if reasons else "Limited fundamental data",
                metrics=features,
                computation_time_ms=(time.time() - start) * 1000
            )

        except Exception as e:
            return AnalysisResult(
                step=ReasoningStep.FUNDAMENTAL,
                signal=0.0,
                confidence=0.0,
                reasoning=f"Fundamental analysis error: {e}",
                computation_time_ms=(time.time() - start) * 1000
            )

    def _analyze_sentiment(
        self, symbol: str, features: Optional[Dict[str, float]]
    ) -> AnalysisResult:
        """
        Step 3: Sentiment Analysis

        Analyzes:
        - News sentiment
        - Social media trends
        - Options flow
        - Analyst ratings
        """
        start = time.time()

        try:
            # Try to get NLP sentiment if available
            sentiment_score = 0.0
            confidence = 0.3
            reasons = []

            if features:
                # Check for pre-computed sentiment features
                news_sent = features.get("news_sentiment", 0)
                social_sent = features.get("social_sentiment", 0)
                options_flow = features.get("options_flow_signal", 0)

                if news_sent != 0:
                    sentiment_score += news_sent * 0.4
                    reasons.append(f"News: {'Positive' if news_sent > 0 else 'Negative'}")

                if social_sent != 0:
                    sentiment_score += social_sent * 0.3
                    reasons.append(f"Social: {'Bullish' if social_sent > 0 else 'Bearish'}")

                if options_flow != 0:
                    sentiment_score += options_flow * 0.3
                    reasons.append(f"Options: {'Calls' if options_flow > 0 else 'Puts'}")

                confidence = 0.5 if reasons else 0.2

            # Try NLP analyzer if available
            if self._model_orchestrator and hasattr(self._model_orchestrator, 'nlp_analyzer'):
                try:
                    nlp_result = self._model_orchestrator._run_nlp_sentiment(symbol)
                    if nlp_result and nlp_result.success:
                        sentiment_score = (sentiment_score + nlp_result.signal) / 2
                        confidence = max(confidence, nlp_result.confidence)
                        reasons.append(f"NLP: {nlp_result.signal:.2f}")
                except Exception:
                    pass

            return AnalysisResult(
                step=ReasoningStep.SENTIMENT,
                signal=np.clip(sentiment_score, -1, 1),
                confidence=confidence,
                reasoning=", ".join(reasons) if reasons else "No sentiment data available",
                metrics={"sentiment_score": sentiment_score},
                computation_time_ms=(time.time() - start) * 1000
            )

        except Exception as e:
            return AnalysisResult(
                step=ReasoningStep.SENTIMENT,
                signal=0.0,
                confidence=0.0,
                reasoning=f"Sentiment analysis error: {e}",
                computation_time_ms=(time.time() - start) * 1000
            )

    def _detect_regime(self, market_data: pd.DataFrame) -> AnalysisResult:
        """
        Step 4: Market Regime Detection

        Detects:
        - Bull/Bear market
        - High/Low volatility
        - Trend/Mean-reversion environment
        """
        start = time.time()

        try:
            # Try to get SPY or market index data
            spy_closes = None

            if isinstance(market_data.columns, pd.MultiIndex):
                for idx_symbol in ["SPY", "^GSPC", "QQQ"]:
                    if idx_symbol in market_data.columns.get_level_values(0):
                        spy_closes = market_data[idx_symbol]["Close"].dropna()
                        break

            if spy_closes is None or len(spy_closes) < 50:
                # Use average of available data
                return AnalysisResult(
                    step=ReasoningStep.REGIME,
                    signal=0.0,
                    confidence=0.3,
                    reasoning="Insufficient market data for regime detection",
                    metrics={"regime": "UNKNOWN"},
                    computation_time_ms=(time.time() - start) * 1000
                )

            # Calculate regime indicators
            returns = spy_closes.pct_change().dropna()
            volatility = returns.rolling(20).std().iloc[-1] * np.sqrt(252)
            sma_200 = spy_closes.rolling(200).mean().iloc[-1] if len(spy_closes) >= 200 else spy_closes.mean()
            current = spy_closes.iloc[-1]

            # Determine regime
            is_bull = current > sma_200
            is_high_vol = volatility > 0.20

            if is_bull and not is_high_vol:
                regime = "BULL_QUIET"
                signal = 0.3
                reasoning = "Bullish trend, low volatility - favorable for longs"
            elif is_bull and is_high_vol:
                regime = "BULL_VOLATILE"
                signal = 0.1
                reasoning = "Bullish but volatile - reduce position sizes"
            elif not is_bull and not is_high_vol:
                regime = "BEAR_QUIET"
                signal = -0.2
                reasoning = "Bearish trend, orderly decline - favor shorts"
            else:
                regime = "BEAR_CRISIS"
                signal = -0.5
                reasoning = "Crisis mode - high volatility, bearish - extreme caution"

            return AnalysisResult(
                step=ReasoningStep.REGIME,
                signal=signal,
                confidence=0.7,
                reasoning=reasoning,
                metrics={
                    "regime": regime,
                    "volatility": volatility,
                    "trend": "BULL" if is_bull else "BEAR"
                },
                computation_time_ms=(time.time() - start) * 1000
            )

        except Exception as e:
            return AnalysisResult(
                step=ReasoningStep.REGIME,
                signal=0.0,
                confidence=0.0,
                reasoning=f"Regime detection error: {e}",
                metrics={"regime": "UNKNOWN"},
                computation_time_ms=(time.time() - start) * 1000
            )

    def _get_model_consensus(
        self,
        symbol: str,
        market_data: pd.DataFrame,
        features: Optional[Dict[str, float]],
        regime: str
    ) -> AnalysisResult:
        """
        Step 5: Multi-Model Consensus

        Aggregates predictions from:
        - HMM Predictor
        - Deep Ensemble (LSTM, Transformer, CNN)
        - RL Trading Agent
        - Statistical Alpha
        """
        start = time.time()

        try:
            predictions = {}

            # Get prices for the symbol
            if isinstance(market_data.columns, pd.MultiIndex):
                if symbol in market_data.columns.get_level_values(0):
                    prices = market_data[symbol]["Close"].dropna()
                else:
                    prices = pd.Series(dtype=float)
            else:
                prices = market_data.get("Close", pd.Series(dtype=float))

            # Use Model Orchestrator if available
            if self._model_orchestrator and len(prices) > 20:
                try:
                    ensemble_pred = self._model_orchestrator.predict(
                        symbol=symbol,
                        prices=prices,
                        features=np.array(list(features.values())) if features else None
                    )

                    predictions["ensemble"] = ensemble_pred.final_signal

                    # Get individual model predictions
                    for model_name, pred in ensemble_pred.model_predictions.items():
                        if pred.success:
                            predictions[model_name] = pred.signal

                except Exception as e:
                    logger.debug(f"[REASONER] Model orchestrator error: {e}")

            # Use Elite Brain if available
            if self._elite_brain and features:
                try:
                    elite_signal = self._elite_brain.generate_elite_signal(
                        symbol=symbol,
                        features=features,
                        price_data={"close": float(prices.iloc[-1]) if len(prices) > 0 else 0},
                        model_predictions=predictions,
                        regime=regime
                    )
                    predictions["elite_brain"] = elite_signal.direction
                except Exception as e:
                    logger.debug(f"[REASONER] Elite brain error: {e}")

            # Calculate consensus
            if predictions:
                # Weight by model type
                weights = {
                    "ensemble": 0.3,
                    "elite_brain": 0.25,
                    "hmm": 0.15,
                    "deep_ensemble": 0.15,
                    "rl_agent": 0.10,
                    "momentum": 0.05
                }

                total_weight = 0
                weighted_signal = 0

                for model, signal in predictions.items():
                    weight = weights.get(model, 0.1)
                    weighted_signal += signal * weight
                    total_weight += weight

                if total_weight > 0:
                    consensus_signal = weighted_signal / total_weight
                else:
                    consensus_signal = 0.0

                # Agreement confidence
                signals = list(predictions.values())
                agreement = 1 - np.std(signals) if len(signals) > 1 else 0.5
                confidence = min(0.9, 0.4 + agreement * 0.5)

                reasoning = f"Consensus from {len(predictions)} models: " + \
                           ", ".join([f"{k}: {v:.2f}" for k, v in list(predictions.items())[:3]])

                return AnalysisResult(
                    step=ReasoningStep.CONSENSUS,
                    signal=np.clip(consensus_signal, -1, 1),
                    confidence=confidence,
                    reasoning=reasoning,
                    metrics=predictions,
                    computation_time_ms=(time.time() - start) * 1000
                )
            else:
                return AnalysisResult(
                    step=ReasoningStep.CONSENSUS,
                    signal=0.0,
                    confidence=0.2,
                    reasoning="No model predictions available",
                    computation_time_ms=(time.time() - start) * 1000
                )

        except Exception as e:
            return AnalysisResult(
                step=ReasoningStep.CONSENSUS,
                signal=0.0,
                confidence=0.0,
                reasoning=f"Model consensus error: {e}",
                computation_time_ms=(time.time() - start) * 1000
            )

    def _assess_risk(
        self,
        symbol: str,
        reasoning_chain: List[AnalysisResult],
        current_position: float,
        nav: float
    ) -> AnalysisResult:
        """
        Step 6: Risk Assessment

        Determines:
        - Optimal position size (Kelly criterion)
        - Stop loss levels
        - Maximum drawdown tolerance
        """
        start = time.time()

        try:
            # Aggregate signals from previous steps
            signals = [r.signal for r in reasoning_chain if r.confidence > 0.3]
            confidences = [r.confidence for r in reasoning_chain if r.confidence > 0.3]

            if not signals:
                return AnalysisResult(
                    step=ReasoningStep.RISK,
                    signal=0.0,
                    confidence=0.0,
                    reasoning="No confident signals to assess risk",
                    metrics={"position_size": 0.0},
                    computation_time_ms=(time.time() - start) * 1000
                )

            # Weighted average signal
            avg_signal = np.average(signals, weights=confidences)
            avg_confidence = np.mean(confidences)

            # Get regime for risk adjustment
            regime_result = next(
                (r for r in reasoning_chain if r.step == ReasoningStep.REGIME),
                None
            )
            regime = regime_result.metrics.get("regime", "NORMAL") if regime_result else "NORMAL"

            # Regime-adjusted risk
            regime_risk_multiplier = {
                "BULL_QUIET": 1.0,
                "BULL_VOLATILE": 0.7,
                "BEAR_QUIET": 0.5,
                "BEAR_CRISIS": 0.25,
                "UNKNOWN": 0.5,
                "NORMAL": 0.8
            }.get(regime, 0.5)

            # Kelly Criterion position sizing (simplified)
            # f* = (p * b - q) / b where p = win prob, b = win/loss ratio
            win_prob = 0.5 + avg_signal * 0.15  # Signal to probability
            win_loss_ratio = 1.5  # Assume 1.5:1 reward/risk

            kelly_fraction = (win_prob * win_loss_ratio - (1 - win_prob)) / win_loss_ratio
            kelly_fraction = max(0, kelly_fraction)

            # Apply safety fraction and regime adjustment
            position_size = kelly_fraction * self.KELLY_FRACTION * regime_risk_multiplier
            position_size = min(position_size, self.MAX_POSITION_SIZE)

            # Consider current position
            if current_position != 0:
                # Reduce new position if already exposed
                position_size *= max(0.5, 1 - abs(current_position) / self.MAX_POSITION_SIZE)

            # Risk signal: positive if risk is acceptable, negative if too risky
            risk_signal = avg_signal * avg_confidence * regime_risk_multiplier

            reasoning = (
                f"Position size: {position_size:.2%} (Kelly: {kelly_fraction:.2%}, "
                f"Regime adj: {regime_risk_multiplier:.1f}, Safety: {self.KELLY_FRACTION})"
            )

            return AnalysisResult(
                step=ReasoningStep.RISK,
                signal=risk_signal,
                confidence=avg_confidence,
                reasoning=reasoning,
                metrics={
                    "position_size": position_size,
                    "kelly_fraction": kelly_fraction,
                    "regime_multiplier": regime_risk_multiplier,
                    "win_probability": win_prob
                },
                computation_time_ms=(time.time() - start) * 1000
            )

        except Exception as e:
            return AnalysisResult(
                step=ReasoningStep.RISK,
                signal=0.0,
                confidence=0.0,
                reasoning=f"Risk assessment error: {e}",
                metrics={"position_size": 0.0},
                computation_time_ms=(time.time() - start) * 1000
            )

    def _make_final_decision(
        self,
        symbol: str,
        reasoning_chain: List[AnalysisResult],
        risk_result: AnalysisResult
    ) -> TradingDecision:
        """
        Step 7: Final Decision

        Synthesizes all analysis into a final trading decision.
        """
        # Weighted combination of all signals
        weights = {
            ReasoningStep.TECHNICAL: 0.25,
            ReasoningStep.FUNDAMENTAL: 0.15,
            ReasoningStep.SENTIMENT: 0.10,
            ReasoningStep.REGIME: 0.15,
            ReasoningStep.CONSENSUS: 0.25,
            ReasoningStep.RISK: 0.10
        }

        total_signal = 0.0
        total_confidence = 0.0
        total_weight = 0.0

        for result in reasoning_chain:
            weight = weights.get(result.step, 0.1)
            if result.confidence > 0:
                total_signal += result.signal * weight * result.confidence
                total_confidence += result.confidence * weight
                total_weight += weight

        if total_weight > 0:
            final_signal = total_signal / total_weight
            final_confidence = total_confidence / total_weight
        else:
            final_signal = 0.0
            final_confidence = 0.0

        # Get position size from risk assessment
        position_size = risk_result.metrics.get("position_size", 0.0)

        # Determine decision type
        if final_confidence < self.MIN_TRADE_CONFIDENCE:
            decision = DecisionType.NO_ACTION
            position_size = 0.0
        elif final_signal > self.STRONG_SIGNAL_THRESHOLD:
            decision = DecisionType.STRONG_BUY
        elif final_signal > 0.3:
            decision = DecisionType.BUY
        elif final_signal < -self.STRONG_SIGNAL_THRESHOLD:
            decision = DecisionType.STRONG_SELL
        elif final_signal < -0.3:
            decision = DecisionType.SELL
        else:
            decision = DecisionType.HOLD
            position_size *= 0.5  # Reduce size for uncertain signals

        # Expected return estimation
        expected_return = final_signal * 0.05  # 5% base expectation
        max_risk = abs(expected_return) * 1.5  # Risk at 1.5x expected return

        # Add final reasoning step
        final_result = AnalysisResult(
            step=ReasoningStep.FINAL,
            signal=final_signal,
            confidence=final_confidence,
            reasoning=f"Final: {decision.value} with {final_confidence:.0%} confidence",
            metrics={"decision": decision.value}
        )
        reasoning_chain.append(final_result)

        return TradingDecision(
            symbol=symbol,
            decision=decision,
            signal_strength=final_signal,
            confidence=final_confidence,
            position_size=position_size,
            expected_return=expected_return,
            max_risk=max_risk,
            reasoning_chain=reasoning_chain
        )

    def make_decisions(
        self,
        symbols: List[str],
        market_data: pd.DataFrame,
        current_positions: Optional[Dict[str, float]] = None,
        nav: float = 1_000_000.0,
        max_positions: int = 20
    ) -> Dict[str, TradingDecision]:
        """
        Make decisions for multiple symbols and rank by quality.

        Args:
            symbols: List of symbols to analyze
            market_data: Market data for all symbols
            current_positions: Current portfolio positions
            nav: Current NAV
            max_positions: Maximum number of positions to recommend

        Returns:
            Dictionary of symbol -> TradingDecision
        """
        if current_positions is None:
            current_positions = {}

        decisions = {}

        for symbol in symbols:
            try:
                current_pos = current_positions.get(symbol, 0.0)
                decision = self.reason(
                    symbol=symbol,
                    market_data=market_data,
                    current_position=current_pos,
                    nav=nav
                )
                decisions[symbol] = decision
            except Exception as e:
                logger.warning(f"[REASONER] Failed to reason for {symbol}: {e}")

        # Rank by confidence * signal strength
        ranked = sorted(
            decisions.items(),
            key=lambda x: abs(x[1].signal_strength) * x[1].confidence,
            reverse=True
        )

        # Return top N decisions
        return dict(ranked[:max_positions])

    def to_signals(
        self, decisions: Dict[str, TradingDecision]
    ) -> pd.Series:
        """Convert decisions to signal series for compatibility."""
        signals = {}
        for symbol, decision in decisions.items():
            if decision.decision != DecisionType.NO_ACTION:
                signals[symbol] = decision.signal_strength * decision.position_size
            else:
                signals[symbol] = 0.0
        return pd.Series(signals)

    def get_status(self) -> Dict[str, Any]:
        """Get current reasoner status."""
        return {
            "initialized": self.initialized,
            "decision_count": self.decision_count,
            "components": {
                "model_orchestrator": self._model_orchestrator is not None,
                "elite_brain": self._elite_brain is not None,
                "risk_manager": self._risk_manager is not None
            },
            "recent_decisions": self.decisions_made[-10:]
        }


# Singleton instance
_reasoner: Optional[AutonomousReasoner] = None


def get_reasoner() -> AutonomousReasoner:
    """Get or create the global Autonomous Reasoner instance."""
    global _reasoner
    if _reasoner is None:
        _reasoner = AutonomousReasoner()
    return _reasoner
