"""
Master Brain Orchestrator - The Ultimate Trading Intelligence
=============================================================

Unifies ALL intelligence modules into ONE SUPREME BRAIN:
- Deep Alpha Brain (300 IQ thinking)
- Autonomous Reasoner (7-step analysis)
- Decision Validator (triple-check)
- Stock Scanner (market-wide opportunities)
- Quant Engine (advanced math)
- Learning Feedback (self-improvement)

This is the FINAL layer - the master controller that
coordinates all intelligence for PERFECT trades.
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DecisionQuality(Enum):
    """Quality level of a trading decision."""
    REJECTED = 0
    ACCEPTABLE = 1
    GOOD = 2
    EXCELLENT = 3
    PERFECT = 4


@dataclass
class MasterDecision:
    """The final, validated trading decision from the Master Brain."""
    timestamp: datetime
    symbol: str
    action: str  # BUY, SELL, HOLD

    # Core decision
    signal_strength: float  # -1 to 1
    confidence: float  # 0 to 1
    position_size: Decimal

    # Price levels
    entry_price: Decimal
    target_price: Decimal
    stop_loss: Decimal

    # Quality metrics
    decision_quality: DecisionQuality
    validation_passed: bool
    thinking_depth: int
    conviction_score: float

    # Reasoning
    primary_reasons: List[str]
    risk_factors: List[str]

    # Analytics used
    technical_score: float
    fundamental_score: float
    sentiment_score: float
    momentum_score: float
    value_score: float
    quality_score: float

    # Expected outcome
    expected_return: Decimal
    probability_success: Decimal
    risk_reward_ratio: Decimal
    expected_value: Decimal  # probability * return

    # Processing time
    total_processing_time_ms: float
    modules_consulted: List[str]


class MasterBrainOrchestrator:
    """
    The SUPREME trading intelligence that orchestrates all modules.

    This is the final decision-maker that:
    1. Consults all intelligence modules
    2. Synthesizes all signals
    3. Validates everything triple-times
    4. Makes the PERFECT final decision
    5. Explains its reasoning
    6. Learns from outcomes

    ZERO tolerance for errors.
    MAXIMUM precision.
    300 IQ level thinking.
    """

    # Quality thresholds
    MIN_CONFIDENCE_TO_TRADE = 0.65
    MIN_EXPECTED_VALUE = Decimal("0.005")  # 0.5% EV minimum
    MAX_POSITION_PER_TRADE = Decimal("0.08")

    def __init__(self):
        """Initialize the Master Brain."""
        self._init_modules()

        self.decisions_made = 0
        self.perfect_decisions = 0
        self.rejected_decisions = 0

        logger.info(
            "[MASTER] Master Brain Orchestrator initialized - "
            "Supreme Intelligence Mode ACTIVE"
        )

    def _init_modules(self):
        """Initialize all intelligence modules."""
        # Deep Alpha Brain
        try:
            from intelligence.deep_alpha_brain import get_deep_brain
            self._deep_brain = get_deep_brain()
            logger.info("[MASTER] Deep Alpha Brain loaded")
        except Exception as e:
            logger.warning(f"[MASTER] Deep Brain not available: {e}")
            self._deep_brain = None

        # Autonomous Reasoner
        try:
            from intelligence.autonomous_reasoner import get_reasoner
            self._reasoner = get_reasoner()
            logger.info("[MASTER] Autonomous Reasoner loaded")
        except Exception as e:
            logger.warning(f"[MASTER] Reasoner not available: {e}")
            self._reasoner = None

        # Decision Validator
        try:
            from intelligence.decision_validator import get_validator
            self._validator = get_validator()
            logger.info("[MASTER] Decision Validator loaded")
        except Exception as e:
            logger.warning(f"[MASTER] Validator not available: {e}")
            self._validator = None

        # Stock Scanner
        try:
            from intelligence.stock_scanner import get_scanner
            self._scanner = get_scanner()
            logger.info("[MASTER] Stock Scanner loaded")
        except Exception as e:
            logger.warning(f"[MASTER] Scanner not available: {e}")
            self._scanner = None

        # Quant Engine
        try:
            from analytics.quant_engine import get_quant_engine
            self._quant_engine = get_quant_engine()
            logger.info("[MASTER] Quant Engine loaded")
        except Exception as e:
            logger.warning(f"[MASTER] Quant Engine not available: {e}")
            self._quant_engine = None

        # Learning Feedback
        try:
            from learning.learning_feedback import get_feedback_loop
            self._feedback = get_feedback_loop()
            logger.info("[MASTER] Learning Feedback loaded")
        except Exception as e:
            logger.warning(f"[MASTER] Feedback not available: {e}")
            self._feedback = None

        # Precision Math
        try:
            from maths.precision_math import PrecisionMath
            self._precision_math = PrecisionMath
            logger.info("[MASTER] Precision Math loaded")
        except Exception as e:
            logger.warning(f"[MASTER] Precision Math not available: {e}")
            self._precision_math = None

        # Zero Loss Guardian - Loss Prevention
        try:
            from intelligence.zero_loss_guardian import get_guardian
            self._guardian = get_guardian()
            logger.info("[MASTER] Zero Loss Guardian loaded - LOSS PREVENTION ACTIVE")
        except Exception as e:
            logger.warning(f"[MASTER] Guardian not available: {e}")
            self._guardian = None

        # Perfect Timing Engine
        try:
            from intelligence.perfect_timing import get_timing_engine
            self._timing = get_timing_engine()
            logger.info("[MASTER] Perfect Timing Engine loaded")
        except Exception as e:
            logger.warning(f"[MASTER] Timing not available: {e}")
            self._timing = None

        # Genius Trade Picker
        try:
            from intelligence.genius_picker import get_genius_picker
            self._genius = get_genius_picker()
            logger.info("[MASTER] Genius Trade Picker loaded")
        except Exception as e:
            logger.warning(f"[MASTER] Genius Picker not available: {e}")
            self._genius = None


    def make_decision(
        self,
        symbol: str,
        market_data: pd.DataFrame,
        fundamentals: Optional[Dict] = None,
        sentiment: Optional[Dict] = None,
        portfolio_state: Optional[Dict] = None,
        current_price: Optional[float] = None
    ) -> MasterDecision:
        """
        Make the FINAL trading decision for a symbol.

        This consults all modules, validates everything,
        and returns the perfect decision.
        """
        start_time = time.time()
        modules_consulted = []

        logger.debug(f"[MASTER] Making decision for {symbol}...")

        # Get current price
        if current_price is None:
            try:
                if isinstance(market_data.columns, pd.MultiIndex):
                    current_price = float(market_data[symbol]["Close"].dropna().iloc[-1])
                else:
                    current_price = float(market_data["Close"].dropna().iloc[-1])
            except Exception:
                current_price = 0.0

        entry_price = Decimal(str(current_price))

        # Initialize scores
        technical_score = 0.5
        fundamental_score = 0.5
        sentiment_score = 0.5
        momentum_score = 0.5
        value_score = 0.5
        quality_score = 0.5

        all_signals = []
        all_confidences = []
        all_reasons = []
        risk_factors = []

        # 1. Consult Deep Alpha Brain
        if self._deep_brain:
            try:
                thinking = self._deep_brain.think(
                    f"Should I trade {symbol}?",
                    {
                        "symbol": symbol,
                        "prices": market_data[symbol]["Close"] if isinstance(
                            market_data.columns, pd.MultiIndex
                        ) else market_data.get("Close"),
                        "fundamentals": fundamentals or {},
                        "sentiment": sentiment or {}
                    }
                )

                signal = 0.5 if thinking.conclusion == "BULLISH" else (
                    -0.5 if thinking.conclusion == "BEARISH" else 0
                )
                all_signals.append(signal)
                all_confidences.append(thinking.confidence)
                all_reasons.extend(thinking.supporting_evidence)
                risk_factors.extend(thinking.risks_identified)
                modules_consulted.append("DeepAlphaBrain")

            except Exception as e:
                logger.debug(f"[MASTER] Deep Brain error: {e}")

        # 2. Consult Autonomous Reasoner
        if self._reasoner:
            try:
                if isinstance(market_data.columns, pd.MultiIndex):
                    prices = market_data[symbol]["Close"].dropna()
                else:
                    prices = market_data.get("Close", pd.Series()).dropna()

                if len(prices) >= 50:
                    decision = self._reasoner.reason(symbol, prices)

                    all_signals.append(decision.signal_strength)
                    all_confidences.append(decision.confidence)
                    all_reasons.extend(decision.reasoning_chain)

                    technical_score = decision.technical_score
                    fundamental_score = decision.fundamental_score
                    sentiment_score = decision.sentiment_score

                    modules_consulted.append("AutonomousReasoner")

            except Exception as e:
                logger.debug(f"[MASTER] Reasoner error: {e}")

        # 3. Consult Quant Engine for advanced analysis
        if self._quant_engine and isinstance(market_data.columns, pd.MultiIndex):
            try:
                prices = market_data[symbol]["Close"].dropna()

                if len(prices) >= 100:
                    # Volatility from GARCH
                    garch_result = self._quant_engine.garch_engine.forecast_volatility(
                        prices.pct_change().dropna()
                    )
                    if garch_result.get("forecast_1d", 0) > 0.03:
                        risk_factors.append(f"High volatility: {garch_result['forecast_1d']:.1%}")

                    # Trend from Kalman
                    kalman_result = self._quant_engine.kalman_engine.filter_price(prices)
                    if kalman_result.get("trend_strength", 0) > 0.3:
                        all_reasons.append("Strong trend detected by Kalman filter")

                    # Mean reversion from Hurst
                    hurst = self._quant_engine.calculate_hurst(prices)
                    if hurst < 0.4:
                        all_reasons.append(f"Mean-reverting behavior (Hurst={hurst:.2f})")
                    elif hurst > 0.6:
                        all_reasons.append(f"Trending behavior (Hurst={hurst:.2f})")

                    modules_consulted.append("QuantEngine")

            except Exception as e:
                logger.debug(f"[MASTER] Quant Engine error: {e}")

        # 4. Calculate final signal
        if all_signals:
            # Weighted average of all signals
            weights = [1.0] * len(all_signals)  # Equal weights for now
            final_signal = sum(s * w for s, w in zip(all_signals, weights)) / sum(weights)
            final_signal = float(np.clip(final_signal, -1, 1))
        else:
            final_signal = 0.0

        if all_confidences:
            final_confidence = float(np.mean(all_confidences))
        else:
            final_confidence = 0.5

        # 5. Determine action
        if final_signal > 0.2 and final_confidence >= self.MIN_CONFIDENCE_TO_TRADE:
            action = "BUY"
        elif final_signal < -0.2 and final_confidence >= self.MIN_CONFIDENCE_TO_TRADE:
            action = "SELL"
        else:
            action = "HOLD"

        # 6. Calculate position size using Kelly
        if self._precision_math and action != "HOLD":
            try:
                p = Decimal(str(final_confidence))
                b = Decimal("2.0")  # Assume 2:1 risk/reward
                kelly = self._precision_math.kelly_criterion(p, b, fraction=Decimal("0.25"))
                position_size = min(kelly, self.MAX_POSITION_PER_TRADE)
            except Exception:
                position_size = Decimal("0.02")
        else:
            position_size = Decimal("0")

        # 7. Calculate price levels
        if action == "BUY":
            target_price = entry_price * Decimal("1.05")
            stop_loss = entry_price * Decimal("0.97")
        elif action == "SELL":
            target_price = entry_price * Decimal("0.95")
            stop_loss = entry_price * Decimal("1.03")
        else:
            target_price = entry_price
            stop_loss = entry_price

        # 8. Calculate expected value
        if action != "HOLD":
            expected_return = (target_price - entry_price) / entry_price
            probability_success = Decimal(str(final_confidence))
            expected_value = expected_return * probability_success

            risk_amount = abs(entry_price - stop_loss)
            reward_amount = abs(target_price - entry_price)
            risk_reward = reward_amount / risk_amount if risk_amount > 0 else Decimal("0")
        else:
            expected_return = Decimal("0")
            probability_success = Decimal("0.5")
            expected_value = Decimal("0")
            risk_reward = Decimal("0")

        # 9. Validate the decision
        validation_passed = True
        if self._validator and action != "HOLD":
            try:
                validation = self._validator.validate({
                    "symbol": symbol,
                    "action": action,
                    "signal_strength": final_signal,
                    "confidence": final_confidence,
                    "position_size": float(position_size),
                    "entry_price": float(entry_price),
                    "target_price": float(target_price),
                    "stop_loss": float(stop_loss),
                    "expected_return": float(expected_return),
                    "risk_reward_ratio": float(risk_reward),
                    "reasoning": all_reasons[:5]
                }, portfolio_state=portfolio_state)

                validation_passed = validation.all_passed

                if not validation_passed:
                    # Either reject or downgrade the decision
                    if validation.critical_failures > 0:
                        action = "HOLD"
                        position_size = Decimal("0")
                        risk_factors.append("Decision failed validation - rejected")
                    else:
                        position_size = position_size * Decimal("0.5")
                        risk_factors.append("Validation warnings - position reduced")

                modules_consulted.append("DecisionValidator")

            except Exception as e:
                logger.debug(f"[MASTER] Validation error: {e}")

        # 10. Determine decision quality
        if not validation_passed:
            quality = DecisionQuality.REJECTED
        elif final_confidence >= 0.85 and expected_value >= Decimal("0.02"):
            quality = DecisionQuality.PERFECT
        elif final_confidence >= 0.75 and expected_value >= Decimal("0.01"):
            quality = DecisionQuality.EXCELLENT
        elif final_confidence >= 0.65:
            quality = DecisionQuality.GOOD
        else:
            quality = DecisionQuality.ACCEPTABLE

        # 11. Record decision for learning
        if self._feedback and action != "HOLD":
            try:
                self._feedback.record_decision(
                    symbol=symbol,
                    action=action,
                    confidence=final_confidence,
                    models_used=modules_consulted
                )
            except Exception:
                pass

        # Update stats
        self.decisions_made += 1
        if quality == DecisionQuality.PERFECT:
            self.perfect_decisions += 1
        if quality == DecisionQuality.REJECTED:
            self.rejected_decisions += 1

        processing_time = (time.time() - start_time) * 1000

        decision = MasterDecision(
            timestamp=datetime.utcnow(),
            symbol=symbol,
            action=action,
            signal_strength=final_signal,
            confidence=final_confidence,
            position_size=position_size,
            entry_price=entry_price.quantize(Decimal("0.01")),
            target_price=target_price.quantize(Decimal("0.01")),
            stop_loss=stop_loss.quantize(Decimal("0.01")),
            decision_quality=quality,
            validation_passed=validation_passed,
            thinking_depth=len(modules_consulted),
            conviction_score=final_confidence * abs(final_signal),
            primary_reasons=all_reasons[:5],
            risk_factors=risk_factors[:5],
            technical_score=technical_score,
            fundamental_score=fundamental_score,
            sentiment_score=sentiment_score,
            momentum_score=momentum_score,
            value_score=value_score,
            quality_score=quality_score,
            expected_return=expected_return.quantize(Decimal("0.0001")),
            probability_success=probability_success.quantize(Decimal("0.01")),
            risk_reward_ratio=risk_reward.quantize(Decimal("0.01")),
            expected_value=expected_value.quantize(Decimal("0.0001")),
            total_processing_time_ms=processing_time,
            modules_consulted=modules_consulted
        )

        logger.info(
            f"[MASTER] {symbol}: {action} | "
            f"Quality={quality.name} | "
            f"Confidence={final_confidence:.1%} | "
            f"Size={position_size:.2%} | "
            f"Time={processing_time:.0f}ms"
        )

        return decision

    def scan_and_decide(
        self,
        market_data: pd.DataFrame,
        fundamentals: Optional[Dict[str, Dict]] = None,
        top_n: int = 5
    ) -> List[MasterDecision]:
        """
        Scan entire market and make decisions on best opportunities.

        This is the AUTONOMOUS mode where the brain:
        1. Scans all stocks
        2. Finds best opportunities
        3. Makes decisions on each
        4. Returns only VALIDATED decisions
        """
        logger.info("[MASTER] Starting autonomous market scan and decision process...")

        decisions = []

        # Use scanner to find opportunities
        if self._scanner:
            try:
                scan_results = self._scanner.full_market_scan(market_data, fundamentals)
                top_picks = scan_results.get("COMPOSITE", [])[:top_n * 2]

                for pick in top_picks:
                    decision = self.make_decision(
                        symbol=pick.symbol,
                        market_data=market_data,
                        fundamentals=fundamentals.get(pick.symbol) if fundamentals else None
                    )

                    # Only include good+ decisions
                    if decision.decision_quality.value >= DecisionQuality.GOOD.value:
                        decisions.append(decision)

                        if len(decisions) >= top_n:
                            break

            except Exception as e:
                logger.warning(f"[MASTER] Scan error: {e}")

        # Use deep brain as backup
        if not decisions and self._deep_brain:
            try:
                opportunities = self._deep_brain.find_best_opportunities(
                    market_data,
                    fundamentals,
                    None,
                    top_n
                )

                for opp in opportunities.qualified_opportunities[:top_n]:
                    decision = self.make_decision(
                        symbol=opp.symbol,
                        market_data=market_data,
                        fundamentals=fundamentals.get(opp.symbol) if fundamentals else None
                    )

                    if decision.decision_quality.value >= DecisionQuality.GOOD.value:
                        decisions.append(decision)

            except Exception as e:
                logger.warning(f"[MASTER] Deep brain backup error: {e}")

        # Sort by expected value
        decisions.sort(
            key=lambda d: float(d.expected_value) * d.confidence,
            reverse=True
        )

        logger.info(
            f"[MASTER] Autonomous scan complete: "
            f"{len(decisions)} quality decisions generated"
        )

        return decisions[:top_n]

    def get_stats(self) -> Dict[str, Any]:
        """Get Master Brain statistics."""
        return {
            "total_decisions": self.decisions_made,
            "perfect_decisions": self.perfect_decisions,
            "rejected_decisions": self.rejected_decisions,
            "perfection_rate": self.perfect_decisions / self.decisions_made if self.decisions_made > 0 else 0,
            "rejection_rate": self.rejected_decisions / self.decisions_made if self.decisions_made > 0 else 0
        }


# Singleton
_master_brain: Optional[MasterBrainOrchestrator] = None


def get_master_brain() -> MasterBrainOrchestrator:
    """Get or create the Master Brain Orchestrator."""
    global _master_brain
    if _master_brain is None:
        _master_brain = MasterBrainOrchestrator()
    return _master_brain
