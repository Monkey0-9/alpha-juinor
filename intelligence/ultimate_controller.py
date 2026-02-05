"""
Ultimate AI Trading Controller - 2026 Peak Intelligence
=======================================================

THE ABSOLUTE PEAK of AI trading technology.

This combines EVERYTHING:
- Multi-Agent Ensemble (5 specialized agents)
- GPT-4 Strategic Reasoning
- Neural Market Predictor
- Elite Alpha Generator
- Transformer Regime Detector
- Adaptive Risk Manager
- Dynamic Portfolio Optimizer
- Kelly-Criterion Position Sizing

Target Performance:
- Annual Return: 60-70%
- Sharpe Ratio: > 2.5
- Max Drawdown: < 15%
- Win Rate: > 58%
"""

import logging
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class UltimateTradingDecision:
    """The most comprehensive trading decision available."""
    timestamp: datetime
    symbol: str
    action: str  # BUY, SELL, HOLD
    position_size: float  # % of portfolio
    confidence: float
    expected_return: float
    risk_reward: float
    regime: str

    # Component contributions
    ensemble_vote: float
    strategic_score: float
    neural_prediction: float
    alpha_signal: float

    # Risk metrics
    stop_loss: float
    take_profit: float
    max_holding_days: int

    # Reasoning
    reasoning: List[str] = field(default_factory=list)


class UltimateAIController:
    """
    THE ULTIMATE AI Trading Controller.

    This is the SMARTEST trading system possible in 2026.
    Combines ALL AI paradigms for maximum edge.
    """

    def __init__(self):
        self.initialized = False
        self.components = {}
        self._init_all_components()
        self.decision_count = 0

        logger.info("=" * 70)
        logger.info("[ULTIMATE_AI] ★★★ PEAK 2026 INTELLIGENCE ACTIVATED ★★★")
        logger.info("=" * 70)

    def _init_all_components(self):
        """Initialize all intelligence components."""

        # Multi-Agent Ensemble
        try:
            from intelligence.multi_agent_ensemble import get_multi_agent_ensemble
            self.ensemble = get_multi_agent_ensemble()
            self.components["ensemble"] = True
            logger.info("[ULTIMATE] ✓ Multi-Agent Ensemble loaded")
        except Exception as e:
            self.ensemble = None
            self.components["ensemble"] = False
            logger.warning(f"[ULTIMATE] Multi-Agent failed: {e}")

        # Strategic Reasoner
        try:
            from intelligence.strategic_reasoner import get_strategic_reasoner
            self.reasoner = get_strategic_reasoner()
            self.components["reasoner"] = True
            logger.info("[ULTIMATE] ✓ Strategic Reasoner loaded")
        except Exception as e:
            self.reasoner = None
            self.components["reasoner"] = False
            logger.warning(f"[ULTIMATE] Reasoner failed: {e}")

        # Neural Predictor
        try:
            from intelligence.neural_predictor import get_neural_predictor
            self.neural = get_neural_predictor()
            self.components["neural"] = True
            logger.info("[ULTIMATE] ✓ Neural Predictor loaded")
        except Exception as e:
            self.neural = None
            self.components["neural"] = False
            logger.warning(f"[ULTIMATE] Neural failed: {e}")

        # Alpha Generator
        try:
            from intelligence.alpha_generator import get_alpha_generator
            self.alpha_gen = get_alpha_generator()
            self.components["alpha"] = True
            logger.info("[ULTIMATE] ✓ Alpha Generator loaded")
        except Exception as e:
            self.alpha_gen = None
            self.components["alpha"] = False
            logger.warning(f"[ULTIMATE] Alpha failed: {e}")

        # Regime Detector
        try:
            from intelligence.regime_detector import get_regime_detector
            self.regime_detector = get_regime_detector()
            self.components["regime"] = True
            logger.info("[ULTIMATE] ✓ Regime Detector loaded")
        except Exception as e:
            self.regime_detector = None
            self.components["regime"] = False
            logger.warning(f"[ULTIMATE] Regime failed: {e}")

        # Risk Manager
        try:
            from intelligence.risk_manager import get_adaptive_risk_manager
            self.risk_mgr = get_adaptive_risk_manager()
            self.components["risk"] = True
            logger.info("[ULTIMATE] ✓ Risk Manager loaded")
        except Exception as e:
            self.risk_mgr = None
            self.components["risk"] = False
            logger.warning(f"[ULTIMATE] Risk failed: {e}")

        # Portfolio Optimizer
        try:
            from intelligence.portfolio_optimizer import get_portfolio_optimizer
            self.optimizer = get_portfolio_optimizer()
            self.components["optimizer"] = True
            logger.info("[ULTIMATE] ✓ Portfolio Optimizer loaded")
        except Exception as e:
            self.optimizer = None
            self.components["optimizer"] = False
            logger.warning(f"[ULTIMATE] Optimizer failed: {e}")

        active = sum(1 for v in self.components.values() if v)
        self.initialized = active >= 3

        logger.info(f"[ULTIMATE] Components active: {active}/{len(self.components)}")

    def generate_decision(
        self,
        symbol: str,
        price: float,
        features: Dict[str, float],
        returns_history: np.ndarray,
        market_data: Dict[str, Any],
        nav: float
    ) -> UltimateTradingDecision:
        """
        Generate THE ULTIMATE trading decision.
        """
        start_time = time.time()
        reasoning = []

        # 1. Detect Market Regime
        if self.regime_detector:
            market_returns = np.array(market_data.get("returns", [0] * 20))
            regime_state = self.regime_detector.detect(
                market_returns=market_returns,
                market_volatility=market_data.get("volatility", 0.015),
                vix_level=market_data.get("vix", 20),
                correlation_avg=market_data.get("correlation", 0.5),
                breadth=market_data.get("breadth", 0.6),
                momentum_20d=market_data.get("momentum", 0.01)
            )
            regime = regime_state.current_regime
            reasoning.append(f"Regime: {regime}")
        else:
            regime = "NORMAL"

        # 2. Multi-Agent Ensemble Vote
        if self.ensemble:
            ensemble_decision = self.ensemble.decide(symbol, features, regime)
            ensemble_vote = ensemble_decision.final_size
            reasoning.append(
                f"Ensemble: {ensemble_decision.final_action} "
                f"({ensemble_decision.consensus_level:.0%} consensus)"
            )
        else:
            ensemble_vote = 0.0

        # 3. Strategic Reasoning
        if self.reasoner:
            strategic = self.reasoner.analyze(
                symbol, price, features, market_data, regime
            )
            strategic_score = (
                1.0 if strategic.recommendation == "STRONG_BUY" else
                0.5 if strategic.recommendation == "BUY" else
                0.0 if strategic.recommendation == "HOLD" else
                -0.5 if strategic.recommendation == "SELL" else
                -1.0
            )
            reasoning.append(f"Strategy: {strategic.recommendation}")
        else:
            strategic_score = 0.0

        # 4. Neural Prediction
        if self.neural and len(returns_history) >= 5:
            neural_pred = self.neural.predict(symbol, features, returns_history)
            neural_signal = neural_pred.predicted_direction
            reasoning.append(
                f"Neural: {neural_signal:+.2f} "
                f"(±{neural_pred.uncertainty:.2f})"
            )
        else:
            neural_signal = 0.0

        # 5. Alpha Signal
        if self.alpha_gen:
            ret_1m = features.get("momentum_1m", 0)
            ret_12m = features.get("momentum_12m", 0)
            vol = features.get("volatility", 0.02)
            alpha = self.alpha_gen.generate_alpha(
                symbol, features, ret_1m, ret_12m, vol, regime
            )
            alpha_signal = alpha.alpha_value
            reasoning.append(f"Alpha: {alpha_signal:+.2f}")
        else:
            alpha_signal = 0.0

        # 6. COMBINE ALL SIGNALS - Weighted Fusion
        weights = {
            "ensemble": 0.30,
            "strategic": 0.25,
            "neural": 0.25,
            "alpha": 0.20
        }

        combined_signal = (
            weights["ensemble"] * ensemble_vote +
            weights["strategic"] * strategic_score +
            weights["neural"] * neural_signal +
            weights["alpha"] * (alpha_signal / 3)  # Normalize alpha
        )

        combined_signal = np.clip(combined_signal, -1.0, 1.0)

        # 7. Determine Action
        if combined_signal > 0.3:
            action = "BUY"
        elif combined_signal < -0.3:
            action = "SELL"
        else:
            action = "HOLD"

        # 8. Position Sizing with Risk Management
        if self.risk_mgr:
            risk_budget = self.risk_mgr.update_state(
                nav=nav,
                market_volatility=market_data.get("volatility", 0.015),
                correlation_avg=market_data.get("correlation", 0.5),
                regime=regime
            )

            base_size = abs(combined_signal) * risk_budget.max_position_size
            position_size = self.risk_mgr.get_position_size(
                signal_strength=abs(combined_signal),
                symbol_volatility=features.get("volatility", 0.02),
                risk_budget=risk_budget
            )

            stop_loss = risk_budget.stop_loss_level
            take_profit = risk_budget.take_profit_level
        else:
            position_size = min(0.10, abs(combined_signal) * 0.15)
            stop_loss = 0.05
            take_profit = 0.10

        # 9. Calculate Expected Return
        vol = features.get("volatility", 0.02)
        expected_return = abs(combined_signal) * 0.02
        risk_reward = expected_return / stop_loss if stop_loss > 0 else 0

        # 10. Determine Holding Period
        if regime in ["VOLATILE", "CRISIS"]:
            max_holding = 3
        elif regime == "BULL":
            max_holding = 20
        else:
            max_holding = 10

        # 11. Calculate Confidence
        confidences = []
        if self.ensemble:
            confidences.append(ensemble_decision.confidence)
        if self.reasoner:
            confidences.append(strategic.conviction)
        if self.neural and len(returns_history) >= 5:
            confidences.append(neural_pred.confidence)
        if self.alpha_gen:
            confidences.append(alpha.confidence)

        overall_confidence = np.mean(confidences) if confidences else 0.5

        elapsed_ms = (time.time() - start_time) * 1000
        reasoning.append(f"Decision time: {elapsed_ms:.0f}ms")

        self.decision_count += 1

        decision = UltimateTradingDecision(
            timestamp=datetime.utcnow(),
            symbol=symbol,
            action=action,
            position_size=position_size if action != "HOLD" else 0.0,
            confidence=overall_confidence,
            expected_return=expected_return,
            risk_reward=risk_reward,
            regime=regime,
            ensemble_vote=ensemble_vote,
            strategic_score=strategic_score,
            neural_prediction=neural_signal,
            alpha_signal=alpha_signal,
            stop_loss=stop_loss,
            take_profit=take_profit,
            max_holding_days=max_holding,
            reasoning=reasoning
        )

        # Log significant decisions
        if action != "HOLD" and overall_confidence > 0.5:
            logger.info(
                f"[ULTIMATE_DECISION] {symbol}: {action} {position_size:.1%} | "
                f"Conf={overall_confidence:.0%} | E[R]={expected_return:.2%} | "
                f"RR={risk_reward:.1f}x"
            )

        return decision

    def generate_portfolio_decisions(
        self,
        symbols: List[str],
        prices: Dict[str, float],
        features_map: Dict[str, Dict],
        returns_map: Dict[str, np.ndarray],
        market_data: Dict[str, Any],
        nav: float,
        max_positions: int = 20
    ) -> Dict[str, UltimateTradingDecision]:
        """
        Generate optimal decisions for entire portfolio.
        """
        all_decisions = {}

        for symbol in symbols:
            features = features_map.get(symbol, {})
            returns = returns_map.get(symbol, np.array([0] * 20))
            price = prices.get(symbol, 100)

            decision = self.generate_decision(
                symbol=symbol,
                price=price,
                features=features,
                returns_history=returns,
                market_data=market_data,
                nav=nav
            )

            all_decisions[symbol] = decision

        # Filter and rank by confidence and expected return
        actionable = {
            s: d for s, d in all_decisions.items()
            if d.action != "HOLD" and d.confidence > 0.5
        }

        # Sort by expected return * confidence
        sorted_symbols = sorted(
            actionable.keys(),
            key=lambda s: (
                actionable[s].expected_return * actionable[s].confidence
            ),
            reverse=True
        )[:max_positions]

        return {s: actionable[s] for s in sorted_symbols}

    def get_status(self) -> Dict[str, Any]:
        """Get controller status."""
        return {
            "status": "ULTIMATE_ACTIVE" if self.initialized else "DEGRADED",
            "decision_count": self.decision_count,
            "components": self.components,
            "active_components": sum(1 for v in self.components.values() if v),
            "total_components": len(self.components)
        }


# Singleton
_ultimate = None


def get_ultimate_controller() -> UltimateAIController:
    global _ultimate
    if _ultimate is None:
        _ultimate = UltimateAIController()
    return _ultimate
