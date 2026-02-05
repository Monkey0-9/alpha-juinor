"""
Master Intelligence Controller - 2026 Elite
============================================

The BRAIN of the Top 1% Trading System.

This controller orchestrates ALL intelligence components:
- Elite AI Brain (signal generation)
- Return Predictor (forecasting)
- Portfolio Optimizer (allocation)
- Alpha Generator (factor alpha)
- Regime Detector (market state)
- Risk Manager (dynamic risk)

Target: 60-70% Annual Return, Sharpe > 2.5, Max DD < 15%
"""

import logging
import time
from typing import Dict, List, Any
from dataclasses import dataclass
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MasterDecision:
    """Complete trading decision from Master Intelligence."""
    timestamp: datetime
    regime: str
    risk_budget_used: float
    signals: Dict[str, float]
    allocations: Dict[str, float]
    expected_return: float
    expected_sharpe: float
    confidence: float
    reasoning: List[str]


class MasterIntelligenceController:
    """
    THE Master Controller for Top 1% Trading Intelligence.

    This is the SMARTEST trading system achievable in 2026.
    """

    def __init__(self):
        self._init_components()
        self.last_decision = None
        self.decision_count = 0
        logger.info("=" * 60)
        logger.info("[MASTER_INTEL] 2026 Elite Intelligence Controller ACTIVE")
        logger.info("=" * 60)

    def _init_components(self):
        """Initialize all intelligence components."""
        try:
            from intelligence.elite_brain import get_elite_brain
            self.brain = get_elite_brain()
        except Exception as e:
            logger.warning(f"Elite Brain unavailable: {e}")
            self.brain = None

        try:
            from intelligence.return_predictor import get_return_predictor
            self.predictor = get_return_predictor()
        except Exception as e:
            logger.warning(f"Return Predictor unavailable: {e}")
            self.predictor = None

        try:
            from intelligence.portfolio_optimizer import get_portfolio_optimizer
            self.optimizer = get_portfolio_optimizer()
        except Exception as e:
            logger.warning(f"Portfolio Optimizer unavailable: {e}")
            self.optimizer = None

        try:
            from intelligence.alpha_generator import get_alpha_generator
            self.alpha_gen = get_alpha_generator()
        except Exception as e:
            logger.warning(f"Alpha Generator unavailable: {e}")
            self.alpha_gen = None

        try:
            from intelligence.regime_detector import get_regime_detector
            self.regime_detector = get_regime_detector()
        except Exception as e:
            logger.warning(f"Regime Detector unavailable: {e}")
            self.regime_detector = None

        try:
            from intelligence.risk_manager import get_adaptive_risk_manager
            self.risk_mgr = get_adaptive_risk_manager()
        except Exception as e:
            logger.warning(f"Risk Manager unavailable: {e}")
            self.risk_mgr = None

    def generate_master_decision(
        self,
        symbols: List[str],
        features_map: Dict[str, Dict],
        prices: Dict[str, float],
        returns_map: Dict[str, Dict],
        nav: float,
        current_positions: Dict[str, float],
        market_data: Dict[str, Any]
    ) -> MasterDecision:
        """
        Generate the MASTER trading decision using ALL intelligence.
        """
        start_time = time.time()
        reasoning = []

        # 1. Detect Market Regime
        if self.regime_detector:
            market_returns = np.array(
                market_data.get("market_returns", [0.0] * 20)
            )
            regime_state = self.regime_detector.detect(
                market_returns=market_returns,
                market_volatility=market_data.get("volatility", 0.015),
                vix_level=market_data.get("vix", 20),
                correlation_avg=market_data.get("correlation", 0.5),
                breadth=market_data.get("breadth", 0.6),
                momentum_20d=market_data.get("momentum", 0.01)
            )
            regime = regime_state.current_regime
            reasoning.append(
                f"Regime: {regime} ({regime_state.confidence:.0%} conf)"
            )
        else:
            regime = "NORMAL"
            reasoning.append("Regime: NORMAL (fallback)")

        # 2. Update Risk Budget
        if self.risk_mgr:
            risk_budget = self.risk_mgr.update_state(
                nav=nav,
                market_volatility=market_data.get("volatility", 0.015),
                correlation_avg=market_data.get("correlation", 0.5),
                regime=regime
            )
            reasoning.append(
                f"Risk Budget: {risk_budget.max_position_size:.1%} max pos"
            )
        else:
            risk_budget = None

        # 3. Generate Alpha Signals
        if self.alpha_gen:
            alpha_signals = self.alpha_gen.get_portfolio_alphas(
                symbols=symbols,
                features_map=features_map,
                returns_map=returns_map,
                regime=regime,
                top_n=30
            )
            reasoning.append(f"Alpha signals: {len(alpha_signals)} generated")
        else:
            alpha_signals = {}

        # 4. Generate Return Predictions
        predictions = {}
        if self.predictor:
            for sym in list(alpha_signals.keys())[:20]:
                features = features_map.get(sym, {})
                hist_returns = np.array(
                    list(returns_map.get(sym, {}).values())[:60]
                )
                if len(hist_returns) < 20:
                    hist_returns = np.zeros(20)

                pred = self.predictor.predict(
                    symbol=sym,
                    features=features,
                    historical_returns=hist_returns,
                    regime=regime
                )
                predictions[sym] = pred

        # 5. Combine into Trading Signals using Elite Brain
        trading_signals = {}
        confidences = {}

        if self.brain:
            for sym in alpha_signals:
                alpha = alpha_signals[sym]
                feat = features_map.get(sym, {})

                # Build model predictions from alpha and predictor
                model_preds = {
                    "alpha_gen": alpha.alpha_value,
                    "return_pred": predictions.get(sym, type('', (), {'pred_5d': 0})()).pred_5d if sym in predictions else 0,
                    "momentum": feat.get("momentum_20d", 0),
                    "mean_reversion": -feat.get("return_5d", 0)
                }

                elite_signal = self.brain.generate_elite_signal(
                    symbol=sym,
                    features=feat,
                    price_data={"price": prices.get(sym, 100)},
                    model_predictions=model_preds,
                    regime=regime
                )

                trading_signals[sym] = elite_signal.direction
                confidences[sym] = elite_signal.confidence
        else:
            for sym, alpha in alpha_signals.items():
                trading_signals[sym] = alpha.alpha_value
                confidences[sym] = alpha.confidence

        # 6. Optimize Portfolio Allocation
        if self.optimizer and trading_signals:
            # Build expected returns and volatilities
            exp_returns = {}
            volatilities = {}
            correlations = {}

            for sym in trading_signals:
                if sym in predictions:
                    exp_returns[sym] = predictions[sym].pred_5d
                else:
                    exp_returns[sym] = trading_signals[sym] * 0.01

                volatilities[sym] = features_map.get(sym, {}).get(
                    "volatility", 0.02
                )

            # Simple correlation assumption
            sym_list = list(trading_signals.keys())
            for i, s1 in enumerate(sym_list):
                for j, s2 in enumerate(sym_list):
                    if i != j:
                        correlations[(s1, s2)] = 0.3

            allocation = self.optimizer.optimize(
                signals=trading_signals,
                expected_returns=exp_returns,
                volatilities=volatilities,
                correlations=correlations,
                current_weights=current_positions,
                regime=regime
            )

            final_weights = allocation.weights
            exp_return = allocation.expected_return
            exp_sharpe = allocation.expected_sharpe

            reasoning.append(
                f"Portfolio: {len(final_weights)} positions, "
                f"E[R]={exp_return:.2%}, Sharpe={exp_sharpe:.2f}"
            )
        else:
            final_weights = {}
            exp_return = 0.0
            exp_sharpe = 0.0

        # Apply risk manager position sizing
        if self.risk_mgr and risk_budget:
            for sym in list(final_weights.keys()):
                vol = volatilities.get(sym, 0.02)
                max_size = self.risk_mgr.get_position_size(
                    signal_strength=trading_signals.get(sym, 0),
                    symbol_volatility=vol,
                    risk_budget=risk_budget
                )
                if final_weights[sym] > max_size:
                    final_weights[sym] = max_size

        # Calculate overall confidence
        if confidences:
            avg_conf = np.mean(list(confidences.values()))
        else:
            avg_conf = 0.5

        elapsed = (time.time() - start_time) * 1000
        reasoning.append(f"Decision time: {elapsed:.0f}ms")

        decision = MasterDecision(
            timestamp=datetime.utcnow(),
            regime=regime,
            risk_budget_used=sum(final_weights.values()),
            signals=trading_signals,
            allocations=final_weights,
            expected_return=exp_return,
            expected_sharpe=exp_sharpe,
            confidence=avg_conf,
            reasoning=reasoning
        )

        self.last_decision = decision
        self.decision_count += 1

        # Log decision
        logger.info(
            f"[MASTER_DECISION] #{self.decision_count} | "
            f"Regime={regime} | Positions={len(final_weights)} | "
            f"E[R]={exp_return:.2%} | Sharpe={exp_sharpe:.2f} | "
            f"Conf={avg_conf:.0%}"
        )

        return decision

    def get_status(self) -> Dict[str, Any]:
        """Get controller status."""
        return {
            "status": "ELITE_ACTIVE",
            "decision_count": self.decision_count,
            "components": {
                "brain": self.brain is not None,
                "predictor": self.predictor is not None,
                "optimizer": self.optimizer is not None,
                "alpha_gen": self.alpha_gen is not None,
                "regime_detector": self.regime_detector is not None,
                "risk_mgr": self.risk_mgr is not None
            },
            "last_regime": self.last_decision.regime if self.last_decision else "UNKNOWN"
        }


# Singleton
_controller = None


def get_master_controller() -> MasterIntelligenceController:
    global _controller
    if _controller is None:
        _controller = MasterIntelligenceController()
    return _controller
