"""
Intelligence Package - 2026 Ultimate Edition
=============================================

THE MOST ADVANCED AI Trading Intelligence Available.

This package contains:
1. UltimateAIController - The master brain
2. Multi-Agent Ensemble - 5 specialized agents
3. Strategic Reasoner - GPT-4 level reasoning
4. Neural Predictor - Deep learning predictions
5. Alpha Generator - 8 alpha families
6. Regime Detector - Market state classification
7. Risk Manager - Dynamic position sizing
8. Portfolio Optimizer - Black-Litterman + HRP
9. Elite Brain - Kelly criterion sizing
10. Return Predictor - Multi-horizon forecasting
"""

# Ultimate Controller (use this!)
from intelligence.ultimate_controller import (
    UltimateAIController,
    UltimateTradingDecision,
    get_ultimate_controller
)

# Multi-Agent System
from intelligence.multi_agent_ensemble import (
    MultiAgentEnsemble,
    EnsembleDecision,
    get_multi_agent_ensemble
)

# Strategic Reasoning
from intelligence.strategic_reasoner import (
    GPT4StrategicReasoner,
    StrategicAnalysis,
    get_strategic_reasoner
)

# Neural Prediction
from intelligence.neural_predictor import (
    NeuralMarketPredictor,
    NeuralPrediction,
    get_neural_predictor
)

# Alpha Generation
from intelligence.alpha_generator import (
    EliteAlphaGenerator,
    AlphaSignal,
    get_alpha_generator
)

# Regime Detection
from intelligence.regime_detector import (
    TransformerRegimeDetector,
    RegimeState,
    get_regime_detector
)

# Risk Management
from intelligence.risk_manager import (
    AdaptiveRiskManager,
    RiskBudget,
    get_adaptive_risk_manager
)

# Portfolio Optimization
from intelligence.portfolio_optimizer import (
    DynamicPortfolioOptimizer,
    PortfolioAllocation,
    get_portfolio_optimizer
)

# Elite Brain
from intelligence.elite_brain import (
    EliteAIBrain,
    EliteSignal,
    get_elite_brain
)

# Return Prediction
from intelligence.return_predictor import (
    AdvancedReturnPredictor,
    ReturnPrediction,
    get_return_predictor
)


__all__ = [
    # Ultimate Controller
    "UltimateAIController",
    "UltimateTradingDecision",
    "get_ultimate_controller",
    # Multi-Agent
    "MultiAgentEnsemble",
    "EnsembleDecision",
    "get_multi_agent_ensemble",
    # Strategic
    "GPT4StrategicReasoner",
    "StrategicAnalysis",
    "get_strategic_reasoner",
    # Neural
    "NeuralMarketPredictor",
    "NeuralPrediction",
    "get_neural_predictor",
    # Alpha
    "EliteAlphaGenerator",
    "AlphaSignal",
    "get_alpha_generator",
    # Regime
    "TransformerRegimeDetector",
    "RegimeState",
    "get_regime_detector",
    # Risk
    "AdaptiveRiskManager",
    "RiskBudget",
    "get_adaptive_risk_manager",
    # Portfolio
    "DynamicPortfolioOptimizer",
    "PortfolioAllocation",
    "get_portfolio_optimizer",
    # Elite Brain
    "EliteAIBrain",
    "EliteSignal",
    "get_elite_brain",
    # Return Predictor
    "AdvancedReturnPredictor",
    "ReturnPrediction",
    "get_return_predictor",
]
