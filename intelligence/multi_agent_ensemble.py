"""
Multi-Agent Trading Ensemble - 2026 Ultimate
=============================================

A collaborative multi-agent system where specialized AI agents
work together to make the BEST trading decisions.

Agents:
1. Alpha Hunter - Finds alpha opportunities
2. Risk Guardian - Protects capital
3. Execution Optimizer - Minimizes costs
4. Macro Strategist - Reads the big picture
5. Contrarian Thinker - Challenges consensus
6. Arbiter - Makes final decisions

This is CUTTING-EDGE 2026 AI technology.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class AgentType(Enum):
    ALPHA_HUNTER = "alpha_hunter"
    RISK_GUARDIAN = "risk_guardian"
    EXECUTION_OPT = "execution_optimizer"
    MACRO_STRATEGIST = "macro_strategist"
    CONTRARIAN = "contrarian_thinker"
    ARBITER = "arbiter"


@dataclass
class AgentVote:
    """Vote from a single agent."""
    agent: AgentType
    action: str  # BUY, SELL, HOLD
    size: float  # -1 to 1
    confidence: float
    reasoning: str


@dataclass
class EnsembleDecision:
    """Final decision from the multi-agent ensemble."""
    symbol: str
    final_action: str
    final_size: float
    confidence: float
    votes: List[AgentVote]
    consensus_level: float  # 0-1 agreement
    dissenting_views: List[str]
    timestamp: datetime = field(default_factory=datetime.utcnow)


class TradingAgent:
    """Base class for trading agents."""

    def __init__(self, agent_type: AgentType, weight: float = 1.0):
        self.type = agent_type
        self.weight = weight

    def vote(
        self,
        symbol: str,
        features: Dict[str, float],
        regime: str
    ) -> AgentVote:
        raise NotImplementedError


class AlphaHunterAgent(TradingAgent):
    """Agent focused on finding alpha opportunities."""

    def __init__(self):
        super().__init__(AgentType.ALPHA_HUNTER, weight=1.5)

    def vote(
        self,
        symbol: str,
        features: Dict[str, float],
        regime: str
    ) -> AgentVote:
        # Combine multiple alpha signals
        momentum = features.get("momentum_1m", 0)
        earnings = features.get("earnings_surprise", 0)
        sentiment = features.get("sentiment", 0)

        alpha_score = 0.4 * momentum + 0.35 * earnings + 0.25 * sentiment

        if alpha_score > 0.3:
            action = "BUY"
            size = min(1.0, alpha_score)
        elif alpha_score < -0.3:
            action = "SELL"
            size = max(-1.0, alpha_score)
        else:
            action = "HOLD"
            size = 0.0

        return AgentVote(
            agent=self.type,
            action=action,
            size=size,
            confidence=abs(alpha_score),
            reasoning=f"Alpha score: {alpha_score:.2f}"
        )


class RiskGuardianAgent(TradingAgent):
    """Agent focused on protecting capital."""

    def __init__(self):
        super().__init__(AgentType.RISK_GUARDIAN, weight=1.3)

    def vote(
        self,
        symbol: str,
        features: Dict[str, float],
        regime: str
    ) -> AgentVote:
        volatility = features.get("volatility", 0.02)
        drawdown = features.get("drawdown", 0)

        # Risk score (higher = more dangerous)
        risk_score = volatility * 20 + abs(drawdown) * 5

        if regime in ["CRISIS", "VOLATILE"]:
            risk_score *= 1.5

        # Convert to conservative action
        if risk_score > 0.8:
            action = "SELL"
            size = -0.5
            reasoning = f"HIGH RISK: {risk_score:.2f}"
        elif risk_score > 0.5:
            action = "HOLD"
            size = 0.0
            reasoning = f"Elevated risk: {risk_score:.2f}"
        else:
            action = "HOLD"
            size = 0.0
            reasoning = f"Risk acceptable: {risk_score:.2f}"

        return AgentVote(
            agent=self.type,
            action=action,
            size=size,
            confidence=0.8,
            reasoning=reasoning
        )


class MacroStrategistAgent(TradingAgent):
    """Agent focused on macro environment."""

    def __init__(self):
        super().__init__(AgentType.MACRO_STRATEGIST, weight=1.2)

    def vote(
        self,
        symbol: str,
        features: Dict[str, float],
        regime: str
    ) -> AgentVote:
        # Regime-based macro view
        regime_views = {
            "BULL": ("BUY", 0.6, "Bullish macro environment"),
            "BEAR": ("SELL", -0.4, "Bearish macro headwinds"),
            "VOLATILE": ("HOLD", 0.0, "Uncertain macro, stay cautious"),
            "SIDEWAYS": ("HOLD", 0.2, "Range-bound, select opportunities"),
            "CRISIS": ("SELL", -0.8, "RISK OFF - protect capital"),
            "RECOVERY": ("BUY", 0.7, "Recovery phase - accumulate")
        }

        action, size, reasoning = regime_views.get(
            regime, ("HOLD", 0.0, "Neutral macro")
        )

        return AgentVote(
            agent=self.type,
            action=action,
            size=size,
            confidence=0.7,
            reasoning=reasoning
        )


class ContrarianAgent(TradingAgent):
    """Agent that challenges consensus - finds crowded trades."""

    def __init__(self):
        super().__init__(AgentType.CONTRARIAN, weight=0.8)

    def vote(
        self,
        symbol: str,
        features: Dict[str, float],
        regime: str
    ) -> AgentVote:
        rsi = features.get("rsi", 50)
        sentiment = features.get("sentiment", 0)
        short_interest = features.get("short_interest", 0)

        # Look for extremes to fade
        if rsi > 80 and sentiment > 0.7:
            action = "SELL"
            size = -0.5
            reasoning = "Overbought + crowded long - fade it"
        elif rsi < 20 and sentiment < -0.7:
            action = "BUY"
            size = 0.5
            reasoning = "Oversold + extreme fear - contrarian buy"
        elif short_interest > 0.3:
            action = "BUY"
            size = 0.3
            reasoning = "High short interest - squeeze potential"
        else:
            action = "HOLD"
            size = 0.0
            reasoning = "No extreme positioning"

        return AgentVote(
            agent=self.type,
            action=action,
            size=size,
            confidence=0.5,
            reasoning=reasoning
        )


class ExecutionOptimizerAgent(TradingAgent):
    """Agent focused on execution quality."""

    def __init__(self):
        super().__init__(AgentType.EXECUTION_OPT, weight=0.6)

    def vote(
        self,
        symbol: str,
        features: Dict[str, float],
        regime: str
    ) -> AgentVote:
        spread = features.get("spread", 0.001)
        volume = features.get("volume_ratio", 1.0)

        # Execution feasibility
        if spread > 0.005 or volume < 0.3:
            action = "HOLD"
            size = 0.0
            reasoning = f"Poor liquidity: spread={spread:.3f}, vol={volume:.1f}"
        else:
            action = "HOLD"  # No directional view, just feasibility
            size = 0.0
            reasoning = "Execution conditions acceptable"

        return AgentVote(
            agent=self.type,
            action=action,
            size=size,
            confidence=0.9,
            reasoning=reasoning
        )


class MultiAgentEnsemble:
    """
    Multi-agent ensemble for collaborative trading decisions.
    """

    def __init__(self):
        self.agents = [
            AlphaHunterAgent(),
            RiskGuardianAgent(),
            MacroStrategistAgent(),
            ContrarianAgent(),
            ExecutionOptimizerAgent()
        ]
        logger.info(
            f"[MULTI_AGENT] Ensemble initialized with {len(self.agents)} agents"
        )

    def decide(
        self,
        symbol: str,
        features: Dict[str, float],
        regime: str
    ) -> EnsembleDecision:
        """
        Make collaborative decision with all agents.
        """
        votes = []

        # Collect votes from all agents
        for agent in self.agents:
            vote = agent.vote(symbol, features, regime)
            votes.append(vote)

        # Weighted voting
        weighted_size = 0.0
        total_weight = 0.0

        for vote, agent in zip(votes, self.agents):
            weighted_size += vote.size * agent.weight * vote.confidence
            total_weight += agent.weight * vote.confidence

        if total_weight > 0:
            final_size = weighted_size / total_weight
        else:
            final_size = 0.0

        final_size = np.clip(final_size, -1.0, 1.0)

        # Determine action from size
        if final_size > 0.2:
            final_action = "BUY"
        elif final_size < -0.2:
            final_action = "SELL"
        else:
            final_action = "HOLD"

        # Calculate consensus
        action_counts = {"BUY": 0, "SELL": 0, "HOLD": 0}
        for vote in votes:
            action_counts[vote.action] += 1

        max_votes = max(action_counts.values())
        consensus_level = max_votes / len(votes)

        # Find dissenters
        majority_action = max(action_counts, key=action_counts.get)
        dissenting = [
            v.reasoning for v in votes
            if v.action != majority_action
        ]

        # Confidence from consensus and individual confidences
        avg_conf = np.mean([v.confidence for v in votes])
        final_conf = avg_conf * consensus_level

        return EnsembleDecision(
            symbol=symbol,
            final_action=final_action,
            final_size=final_size,
            confidence=final_conf,
            votes=votes,
            consensus_level=consensus_level,
            dissenting_views=dissenting
        )


# Singleton
_ensemble = None


def get_multi_agent_ensemble() -> MultiAgentEnsemble:
    global _ensemble
    if _ensemble is None:
        _ensemble = MultiAgentEnsemble()
    return _ensemble
