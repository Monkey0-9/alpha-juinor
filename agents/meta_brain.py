"""
Meta-Brain Decision Engine.

Aggregates all agent outputs into final trading decisions using:
- Confidence-weighted ensemble aggregation
- Disagreement penalty (exp(-β · Var(μ_k)))
- Robust z-score calculation
- Fractional Kelly capital allocation
- Final decision rules: EXECUTE_BUY, EXECUTE_SELL, HOLD, REJECT
"""

import logging
import hashlib
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)

# Decision constants - Must match database constraints (EXECUTE, HOLD, REJECT, ERROR)
# We use a tuple of (action, side) to distinguish BUY vs SELL for EXECUTE
DECISION_EXECUTE = "EXECUTE"
DECISION_HOLD = "HOLD"
DECISION_REJECT = "REJECT"
DECISION_ERROR = "ERROR"

# Action constants for execution layer
ACTION_BUY = "BUY"
ACTION_SELL = "SELL"
ACTION_HOLD = "HOLD"

# Mapping for backwards compatibility
DECISION_BUY = "EXECUTE"  # Will be paired with ACTION_BUY
DECISION_SELL = "EXECUTE"  # Will be paired with ACTION_SELL

# Meta-Brain configuration
DEFAULT_BETA = 0.5  # Disagreement penalty strength
DEFAULT_GAMMA = 0.3  # Kelly fractional factor
RISK_FREE_RATE = 0.02  # Annual risk-free rate


@dataclass
class AgentContribution:
    """Individual agent contribution to the ensemble"""
    agent_name: str
    mu: float
    sigma: float
    confidence: float
    weight: float = 0.0
    metadata: Dict[str, Any] = None


@dataclass
class SymbolDecision:
    """Final decision for a symbol"""
    symbol: str
    cycle_id: str
    final_decision: str
    reason_codes: List[str]

    # Aggregated metrics
    mu_hat: float = 0.0
    sigma_hat: float = 0.0
    conviction: float = 0.0
    z_score: float = 0.0

    # Position sizing
    position_size: float = 0.0
    kelly_fraction: float = 0.0

    # Execution parameters
    stop_loss: Optional[float] = None
    trailing_params: Dict[str, float] = None

    # Full provenance
    agent_results: Dict[str, AgentContribution] = None
    risk_checks: Dict[str, Any] = None
    provider_confidence: float = 0.5
    data_quality_score: float = 1.0

    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'cycle_id': self.cycle_id,
            'final_decision': self.final_decision,
            'reason_codes': self.reason_codes,
            'mu_hat': self.mu_hat,
            'sigma_hat': self.sigma_hat,
            'conviction': self.conviction,
            'z_score': self.z_score,
            'position_size': self.position_size,
            'kelly_fraction': self.kelly_fraction,
            'stop_loss': self.stop_loss,
            'trailing_params': self.trailing_params,
            'agent_results': {
                name: {
                    'agent_name': contrib.agent_name,
                    'mu': contrib.mu,
                    'sigma': contrib.sigma,
                    'confidence': contrib.confidence,
                    'weight': contrib.weight
                }
                for name, contrib in (self.agent_results or {}).items()
            },
            'risk_checks': self.risk_checks,
            'provider_confidence': self.provider_confidence,
            'data_quality_score': self.data_quality_score,
            'timestamp': self.timestamp,
            'metadata': self.metadata
        }


class MetaBrain:
    """
    Meta-Brain: Portfolio decision engine that aggregates all agent outputs.

    Core algorithm:
    1. Collect outputs from all agents (Momentum, MeanReversion, Vol, Sentiment, etc.)
    2. Compute confidence-weighted ensemble μ̂ᵢ = Σ_k w_k · αᵢ,k
    3. Apply disagreement penalty: reduce μ̂ᵢ by exp(-β · Var(μ_k))
    4. Compute robust z-score: zᵢ = (μ̂ᵢ − median(μ̂)) / MAD(μ̂)
    5. Compute score Sᵢ = μ̂ᵢ / σ̂ᵢ
    6. Apply fractional Kelly sizing: fᵢ = γ·μ̂ᵢ/σ̂ᵢ²
    7. Apply opportunity-cost check
    8. Apply risk rules and final decision
    """

    def __init__(
        self,
        beta: float = DEFAULT_BETA,
        gamma: float = DEFAULT_GAMMA,
        risk_free_rate: float = RISK_FREE_RATE,
        min_confidence_threshold: float = 0.3,
        min_data_quality_threshold: float = 0.6,
        max_position_size: float = 0.10,
        max_total_leverage: float = 1.0,
        allow_short: bool = False,
        cvar_limit: float = 0.05,  # 5% Portfolio Tail Risk Limit
        risk_lambda: float = 0.2    # Risk aversion for CVaR penalty
    ):
        """
        Initialize Meta-Brain.

        Args:
            beta: Disagreement penalty strength
            gamma: Kelly fractional factor
            risk_free_rate: Annual risk-free rate for Sharpe calc
            min_confidence_threshold: Minimum agent confidence to consider
            min_data_quality_threshold: Minimum data quality score
            max_position_size: Maximum position size (fraction of portfolio)
            max_total_leverage: Maximum portfolio leverage
            allow_short: Allow short positions
            cvar_limit: Portfolio-level CVaR limit
            risk_lambda: λ for CVaR optimization
        """
        self.beta = beta
        self.gamma = gamma
        self.risk_free_rate = risk_free_rate
        self.min_confidence_threshold = min_confidence_threshold
        self.min_data_quality_threshold = min_data_quality_threshold
        self.max_position_size = max_position_size
        self.max_total_leverage = max_total_leverage
        self.allow_short = allow_short
        self.cvar_limit = cvar_limit
        self.risk_lambda = risk_lambda

        # Initialize Institutional Decision Agent for enhanced proposals
        try:
            from agents.institutional_decision_agent import InstitutionalDecisionAgent
            self.institutional_agent = InstitutionalDecisionAgent(
                cvar_limit=cvar_limit,
                per_symbol_cap=max_position_size,
                strategy_max_pct=max_position_size * 2
            )
            logger.info("[META_BRAIN] Institutional Decision Agent initialized")
        except ImportError:
            self.institutional_agent = None
            logger.warning("[META_BRAIN] Institutional Decision Agent not available")

    def make_decisions(
        self,
        cycle_id: str,
        symbol_agent_outputs: Dict[str, List[Dict[str, Any]]],
        symbol_features: Dict[str, Dict[str, Any]],
        symbol_positions: Dict[str, float],
        portfolio_state: Dict[str, Any],
        risk_state: Dict[str, Any]
    ) -> Dict[str, SymbolDecision]:
        """
        Make decisions for all symbols.

        Args:
            cycle_id: Unique cycle identifier
            symbol_agent_outputs: Dict[symbol] -> List of agent outputs
            symbol_features: Dict[symbol] -> Feature dict
            symbol_positions: Dict[symbol] -> Current position size
            portfolio_state: Current portfolio state (NAV, cash, etc.)
            risk_state: Current risk state (regime, CVaR, etc.)

        Returns:
            Dict[symbol] -> SymbolDecision
        """
        decisions = {}

        for symbol, agent_outputs in symbol_agent_outputs.items():
            try:
                decision = self._make_symbol_decision(
                    cycle_id=cycle_id,
                    symbol=symbol,
                    agent_outputs=agent_outputs,
                    features=symbol_features.get(symbol, {}),
                    current_position=symbol_positions.get(symbol, 0.0),
                    portfolio_state=portfolio_state,
                    risk_state=risk_state
                )
                decisions[symbol] = decision
            except Exception as e:
                logger.error(f"Decision failed for {symbol}: {e}")
                decisions[symbol] = self._create_reject_decision(
                    cycle_id, symbol, ["decision_error", str(e)]
                )

        return decisions

    def _make_symbol_decision(
        self,
        cycle_id: str,
        symbol: str,
        agent_outputs: List[Dict[str, Any]],
        features: Dict[str, Any],
        current_position: float,
        portfolio_state: Dict[str, Any],
        risk_state: Dict[str, Any]
    ) -> SymbolDecision:
        """Make decision for a single symbol"""

        reason_codes = []
        agent_contributions = {}

        # 1. Filter and process agent outputs
        valid_outputs = []
        for output in agent_outputs:
            # Check minimum confidence
            confidence = output.get('confidence', 0.0)
            if confidence < self.min_confidence_threshold:
                reason_codes.append(f"low_confidence_{output.get('agent_name', 'unknown')}")
                continue

            # Check for valid mu/sigma
            mu = output.get('mu', 0.0)
            sigma = output.get('sigma', 0.1)  # Default to 0.1 if missing

            valid_outputs.append({
                'agent_name': output.get('agent_name', 'unknown'),
                'mu': mu,
                'sigma': max(sigma, 0.01),  # Minimum sigma
                'confidence': confidence,
                'metadata': output.get('metadata', {})
            })

        if not valid_outputs:
            reason_codes.append("no_valid_agents")
            return self._create_reject_decision(
                cycle_id, symbol, reason_codes + ["insufficient_agent_coverage"]
            )

        # 2. Filter by data quality (from features)
        data_quality = features.get('_provenance', {}).get('data_quality', 1.0)
        if data_quality < self.min_data_quality_threshold:
            reason_codes.append("low_data_quality")
            return self._create_reject_decision(
                cycle_id, symbol, reason_codes
            )

        # 3. Compute confidence-weighted ensemble
        total_weight = sum(o['confidence'] for o in valid_outputs)
        if total_weight == 0:
            total_weight = len(valid_outputs)
            for o in valid_outputs:
                o['confidence'] = 1.0 / len(valid_outputs)

        # Compute weighted average mu
        mu_hat = sum(o['mu'] * o['confidence'] for o in valid_outputs) / total_weight

        # Compute weighted average sigma (uncertainty)
        sigma_hat = sum(o['sigma'] * o['confidence'] for o in valid_outputs) / total_weight

        # 4. Apply disagreement penalty
        mus = [o['mu'] for o in valid_outputs]
        mu_var = np.var(mus)

        # Disagreement penalty factor
        disagreement_penalty = np.exp(-self.beta * mu_var)
        mu_hat_adjusted = mu_hat * disagreement_penalty

        if mu_var > 0.1:
            reason_codes.append("high_disagreement")

        # Store agent contributions
        for o in valid_outputs:
            agent_contributions[o['agent_name']] = AgentContribution(
                agent_name=o['agent_name'],
                mu=o['mu'],
                sigma=o['sigma'],
                confidence=o['confidence'],
                weight=o['confidence'] / total_weight,
                metadata=o['metadata']
            )

        # 5. Compute robust z-score
        # We'll compute this across all symbols in the batch for cross-sectional context
        # For single symbol, use deviation from mean
        z_score = 0.0  # Will be updated in batch context

        # 6. Compute conviction score (signal strength)
        if sigma_hat > 0:
            conviction = mu_hat_adjusted / sigma_hat
        else:
            conviction = 0.0

        # 7. Apply risk regime adjustment
        is_risk_on = risk_state.get('is_risk_on', True)
        regime_scalar = risk_state.get('regime_scalar', 1.0)

        if not is_risk_on:
            mu_hat_adjusted *= 0.7  # Risk-off haircut
            reason_codes.append("risk_off_regime")

        if regime_scalar < 1.0:
            mu_hat_adjusted *= regime_scalar
            reason_codes.append(f"regime_adjustment_{regime_scalar:.2f}")

        # 8. Compute CVaR for the symbol (Simplified: assume normal distribution for marginal)
        # CVaR_alpha = mu - sigma * (phi(Phi^-1(alpha)) / alpha)
        # For a 95% CVaR (alpha=0.05), ratio is ~2.06
        symbol_cvar = abs(mu_hat_adjusted) + 2.06 * sigma_hat

        # 9. Compute Kelly sizing
        kelly_fraction = self._compute_kelly(mu_hat_adjusted, sigma_hat)

        # 10. Apply Data Confidence Constraints (Institutional Rule)
        # Reduce sizing if data_quality < threshold
        if data_quality < 0.8:
            reduction_factor = (data_quality - self.min_data_quality_threshold) / (0.8 - self.min_data_quality_threshold)
            kelly_fraction *= max(0.0, reduction_factor)
            reason_codes.append(f"data_confidence_haircut_{reduction_factor:.2f}")

        # Apply maximum position limit
        position_size = min(kelly_fraction, self.max_position_size)

        # Apply short constraint
        if not self.allow_short and position_size < 0:
            position_size = 0
            reason_codes.append("short_not_allowed")

        # 11. Apply existing position adjustment
        if current_position > 0:
            # If we already have position, scale new recommendation
            if mu_hat_adjusted > 0:
                # Positive signal, potentially add
                additional = max(0, position_size - current_position)
                position_size = current_position + additional
            else:
                # Negative signal, reduce or close
                position_size = max(0, current_position * 0.5)
                reason_codes.append("position_reduction")
        elif position_size < 0:
            # Short signal but no existing position
            position_size = 0
            reason_codes.append("short_not_allowed")

        # 12. Final decision rules (CVaR-first)
        final_decision = self._apply_decision_rules(
            mu_hat=mu_hat_adjusted,
            conviction=conviction,
            position_size=position_size,
            current_position=current_position,
            risk_state=risk_state,
            reason_codes=reason_codes,
            symbol_cvar=symbol_cvar
        )

        # 11. Generate execution parameters
        stop_loss = None
        trailing_params = None
        action = ACTION_HOLD  # Default action

        # Determine action and generate stop loss based on position_size sign
        if position_size > 0.01:
            action = ACTION_BUY
        elif position_size < -0.01 and self.allow_short:
            action = ACTION_SELL
        elif current_position > 0 and position_size < current_position * 0.9:
            action = ACTION_SELL

        if action in [ACTION_BUY, ACTION_SELL]:
            # Generate stop loss based on volatility
            atr_pct = features.get('atr_20pct', 0.02)
            if atr_pct > 0:
                stop_loss = atr_pct * 1.5 if action == ACTION_BUY else atr_pct * 1.5

            trailing_params = {
                'trailing_stop_pct': atr_pct * 2,
                'trailing_activation_pct': atr_pct
            }

        return SymbolDecision(
            symbol=symbol,
            cycle_id=cycle_id,
            final_decision=final_decision,
            reason_codes=reason_codes,
            mu_hat=mu_hat_adjusted,
            sigma_hat=sigma_hat,
            conviction=conviction,
            z_score=z_score,
            position_size=position_size,
            kelly_fraction=kelly_fraction,
            stop_loss=stop_loss,
            trailing_params=trailing_params,
            agent_results=agent_contributions,
            risk_checks=risk_state.get('violations', []),
            provider_confidence=features.get('provider_confidence', 0.5),
            data_quality_score=data_quality,
            metadata={
                'feature_hash': features.get('_provenance', {}).get('raw_hash', ''),
                'n_agents': len(valid_outputs),
                'disagreement_var': mu_var,
                'action': action  # Track BUY/SELL/HOLD action separately
            }
        )

    def _compute_kelly(self, mu: float, sigma: float) -> float:
        """
        Compute fractional Kelly fraction.

        f* = γ * μ / σ²

        Where:
        - μ: Expected return (annualized)
        - σ: Volatility (annualized)
        - γ: Fractional Kelly factor
        """
        if sigma <= 0:
            return 0.0

        # Kelly fraction
        kelly = (mu - self.risk_free_rate) / (sigma ** 2)

        # Clip to reasonable bounds
        kelly = np.clip(kelly, -2.0, 2.0)

        # Apply fractional Kelly
        kelly_fraction = self.gamma * kelly

        return kelly_fraction

    def _apply_decision_rules(
        self,
        mu_hat: float,
        conviction: float,
        position_size: float,
        current_position: float,
        risk_state: Dict[str, Any],
        reason_codes: List[str],
        symbol_cvar: float = 0.0
    ) -> str:
        """
        Apply final decision rules (CVaR-First).

        Rules:
            1. If risk_override in risk_state -> REJECT
            2. CVaR-First: If marginal CVaR increases portfolio risk above limit -> REJECT
            3. If risk rules/CVaR exceed thresholds -> REJECT
            4. If position exists and model recommends reduction -> EXECUTE_SELL
            5. If model recommends increase and risk checks ok -> EXECUTE_BUY
            6. Otherwise -> HOLD
        """
        # Rule 1: Risk override
        if risk_state.get('risk_override', False):
            reason_codes.append("risk_override")
            return DECISION_REJECT

        # Rule 2: CVaR-First Limit Check
        portfolio_cvar = risk_state.get('portfolio_cvar', 0.0)
        # Simplified marginal impact check
        if (portfolio_cvar + symbol_cvar * position_size) > self.cvar_limit:
            reason_codes.append(f"cvar_limit_breach_{self.cvar_limit}")
            return DECISION_REJECT

        # Rule 3: CVaR/risk threshold breach in risk_state
        if risk_state.get('cvar_breach', False):
            reason_codes.append("cvar_breach")
            return DECISION_REJECT

        # Rule 3: Leverage limit
        portfolio_leverage = risk_state.get('portfolio_leverage', 0.0)
        if portfolio_leverage > self.max_total_leverage:
            reason_codes.append("leverage_limit")
            return DECISION_REJECT

        # Rule 4: Position reduction recommendation (SELL signal)
        if current_position > 0 and position_size < current_position * 0.9:
            reason_codes.append("position_reduction_recommended")
            # Return EXECUTE with SELL action - metadata will track the side
            return DECISION_EXECUTE

        # Rule 5: Position increase recommendation (BUY signal)
        if position_size > 0.01:
            # Additional checks for buy
            if mu_hat > 0 and conviction > 0.0:
                return DECISION_EXECUTE
            elif mu_hat < 0 and not self.allow_short:
                return DECISION_HOLD
            elif mu_hat < 0 and self.allow_short:
                return DECISION_EXECUTE
            else:
                return DECISION_EXECUTE if position_size > 0.01 else DECISION_HOLD

        # Default: HOLD
        return DECISION_HOLD

    def _create_reject_decision(
        self,
        cycle_id: str,
        symbol: str,
        reason_codes: List[str]
    ) -> SymbolDecision:
        """Create a REJECT decision with reason codes"""
        return SymbolDecision(
            symbol=symbol,
            cycle_id=cycle_id,
            final_decision=DECISION_REJECT,
            reason_codes=reason_codes,
            mu_hat=0.0,
            sigma_hat=0.1,
            conviction=0.0,
            position_size=0.0,
            kelly_fraction=0.0,
            agent_results={},
            risk_checks=[],
            data_quality_score=0.0,
            metadata={'reject': True, 'reason_codes': reason_codes}
        )

    def compute_batch_z_scores(
        self,
        decisions: Dict[str, SymbolDecision]
    ) -> Dict[str, float]:
        """
        Compute robust z-scores across all decisions in the batch.

        Uses Median Absolute Deviation (MAD) for robustness.
        """
        if not decisions:
            return {}

        mu_hats = np.array([d.mu_hat for d in decisions.values()])

        # Median
        median = np.median(mu_hats)

        # MAD
        mad = np.median(np.abs(mu_hats - median))

        # Avoid division by zero
        if mad < 1e-6:
            return {symbol: 0.0 for symbol in decisions}

        # Z-scores
        z_scores = {}
        for symbol, decision in decisions.items():
            z = (decision.mu_hat - median) / (1.4826 * mad)  # 1.4826 makes MAD consistent with std for normal
            z_scores[symbol] = float(z)
            decision.z_score = z

        return z_scores


class EnsembleAggregator:
    """
    Helper class for running ensemble aggregation across multiple cycles.
    """

    def __init__(self, meta_brain: MetaBrain):
        self.meta_brain = meta_brain
        self.cycle_history = []

    def add_cycle(self, cycle_id: str, decisions: Dict[str, SymbolDecision]) -> Dict[str, Any]:
        """Add a cycle's decisions to history and compute metrics"""
        metrics = {
            'cycle_id': cycle_id,
            'n_decisions': len(decisions),
            'decision_counts': {
                'buy': sum(1 for d in decisions.values() if d.final_decision == DECISION_BUY),
                'sell': sum(1 for d in decisions.values() if d.final_decision == DECISION_SELL),
                'hold': sum(1 for d in decisions.values() if d.final_decision == DECISION_HOLD),
                'reject': sum(1 for d in decisions.values() if d.final_decision == DECISION_REJECT)
            },
            'avg_conviction': np.mean([d.conviction for d in decisions.values()]),
            'avg_kelly': np.mean([d.kelly_fraction for d in decisions.values()]),
            'high_conviction_buys': [
                d.symbol for d in decisions.values()
                if d.final_decision == DECISION_BUY and d.conviction > 1.0
            ]
        }

        self.cycle_history.append(metrics)
        return metrics

    def get_ensemble_stats(self) -> Dict[str, Any]:
        """Get ensemble statistics across cycles"""
        if not self.cycle_history:
            return {}

        return {
            'n_cycles': len(self.cycle_history),
            'avg_buy_rate': np.mean([m['decision_counts']['buy'] for m in self.cycle_history]),
            'avg_reject_rate': np.mean([m['decision_counts']['reject'] for m in self.cycle_history]),
            'avg_conviction': np.mean([m['avg_conviction'] for m in self.cycle_history]),
            'high_confidence_signals': [
                signal
                for cycle in self.cycle_history
                for signal in cycle['high_conviction_buys']
            ]
        }

