"""
INSTITUTIONAL ARCHITECTURE SPECIFICATION v3.0
==============================================
Top-1% Hedge Fund Grade Production Architecture

NON-NEGOTIABLE PRINCIPLES:
- Survival > Profit
- Governance > Speed
- Capital efficiency > Signals
- CVaR dominates Sharpe
- Models are assumed to decay
- Every decision must be auditable
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Tuple
from enum import Enum
import json


# =============================================================================
# PHASE 1: DATA GOVERNANCE - Symbol Classification
# =============================================================================

class AssetClass(Enum):
    STOCKS = "stocks"
    FX = "fx"
    CRYPTO = "crypto"
    COMMODITIES = "commodities"


def classify_symbol(symbol: str) -> AssetClass:
    """
    Classify symbol by asset class.
    PERSISTED PERMANENTLY in database.
    """
    if symbol.endswith("=X"):
        return AssetClass.FX
    if symbol.endswith("=F"):
        return AssetClass.COMMODITIES
    if "-USD" in symbol or "-USDT" in symbol:
        return AssetClass.CRYPTO
    return AssetClass.STOCKS


# =============================================================================
# PHASE 1: Provider Capability Matrix (IMMUTABLE)
# =============================================================================

PROVIDER_CAPABILITIES = {
    "alpaca": {
        "stocks": True,
        "fx": False,
        "crypto": True,
        "commodities": False,
        "max_history_days": 730,
        "requires_entitlement": True,
        "live_quotes": True,
        "execution": True
    },
    "yahoo": {
        "stocks": True,
        "fx": True,
        "crypto": True,
        "commodities": True,
        "max_history_days": 5000,
        "requires_entitlement": False,
        "live_quotes": False,
        "execution": False
    },
    "polygon": {
        "stocks": True,
        "fx": True,
        "crypto": True,
        "commodities": False,
        "max_history_days": 5000,
        "requires_entitlement": True,
        "live_quotes": True,
        "execution": False
    },
    "binance": {
        "stocks": False,
        "fx": False,
        "crypto": True,
        "commodities": False,
        "max_history_days": 1000,
        "requires_entitlement": False,
        "live_quotes": True,
        "execution": False
    },
    "fred": {
        "stocks": False,
        "fx": False,
        "crypto": False,
        "commodities": False,
        "max_history_days": None,
        "requires_entitlement": False,
        "macro_data": True,
        "execution": False
    }
}


# =============================================================================
# PHASE 1: Strict Provider Routing
# =============================================================================

def select_provider(
    symbol: str,
    history_days: int,
    purpose: str = "history",
    entitled_providers: Optional[List[str]] = None
) -> str:
    """
    Strict provider selection based on capability matrix.

    Rules:
    1. Check asset class capability
    2. Verify history requirement fits
    3. Check entitlement status
    4. Return NO_VALID_PROVIDER if no match

    NO_ERROR on NO_VALID_PROVIDER - just REJECT symbol.
    """
    import numpy as np

    if entitled_providers is None:
        entitled_providers = []

    asset = classify_symbol(symbol)
    provider_priority = ["yahoo", "polygon", "alpaca", "binance", "fred"]

    for provider in provider_priority:
        caps = PROVIDER_CAPABILITIES.get(provider, {})

        if not caps.get(asset.value, False) and not caps.get("macro_data", False):
            continue

        max_days = caps.get("max_history_days")
        if max_days is not None and history_days > max_days:
            continue

        if caps.get("requires_entitlement", False):
            if provider not in entitled_providers:
                continue

        if purpose == "execution" and not caps.get("execution", False):
            continue
        if purpose == "live_quotes" and not caps.get("live_quotes", False):
            continue
        if purpose == "history" and caps.get("execution", False):
            continue

        return provider

    return "NO_VALID_PROVIDER"


# =============================================================================
# PHASE 2: Data Separation (ABSOLUTE)
# =============================================================================

DATA_PURPOSE_PROVIDER_MAP = {
    "5y_history": ["yahoo", "polygon"],
    "fx_history": ["yahoo"],
    "commodities_history": ["yahoo"],
    "crypto_history": ["yahoo", "binance"],
    "live_quotes": ["alpaca"],
    "execution": ["alpaca"],
    "macro_data": ["fred"]
}


# =============================================================================
# PHASE 4: Capital Competition - Auction Engine Inputs
# =============================================================================

@dataclass
class CapitalAuctionInput:
    """Input to the capital auction engine."""
    symbol: str
    asset_class: AssetClass
    mu: float
    sigma: float
    sharpe_annual: float
    cvar_95: float
    marginal_cvar: float
    p_loss: float
    data_quality_score: float
    history_days: int
    provider_confidence: float
    adv_usd: float
    liquidity_cost_bps: float
    market_impact_bps: float
    correlation_risk: float
    sector_exposure: float
    holding_period_days: int
    time_decay_rate: float
    model_age_days: int
    rolling_forecast_error: float
    autocorr_flip_detected: bool
    strategy_id: str
    strategy_lifecycle_stage: str
    strategy_allocation_pct: float
    reason_codes: List[str] = field(default_factory=list)


@dataclass
class CapitalAuctionOutput:
    """Output from capital auction engine."""
    symbol: str
    allocated: bool
    weight: float = 0.0
    rank: int = 0
    decision: str = "REJECT"
    reason_codes: List[str] = field(default_factory=list)
    mu_contribution: float = 0.0
    cvar_contribution: float = 0.0
    liquidity_cost: float = 0.0
    correlation_penalty: float = 0.0
    data_quality_penalty: float = 0.0
    model_decay_penalty: float = 0.0
    vetoed: bool = False
    veto_reason: str = ""


# =============================================================================
# PHASE 5: CVaR-First Risk Dominance
# =============================================================================

@dataclass
class CVaRConfig:
    """CVaR configuration."""
    confidence_level: float = 0.95
    portfolio_limit: float = 0.06
    marginal_limit: float = 0.01
    scale_on_breach: bool = True
    min_scale_factor: float = 0.0


# =============================================================================
# PHASE 6: Model Decay Tracking
# =============================================================================

@dataclass
class ModelHealthMetrics:
    """Required for every model output."""
    mu: float = 0.0
    sigma: float = 0.0
    p_loss: float = 0.5
    cvar_95: float = 0.0
    confidence: float = 0.0
    model_disagreement_var: float = 0.0
    disagreement_penalty: float = 1.0
    mu_adjusted: float = 0.0
    model_age_days: int = 0
    rolling_forecast_error: float = 0.0
    autocorr_flip_detected: bool = False
    age_decay_factor: float = 1.0
    error_decay_factor: float = 1.0
    final_decay_factor: float = 1.0


def compute_model_disagreement_penalty(mus: List[float], beta: float = 0.5) -> float:
    """
    Compute disagreement penalty factor.
    mu_adj = mu * exp(-beta * Var(mu_models))
    """
    import numpy as np
    if len(mus) <= 1:
        return 1.0
    mu_var = float(np.var(mus))
    return np.exp(-beta * mu_var)


def compute_decay_factors(
    model_age_days: int,
    rolling_error: float,
    autocorr_flip: bool,
    max_age_days: int = 90,
    error_threshold: float = 0.02
) -> Tuple[float, float, float]:
    """Compute model decay factors."""
    import numpy as np
    if model_age_days <= max_age_days:
        age_decay = 1.0
    else:
        age_decay = max(0.0, 1.0 - (model_age_days - max_age_days) / max_age_days)

    if rolling_error <= error_threshold:
        error_decay = 1.0
    else:
        error_decay = np.exp(-(rolling_error - error_threshold) * 10)

    if autocorr_flip:
        error_decay *= 0.5

    final = age_decay * error_decay
    return age_decay, error_decay, final


# =============================================================================
# PHASE 7: Regime as Probability Flow
# =============================================================================

@dataclass
class RegimeProbabilityState:
    """Track regime as probability distribution."""
    p_normal_5d: float = 0.8
    p_volatile_5d: float = 0.15
    p_crisis_5d: float = 0.05
    current_belief_normal: float = 0.8
    current_belief_volatile: float = 0.15
    current_belief_crisis: float = 0.05
    spy_ma200_position: float = 0.0
    volatility_percentile: float = 0.5
    vix_level: float = 20.0
    correlation_spike: float = 0.0
    transition_matrix: Optional[List[List[float]]] = None

    def __post_init__(self):
        if self.transition_matrix is None:
            self.transition_matrix = [
                [0.90, 0.08, 0.02],
                [0.30, 0.60, 0.10],
                [0.40, 0.30, 0.30],
            ]

    def get_crisis_prob(self, days_ahead: int = 5) -> float:
        if days_ahead <= 0:
            return self.current_belief_crisis
        base = self.current_belief_crisis
        return min(1.0, base * (1 + 0.1 * days_ahead))

    def should_derexik(self, threshold: float = 0.15) -> bool:
        return self.get_crisis_prob(5) > threshold


# =============================================================================
# PHASE 8: Execution Realism
# =============================================================================

class ExecutionRegime(Enum):
    CALM = "calm"
    VOLATILE = "volatile"
    CRISIS = "crisis"


@dataclass
class ExecutionImpactEstimate:
    """Realistic execution impact estimation."""
    temporary_impact_bps: float = 0.0
    permanent_impact_bps: float = 0.0
    spread_cost_bps: float = 0.0
    total_cost_bps: float = 0.0
    liquidity_decay_rate: float = 0.0
    regime: ExecutionRegime = ExecutionRegime.CALM
    confidence: float = 1.0
    reason_codes: List[str] = field(default_factory=list)


def estimate_execution_impact(
    symbol: str,
    order_size_usd: float,
    adv_usd: float,
    volatility: float,
    spread_bps: float,
    regime: ExecutionRegime,
    market_depth_factor: float = 1.0
) -> ExecutionImpactEstimate:
    """Estimate execution impact with realistic models."""
    import numpy as np

    if adv_usd <= 0:
        return ExecutionImpactEstimate(reason_codes=["NO_LIQUIDITY_DATA"])

    size_adv = order_size_usd / adv_usd
    temp_impact = volatility * np.sqrt(size_adv) * 0.1 * 10000 * market_depth_factor
    perm_impact = temp_impact * 0.5
    spread_cost = spread_bps / 2
    total = temp_impact + perm_impact + spread_cost

    if regime == ExecutionRegime.VOLATILE:
        temp_impact *= 1.5
        perm_impact *= 1.5
        market_depth_factor *= 0.8
    elif regime == ExecutionRegime.CRISIS:
        temp_impact *= 3.0
        perm_impact *= 2.0
        market_depth_factor *= 0.5

    return ExecutionImpactEstimate(
        temporary_impact_bps=temp_impact,
        permanent_impact_bps=perm_impact,
        spread_cost_bps=spread_cost,
        total_cost_bps=total,
        liquidity_decay_rate=size_adv * 0.1,
        regime=regime,
        confidence=1.0 - min(size_adv * 0.5, 0.3),
        reason_codes=["IMPACT_ESTIMATED"]
    )


# =============================================================================
# PHASE 9: Governance & Audit (VETO POWER)
# =============================================================================

@dataclass
class GovernanceDecision:
    """Every trading decision MUST emit this structure."""
    decision: str
    reason_codes: List[str] = field(default_factory=list)
    mu: float = 0.0
    sigma: float = 0.0
    cvar: float = 0.0
    data_quality: float = 0.0
    model_confidence: float = 0.0
    vetoed: bool = False
    veto_reason: str = ""
    cycle_id: str = ""
    timestamp: str = ""
    symbol: str = ""
    position_size: float = 0.0
    expected_return: float = 0.0
    expected_risk: float = 0.0
    strategy_id: str = ""
    strategy_stage: str = ""
    cvar_limit_check: bool = True
    leverage_limit_check: bool = True
    drawdown_limit_check: bool = True
    correlation_limit_check: bool = True
    sector_limit_check: bool = True
    veto_triggers: Dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision,
            "reason_codes": self.reason_codes,
            "mu": self.mu,
            "sigma": self.sigma,
            "cvar": self.cvar,
            "data_quality": self.data_quality,
            "model_confidence": self.model_confidence,
            "vetoed": self.vetoed,
            "veto_reason": self.veto_reason,
            "cycle_id": self.cycle_id,
            "timestamp": self.timestamp,
            "symbol": self.symbol,
            "position_size": self.position_size,
            "expected_return": self.expected_return,
            "expected_risk": self.expected_risk,
            "strategy_id": self.strategy_id,
            "strategy_stage": self.strategy_stage,
            "risk_checks": {
                "cvar": self.cvar_limit_check,
                "leverage": self.leverage_limit_check,
                "drawdown": self.drawdown_limit_check,
                "correlation": self.correlation_limit_check,
                "sector": self.sector_limit_check
            },
            "veto_triggers": self.veto_triggers
        }


# =============================================================================
# PHASE 10: Strategy Lifecycle
# =============================================================================

class StrategyLifecycle(Enum):
    INCUBATING = "INCUBATING"
    SCALING = "SCALING"
    HARVESTING = "HARVESTING"
    DECOMMISSIONED = "DECOMMISSIONED"


@dataclass
class StrategyLifecycleState:
    """Track strategy lifecycle stage and capital allocation."""
    strategy_id: str
    stage: StrategyLifecycle = StrategyLifecycle.INCUBATING
    max_capital_pct: Dict[StrategyLifecycle, float] = field(default_factory=lambda: {
        StrategyLifecycle.INCUBATING: 0.02,
        StrategyLifecycle.SCALING: 0.05,
        StrategyLifecycle.HARVESTING: 0.10,
        StrategyLifecycle.DECOMMISSIONED: 0.0
    })
    total_return: float = 0.0
    sharpe_rolling: float = 0.0
    max_drawdown: float = 0.0
    profit_factor: float = 0.0
    signal_stale_hours: float = 0.0
    error_rate: float = 0.0
    last_profit_date: str = ""
    decay_rate: float = 0.95

    def get_current_capital_limit(self, portfolio_nav: float) -> float:
        return portfolio_nav * self.max_capital_pct[self.stage]

    def should_decommission(self) -> bool:
        if self.stage == StrategyLifecycle.DECOMMISSIONED:
            return True
        if self.sharpe_rolling < 0.3:
            return True
        if self.max_drawdown > 0.15:
            return True
        if self.profit_factor < 0.8:
            return True
        if self.error_rate > 0.05:
            return True
        return False


# =============================================================================
# EXECUTION ORDER (NON-NEGOTIABLE SEQUENCE)
# =============================================================================

EXECUTION_ORDER = [
    "1. Nightly ingestion",
    "2. DB verification",
    "3. Data governance checks",
    "4. Capital competition",
    "5. CVaR risk gates",
    "6. Execution simulation",
    "7. Live trading"
]

VIOLATION_ACTION = "STOP_SYSTEM"


# =============================================================================
# FINAL MANDATE
# =============================================================================

FINAL_MANDATE = """
Your goal is not profit.

Your goal is:

- Never blow up
- Never violate entitlements
- Never trade blind
- Never trust a model
- Never hide risk
- Never rely on luck

If done correctly, this system:

- Survives crises
- Scales safely
- Looks institutional
- Earns trust
- Competes with top hedge funds
"""

