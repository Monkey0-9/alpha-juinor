#!/usr/bin/env python3
"""
agents/institutional_decision_agent.py

Institutional Trading Decision Agent (Read-Only)

Produces BUY/SELL/HOLD/REJECT proposals with:
- Ensemble scoring (mean-reversion + momentum + volatility + liquidity)
- Position sizing (volatility-target + conservative Kelly)
- Risk gates (CVaR, regime, entanglement, data confidence, execution cost)
- Full audit trail with deterministic JSON output

ABSOLUTE NON-NEGOTIABLE RULES:
1. READ-ONLY: No broker calls, no order execution
2. NO GUARANTEES: Probability-weighted expectations only
3. DECISION BOUNDARY: Proposals only, final sizing by PM Brain
4. RISK FIRST: Violations -> REJECT
5. DETERMINISTIC: run_id + seed -> identical output
6. EXPLAINABILITY: Step-by-step rationale required
"""

import hashlib
import json
import logging
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum

import numpy as np

logger = logging.getLogger("INSTITUTIONAL_DECISION_AGENT")

# =============================================================================
# CONSTANTS
# =============================================================================

CONTRACT_VERSION = "1.0"
AGENT_ID = "agent_institutional_v1"

# Decision types
class Decision(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    REJECT = "REJECT"

# Signal types
class PrimarySignal(str, Enum):
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    EVENT = "event"

# Entry types
class EntryType(str, Enum):
    LIMIT = "LIMIT"
    TWAP = "TWAP"
    PASSIVE = "PASSIVE"

# =============================================================================
# DATA CLASSES - OUTPUT SCHEMA
# =============================================================================

@dataclass
class PriceLimits:
    min: float
    max: float

@dataclass
class EntryZone:
    low: float
    high: float
    type: str  # LIMIT|TWAP|PASSIVE

@dataclass
class TakeProfitTier:
    pct: float
    qty_frac: float

@dataclass
class ExitLogic:
    stop_loss_price: float
    stop_loss_pct: float
    take_profit_tiers: List[Dict[str, float]]
    trailing_pct: float
    time_in_trade_limit_minutes: int

@dataclass
class RiskChecks:
    cvar_ok: bool
    entanglement_ok: bool
    data_confidence_ok: bool
    execution_cost_ok: bool
    regime_ok: bool = True

@dataclass
class Explanation:
    why_price_is_mispriced: str
    indicators_used: Dict[str, float]
    expected_edge_bps: float
    expected_return_pct: float
    model_versions: Dict[str, str]
    top_drivers: List[str] = field(default_factory=list)
    risk_contribution_pct: float = 0.0
    sensitivity: Dict[str, float] = field(default_factory=dict)

@dataclass
class InstitutionalProposal:
    """
    Strict JSON output schema per user specification.
    All fields must be filled.
    """
    run_id: str
    seed: int
    timestamp: str
    agent_id: str
    decision: str  # BUY|SELL|HOLD|REJECT
    confidence: float
    ensemble_score: float
    primary_signal: str
    suggested_notional_pct: float
    suggested_qty: int
    price_limits: Dict[str, float]
    entry_zone: Dict[str, Any]
    exit_logic: Dict[str, Any]
    risk_checks: Dict[str, bool]
    explain: Dict[str, Any]
    warnings: List[str]
    contract_version: str
    schema_hash: str
    signature: str

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(asdict(self), separators=(',', ':'))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

# =============================================================================
# INPUT VALIDATION
# =============================================================================

REQUIRED_INPUT_FIELDS = [
    "run_id", "seed", "timestamp", "symbol", "price", "nav_usd",
    "market", "features", "position_state", "execution", "risk"
]

REQUIRED_FEATURE_FIELDS = [
    "rsi_14", "rsi_3", "ema_9", "ema_21", "boll_z", "atr_pct", "macd_hist"
]

def validate_input(input_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate input data against required schema.

    Returns:
        Tuple of (is_valid, list of errors)
    """
    errors = []

    # Check required top-level fields
    for field in REQUIRED_INPUT_FIELDS:
        if field not in input_data:
            errors.append(f"Missing required field: {field}")

    # Check features
    features = input_data.get("features", {})
    for field in REQUIRED_FEATURE_FIELDS:
        if field not in features:
            errors.append(f"Missing required feature: {field}")

    # Validate ranges
    price = input_data.get("price", 0)
    if price <= 0:
        errors.append(f"Invalid price: {price}")

    nav = input_data.get("nav_usd", 0)
    if nav <= 0:
        errors.append(f"Invalid nav_usd: {nav}")

    return len(errors) == 0, errors

def compute_input_hash(input_data: Dict[str, Any]) -> str:
    """Compute deterministic hash of input data."""
    # Sort keys for determinism
    serialized = json.dumps(input_data, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]

# =============================================================================
# ENSEMBLE SCORING
# =============================================================================

def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value to range."""
    return max(min_val, min(max_val, value))

def compute_mean_reversion_component(boll_z: float, rsi_3: float) -> float:
    """
    Compute mean reversion signal component.

    Formula: clamp((boll_z * -0.3) + ((30 - rsi_3)/30 * 0.5), -1, 1)

    High score when:
    - boll_z is negative (price below lower band)
    - rsi_3 is low (oversold)
    """
    boll_signal = boll_z * -0.3  # Invert: low boll_z -> positive signal
    rsi_signal = ((30 - rsi_3) / 30) * 0.5  # Oversold -> positive signal
    return clamp(boll_signal + rsi_signal, -1.0, 1.0)

def compute_momentum_component(ema_9: float, ema_21: float, macd_hist: float) -> float:
    """
    Compute momentum signal component.

    Formula: clamp((ema_gap_pct * 0.6) + (macd_hist * 0.4), -1, 1)

    High score when:
    - EMA9 > EMA21 (uptrend)
    - MACD histogram positive
    """
    if ema_21 <= 0:
        return 0.0

    ema_gap_pct = (ema_9 / ema_21) - 1.0

    # Scale ema_gap to reasonable range (e.g., 5% gap = +1)
    ema_signal = ema_gap_pct * 20.0 * 0.6  # 5% gap = 0.6
    macd_signal = macd_hist * 0.4

    return clamp(ema_signal + macd_signal, -1.0, 1.0)

def compute_liquidity_component(
    adv_usd: float,
    volume_z: float = 0.0,
    adv_threshold: float = 5_000_000
) -> float:
    """
    Compute liquidity component.

    +0.3 if ADV large & volume_z > 1
    -0.2 otherwise
    """
    adv_large = adv_usd >= adv_threshold
    volume_high = volume_z > 1.0

    if adv_large and volume_high:
        return 0.3
    elif adv_large:
        return 0.1
    else:
        return -0.2

def compute_volatility_component(atr_pct: float, threshold: float = 0.03) -> float:
    """
    Compute volatility component.

    -0.5 if atr_pct > threshold (too risky)
    +0.1 otherwise
    """
    if atr_pct > threshold:
        return -0.5
    else:
        return 0.1

def compute_ensemble_score(
    mean_rev: float,
    momentum: float,
    liquidity: float,
    volatility: float,
    weights: Tuple[float, float, float, float] = (0.35, 0.30, 0.15, 0.20)
) -> float:
    """
    Compute final ensemble score as weighted sum.

    Default weights:
    - Mean reversion: 35%
    - Momentum: 30%
    - Liquidity: 15%
    - Volatility: 20%
    """
    components = [mean_rev, momentum, liquidity, volatility]
    weighted_sum = sum(c * w for c, w in zip(components, weights))

    # Normalize to [-1, 1]
    return clamp(weighted_sum, -1.0, 1.0)

def determine_primary_signal(
    mean_rev: float,
    momentum: float,
    volatility: float
) -> PrimarySignal:
    """Determine the primary signal driver."""
    scores = {
        PrimarySignal.MEAN_REVERSION: abs(mean_rev),
        PrimarySignal.MOMENTUM: abs(momentum),
        PrimarySignal.VOLATILITY: abs(volatility)
    }
    return max(scores, key=scores.get)

# =============================================================================
# POSITION SIZING
# =============================================================================

def compute_volatility_target_size(
    nav_usd: float,
    realized_vol_annual: float,
    strategy_target_vol: float = 0.15
) -> float:
    """
    Compute position size based on volatility targeting.

    target_vol_pct = strategy_target_vol / realized_vol_annual
    notional_vol_target = NAV * target_vol_pct

    Returns: suggested notional as fraction of NAV
    """
    if realized_vol_annual <= 0:
        return 0.0

    target_vol_pct = strategy_target_vol / realized_vol_annual
    return clamp(target_vol_pct, 0.0, 1.0)

def compute_kelly_fraction(
    expected_edge_bps: float,
    estimated_variance: float,
    max_kelly: float = 0.25,
    kelly_fraction_factor: float = 0.5
) -> float:
    """
    Compute conservative Kelly fraction.

    edge = expected_edge_bps / 10000
    kelly_naive = edge / estimated_variance
    kelly_fraction = min(0.25, 0.5 * kelly_naive)

    Args:
        expected_edge_bps: Expected edge in basis points
        estimated_variance: Estimated return variance (sigma^2)
        max_kelly: Maximum Kelly fraction (default 0.25 = quarter Kelly)
        kelly_fraction_factor: Fraction of naive Kelly to use (default 0.5)

    Returns:
        Conservative Kelly fraction
    """
    if estimated_variance <= 0:
        return 0.0

    edge = expected_edge_bps / 10000.0

    # Shrink edge for conservatism
    edge_shrunk = edge * 0.5

    # Inflate variance for conservatism
    variance_inflated = estimated_variance * 1.5

    kelly_naive = edge_shrunk / variance_inflated

    return min(max_kelly, kelly_fraction_factor * kelly_naive)

def compute_final_position_size(
    kelly_fraction: float,
    vol_target_size: float,
    per_symbol_cap: float = 0.02,
    strategy_max_pct: float = 0.05
) -> float:
    """
    Compute final position size with all caps applied.

    final_pct = min(per_symbol_cap, kelly_fraction, vol_target_size)
    """
    return min(per_symbol_cap, strategy_max_pct, kelly_fraction, vol_target_size)

# =============================================================================
# RISK GATES
# =============================================================================

def check_regime_gate(
    regime: str,
    decision: Decision
) -> Tuple[bool, str]:
    """
    Regime gate: do not buy in RISK_OFF unless hedged.
    """
    if decision == Decision.BUY and regime in ["RISK_OFF", "BEAR", "CRISIS"]:
        return False, f"regime_block:{regime}"
    return True, ""

def check_cvar_gate(
    portfolio_cvar: float,
    symbol_cvar_contribution: float,
    position_size: float,
    cvar_limit: float = 0.05
) -> Tuple[bool, str]:
    """
    CVaR gate: marginal CVaR contribution must not exceed limit.
    """
    marginal_cvar = portfolio_cvar + (symbol_cvar_contribution * position_size)

    if marginal_cvar > cvar_limit:
        return False, f"cvar_breach:{marginal_cvar:.4f}>{cvar_limit}"
    return True, ""

def check_entanglement_gate(
    global_entanglement_score: float,
    threshold: float = 0.7
) -> Tuple[bool, float]:
    """
    Entanglement gate: if high correlation, reduce sizing.

    Returns:
        Tuple of (ok, sizing_multiplier)
    """
    if global_entanglement_score > threshold:
        # Scale down sizing proportionally
        multiplier = max(0.3, 1.0 - (global_entanglement_score - threshold))
        return True, multiplier
    return True, 1.0

def check_data_confidence_gate(
    data_confidence: float,
    threshold: float = 0.6
) -> Tuple[bool, str]:
    """
    Data confidence gate: reject if data quality too low.
    """
    if data_confidence < threshold:
        return False, f"low_data_confidence:{data_confidence:.2f}<{threshold}"
    return True, ""

def check_execution_gate(
    slippage_bps: float,
    spread_bps: float,
    expected_edge_bps: float,
    buffer_bps: float = 5.0
) -> Tuple[bool, str]:
    """
    Execution gate: costs must be less than edge minus buffer.
    """
    total_cost = slippage_bps + spread_bps
    net_edge = expected_edge_bps - total_cost - buffer_bps

    if net_edge <= 0:
        return False, f"execution_cost_exceeds_edge:{total_cost}bps+{buffer_bps}bps>{expected_edge_bps}bps"
    return True, ""

# =============================================================================
# ENTRY/EXIT LOGIC
# =============================================================================

def compute_entry_zone(
    price: float,
    atr_pct: float,
    entry_type: str = "LIMIT"
) -> Dict[str, Any]:
    """
    Compute entry price zone.

    For LIMIT orders, set a range around current price.
    """
    return {
        "low": round(price * (1 - atr_pct * 0.5), 2),
        "high": round(price * (1 + atr_pct * 0.3), 2),
        "type": entry_type
    }

def compute_stop_loss(
    price: float,
    atr_pct: float,
    n_atr: float = 1.5,
    min_stop_pct: float = 0.005
) -> Tuple[float, float]:
    """
    Compute stop loss price and percentage.

    stop = price * (1 - n_atr * atr_pct)
    Minimum stop: 0.5%
    """
    stop_pct = max(n_atr * atr_pct, min_stop_pct)
    stop_price = round(price * (1 - stop_pct), 2)
    return stop_price, round(stop_pct, 4)

def compute_take_profit_tiers(
    expected_return_pct: float
) -> List[Dict[str, float]]:
    """
    Compute take profit tiers.

    Default: 50% at +3%, 50% at +8% (or scaled by expected return)
    """
    tier1_pct = max(0.03, expected_return_pct * 0.5)
    tier2_pct = max(0.08, expected_return_pct * 1.5)

    return [
        {"pct": round(tier1_pct, 4), "qty_frac": 0.5},
        {"pct": round(tier2_pct, 4), "qty_frac": 0.5}
    ]

def compute_trailing_stop(atr_pct: float, n_atr: float = 1.2) -> float:
    """
    Compute trailing stop percentage.

    trailing = n_atr * atr_pct
    """
    return round(n_atr * atr_pct, 4)

def compute_exit_logic(
    price: float,
    atr_pct: float,
    expected_return_pct: float,
    time_limit_minutes: int = 1440  # 24 hours
) -> Dict[str, Any]:
    """
    Compute full exit logic object.
    """
    stop_price, stop_pct = compute_stop_loss(price, atr_pct)

    return {
        "stop_loss_price": stop_price,
        "stop_loss_pct": stop_pct,
        "take_profit_tiers": compute_take_profit_tiers(expected_return_pct),
        "trailing_pct": compute_trailing_stop(atr_pct),
        "time_in_trade_limit_minutes": time_limit_minutes
    }

# =============================================================================
# SIGNATURE & DETERMINISM
# =============================================================================

def compute_signature(
    run_id: str,
    seed: int,
    decision: str,
    ensemble_score: float
) -> str:
    """
    Compute deterministic signature for proposal.
    """
    payload = f"{run_id}|{seed}|{decision}|{ensemble_score:.6f}"
    return hashlib.sha256(payload.encode()).hexdigest()[:32]

def set_deterministic_seed(run_id: str, seed: int) -> int:
    """
    Set deterministic seed derived from run_id + seed.
    """
    combined = f"{run_id}:{seed}"
    derived = int(hashlib.md5(combined.encode()).hexdigest()[:8], 16)
    np.random.seed(derived)
    return derived

# =============================================================================
# MAIN AGENT CLASS
# =============================================================================

class InstitutionalDecisionAgent:
    """
    Institutional Trading Decision Agent.

    Produces BUY/SELL/HOLD/REJECT proposals with full audit trail.
    READ-ONLY: Does not execute trades.
    """

    def __init__(
        self,
        cvar_limit: float = 0.05,
        per_symbol_cap: float = 0.02,
        strategy_max_pct: float = 0.05,
        data_confidence_threshold: float = 0.6,
        execution_buffer_bps: float = 5.0,
        buy_threshold: float = 0.65,
        sell_threshold: float = -0.5
    ):
        """
        Initialize the institutional decision agent.

        Args:
            cvar_limit: Maximum portfolio CVaR (default 5%)
            per_symbol_cap: Maximum position per symbol (default 2%)
            strategy_max_pct: Maximum strategy allocation (default 5%)
            data_confidence_threshold: Minimum data quality (default 0.6)
            execution_buffer_bps: Buffer for execution cost check (default 5bps)
            buy_threshold: Ensemble score threshold for BUY (default 0.65)
            sell_threshold: Ensemble score threshold for SELL (default -0.5)
        """
        self.cvar_limit = cvar_limit
        self.per_symbol_cap = per_symbol_cap
        self.strategy_max_pct = strategy_max_pct
        self.data_confidence_threshold = data_confidence_threshold
        self.execution_buffer_bps = execution_buffer_bps
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

        logger.info(f"[INIT] InstitutionalDecisionAgent initialized | cvar_limit={cvar_limit} | per_symbol_cap={per_symbol_cap}")

    def compute_proposal(self, input_data: Dict[str, Any]) -> InstitutionalProposal:
        """
        Compute trading proposal from input data.

        Args:
            input_data: Full input object per schema

        Returns:
            InstitutionalProposal with decision and full audit trail
        """
        warnings = []

        # 1. Validate input
        is_valid, errors = validate_input(input_data)
        if not is_valid:
            return self._create_reject_proposal(
                input_data, "INVALID_INPUT", errors
            )

        # 2. Set deterministic seed
        run_id = input_data["run_id"]
        seed = input_data["seed"]
        set_deterministic_seed(run_id, seed)

        # 3. Extract data
        symbol = input_data["symbol"]
        price = input_data["price"]
        nav_usd = input_data["nav_usd"]
        timestamp = input_data["timestamp"]

        features = input_data["features"]
        market = input_data.get("market", {})
        position_state = input_data.get("position_state", {})
        execution = input_data.get("execution", {})
        risk = input_data.get("risk", {})
        models = input_data.get("models", {})

        # 4. Compute ensemble score
        mean_rev = compute_mean_reversion_component(
            features.get("boll_z", 0),
            features.get("rsi_3", 50)
        )

        momentum = compute_momentum_component(
            features.get("ema_9", price),
            features.get("ema_21", price),
            features.get("macd_hist", 0)
        )

        liquidity = compute_liquidity_component(
            execution.get("adv_usd", 0),
            features.get("volume_z", 0)
        )

        volatility = compute_volatility_component(
            features.get("atr_pct", 0.02)
        )

        ensemble_score = compute_ensemble_score(mean_rev, momentum, liquidity, volatility)
        primary_signal = determine_primary_signal(mean_rev, momentum, volatility)

        # 5. Check risk gates
        risk_checks = {
            "cvar_ok": True,
            "entanglement_ok": True,
            "data_confidence_ok": True,
            "execution_cost_ok": True,
            "regime_ok": True
        }

        reject_reasons = []

        # Regime gate
        regime = market.get("regime", "RISK_ON")
        decision_intent = Decision.BUY if ensemble_score >= self.buy_threshold else Decision.HOLD
        regime_ok, regime_reason = check_regime_gate(regime, decision_intent)
        risk_checks["regime_ok"] = regime_ok
        if not regime_ok:
            reject_reasons.append(regime_reason)
            warnings.append(f"Regime gate blocked: {regime}")

        # Data confidence gate
        data_confidence = features.get("data_confidence", 0.8)
        data_ok, data_reason = check_data_confidence_gate(
            data_confidence, self.data_confidence_threshold
        )
        risk_checks["data_confidence_ok"] = data_ok
        if not data_ok:
            reject_reasons.append(data_reason)

        # Execution cost gate
        expected_edge_bps = models.get("expected_edge_bps", 0)
        exec_ok, exec_reason = check_execution_gate(
            execution.get("slippage_bps", 0),
            execution.get("spread_bps", 0),
            expected_edge_bps,
            self.execution_buffer_bps
        )
        risk_checks["execution_cost_ok"] = exec_ok
        if not exec_ok:
            reject_reasons.append(exec_reason)
            warnings.append("Execution costs exceed expected edge")

        # CVaR gate
        portfolio_cvar = risk.get("cvar_95_pct", 0) / 100.0
        symbol_cvar = models.get("expected_edge_bps", 0) / 100.0  # Simplified
        cvar_ok, cvar_reason = check_cvar_gate(
            portfolio_cvar, symbol_cvar, 0.01, self.cvar_limit
        )
        risk_checks["cvar_ok"] = cvar_ok
        if not cvar_ok:
            reject_reasons.append(cvar_reason)

        # Entanglement gate
        entanglement_score = risk.get("global_entanglement_score", 0)
        ent_ok, ent_multiplier = check_entanglement_gate(entanglement_score)
        risk_checks["entanglement_ok"] = ent_ok

        # 6. Determine decision
        if reject_reasons:
            return self._create_reject_proposal(
                input_data, "RISK_GATE_FAILED", reject_reasons
            )

        if ensemble_score >= self.buy_threshold:
            decision = Decision.BUY
        elif ensemble_score <= self.sell_threshold:
            decision = Decision.SELL
        elif position_state.get("has_position", False) and ensemble_score < 0:
            decision = Decision.SELL
        else:
            decision = Decision.HOLD

        # 7. Compute position sizing
        atr_pct = features.get("atr_pct", 0.02)
        realized_vol = atr_pct * math.sqrt(252)  # Annualize

        vol_target_size = compute_volatility_target_size(nav_usd, realized_vol)

        estimated_variance = (atr_pct ** 2) * 252  # Annualized variance
        kelly_fraction = compute_kelly_fraction(expected_edge_bps, estimated_variance)

        suggested_notional_pct = compute_final_position_size(
            kelly_fraction * ent_multiplier,
            vol_target_size,
            self.per_symbol_cap,
            self.strategy_max_pct
        )

        if decision in [Decision.HOLD, Decision.REJECT]:
            suggested_notional_pct = 0.0

        suggested_qty = int((nav_usd * suggested_notional_pct) / price) if price > 0 else 0

        # 8. Compute entry/exit logic
        entry_zone = compute_entry_zone(price, atr_pct, "LIMIT")
        exit_logic = compute_exit_logic(
            price, atr_pct,
            expected_return_pct=expected_edge_bps / 10000.0
        )

        # 9. Compute confidence
        confidence = min(1.0, abs(ensemble_score) * 1.2)

        # 10. Build explanation
        explain = {
            "why_price_is_mispriced": self._generate_rationale(
                features, ensemble_score, primary_signal
            ),
            "indicators_used": {
                "rsi_3": features.get("rsi_3", 0),
                "boll_z": features.get("boll_z", 0),
                "ema_gap_pct": (features.get("ema_9", 1) / features.get("ema_21", 1)) - 1,
                "atr_pct": atr_pct
            },
            "expected_edge_bps": expected_edge_bps,
            "expected_return_pct": expected_edge_bps / 10000.0,
            "model_versions": {"mean_rev": "v1.0", "momentum": "v1.0"},
            "top_drivers": self._get_top_drivers(mean_rev, momentum, volatility),
            "components": {
                "mean_reversion": round(mean_rev, 4),
                "momentum": round(momentum, 4),
                "liquidity": round(liquidity, 4),
                "volatility": round(volatility, 4)
            }
        }

        # 11. Compute signature and hash
        schema_hash = f"sha256:{compute_input_hash(input_data)}"
        signature = compute_signature(run_id, seed, decision.value, ensemble_score)

        # 12. Build and return proposal
        proposal = InstitutionalProposal(
            run_id=run_id,
            seed=seed,
            timestamp=timestamp,
            agent_id=AGENT_ID,
            decision=decision.value,
            confidence=round(confidence, 4),
            ensemble_score=round(ensemble_score, 4),
            primary_signal=primary_signal.value,
            suggested_notional_pct=round(suggested_notional_pct, 6),
            suggested_qty=suggested_qty,
            price_limits={"min": entry_zone["low"], "max": entry_zone["high"]},
            entry_zone=entry_zone,
            exit_logic=exit_logic,
            risk_checks=risk_checks,
            explain=explain,
            warnings=warnings,
            contract_version=CONTRACT_VERSION,
            schema_hash=schema_hash,
            signature=signature
        )

        logger.info(f"[PROPOSAL] {symbol} | {decision.value} | conf={confidence:.2f} | ensemble={ensemble_score:.2f}")

        return proposal

    def _create_reject_proposal(
        self,
        input_data: Dict[str, Any],
        reason: str,
        details: List[str]
    ) -> InstitutionalProposal:
        """Create a REJECT proposal with reasons."""
        run_id = input_data.get("run_id", "unknown")
        seed = input_data.get("seed", 0)
        timestamp = input_data.get("timestamp", datetime.utcnow().isoformat() + "Z")
        price = input_data.get("price", 0)

        schema_hash = f"sha256:{compute_input_hash(input_data)}"
        signature = compute_signature(run_id, seed, "REJECT", 0.0)

        return InstitutionalProposal(
            run_id=run_id,
            seed=seed,
            timestamp=timestamp,
            agent_id=AGENT_ID,
            decision="REJECT",
            confidence=0.0,
            ensemble_score=0.0,
            primary_signal="",
            suggested_notional_pct=0.0,
            suggested_qty=0,
            price_limits={"min": price, "max": price},
            entry_zone={"low": price, "high": price, "type": "LIMIT"},
            exit_logic={
                "stop_loss_price": 0,
                "stop_loss_pct": 0,
                "take_profit_tiers": [],
                "trailing_pct": 0,
                "time_in_trade_limit_minutes": 0
            },
            risk_checks={
                "cvar_ok": False,
                "entanglement_ok": False,
                "data_confidence_ok": False,
                "execution_cost_ok": False
            },
            explain={
                "why_price_is_mispriced": "",
                "indicators_used": {},
                "expected_edge_bps": 0,
                "expected_return_pct": 0,
                "model_versions": {},
                "reject_reason": reason,
                "reject_details": details
            },
            warnings=details,
            contract_version=CONTRACT_VERSION,
            schema_hash=schema_hash,
            signature=signature
        )

    def _generate_rationale(
        self,
        features: Dict[str, float],
        ensemble_score: float,
        primary_signal: PrimarySignal
    ) -> str:
        """Generate human-readable rationale."""
        boll_z = features.get("boll_z", 0)
        rsi_3 = features.get("rsi_3", 50)

        parts = []

        if primary_signal == PrimarySignal.MEAN_REVERSION:
            if boll_z < -2:
                parts.append(f"Price at Bollinger Z={boll_z:.1f} (oversold)")
            if rsi_3 < 30:
                parts.append(f"RSI(3)={rsi_3:.1f} indicates short-term oversold")
        elif primary_signal == PrimarySignal.MOMENTUM:
            ema_9 = features.get("ema_9", 0)
            ema_21 = features.get("ema_21", 0)
            if ema_9 > ema_21:
                parts.append("EMA9 > EMA21 indicates uptrend")
            else:
                parts.append("EMA9 < EMA21 indicates downtrend")

        if not parts:
            parts.append(f"Ensemble score {ensemble_score:.2f} based on multi-factor model")

        return "; ".join(parts)

    def _get_top_drivers(
        self,
        mean_rev: float,
        momentum: float,
        volatility: float
    ) -> List[str]:
        """Get top 3 decision drivers."""
        drivers = [
            (abs(mean_rev), f"mean_reversion={mean_rev:.2f}"),
            (abs(momentum), f"momentum={momentum:.2f}"),
            (abs(volatility), f"volatility={volatility:.2f}")
        ]
        drivers.sort(key=lambda x: x[0], reverse=True)
        return [d[1] for d in drivers[:3]]


# =============================================================================
# CONVENIENCE FACTORY
# =============================================================================

_agent_instance: Optional[InstitutionalDecisionAgent] = None

def get_institutional_agent() -> InstitutionalDecisionAgent:
    """Get singleton agent instance."""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = InstitutionalDecisionAgent()
    return _agent_instance


# =============================================================================
# TEST HELPER
# =============================================================================

def create_sample_input() -> Dict[str, Any]:
    """Create sample input for testing."""
    return {
        "run_id": "r-20260128-test",
        "seed": 12345,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "symbol": "AAPL",
        "price": 182.35,
        "nav_usd": 1000000,
        "market": {
            "regime": "RISK_ON",
            "vix": 18.2,
            "liquidity": "NORMAL"
        },
        "features": {
            "rsi_14": 28.3,
            "rsi_3": 18.2,
            "ema_9": 179.1,
            "ema_21": 186.7,
            "boll_z": -2.4,
            "atr_pct": 0.018,
            "vwap": 184.9,
            "macd_hist": -0.45,
            "obv_z": 2.1,
            "volume_z": 1.5,
            "data_confidence": 0.9
        },
        "historical": {
            "returns_window": [],
            "price_series": []
        },
        "position_state": {
            "has_position": False,
            "entry_price": None,
            "unrealized_pct": None,
            "qty": 0
        },
        "execution": {
            "slippage_bps": 6,
            "spread_bps": 3,
            "adv_usd": 10000000
        },
        "models": {
            "mean_reversion_score": 0.82,
            "trend_score": 0.31,
            "expected_edge_bps": 42,
            "win_rate_est": 0.57
        },
        "risk": {
            "cvar_95_pct": 2.8,
            "max_strategy_weight_pct": 0.05,
            "per_symbol_max_pct": 0.02,
            "global_drawdown_pct": 0.06,
            "global_entanglement_score": 0.3
        },
        "contracts": {
            "contract_version": "1.0",
            "schema_hash": "sha256:test"
        }
    }


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)

    agent = InstitutionalDecisionAgent()
    sample_input = create_sample_input()

    proposal = agent.compute_proposal(sample_input)

    print("\n" + "=" * 80)
    print("INSTITUTIONAL DECISION AGENT - TEST OUTPUT")
    print("=" * 80)
    print(json.dumps(proposal.to_dict(), indent=2))
    print("=" * 80)
