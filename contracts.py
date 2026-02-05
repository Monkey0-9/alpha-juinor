from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
import pandas as pd
from enum import Enum
import logging

logger = logging.getLogger(__name__)

# ===================================================================
# ALPHA DISTRIBUTIONAL CONTRACT (Elite-Tier)
# ===================================================================

@dataclass
class AlphaDistribution:
    """
    Mandatory output contract for all alpha families.

    Attributes:
        mu: Expected return (daily)
        sigma: Volatility (daily)
        p_loss: Probability of loss [0,1]
        cvar_95: Conditional Value at Risk at 95% confidence
        confidence: Signal confidence [0,1]
    """
    mu: float
    sigma: float
    p_loss: float
    cvar_95: float
    confidence: float

    def validate(self) -> bool:
        """Validate contract compliance"""
        errors = []

        if not (-1.0 <= self.mu <= 1.0):
            errors.append(f"mu out of range: {self.mu}")
        if not (0.0 <= self.sigma <= 1.0):
            errors.append(f"sigma out of range: {self.sigma}")
        if not (0.0 <= self.p_loss <= 1.0):
            errors.append(f"p_loss out of range: {self.p_loss}")
        if not (-1.0 <= self.cvar_95 <= 0.0):
            errors.append(f"cvar_95 out of range: {self.cvar_95}")
        if not (0.0 <= self.confidence <= 1.0):
            errors.append(f"confidence out of range: {self.confidence}")
        if self.sigma <= 0:
            errors.append("sigma must be positive")

        if errors:
            logger.error(f"Alpha contract validation failed: {errors}")
            return False
        return True

# ===================================================================
# DECISION CONTRACTS
# ===================================================================

class decision_enum(Enum):
    EXECUTE = "EXECUTE"
    HOLD = "HOLD"
    REJECT = "REJECT"
    ERROR = "ERROR"

@dataclass
class AgentResult:
    """
    Standard output for ALL Alpha Agents.
    Enforces the Production Interface.
    """
    symbol: str
    name: str
    mu: float  # Forecasted return (normalized)
    sigma: float  # Forecasted volatility / uncertainty
    confidence: float  # 0.0 to 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return {
            "symbol": self.symbol,
            "name": self.name,
            "mu": self.mu,
            "sigma": self.sigma,
            "confidence": self.confidence,
            "metadata": self.metadata
        }

@dataclass
class ProviderMetadata:
    """Metadata about data provider used for a symbol"""
    name: str
    confidence: float
    latency_ms: float = 0.0
    quality_score: float = 1.0

@dataclass(frozen=True)
class AllocationRequest:
    """
    Immutable request from PM Brain to Allocator.
    Enforces strict interface contract.
    """
    symbol: str
    mu: float
    sigma: float
    confidence: float
    liquidity: float
    regime: str
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DecisionRecord:
    """
    Audit record matching institutional schema.
    This is what gets written to the audit log/database.
    """
    cycle_id: str
    symbol: str
    timestamp: str
    data_providers: Dict[str, float]   # provider -> confidence
    alphas: Dict[str, float]           # agent -> mu
    sigmas: Dict[str, float]           # agent -> sigma
    conviction: float
    conviction_zscore: float
    risk_checks: List[str]
    pm_override: str
    final_decision: str  # EXECUTE/HOLD/REJECT/ERROR
    order: Optional[Dict[str, Any]]
    reason_codes: List[str]
    raw_traceback: Optional[str]

@dataclass
class OrderInfo:
    action: str # BUY, SELL
    quantity: float
    order_type: str = "MARKET"
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "DAY"

@dataclass
class Decision:
    """
    Final per-symbol decision produced by the Cycle Orchestrator.
    This is the Atomic Unit of the system.
    """
    cycle_id: str
    symbol: str
    final_decision: decision_enum
    reason_codes: List[str]

    # Aggregated Metrics (FIXED: removed duplicate)
    mu_hat: float = 0.0
    sigma_hat: float = 0.0
    conviction: float = 0.0

    # Execution Details
    order: Optional[OrderInfo] = None

    # Debug/Audit
    agent_results: Dict[str, Any] = field(default_factory=dict)
    risk_results: Dict[str, Any] = field(default_factory=dict)

    # Audit & Debug
    traceback: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_audit_record(self) -> DecisionRecord:
        """
        Generates the Immutable Audit Record matching the Institutional Schema.
        Returns DecisionRecord dataclass.
        """
        # Extract provider metadata
        provider_meta = self.metadata.get("provider", {})
        if isinstance(provider_meta, dict):
            data_providers = {provider_meta.get("name", "Unknown"): provider_meta.get("confidence", 0.5)}
        else:
            data_providers = {"Unknown": 0.5}

        # Extract alpha results
        alphas = {name: result.get("mu", 0.0) for name, result in self.agent_results.items()}
        sigmas = {name: result.get("sigma", 0.0) for name, result in self.agent_results.items()}

        # Extract risk checks
        risk_checks = self.risk_results.get("checks", []) if isinstance(self.risk_results, dict) else []

        # PM override
        pm_override = self.metadata.get("pm_override", "ALLOW")

        # Conviction z-score
        conviction_zscore = self.metadata.get("conviction_zscore", self.conviction)

        return DecisionRecord(
            cycle_id=self.cycle_id,
            symbol=self.symbol,
            timestamp=str(pd.Timestamp.utcnow()),
            data_providers=data_providers,
            alphas=alphas,
            sigmas=sigmas,
            conviction=self.conviction,
            conviction_zscore=conviction_zscore,
            risk_checks=risk_checks,
            pm_override=pm_override,
            final_decision=self.final_decision.value,
            order=self.order.__dict__ if self.order else None,
            reason_codes=self.reason_codes,
            raw_traceback=self.traceback
        )

class BaseAgent:
    """
    Base class for all 50+ AI Agents.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = self.__class__.__name__

    def evaluate(self, symbol: str, data: Any, **kwargs) -> AgentResult:
        """
        Must be implemented by all agents.
        """
        raise NotImplementedError("Agents must implement evaluate()")
