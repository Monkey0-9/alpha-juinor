"""
contracts/allocation.py

Allocation contract dataclasses for capital auction and order management.

CONTRACT VERSIONING: All contracts include version and schema hash for
breaking change detection and forced backtest requirements.
"""

import hashlib
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, Optional
from datetime import datetime

# ============================================================================
# CONTRACT VERSIONING
# ============================================================================

ALLOCATION_REQUEST_VERSION = "1.0.0"
ORDER_INFO_VERSION = "1.0.0"
DECISION_RECORD_VERSION = "1.0.0"

def _compute_schema_hash(fields: list) -> str:
    """Compute deterministic hash of field names for schema validation."""
    canonical = ",".join(sorted(fields))
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()[:16]

# Pre-computed hashes (update when schema changes)
ALLOCATION_REQUEST_HASH = _compute_schema_hash([
    "symbol", "mu", "sigma", "confidence", "regime", "cvar_95",
    "provider", "liquidity", "metadata", "timestamp",
    "contract_version", "schema_hash"
])
ORDER_INFO_HASH = _compute_schema_hash([
    "symbol", "side", "quantity", "target_weight", "reason", "metadata",
    "contract_version", "schema_hash"
])
DECISION_RECORD_HASH = _compute_schema_hash([
    "cycle_id", "symbol", "timestamp", "data_providers", "alphas", "sigmas",
    "conviction", "conviction_zscore", "mu", "sigma", "price", "risk_checks", "pm_override",
    "final_decision", "decision", "reason_codes", "order", "raw_traceback",
    "quantum_state", "regime", "entanglement_score",
    "contract_version", "schema_hash"
])


@dataclass
class AllocationRequest:
    """
    Request for capital allocation from a strategy.

    Used by CapitalAuctionEngine to compete for capital.

    CONTRACT VERSION: Breaking changes require forced backtest.
    """
    symbol: str
    mu: float                    # Expected daily return
    sigma: float                 # Expected daily volatility
    confidence: float            # Model confidence [0, 1]
    regime: str = "UNKNOWN"      # Current regime label
    cvar_95: float = -0.02       # 95% CVaR (optional)
    provider: str = "unknown"    # Alpha provider
    liquidity: float = 0.0       # Available liquidity or ADV
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""
    # Contract versioning
    contract_version: str = ALLOCATION_REQUEST_VERSION
    schema_hash: str = ALLOCATION_REQUEST_HASH

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat() + 'Z'
        # Clamp confidence
        self.confidence = max(0.0, min(1.0, self.confidence))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)




@dataclass
class OrderInfo:
    """
    Order information for execution.

    CONTRACT VERSION: Breaking changes require forced backtest.
    """
    symbol: str
    side: str                    # "BUY" or "SELL"
    quantity: float
    target_weight: float
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Contract versioning
    contract_version: str = ORDER_INFO_VERSION
    schema_hash: str = ORDER_INFO_HASH

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DecisionRecord:
    """
    Audit decision record for tracking trading decisions.

    Used by decision_log.py for compliance and audit trail.

    CONTRACT VERSION: Breaking changes require forced backtest.
    """
    cycle_id: str
    symbol: str
    timestamp: str
    data_providers: Dict[str, Any] = field(default_factory=dict)
    alphas: Dict[str, float] = field(default_factory=dict)
    sigmas: Dict[str, float] = field(default_factory=dict)
    conviction: float = 0.0
    conviction_zscore: float = 0.0
    mu: float = 0.0        # Aggregated expected return
    sigma: float = 0.0     # Aggregated volatility
    price: float = 0.0     # Execution reference price
    risk_checks: list = field(default_factory=list)
    pm_override: str = "NONE"
    final_decision: str = "HOLD"
    decision: str = "HOLD"  # Alias for final_decision
    reason_codes: list = field(default_factory=list)
    order: Optional[Dict[str, Any]] = None
    raw_traceback: Optional[str] = None

    # Quantum Fields
    quantum_state: Optional[Dict[str, Any]] = field(default_factory=dict)
    regime: str = "UNCERTAIN"
    entanglement_score: float = 0.0

    # Contract versioning
    contract_version: str = DECISION_RECORD_VERSION
    schema_hash: str = DECISION_RECORD_HASH

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat() + 'Z'
        # Sync decision and final_decision
        if self.decision != "HOLD":
            self.final_decision = self.decision
        elif self.final_decision != "HOLD":
            self.decision = self.final_decision

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)



REJECTED_ASSET_HASH = _compute_schema_hash([
    "symbol", "reason", "mu", "sigma", "score", "timestamp",
    "contract_version", "schema_hash"
])

@dataclass
class RejectedAsset:
    """Record of why an asset was excluded from the portfolio."""
    symbol: str
    reason: str
    mu: float
    sigma: float
    score: float
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    contract_version: str = "1.0.0"
    schema_hash: str = REJECTED_ASSET_HASH

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


