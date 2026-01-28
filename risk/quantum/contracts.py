"""
risk/quantum/contracts.py

Data contracts for quantum risk modules.
Enforces versioning and schema hashing.
"""

import hashlib
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Any, Optional
from datetime import datetime

# ============================================================================
# CONTRACT VERSIONING
# ============================================================================

QUANTUM_CONTRACT_VERSION = "1.0.0"

def _compute_schema_hash(fields: list) -> str:
    canonical = ",".join(sorted(fields))
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()[:16]

STATE_HASH = _compute_schema_hash([
    "regime_belief", "entropy", "transition_matrix_id",
    "timestamp", "contract_version", "schema_hash"
])

ENTANGLEMENT_HASH = _compute_schema_hash([
    "global_index", "asset_centrality", "adjacency_matrix_id",
    "threshold_breach", "timestamp", "contract_version", "schema_hash"
])

PATH_INTEGRAL_HASH = _compute_schema_hash([
    "stressed_cvar", "samples", "shock_magnitude", "passed",
    "timestamp", "contract_version", "schema_hash"
])

# ============================================================================
# DATACLASSES
# ============================================================================

@dataclass
class QuantumState:
    """Regime belief state vector p(t)."""
    regime_belief: List[float]  # Probability vector sum=1
    entropy: float
    transition_matrix_id: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    contract_version: str = QUANTUM_CONTRACT_VERSION
    schema_hash: str = STATE_HASH

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class EntanglementReport:
    """Market entanglement (correlation) structure."""
    global_index: float  # Spectral radius / top eigenvalue
    asset_centrality: Dict[str, float]
    adjacency_matrix_id: str
    threshold_breach: bool
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    contract_version: str = QUANTUM_CONTRACT_VERSION
    schema_hash: str = ENTANGLEMENT_HASH

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class PathIntegralResult:
    """Result of Feynman-Kac path integral stress test."""
    stressed_cvar: float
    samples: int
    shock_magnitude: float
    passed: bool
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    contract_version: str = QUANTUM_CONTRACT_VERSION
    schema_hash: str = PATH_INTEGRAL_HASH

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
