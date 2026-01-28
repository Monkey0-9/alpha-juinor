"""
risk/quantum package - Advanced Physics-based Risk Modules.

Modules:
- state_space: Regime belief evolution p(t+1) = U(t)p(t)
- entanglement: Mutual information based correlation structure
- path_integral: Importance sampling for tail risk (Feynman-Kac)
- decision_operators: Non-commutative action sequences

Rules:
1. READ-ONLY for execution.
2. Deterministic (seeded).
3. Contract versioned outputs.
"""

from .contracts import (
    QuantumState,
    EntanglementReport,
    PathIntegralResult
)

__version__ = "1.0.0"
