import logging
import os
from typing import Any, Dict, List

import numpy as np
import yaml

# Try importing Quantum libraries
try:
    import pennylane as qml
    HAS_PENNYLANE = True
except ImportError:
    HAS_PENNYLANE = False

logger = logging.getLogger("QC_INTERFACE")

class QuantumBackend:
    """
    Unified Interface for Quantum Execution.
    Currently supports PennyLane (Simulator).
    """

    def __init__(self, config_path: str = "configs/quantum_config.yaml"):
        self.config = self._load_config(config_path)
        self.provider = self.config.get("backend", {}).get("provider", "pennylane")
        self.device_name = self.config.get("backend", {}).get("device", "default.qubit")
        self.n_qubits = self.config.get("backend", {}).get("n_qubits", 8)
        self.shots = self.config.get("backend", {}).get("shots", 1024)

        self.device = None
        self._init_backend()

    def _load_config(self, path: str) -> Dict[str, Any]:
        if not os.path.exists(path):
            logger.warning(f"Config {path} not found. Using defaults.")
            return {}
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def _init_backend(self):
        if self.provider == "pennylane":
            if not HAS_PENNYLANE:
                logger.warning("PennyLane not installed. Quantum Ops disabled.")
                self.device = None
                return

            logger.info(f"Initializing PennyLane device: {self.device_name} ({self.n_qubits} qubits)")
            self.device = qml.device(self.device_name, wires=self.n_qubits, shots=self.shots)
        else:
            logger.warning(f"Provider {self.provider} not supported or libs missing.")
            self.device = None

    def get_device(self):
        """Return the native PennyLane device."""
        return self.device

    def run_circuit(self, circuit_func, params, **kwargs):
        """
        if self.device is None:
            logger.warning("Quantum device not available. Returning None.")
            return None

        if self.provider == "pennylane":
            return circuit_func(params, **kwargs)

# Singleton
_BACKEND = None

def get_backend():
    global _BACKEND
    if _BACKEND is None:
        _BACKEND = QuantumBackend()
    return _BACKEND
