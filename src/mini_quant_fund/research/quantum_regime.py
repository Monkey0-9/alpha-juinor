"""
Quantum Regime Detection using VQC
==================================

Variational Quantum Circuit (VQC) for market regime classification.
Uses PennyLane for quantum circuit simulation and PyTorch for optimization.

Hybrid quantum-classical architecture:
1. Classical Encoder (Data -> Features)
2. Angle Embedding (Features -> Qubits)
3. Variational Quantum Circuit (Entanglement + Rotation)
4. Measurement & Classical Decoder (Qubits -> Regime Class)

References:
- Mitarai, K., et al. (2018). "Quantum circuit learning"
- Schuld, M., et al. (2019). "Quantum machine learning in feature Hilbert spaces"
"""

import numpy as np
import torch
import torch.nn as nn
import logging

# Optional import to avoid crashing if PennyLane is not installed
try:
    import pennylane as qml
    HAS_PENNYLANE = True
except ImportError:
    HAS_PENNYLANE = False

logger = logging.getLogger(__name__)


class QuantumRegimeDetector(nn.Module):
    """
    Hybrid Quantum-Classical Neural Network for Regime Detection.
    """

    def __init__(self, n_qubits: int = 4, n_layers: int = 2):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.status = (
            "ACTIVE" if HAS_PENNYLANE else "SIMULATED_CLASSICAL_FALLBACK"
        )

        if HAS_PENNYLANE:
            self.dev = qml.device("default.qubit", wires=n_qubits)
            self.qnode = qml.QNode(self._circuit, self.dev, interface="torch")
            self.weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        else:
            logger.warning("PennyLane not found. Using classical fallback.")

        # Classical pre-processing
        self.cl_layer1 = nn.Linear(10, n_qubits)
        self.tanh = nn.Tanh()

        # Quantum weights (if PL exists, otherwise classical weights)
        if HAS_PENNYLANE:
            self.q_weights = nn.Parameter(torch.randn(n_layers, n_qubits, 3))
        else:
            self.fallback_layer = nn.Linear(n_qubits, n_qubits)

        # Classical post-processing
        self.cl_layer2 = nn.Linear(n_qubits, 4)  # 4 regimes
        self.softmax = nn.Softmax(dim=1)

    def _circuit(self, inputs, weights):
        """Quantum circuit definition."""
        # Angle Embedding
        for i in range(self.n_qubits):
            qml.RX(inputs[i] * np.pi, wires=i)

        # Variational Layers
        for layer_idx in range(self.n_layers):
            for i in range(self.n_qubits):
                qml.Rot(*weights[layer_idx, i], wires=i)
            # Entanglement
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            if self.n_qubits > 1:
                qml.CNOT(wires=[self.n_qubits - 1, 0])

        # Measurement
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def forward(self, x):
        batch_size = x.shape[0]

        # Pre-processing
        x = self.cl_layer1(x)
        x = self.tanh(x)

        # Quantum Layer
        if HAS_PENNYLANE:
            # PennyLane doesn't support batching natively in all versions,
            # so we loop (inefficient but works for prototype)
            q_out = []
            for i in range(batch_size):
                res = self.qnode(x[i], self.q_weights)
                q_out.append(torch.stack(res))
            x_q = torch.stack(q_out)
        else:
            # Fallback
            x_q = self.fallback_layer(x)
            x_q = torch.tanh(x_q)

        # Post-processing
        out = self.cl_layer2(x_q)
        return self.softmax(out)


# Global Accessor
_vqc_detector = None


def get_quantum_detector():
    global _vqc_detector
    if _vqc_detector is None:
        _vqc_detector = QuantumRegimeDetector()
    return _vqc_detector

