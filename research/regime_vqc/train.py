"""
Phase 3 Prototype: Hybrid VQC for Regime Detection.
Demonstrates:
- Feature Engineering (Classical)
- Dimensionality Reduction (Preprocessing)
- Quantum Embedding (Angle Encoding)
- Variational Training (PennyLane + NumPy)
- Experiment Tracking (MLflow)
"""
import sys
import os
import numpy as np
import pandas as pd
import logging

# Add root to path safely
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(root_dir)

# Now standard imports should work without lint errors
from quantum.qc_interface import get_backend, qml  # noqa: E402
from ml.preprocessing import QuantumPreprocessor  # noqa: E402
from ml.tracking import ExperimentTracker  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("REGIME_VQC")


def get_synthetic_data(n_samples=100, n_features=8):
    """Generate synthetic regime data (Vol, Returns, Spread) -> Regime."""
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    # Simple logic: If feature 0 > 0.5 -> Crisis (1)
    y = (X[:, 0] > 0.5).astype(int)
    # Add noise
    mask = np.random.rand(n_samples) < 0.1
    y[mask] = 1 - y[mask]
    columns = [f"f_{i}" for i in range(n_features)]
    return pd.DataFrame(X, columns=columns), y


def train_vqc():
    # 1. Config & Setup
    backend = get_backend()
    dev = backend.get_device()
    n_qubits = backend.n_qubits

    tracker = ExperimentTracker("hybrid_regime_vqc")

    with tracker.start_run("vqc_regime_proto_01") as run:
        logger.info(f"Run ID: {run.info.run_id}")
        tracker.enforce_determinism(seed=42)

        # 2. Data & Preprocessing
        X, y = get_synthetic_data(n_features=n_qubits)
        preprocessor = QuantumPreprocessor(n_qubits=n_qubits, use_pca=True)

        # Transform classical data to Quantum-Ready features [0, PI]
        X_encoded = preprocessor.fit_transform(X)
        y_signed = np.where(y == 0, -1, 1)  # VQC usually outputs [-1, 1]

        logger.info(f"Data shape: {X_encoded.shape}")

        # 3. Define VQC Circuit
        @qml.qnode(dev)
        def circuit(params, x):
            # Angle Embedding
            for i in range(n_qubits):
                qml.RX(x[i], wires=i)

            # Strong Entangling Layers
            qml.StronglyEntanglingLayers(params, wires=range(n_qubits))

            # Measurement
            return qml.expval(qml.PauliZ(0))

        # 4. Variational Optimization
        # 2 layers of StronglyEntangling
        shape = qml.StronglyEntanglingLayers.shape(
            n_layers=2, n_wires=n_qubits
        )
        params = np.random.random(shape)

        opt = qml.NesterovMomentumOptimizer(stepsize=0.1)
        batch_size = 10

        def cost(params, X_batch, y_batch):
            preds = [circuit(params, x) for x in X_batch]
            return np.mean((y_batch - preds) ** 2)

        logger.info("Starting VQC Optimization...")
        for epoch in range(5):  # Short run for prototype
            # Mini-batch
            indices = np.random.randint(0, len(X), batch_size)
            X_batch = X_encoded[indices]
            y_batch = y_signed[indices]

            params, loss = opt.step_and_cost(
                lambda p: cost(p, X_batch, y_batch), params
            )

            logger.info(f"Epoch {epoch}: Loss = {loss:.4f}")
            tracker.log_metrics({"loss": float(loss)}, step=epoch)

        # 5. Save Model Metadata
        tracker.log_artifact("research/regime_vqc/train.py")
        logger.info("Training Complete.")


if __name__ == "__main__":
    train_vqc()
