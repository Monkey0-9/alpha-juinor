"""
Hello VQC: A tiny End-to-End Hybrid Quantum-Classical Loop.
Verifies all phases: Config -> Tracking -> Quantum Sim -> Artifacts
"""
import logging
import os
import sys

import numpy as np

# Add root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ml.tracking import ExperimentTracker
from quantum.qc_interface import get_backend, qml

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HELLO_VQC")

def main():
    logger.info("Initializing Hybrid Stack...")

    # 1. Initialize Backend
    backend = get_backend()
    dev = backend.get_device()

    # 2. Define Circuit (Variational Ansatze)
    @qml.qnode(dev)
    def circuit(params, x):
        # Embed Data (Angle Encoding)
        qml.RX(x[0], wires=0)
        qml.RY(x[1], wires=1)

        # Variational Layer
        qml.Rot(*params[0], wires=0)
        qml.Rot(*params[1], wires=1)

        # Entanglement
        qml.CNOT(wires=[0, 1])

        # Measure
        return qml.expval(qml.PauliZ(0))

    # 3. Initialize Tracking
    tracker = ExperimentTracker("quantum_smoke_tests")

    with tracker.start_run("hello_vqc_01") as run:
        logger.info(f"Run ID: {run.info.run_id}")

        # Log Config
        tracker.enforce_determinism(seed=42)
        tracker.log_params({"n_qubits": backend.n_qubits, "shots": backend.shots})

        # 4. "Training" Loop (Mock)
        logger.info("Starting VQC Execution...")
        params = np.random.uniform(0, np.pi, (2, 3))
        data_inputs = np.array([0.5, 0.2])

        # Run Quantum Circuit
        result = circuit(params, data_inputs)
        logger.info(f"Quantum Result: {result}")

        # Log Metrics
        tracker.log_metrics({"expectation_value": float(result)})

        # 5. Save Artifacts
        result_path = "vqc_result.txt"
        with open(result_path, "w") as f:
            f.write(f"Result: {result}\nParams: {params}\n")
        tracker.log_artifact(result_path)

        logger.info("Success! Hybrid Loop Complete.")

if __name__ == "__main__":
    main()
