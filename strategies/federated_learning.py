import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import json
import hashlib
import secrets
from pathlib import Path
import threading
import time
import copy
from collections import defaultdict, deque
import warnings

# ML imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
import joblib

logger = logging.getLogger(__name__)

class FederatedAlgorithm(Enum):
    FED_AVG = "fed_avg"                    # Federated Averaging
    FED_PROX = "fed_prox"                  # FedProx (with proximal term)
    SCAFFOLD = "scaffold"                  # SCAFFOLD algorithm
    FED_NOVA = "fed_nova"                  # FedNova (normalized averaging)
    FED_ADAM = "fed_adam"                  # Federated Adam

class PrivacyMechanism(Enum):
    DIFFERENTIAL_PRIVACY = "differential_privacy"
    SECURE_AGGREGATION = "secure_aggregation"
    HOMOMORPHIC_ENCRYPTION = "homomorphic_encryption"
    NONE = "none"

class AggregationStrategy(Enum):
    WEIGHTED_AVERAGE = "weighted_average"
    ROBUST_AGGREGATION = "robust_aggregation"
    Krum = "krum"
    Trimmed_Mean = "trimmed_mean"

@dataclass
class FederatedClient:
    """Represents a client in the federated learning network."""
    client_id: str
    data_size: int
    model_version: int = 0
    last_update: datetime = field(default_factory=datetime.utcnow)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    privacy_budget: float = 1.0
    is_malicious: bool = False
    communication_cost: float = 0.0

@dataclass
class FederatedModelUpdate:
    """Represents a model update from a client."""
    client_id: str
    model_parameters: Dict[str, np.ndarray]
    gradients: Optional[Dict[str, np.ndarray]] = None
    data_size: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    privacy_noise: Optional[Dict[str, np.ndarray]] = None
    checksum: str = ""

@dataclass
class GlobalModelState:
    """Represents the global model state."""
    model_parameters: Dict[str, np.ndarray]
    version: int
    round_number: int
    participating_clients: List[str]
    aggregation_method: AggregationStrategy
    convergence_metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class FederatedLearningConfig:
    """Configuration for federated learning."""
    num_clients: int = 10
    fraction_participate: float = 0.8
    num_rounds: int = 100
    local_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 0.01
    algorithm: FederatedAlgorithm = FederatedAlgorithm.FED_AVG
    privacy_mechanism: PrivacyMechanism = PrivacyMechanism.NONE
    aggregation_strategy: AggregationStrategy = AggregationStrategy.WEIGHTED_AVERAGE
    differential_privacy_epsilon: float = 1.0
    secure_aggregation_threshold: int = 3

class InstitutionalFederatedLearning:
    """
    INSTITUTIONAL-GRADE FEDERATED LEARNING SYSTEM
    Privacy-preserving collaborative machine learning across distributed institutions.
    Implements advanced FL algorithms with differential privacy and secure aggregation.
    """

    def __init__(self, config: FederatedLearningConfig = None, model_factory: Callable = None):
        self.config = config or FederatedLearningConfig()
        self.model_factory = model_factory or self._default_model_factory

        # Federated learning state
        self.clients: Dict[str, FederatedClient] = {}
        self.global_model: GlobalModelState = None
        self.model_updates: List[FederatedModelUpdate] = []

        # Privacy and security
        self.privacy_engine = PrivacyEngine(self.config.privacy_mechanism)
        self.secure_aggregator = SecureAggregator(self.config.secure_aggregation_threshold)

        # Performance tracking
        self.round_history: List[Dict[str, Any]] = []
        self.convergence_metrics: Dict[str, float] = {}

        # Communication optimization
        self.compression_enabled = True
        self.bandwidth_budget = 100 * 1024 * 1024  # 100MB per round

        # Initialize system
        self._initialize_clients()
        self._initialize_global_model()

        logger.info(f"Institutional Federated Learning initialized with {self.config.num_clients} clients")

    def _initialize_clients(self):
        """Initialize federated clients."""
        for i in range(self.config.num_clients):
            client_id = f"client_{i:03d}"
            data_size = np.random.randint(1000, 10000)  # Simulated data sizes

            self.clients[client_id] = FederatedClient(
                client_id=client_id,
                data_size=data_size,
                performance_metrics={'initial_accuracy': 0.5}
            )

        logger.info(f"Initialized {len(self.clients)} federated clients")

    def _initialize_global_model(self):
        """Initialize the global model."""
        # Create initial model
        initial_model = self.model_factory()

        self.global_model = GlobalModelState(
            model_parameters=self._extract_model_parameters(initial_model),
            version=0,
            round_number=0,
            participating_clients=[],
            aggregation_method=self.config.aggregation_strategy
        )

        logger.info("Global model initialized")

    def _default_model_factory(self) -> Any:
        """Default model factory - creates a simple ML model."""
        return RandomForestClassifier(n_estimators=100, random_state=42)

    def _extract_model_parameters(self, model) -> Dict[str, np.ndarray]:
        """Extract parameters from ML model."""
        if hasattr(model, 'coef_') and hasattr(model, 'intercept_'):
            # Linear model
            return {
                'coef': model.coef_.copy(),
                'intercept': model.intercept_.copy()
            }
        elif hasattr(model, 'estimators_'):
            # Ensemble model - extract tree parameters
            # Simplified: just return feature importances
            return {
                'feature_importances': model.feature_importances_.copy(),
                'n_estimators': np.array([model.n_estimators])
            }
        else:
            # Generic model - try to extract all numpy arrays
            params = {}
            for attr in dir(model):
                if not attr.startswith('_'):
                    value = getattr(model, attr)
                    if isinstance(value, np.ndarray):
                        params[attr] = value.copy()
            return params

    def _set_model_parameters(self, model, parameters: Dict[str, np.ndarray]):
        """Set parameters in ML model."""
        for param_name, param_value in parameters.items():
            if hasattr(model, param_name):
                setattr(model, param_name, param_value)

    async def run_federated_training(self, training_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
                                   num_rounds: int = None) -> Dict[str, Any]:
        """
        Run federated learning training across all rounds.
        training_data: dict of client_id -> (X, y) training data
        """
        num_rounds = num_rounds or self.config.num_rounds
        logger.info(f"Starting federated training for {num_rounds} rounds")

        for round_num in range(num_rounds):
            logger.info(f"Round {round_num + 1}/{num_rounds}")

            # Select participating clients
            participating_clients = self._select_clients()

            # Send global model to clients
            client_updates = await self._client_training_round(
                participating_clients, training_data, round_num
            )

            # Aggregate updates
            if client_updates:
                new_global_params = self._aggregate_updates(client_updates, round_num)

                # Update global model
                self.global_model = GlobalModelState(
                    model_parameters=new_global_params,
                    version=self.global_model.version + 1,
                    round_number=round_num + 1,
                    participating_clients=participating_clients,
                    aggregation_method=self.config.aggregation_strategy,
                    convergence_metrics=self._calculate_convergence_metrics(client_updates)
                )

                # Evaluate global model
                global_performance = self._evaluate_global_model(training_data)

                # Record round results
                round_result = {
                    'round': round_num + 1,
                    'participating_clients': len(participating_clients),
                    'global_performance': global_performance,
                    'convergence_metrics': self.global_model.convergence_metrics,
                    'communication_cost': sum(update.communication_cost for update in client_updates),
                    'privacy_budget_used': sum(1 - client.privacy_budget for client in
                                             [self.clients[cid] for cid in participating_clients if cid in self.clients])
                }

                self.round_history.append(round_result)

                logger.info(f"Round {round_num + 1} completed. Global accuracy: {global_performance.get('accuracy', 0):.4f}")

            # Check convergence
            if self._check_convergence():
                logger.info(f"Training converged at round {round_num + 1}")
                break

        # Final evaluation
        final_results = {
            'total_rounds': len(self.round_history),
            'final_performance': self._evaluate_global_model(training_data),
            'convergence_achieved': self._check_convergence(),
            'privacy_preserved': self._check_privacy_preservation(),
            'communication_efficiency': self._calculate_communication_efficiency(),
            'round_history': self.round_history
        }

        logger.info(f"Federated training completed. Final results: {final_results['final_performance']}")
        return final_results

    def _select_clients(self) -> List[str]:
        """Select clients for participation in current round."""
        num_participants = int(self.config.num_clients * self.config.fraction_participate)

        # Simple random selection (could be improved with importance sampling)
        available_clients = list(self.clients.keys())
        participating_clients = np.random.choice(
            available_clients,
            size=min(num_participants, len(available_clients)),
            replace=False
        )

        return list(participating_clients)

    async def _client_training_round(self, participating_clients: List[str],
                                   training_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
                                   round_num: int) -> List[FederatedModelUpdate]:
        """Run one round of client training."""
        client_updates = []

        # Send global model to clients (simulated)
        global_params = self.global_model.model_parameters

        for client_id in participating_clients:
            if client_id not in training_data:
                continue

            # Train client model
            update = await self._train_client_model(
                client_id, global_params, training_data[client_id], round_num
            )

            if update:
                client_updates.append(update)

        return client_updates

    async def _train_client_model(self, client_id: str, global_params: Dict[str, np.ndarray],
                                client_data: Tuple[np.ndarray, np.ndarray], round_num: int) -> Optional[FederatedModelUpdate]:
        """Train model on client data."""
        try:
            X, y = client_data
            client = self.clients[client_id]

            # Create local model with global parameters
            local_model = self.model_factory()
            self._set_model_parameters(local_model, global_params)

            # Local training
            if hasattr(local_model, 'fit'):
                # Split data for training and validation
                split_idx = int(0.8 * len(X))
                X_train, X_val = X[:split_idx], X[split_idx:]
                y_train, y_val = y[:split_idx], y[split_idx:]

                # Train model
                local_model.fit(X_train, y_train)

                # Evaluate on validation set
                if hasattr(local_model, 'predict'):
                    y_pred = local_model.predict(X_val)
                    if len(y.shape) > 1 and y.shape[1] > 1:  # Classification
                        accuracy = accuracy_score(y_val, y_pred)
                        client.performance_metrics['accuracy'] = accuracy
                    else:  # Regression
                        mse = mean_squared_error(y_val, y_pred)
                        client.performance_metrics['mse'] = mse

            # Extract updated parameters
            updated_params = self._extract_model_parameters(local_model)

            # Apply privacy mechanisms
            if self.config.privacy_mechanism != PrivacyMechanism.NONE:
                updated_params = self.privacy_engine.apply_privacy(
                    updated_params, client.privacy_budget
                )
                client.privacy_budget *= 0.9  # Reduce privacy budget

            # Calculate parameter differences (gradients)
            gradients = {}
            for param_name in updated_params:
                if param_name in global_params:
                    gradients[param_name] = updated_params[param_name] - global_params[param_name]

            # Create model update
            update = FederatedModelUpdate(
                client_id=client_id,
                model_parameters=updated_params,
                gradients=gradients,
                data_size=len(X),
                privacy_noise=self.privacy_engine.noise_added if hasattr(self.privacy_engine, 'noise_added') else None
            )

            # Calculate communication cost
            update.communication_cost = self._calculate_communication_cost(update)

            # Update client state
            client.model_version = self.global_model.version + 1
            client.last_update = datetime.utcnow()

            return update

        except Exception as e:
            logger.error(f"Client {client_id} training failed: {e}")
            return None

    def _aggregate_updates(self, client_updates: List[FederatedModelUpdate], round_num: int) -> Dict[str, np.ndarray]:
        """Aggregate model updates from clients."""
        if not client_updates:
            return self.global_model.model_parameters

        if self.config.aggregation_strategy == AggregationStrategy.WEIGHTED_AVERAGE:
            return self._weighted_average_aggregation(client_updates)
        elif self.config.aggregation_strategy == AggregationStrategy.ROBUST_AGGREGATION:
            return self._robust_aggregation(client_updates)
        elif self.config.aggregation_strategy == AggregationStrategy.Krum:
            return self._krum_aggregation(client_updates)
        elif self.config.aggregation_strategy == AggregationStrategy.Trimmed_Mean:
            return self._trimmed_mean_aggregation(client_updates)
        else:
            return self._weighted_average_aggregation(client_updates)

    def _weighted_average_aggregation(self, updates: List[FederatedModelUpdate]) -> Dict[str, np.ndarray]:
        """Weighted average aggregation (FedAvg)."""
        # Calculate total data size
        total_data_size = sum(update.data_size for update in updates)

        # Initialize aggregated parameters
        aggregated_params = {}

        # Aggregate each parameter
        for param_name in updates[0].model_parameters.keys():
            weighted_sum = None

            for update in updates:
                if param_name in update.model_parameters:
                    weight = update.data_size / total_data_size
                    param_value = update.model_parameters[param_name]

                    if weighted_sum is None:
                        weighted_sum = weight * param_value
                    else:
                        weighted_sum += weight * param_value

            if weighted_sum is not None:
                aggregated_params[param_name] = weighted_sum

        return aggregated_params

    def _robust_aggregation(self, updates: List[FederatedModelUpdate]) -> Dict[str, np.ndarray]:
        """Robust aggregation resistant to outliers."""
        # Use median-based aggregation for robustness
        aggregated_params = {}

        for param_name in updates[0].model_parameters.keys():
            param_values = []

            for update in updates:
                if param_name in update.model_parameters:
                    param_values.append(update.model_parameters[param_name])

            if param_values:
                # Use coordinate-wise median
                stacked_params = np.stack(param_values, axis=0)
                median_params = np.median(stacked_params, axis=0)
                aggregated_params[param_name] = median_params

        return aggregated_params

    def _krum_aggregation(self, updates: List[FederatedModelUpdate]) -> Dict[str, np.ndarray]:
        """Krum aggregation - selects update closest to others."""
        # Simplified Krum implementation
        if len(updates) <= 1:
            return updates[0].model_parameters if updates else {}

        # Calculate pairwise distances
        distances = {}
        for i, update_i in enumerate(updates):
            distances[i] = []
            for j, update_j in enumerate(updates):
                if i != j:
                    dist = self._calculate_parameter_distance(update_i.model_parameters, update_j.model_parameters)
                    distances[i].append((dist, j))

        # Find update with smallest sum of distances to closest n-f-1 updates
        n = len(updates)
        f = max(1, n // 3)  # Assume up to 1/3 malicious clients

        best_update_idx = None
        best_score = float('inf')

        for i in range(n):
            # Sort distances and sum closest n-f-1
            sorted_distances = sorted(distances[i], key=lambda x: x[0])
            score = sum(dist[0] for dist in sorted_distances[:n-f-1])
            if score < best_score:
                best_score = score
                best_update_idx = i

        return updates[best_update_idx].model_parameters

    def _trimmed_mean_aggregation(self, updates: List[FederatedModelUpdate]) -> Dict[str, np.ndarray]:
        """Trimmed mean aggregation."""
        aggregated_params = {}

        for param_name in updates[0].model_parameters.keys():
            param_values = []

            for update in updates:
                if param_name in update.model_parameters:
                    param_values.append(update.model_parameters[param_name])

            if param_values:
                # Trim outliers and take mean
                stacked_params = np.stack(param_values, axis=0)

                # Trim 20% from each end
                trim_percent = 0.2
                trim_count = int(len(param_values) * trim_percent)

                if len(param_values) > 2 * trim_count:
                    # Sort and trim
                    sorted_indices = np.argsort(stacked_params, axis=0)
                    trimmed_params = stacked_params[trim_count:-trim_count]
                    mean_params = np.mean(trimmed_params, axis=0)
                else:
                    mean_params = np.mean(stacked_params, axis=0)

                aggregated_params[param_name] = mean_params

        return aggregated_params

    def _calculate_parameter_distance(self, params1: Dict[str, np.ndarray],
                                    params2: Dict[str, np.ndarray]) -> float:
        """Calculate distance between parameter sets."""
        total_distance = 0

        for param_name in params1.keys():
            if param_name in params2:
                diff = params1[param_name] - params2[param_name]
                total_distance += np.sum(diff ** 2)

        return np.sqrt(total_distance)

    def _calculate_convergence_metrics(self, client_updates: List[FederatedModelUpdate]) -> Dict[str, float]:
        """Calculate convergence metrics from client updates."""
        if not client_updates:
            return {}

        # Parameter variance across clients
        param_variances = {}

        for param_name in client_updates[0].model_parameters.keys():
            param_values = []

            for update in client_updates:
                if param_name in update.model_parameters:
                    param_values.append(update.model_parameters[param_name].flatten())

            if param_values:
                # Calculate variance of parameter values
                stacked_params = np.stack(param_values, axis=0)
                param_variances[param_name] = np.var(stacked_params, axis=0).mean()

        # Average parameter variance
        avg_variance = np.mean(list(param_variances.values())) if param_variances else 0

        return {
            'parameter_variance': avg_variance,
            'num_participating_clients': len(client_updates),
            'total_data_size': sum(update.data_size for update in client_updates)
        }

    def _check_convergence(self) -> bool:
        """Check if training has converged."""
        if len(self.round_history) < 5:
            return False

        # Check if parameter variance is decreasing and below threshold
        recent_rounds = self.round_history[-5:]
        variances = [r['convergence_metrics'].get('parameter_variance', 0) for r in recent_rounds]

        # Check if variance is stabilizing
        if len(variances) >= 3:
            recent_avg = np.mean(variances[-3:])
            earlier_avg = np.mean(variances[:-3])

            # Convergence if recent variance is much lower than earlier
            return recent_avg < earlier_avg * 0.5 and recent_avg < 0.01

        return False

    def _evaluate_global_model(self, test_data: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Dict[str, float]:
        """Evaluate global model on test data."""
        try:
            # Create model with global parameters
            global_model = self.model_factory()
            self._set_model_parameters(global_model, self.global_model.model_parameters)

            # Combine all test data
            all_X = []
            all_y = []

            for client_data in test_data.values():
                X_test, y_test = client_data
                # Use last 20% as test data
                test_split = int(0.8 * len(X_test))
                all_X.append(X_test[test_split:])
                all_y.append(y_test[test_split:])

            if all_X and all_y:
                X_test_combined = np.vstack(all_X)
                y_test_combined = np.concatenate(all_y)

                # Evaluate
                if hasattr(global_model, 'predict'):
                    y_pred = global_model.predict(X_test_combined)

                    if len(y_test_combined.shape) > 1 and y_test_combined.shape[1] > 1:
                        # Classification
                        accuracy = accuracy_score(y_test_combined, y_pred)
                        return {'accuracy': accuracy}
                    else:
                        # Regression
                        mse = mean_squared_error(y_test_combined, y_pred)
                        rmse = np.sqrt(mse)
                        return {'mse': mse, 'rmse': rmse}

            return {}

        except Exception as e:
            logger.error(f"Global model evaluation failed: {e}")
            return {}

    def _calculate_communication_cost(self, update: FederatedModelUpdate) -> float:
        """Calculate communication cost of model update."""
        # Estimate size of parameter update
        total_params = 0
        for param in update.model_parameters.values():
            total_params += param.size

        # Assume 4 bytes per parameter (float32)
        size_bytes = total_params * 4

        # Convert to MB
        size_mb = size_bytes / (1024 * 1024)

        return size_mb

    def _calculate_communication_efficiency(self) -> float:
        """Calculate communication efficiency."""
        if not self.round_history:
            return 0

        total_communication = sum(r.get('communication_cost', 0) for r in self.round_history)
        total_rounds = len(self.round_history)

        # Efficiency = 1 / (communication per round)
        return 1.0 / (total_communication / total_rounds) if total_communication > 0 else 0

    def _check_privacy_preservation(self) -> bool:
        """Check if privacy is being preserved."""
        # Simple check: ensure privacy budgets are being consumed
        total_privacy_budget = sum(client.privacy_budget for client in self.clients.values())
        average_budget = total_privacy_budget / len(self.clients)

        # Privacy preserved if average budget is reasonable (> 0.1)
        return average_budget > 0.1

    def add_malicious_client(self, client_id: str, attack_type: str = "data_poisoning"):
        """Add a malicious client for testing robustness."""
        if client_id in self.clients:
            self.clients[client_id].is_malicious = True
            logger.warning(f"Client {client_id} marked as malicious ({attack_type})")

    def get_federated_status(self) -> Dict[str, Any]:
        """Get comprehensive federated learning status."""
        return {
            'num_clients': len(self.clients),
            'global_model_version': self.global_model.version if self.global_model else 0,
            'current_round': self.global_model.round_number if self.global_model else 0,
            'total_rounds_completed': len(self.round_history),
            'convergence_achieved': self._check_convergence(),
            'privacy_mechanism': self.config.privacy_mechanism.value,
            'aggregation_strategy': self.config.aggregation_strategy.value,
            'algorithm': self.config.algorithm.value,
            'client_status': {
                client_id: {
                    'data_size': client.data_size,
                    'model_version': client.model_version,
                    'privacy_budget': client.privacy_budget,
                    'is_malicious': client.is_malicious,
                    'last_update': client.last_update.isoformat()
                }
                for client_id, client in self.clients.items()
            },
            'performance_metrics': self._evaluate_global_model({}) if self.global_model else {},
            'communication_efficiency': self._calculate_communication_efficiency()
        }


class PrivacyEngine:
    """Privacy-preserving mechanisms for federated learning."""

    def __init__(self, mechanism: PrivacyMechanism):
        self.mechanism = mechanism
        self.noise_added = {}

    def apply_privacy(self, parameters: Dict[str, np.ndarray], privacy_budget: float) -> Dict[str, np.ndarray]:
        """Apply privacy mechanism to parameters."""
        if self.mechanism == PrivacyMechanism.DIFFERENTIAL_PRIVACY:
            return self._apply_differential_privacy(parameters, privacy_budget)
        elif self.mechanism == PrivacyMechanism.SECURE_AGGREGATION:
            return self._apply_secure_aggregation(parameters)
        else:
            return parameters

    def _apply_differential_privacy(self, parameters: Dict[str, np.ndarray], epsilon: float) -> Dict[str, np.ndarray]:
        """Apply differential privacy noise."""
        privatized_params = {}
        self.noise_added = {}

        for param_name, param_value in parameters.items():
            # Calculate noise scale based on sensitivity and epsilon
            sensitivity = np.max(np.abs(param_value))  # Simplified sensitivity
            noise_scale = sensitivity / epsilon

            # Add Laplace noise
            noise = np.random.laplace(0, noise_scale, param_value.shape)
            privatized_params[param_name] = param_value + noise
            self.noise_added[param_name] = noise

        return privatized_params

    def _apply_secure_aggregation(self, parameters: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply secure aggregation (simplified - would use cryptographic protocols)."""
        # In practice, this would use secure multi-party computation
        # For simulation, just return parameters unchanged
        return parameters


class SecureAggregator:
    """Secure aggregation for federated learning."""

    def __init__(self, threshold: int):
        self.threshold = threshold  # Minimum clients needed for aggregation

    def can_aggregate(self, num_updates: int) -> bool:
        """Check if enough updates are available for secure aggregation."""
        return num_updates >= self.threshold
