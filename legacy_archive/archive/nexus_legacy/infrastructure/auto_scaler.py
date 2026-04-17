import logging
import asyncio
import aiohttp
import json
import time
import statistics
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import threading
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

class ScalingStrategy(Enum):
    HORIZONTAL = "horizontal"  # Scale by adding/removing instances
    VERTICAL = "vertical"     # Scale by changing instance size
    HYBRID = "hybrid"         # Combination of horizontal and vertical

class ScalingTrigger(Enum):
    CPU_UTILIZATION = "cpu_utilization"
    MEMORY_UTILIZATION = "memory_utilization"
    REQUEST_RATE = "request_rate"
    RESPONSE_TIME = "response_time"
    CUSTOM_METRIC = "custom_metric"
    PREDICTIVE = "predictive"

class ScalingDirection(Enum):
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    NO_CHANGE = "no_change"

@dataclass
class ScalingPolicy:
    """Represents a scaling policy for a service."""
    service_name: str
    min_instances: int = 1
    max_instances: int = 10
    target_utilization: float = 0.7
    scale_up_threshold: float = 0.8
    scale_down_threshold: float = 0.3
    cooldown_period: int = 300  # seconds
    scaling_strategy: ScalingStrategy = ScalingStrategy.HORIZONTAL
    triggers: List[ScalingTrigger] = field(default_factory=list)
    custom_metrics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ScalingEvent:
    """Represents a scaling event."""
    event_id: str
    service_name: str
    direction: ScalingDirection
    old_instances: int
    new_instances: int
    trigger: ScalingTrigger
    trigger_value: float
    timestamp: datetime
    reason: str
    success: bool = False

@dataclass
class LoadBalancerConfig:
    """Configuration for load balancer."""
    algorithm: str = "round_robin"  # round_robin, least_connections, ip_hash, etc.
    health_check_interval: int = 30
    health_check_timeout: int = 5
    max_connections: int = 1000
    session_sticky: bool = False

class InstitutionalAutoScaler:
    """
    INSTITUTIONAL-GRADE AUTOMATED SCALING AND LOAD BALANCING SYSTEM
    Intelligent auto-scaling based on multiple metrics and predictive algorithms.
    Advanced load balancing with health checks and traffic optimization.
    """

    def __init__(self, config_file: str = "configs/autoscaling.yaml"):
        self.config_file = Path(config_file)
        self.config = self._load_config()

        # Scaling policies
        self.scaling_policies: Dict[str, ScalingPolicy] = {}
        self.scaling_history: List[ScalingEvent] = []
        self.cooldown_timers: Dict[str, datetime] = {}

        # Load balancer
        self.load_balancer = LoadBalancer(
            config=self.config.get('load_balancer', LoadBalancerConfig())
        )

        # Metrics collectors
        self.metrics_collectors: Dict[str, Callable] = {}
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

        # Predictive scaling
        self.predictive_models: Dict[str, Any] = {}
        self.forecasting_enabled = self.config.get('predictive_scaling', {}).get('enabled', False)

        # Performance baselines
        self.baselines: Dict[str, Dict[str, float]] = {}

        # Initialize components
        self._initialize_scaling_policies()
        self._initialize_metrics_collectors()
        self._initialize_predictive_models()

        logger.info("Institutional Auto-Scaler initialized")

    def _load_config(self) -> Dict[str, Any]:
        """Load autoscaling configuration."""
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                import yaml
                return yaml.safe_load(f)
        else:
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default autoscaling configuration."""
        return {
            'global_settings': {
                'evaluation_interval': 60,  # seconds
                'max_scale_up_percent': 100,  # Max 100% increase at once
                'max_scale_down_percent': 50,  # Max 50% decrease at once
                'enable_predictive_scaling': True,
                'forecast_horizon': 300  # 5 minutes ahead
            },
            'load_balancer': {
                'algorithm': 'least_connections',
                'health_check_interval': 30,
                'health_check_timeout': 5,
                'max_connections': 10000,
                'session_sticky': False
            },
            'predictive_scaling': {
                'enabled': True,
                'model_type': 'arima',
                'training_window': 3600,  # 1 hour
                'prediction_interval': 300  # 5 minutes
            },
            'services': {
                'data-router': {
                    'min_instances': 3,
                    'max_instances': 20,
                    'target_utilization': 0.7,
                    'triggers': ['cpu_utilization', 'request_rate']
                },
                'strategy-engine': {
                    'min_instances': 5,
                    'max_instances': 50,
                    'target_utilization': 0.8,
                    'triggers': ['cpu_utilization', 'memory_utilization', 'custom_metric']
                },
                'execution-simulator': {
                    'min_instances': 2,
                    'max_instances': 15,
                    'target_utilization': 0.6,
                    'triggers': ['cpu_utilization', 'response_time']
                }
            }
        }

    def _initialize_scaling_policies(self):
        """Initialize scaling policies for all services."""
        services_config = self.config.get('services', {})

        for service_name, service_config in services_config.items():
            triggers = [ScalingTrigger(t) for t in service_config.get('triggers', [])]

            policy = ScalingPolicy(
                service_name=service_name,
                min_instances=service_config.get('min_instances', 1),
                max_instances=service_config.get('max_instances', 10),
                target_utilization=service_config.get('target_utilization', 0.7),
                scale_up_threshold=service_config.get('scale_up_threshold', 0.8),
                scale_down_threshold=service_config.get('scale_down_threshold', 0.3),
                cooldown_period=service_config.get('cooldown_period', 300),
                triggers=triggers
            )

            self.scaling_policies[service_name] = policy

    def _initialize_metrics_collectors(self):
        """Initialize metrics collectors for different services."""
        self.metrics_collectors = {
            'cpu_utilization': self._collect_cpu_metrics,
            'memory_utilization': self._collect_memory_metrics,
            'request_rate': self._collect_request_rate_metrics,
            'response_time': self._collect_response_time_metrics,
            'custom_metric': self._collect_custom_metrics
        }

    def _initialize_predictive_models(self):
        """Initialize predictive scaling models."""
        if not self.forecasting_enabled:
            return

        # Initialize forecasting models for each service
        for service_name in self.scaling_policies.keys():
            try:
                from statsmodels.tsa.arima.model import ARIMA
                self.predictive_models[service_name] = {
                    'model': ARIMA,
                    'trained': False,
                    'last_training': None
                }
            except ImportError:
                logger.warning("statsmodels not available, predictive scaling disabled")
                self.forecasting_enabled = False
                break

    async def start_autoscaling(self):
        """Start the autoscaling system."""
        logger.info("Starting autoscaling system...")

        # Start metrics collection
        asyncio.create_task(self._collect_metrics_loop())

        # Start scaling evaluation
        asyncio.create_task(self._evaluate_scaling_loop())

        # Start predictive scaling if enabled
        if self.forecasting_enabled:
            asyncio.create_task(self._predictive_scaling_loop())

        # Start load balancer
        asyncio.create_task(self.load_balancer.start())

        logger.info("Autoscaling system started")

    def add_scaling_policy(self, policy: ScalingPolicy):
        """Add or update a scaling policy."""
        self.scaling_policies[policy.service_name] = policy
        logger.info(f"Added scaling policy for {policy.service_name}")

    def get_scaling_decision(self, service_name: str, current_metrics: Dict[str, float]) -> Tuple[ScalingDirection, int, str]:
        """
        Get scaling decision for a service based on current metrics.
        Returns (direction, target_instances, reason)
        """
        if service_name not in self.scaling_policies:
            return ScalingDirection.NO_CHANGE, 0, "No scaling policy defined"

        policy = self.scaling_policies[service_name]

        # Check cooldown
        if service_name in self.cooldown_timers:
            if datetime.utcnow() < self.cooldown_timers[service_name]:
                return ScalingDirection.NO_CHANGE, 0, "In cooldown period"

        # Get current instance count (would come from actual monitoring)
        current_instances = self._get_current_instance_count(service_name)

        # Evaluate triggers
        scale_up_signals = 0
        scale_down_signals = 0
        reasons = []

        for trigger in policy.triggers:
            trigger_value = current_metrics.get(trigger.value, 0)
            decision, reason = self._evaluate_trigger(trigger, trigger_value, policy)

            if decision == ScalingDirection.SCALE_UP:
                scale_up_signals += 1
                reasons.append(f"{trigger.value}: {reason}")
            elif decision == ScalingDirection.SCALE_DOWN:
                scale_down_signals += 1
                reasons.append(f"{trigger.value}: {reason}")

        # Make final decision
        if scale_up_signals > scale_down_signals:
            target_instances = self._calculate_target_instances(
                current_instances, ScalingDirection.SCALE_UP, policy
            )
            return ScalingDirection.SCALE_UP, target_instances, "; ".join(reasons)
        elif scale_down_signals > scale_up_signals:
            target_instances = self._calculate_target_instances(
                current_instances, ScalingDirection.SCALE_DOWN, policy
            )
            return ScalingDirection.SCALE_DOWN, target_instances, "; ".join(reasons)
        else:
            return ScalingDirection.NO_CHANGE, current_instances, "Metrics within target range"

    def _evaluate_trigger(self, trigger: ScalingTrigger, value: float,
                         policy: ScalingPolicy) -> Tuple[ScalingDirection, str]:
        """Evaluate a single scaling trigger."""
        if trigger == ScalingTrigger.CPU_UTILIZATION:
            if value > policy.scale_up_threshold:
                return ScalingDirection.SCALE_UP, f"CPU {value:.1%} > {policy.scale_up_threshold:.1%}"
            elif value < policy.scale_down_threshold:
                return ScalingDirection.SCALE_DOWN, f"CPU {value:.1%} < {policy.scale_down_threshold:.1%}"

        elif trigger == ScalingTrigger.MEMORY_UTILIZATION:
            if value > policy.scale_up_threshold:
                return ScalingDirection.SCALE_UP, f"Memory {value:.1%} > {policy.scale_up_threshold:.1%}"
            elif value < policy.scale_down_threshold:
                return ScalingDirection.SCALE_DOWN, f"Memory {value:.1%} < {policy.scale_down_threshold:.1%}"

        elif trigger == ScalingTrigger.REQUEST_RATE:
            # Higher request rate = scale up
            baseline = self.baselines.get(policy.service_name, {}).get('request_rate', 100)
            if value > baseline * 1.5:
                return ScalingDirection.SCALE_UP, f"Requests {value:.0f} > baseline {baseline:.0f}"
            elif value < baseline * 0.5:
                return ScalingDirection.SCALE_DOWN, f"Requests {value:.0f} < baseline {baseline:.0f}"

        elif trigger == ScalingTrigger.RESPONSE_TIME:
            # Higher response time = scale up
            baseline = self.baselines.get(policy.service_name, {}).get('response_time', 1.0)
            if value > baseline * 1.5:
                return ScalingDirection.SCALE_UP, f"Response time {value:.2f}s > baseline {baseline:.2f}s"
            elif value < baseline * 0.7:
                return ScalingDirection.SCALE_DOWN, f"Response time {value:.2f}s < baseline {baseline:.2f}s"

        return ScalingDirection.NO_CHANGE, "Within normal range"

    def _calculate_target_instances(self, current: int, direction: ScalingDirection,
                                  policy: ScalingPolicy) -> int:
        """Calculate target number of instances."""
        if direction == ScalingDirection.SCALE_UP:
            # Scale up aggressively but with limits
            max_increase = max(1, int(current * self.config['global_settings']['max_scale_up_percent'] / 100))
            target = min(current + max_increase, policy.max_instances)
        else:
            # Scale down conservatively
            max_decrease = max(1, int(current * self.config['global_settings']['max_scale_down_percent'] / 100))
            target = max(current - max_decrease, policy.min_instances)

        return target

    async def execute_scaling(self, service_name: str, direction: ScalingDirection,
                            target_instances: int, reason: str) -> bool:
        """Execute scaling action."""
        try:
            current_instances = self._get_current_instance_count(service_name)

            if target_instances == current_instances:
                return True

            # Create scaling event
            event = ScalingEvent(
                event_id=f"{service_name}_{int(time.time())}",
                service_name=service_name,
                direction=direction,
                old_instances=current_instances,
                new_instances=target_instances,
                trigger=ScalingTrigger.CUSTOM_METRIC,  # Would be passed from decision
                trigger_value=0.0,  # Would be passed from decision
                timestamp=datetime.utcnow(),
                reason=reason
            )

            # Execute scaling (would integrate with Kubernetes/AWS/GCP APIs)
            success = await self._scale_service_instances(service_name, target_instances)

            event.success = success
            self.scaling_history.append(event)

            if success:
                # Set cooldown
                self.cooldown_timers[service_name] = (
                    datetime.utcnow() + timedelta(seconds=self.scaling_policies[service_name].cooldown_period)
                )

                logger.info(f"Successfully scaled {service_name} from {current_instances} to {target_instances} instances")
            else:
                logger.error(f"Failed to scale {service_name}")

            return success

        except Exception as e:
            logger.error(f"Error executing scaling for {service_name}: {e}")
            return False

    async def _scale_service_instances(self, service_name: str, target_instances: int) -> bool:
        """Scale service instances (integrates with infrastructure APIs)."""
        try:
            # This would integrate with Kubernetes, AWS ECS, GCP Cloud Run, etc.
            # For now, simulate scaling
            await asyncio.sleep(2)  # Simulate API call delay

            # In real implementation:
            # - Kubernetes: kubectl scale deployment
            # - AWS: update ECS service desired count
            # - GCP: update Cloud Run revision

            return True

        except Exception as e:
            logger.error(f"Scaling API call failed: {e}")
            return False

    async def _collect_metrics_loop(self):
        """Continuously collect metrics for all services."""
        while True:
            try:
                for service_name in self.scaling_policies.keys():
                    metrics = await self._collect_service_metrics(service_name)

                    # Store metrics history
                    for metric_name, value in metrics.items():
                        key = f"{service_name}_{metric_name}"
                        self.metrics_history[key].append((datetime.utcnow(), value))

                await asyncio.sleep(self.config['global_settings']['evaluation_interval'])

            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
                await asyncio.sleep(self.config['global_settings']['evaluation_interval'])

    async def _collect_service_metrics(self, service_name: str) -> Dict[str, float]:
        """Collect metrics for a specific service."""
        metrics = {}

        try:
            # Collect standard metrics
            for trigger in self.scaling_policies[service_name].triggers:
                if trigger in self.metrics_collectors:
                    value = await self.metrics_collectors[trigger](service_name)
                    metrics[trigger.value] = value

            # Collect custom metrics
            custom_metrics = self.scaling_policies[service_name].custom_metrics
            for metric_name, config in custom_metrics.items():
                value = await self._collect_custom_metric(service_name, metric_name, config)
                metrics[metric_name] = value

        except Exception as e:
            logger.error(f"Error collecting metrics for {service_name}: {e}")

        return metrics

    async def _collect_cpu_metrics(self, service_name: str) -> float:
        """Collect CPU utilization metrics."""
        # This would integrate with monitoring APIs (Prometheus, CloudWatch, etc.)
        # Simulate realistic CPU usage
        base_cpu = self.baselines.get(service_name, {}).get('cpu_utilization', 0.6)
        variation = np.random.normal(0, 0.1)
        return np.clip(base_cpu + variation, 0, 1)

    async def _collect_memory_metrics(self, service_name: str) -> float:
        """Collect memory utilization metrics."""
        base_memory = self.baselines.get(service_name, {}).get('memory_utilization', 0.7)
        variation = np.random.normal(0, 0.05)
        return np.clip(base_memory + variation, 0, 1)

    async def _collect_request_rate_metrics(self, service_name: str) -> float:
        """Collect request rate metrics."""
        # Simulate request rate based on service type
        if service_name == 'data-router':
            return np.random.normal(500, 100)
        elif service_name == 'strategy-engine':
            return np.random.normal(200, 50)
        else:
            return np.random.normal(100, 20)

    async def _collect_response_time_metrics(self, service_name: str) -> float:
        """Collect response time metrics."""
        base_time = self.baselines.get(service_name, {}).get('response_time', 1.0)
        variation = np.random.normal(0, 0.2)
        return max(0.1, base_time + variation)

    async def _collect_custom_metrics(self, service_name: str) -> float:
        """Collect custom metrics."""
        # Placeholder for custom metric collection
        return 0.5

    async def _collect_custom_metric(self, service_name: str, metric_name: str, config: Dict[str, Any]) -> float:
        """Collect a custom metric based on configuration."""
        # This would implement custom metric collection logic
        return np.random.random()

    async def _evaluate_scaling_loop(self):
        """Continuously evaluate scaling decisions."""
        while True:
            try:
                for service_name, policy in self.scaling_policies.items():
                    # Get current metrics
                    current_metrics = {}
                    for trigger in policy.triggers:
                        key = f"{service_name}_{trigger.value}"
                        if key in self.metrics_history and self.metrics_history[key]:
                            current_metrics[trigger.value] = self.metrics_history[key][-1][1]

                    if current_metrics:
                        # Get scaling decision
                        direction, target_instances, reason = self.get_scaling_decision(
                            service_name, current_metrics
                        )

                        if direction != ScalingDirection.NO_CHANGE:
                            # Execute scaling
                            await self.execute_scaling(service_name, direction, target_instances, reason)

                await asyncio.sleep(self.config['global_settings']['evaluation_interval'])

            except Exception as e:
                logger.error(f"Error in scaling evaluation loop: {e}")
                await asyncio.sleep(self.config['global_settings']['evaluation_interval'])

    async def _predictive_scaling_loop(self):
        """Run predictive scaling analysis."""
        while True:
            try:
                for service_name in self.scaling_policies.keys():
                    prediction = await self._predict_future_load(service_name)

                    if prediction:
                        # Adjust scaling decision based on prediction
                        await self._apply_predictive_scaling(service_name, prediction)

                await asyncio.sleep(self.config['predictive_scaling']['prediction_interval'])

            except Exception as e:
                logger.error(f"Error in predictive scaling loop: {e}")
                await asyncio.sleep(self.config['predictive_scaling']['prediction_interval'])

    async def _predict_future_load(self, service_name: str) -> Optional[Dict[str, Any]]:
        """Predict future load for a service."""
        try:
            # Get historical metrics
            cpu_history = [v for t, v in self.metrics_history.get(f"{service_name}_cpu_utilization", [])]
            request_history = [v for t, v in self.metrics_history.get(f"{service_name}_request_rate", [])]

            if len(cpu_history) < 50:
                return None

            # Simple forecasting using moving average + trend
            window = min(20, len(cpu_history))
            recent_avg = statistics.mean(cpu_history[-window:])
            trend = statistics.mean(cpu_history[-window:]) - statistics.mean(cpu_history[-2*window:-window])

            predicted_cpu = recent_avg + trend
            predicted_requests = statistics.mean(request_history[-window:]) * (1 + trend)

            return {
                'predicted_cpu': predicted_cpu,
                'predicted_requests': predicted_requests,
                'confidence': 0.7,
                'horizon': self.config['predictive_scaling']['prediction_interval']
            }

        except Exception as e:
            logger.error(f"Error predicting load for {service_name}: {e}")
            return None

    async def _apply_predictive_scaling(self, service_name: str, prediction: Dict[str, Any]):
        """Apply predictive scaling adjustments."""
        try:
            policy = self.scaling_policies[service_name]
            current_instances = self._get_current_instance_count(service_name)

            # Proactive scaling based on prediction
            predicted_cpu = prediction['predicted_cpu']

            if predicted_cpu > policy.scale_up_threshold * 1.2:  # 20% buffer
                # Scale up preemptively
                target_instances = min(current_instances + 2, policy.max_instances)
                await self.execute_scaling(
                    service_name, ScalingDirection.SCALE_UP, target_instances,
                    f"Predictive scaling: CPU predicted at {predicted_cpu:.1%}"
                )

        except Exception as e:
            logger.error(f"Error applying predictive scaling for {service_name}: {e}")

    def _get_current_instance_count(self, service_name: str) -> int:
        """Get current instance count for a service."""
        # This would integrate with infrastructure APIs
        # For now, return a simulated value
        return getattr(self, f'{service_name}_instances', 3)

    def get_scaling_status(self) -> Dict[str, Any]:
        """Get comprehensive scaling status."""
        return {
            'policies': {
                name: {
                    'min_instances': policy.min_instances,
                    'max_instances': policy.max_instances,
                    'current_instances': self._get_current_instance_count(name),
                    'target_utilization': policy.target_utilization,
                    'cooldown_remaining': (
                        (self.cooldown_timers.get(name, datetime.utcnow()) - datetime.utcnow()).total_seconds()
                        if name in self.cooldown_timers else 0
                    )
                }
                for name, policy in self.scaling_policies.items()
            },
            'recent_events': [
                {
                    'service': event.service_name,
                    'direction': event.direction.value,
                    'old_instances': event.old_instances,
                    'new_instances': event.new_instances,
                    'timestamp': event.timestamp.isoformat(),
                    'success': event.success,
                    'reason': event.reason
                }
                for event in self.scaling_history[-10:]  # Last 10 events
            ],
            'load_balancer_status': self.load_balancer.get_status(),
            'predictive_enabled': self.forecasting_enabled
        }

    def update_baselines(self, service_name: str, metrics: Dict[str, float]):
        """Update performance baselines for a service."""
        if service_name not in self.baselines:
            self.baselines[service_name] = {}

        for metric_name, value in metrics.items():
            self.baselines[service_name][metric_name] = value

        logger.info(f"Updated baselines for {service_name}: {metrics}")


class LoadBalancer:
    """Advanced load balancer with health checks and multiple algorithms."""

    def __init__(self, config: LoadBalancerConfig):
        self.config = config
        self.backends: Dict[str, Dict[str, Any]] = {}
        self.health_status: Dict[str, bool] = {}
        self.connection_counts: Dict[str, int] = defaultdict(int)
        self.session_sticky_table: Dict[str, str] = {}

    async def start(self):
        """Start the load balancer."""
        asyncio.create_task(self._health_check_loop())
        logger.info("Load balancer started")

    def add_backend(self, backend_id: str, address: str, port: int, weight: int = 1):
        """Add a backend server."""
        self.backends[backend_id] = {
            'address': address,
            'port': port,
            'weight': weight,
            'healthy': True
        }
        self.health_status[backend_id] = True

    def remove_backend(self, backend_id: str):
        """Remove a backend server."""
        if backend_id in self.backends:
            del self.backends[backend_id]
            del self.health_status[backend_id]
            if backend_id in self.connection_counts:
                del self.connection_counts[backend_id]

    def get_backend(self, client_ip: str = None) -> Optional[Tuple[str, str, int]]:
        """Get the next backend using the configured algorithm."""
        healthy_backends = [
            (bid, backend) for bid, backend in self.backends.items()
            if self.health_status.get(bid, False)
        ]

        if not healthy_backends:
            return None

        if self.config.algorithm == 'round_robin':
            return self._round_robin_selection(healthy_backends)
        elif self.config.algorithm == 'least_connections':
            return self._least_connections_selection(healthy_backends)
        elif self.config.algorithm == 'ip_hash':
            return self._ip_hash_selection(healthy_backends, client_ip)
        elif self.config.algorithm == 'weighted_round_robin':
            return self._weighted_round_robin_selection(healthy_backends)
        else:
            return self._round_robin_selection(healthy_backends)

    def _round_robin_selection(self, backends: List[Tuple[str, Dict]]) -> Tuple[str, str, int]:
        """Round-robin backend selection."""
        # Simple round-robin (would need persistent counter in production)
        backend_id, backend = backends[0]  # Simplified
        return backend_id, backend['address'], backend['port']

    def _least_connections_selection(self, backends: List[Tuple[str, Dict]]) -> Tuple[str, str, int]:
        """Least connections backend selection."""
        # Find backend with least connections
        min_connections = min(self.connection_counts.get(bid, 0) for bid, _ in backends)
        candidates = [(bid, backend) for bid, backend in backends
                     if self.connection_counts.get(bid, 0) == min_connections]

        backend_id, backend = candidates[0]
        return backend_id, backend['address'], backend['port']

    def _ip_hash_selection(self, backends: List[Tuple[str, Dict]], client_ip: str) -> Tuple[str, str, int]:
        """IP hash backend selection."""
        if not client_ip:
            return self._round_robin_selection(backends)

        # Hash IP to select backend
        hash_value = hash(client_ip) % len(backends)
        backend_id, backend = backends[hash_value]
        return backend_id, backend['address'], backend['port']

    def _weighted_round_robin_selection(self, backends: List[Tuple[str, Dict]]) -> Tuple[str, str, int]:
        """Weighted round-robin backend selection."""
        # Select based on weights
        total_weight = sum(backend['weight'] for _, backend in backends)
        rand_value = np.random.uniform(0, total_weight)

        current_weight = 0
        for backend_id, backend in backends:
            current_weight += backend['weight']
            if rand_value <= current_weight:
                return backend_id, backend['address'], backend['port']

        # Fallback
        backend_id, backend = backends[0]
        return backend_id, backend['address'], backend['port']

    async def _health_check_loop(self):
        """Continuously check backend health."""
        while True:
            try:
                for backend_id, backend in self.backends.items():
                    healthy = await self._check_backend_health(backend)
                    self.health_status[backend_id] = healthy

                    if not healthy:
                        logger.warning(f"Backend {backend_id} is unhealthy")

                await asyncio.sleep(self.config.health_check_interval)

            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(self.config.health_check_interval)

    async def _check_backend_health(self, backend: Dict[str, Any]) -> bool:
        """Check health of a backend server."""
        try:
            address = backend['address']
            port = backend['port']

            # Simple TCP health check (would be HTTP health check in production)
            reader, writer = await asyncio.open_connection(address, port)
            writer.close()
            await writer.wait_closed()
            return True

        except Exception:
            return False

    def record_connection(self, backend_id: str):
        """Record a new connection to a backend."""
        self.connection_counts[backend_id] += 1

    def remove_connection(self, backend_id: str):
        """Remove a connection from a backend."""
        if self.connection_counts[backend_id] > 0:
            self.connection_counts[backend_id] -= 1

    def get_status(self) -> Dict[str, Any]:
        """Get load balancer status."""
        return {
            'total_backends': len(self.backends),
            'healthy_backends': sum(1 for status in self.health_status.values() if status),
            'connection_counts': dict(self.connection_counts),
            'algorithm': self.config.algorithm,
            'session_sticky': self.config.session_sticky
        }
