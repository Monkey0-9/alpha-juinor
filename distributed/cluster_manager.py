import logging
import asyncio
import aiohttp
import json
import psutil
import platform
from typing import Dict, List, Optional, Any, Callable, Awaitable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import uuid
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

logger = logging.getLogger(__name__)

class NodeStatus(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    OVERLOADED = "overloaded"
    MAINTENANCE = "maintenance"
    FAILED = "failed"

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TaskPriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class ClusterNode:
    """Represents a node in the distributed cluster."""
    node_id: str
    hostname: str
    ip_address: str
    port: int
    status: NodeStatus
    capabilities: Dict[str, Any]
    resources: Dict[str, float]  # CPU, Memory, GPU, etc.
    last_heartbeat: datetime
    tasks_running: int
    tasks_completed: int
    uptime: float

    def __post_init__(self):
        if not self.node_id:
            self.node_id = str(uuid.uuid4())

    def is_available(self) -> bool:
        """Check if node is available for new tasks."""
        return (self.status == NodeStatus.ACTIVE and
                self.tasks_running < self.capabilities.get('max_concurrent_tasks', 10) and
                self.resources.get('cpu_percent', 0) < 90.0 and
                self.resources.get('memory_percent', 0) < 90.0)

    def get_load_score(self) -> float:
        """Calculate load score (lower is better)."""
        cpu_load = self.resources.get('cpu_percent', 0) / 100.0
        memory_load = self.resources.get('memory_percent', 0) / 100.0
        task_load = self.tasks_running / max(1, self.capabilities.get('max_concurrent_tasks', 10))

        return (cpu_load + memory_load + task_load) / 3.0

@dataclass
class DistributedTask:
    """Represents a task to be executed in the cluster."""
    task_id: str
    task_type: str
    priority: TaskPriority
    payload: Dict[str, Any]
    created_at: datetime
    timeout: Optional[float]
    dependencies: List[str]  # Task IDs this task depends on
    assigned_node: Optional[str]
    status: TaskStatus
    progress: float
    result: Optional[Any]
    error: Optional[str]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]

    def __post_init__(self):
        if not self.task_id:
            self.task_id = str(uuid.uuid4())
        if not self.created_at:
            self.created_at = datetime.utcnow()

    def is_expired(self) -> bool:
        """Check if task has exceeded timeout."""
        if not self.timeout or not self.started_at:
            return False
        return (datetime.utcnow() - self.started_at).total_seconds() > self.timeout

    def can_start(self, completed_tasks: List[str]) -> bool:
        """Check if all dependencies are satisfied."""
        return all(dep in completed_tasks for dep in self.dependencies)

class InstitutionalClusterManager:
    """
    INSTITUTIONAL-GRADE DISTRIBUTED COMPUTING SYSTEM
    Manages a cluster of computing nodes for scalable trading operations.
    Supports task distribution, load balancing, fault tolerance, and monitoring.
    """

    def __init__(self, coordinator_host: str = "localhost", coordinator_port: int = 8080,
                 node_id: Optional[str] = None, is_coordinator: bool = False):
        self.coordinator_host = coordinator_host
        self.coordinator_port = coordinator_port
        self.node_id = node_id or str(uuid.uuid4())
        self.is_coordinator = is_coordinator

        # Cluster state
        self.nodes: Dict[str, ClusterNode] = {}
        self.tasks: Dict[str, DistributedTask] = {}
        self.task_queue: List[DistributedTask] = []

        # Node capabilities and resources
        self.capabilities = self._detect_capabilities()
        self.resources = self._get_system_resources()

        # Task execution
        self.task_executor = ThreadPoolExecutor(max_workers=self.capabilities.get('max_threads', 4))
        self.process_executor = ProcessPoolExecutor(max_workers=self.capabilities.get('max_processes', 2))

        # Task handlers
        self.task_handlers: Dict[str, Callable] = {}
        self._register_default_handlers()

        # Networking
        self.http_session: Optional[aiohttp.ClientSession] = None
        self.heartbeat_interval = 30  # seconds
        self.node_timeout = 120  # seconds

        # Monitoring
        self.metrics = {
            'tasks_submitted': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'avg_execution_time': 0.0,
            'node_failures': 0
        }

        # Initialize
        if is_coordinator:
            self._initialize_as_coordinator()
        else:
            self._initialize_as_worker()

        logger.info(f"Cluster Manager initialized (Node: {self.node_id}, Coordinator: {is_coordinator})")

    def _detect_capabilities(self) -> Dict[str, Any]:
        """Detect node capabilities."""
        capabilities = {
            'platform': platform.system(),
            'architecture': platform.machine(),
            'cpu_count': psutil.cpu_count(),
            'cpu_logical': psutil.cpu_count(logical=True),
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'has_gpu': False,
            'max_threads': min(16, psutil.cpu_count(logical=True) * 2),
            'max_processes': min(4, psutil.cpu_count()),
            'max_concurrent_tasks': 10
        }

        # GPU detection (simplified)
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                capabilities['has_gpu'] = True
                capabilities['gpu_count'] = len(gpus)
                capabilities['gpu_memory_gb'] = sum(gpu.memoryTotal for gpu in gpus) / 1024
        except ImportError:
            pass

        return capabilities

    def _get_system_resources(self) -> Dict[str, float]:
        """Get current system resource usage."""
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_used_gb': psutil.virtual_memory().used / (1024**3),
            'disk_percent': psutil.disk_usage('/').percent,
            'network_connections': len(psutil.net_connections())
        }

    def _register_default_handlers(self):
        """Register default task handlers."""
        self.register_task_handler('backtest', self._handle_backtest_task)
        self.register_task_handler('signal_generation', self._handle_signal_task)
        self.register_task_handler('portfolio_optimization', self._handle_optimization_task)
        self.register_task_handler('risk_calculation', self._handle_risk_task)
        self.register_task_handler('data_processing', self._handle_data_task)

    def register_task_handler(self, task_type: str, handler: Callable[[Dict[str, Any]], Any]):
        """Register a handler for a specific task type."""
        self.task_handlers[task_type] = handler
        logger.info(f"Registered handler for task type: {task_type}")

    async def submit_task(self, task_type: str, payload: Dict[str, Any],
                         priority: TaskPriority = TaskPriority.NORMAL,
                         timeout: Optional[float] = None,
                         dependencies: List[str] = None) -> str:
        """
        Submit a task to the cluster for execution.
        Returns task ID.
        """
        task = DistributedTask(
            task_id="",
            task_type=task_type,
            priority=priority,
            payload=payload,
            created_at=datetime.utcnow(),
            timeout=timeout,
            dependencies=dependencies or [],
            assigned_node=None,
            status=TaskStatus.PENDING,
            progress=0.0,
            result=None,
            error=None,
            started_at=None,
            completed_at=None
        )

        self.tasks[task.task_id] = task
        self.metrics['tasks_submitted'] += 1

        if self.is_coordinator:
            self.task_queue.append(task)
            await self._schedule_task(task)
        else:
            # Send to coordinator
            await self._send_task_to_coordinator(task)

        logger.info(f"Task submitted: {task.task_id} ({task_type})")
        return task.task_id

    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a task."""
        task = self.tasks.get(task_id)
        if task:
            return {
                'task_id': task.task_id,
                'status': task.status.value,
                'progress': task.progress,
                'result': task.result,
                'error': task.error,
                'assigned_node': task.assigned_node,
                'started_at': task.started_at.isoformat() if task.started_at else None,
                'completed_at': task.completed_at.isoformat() if task.completed_at else None
            }
        return None

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task."""
        task = self.tasks.get(task_id)
        if task and task.status in [TaskStatus.PENDING, TaskStatus.RUNNING]:
            task.status = TaskStatus.CANCELLED
            if task.assigned_node and task.assigned_node != self.node_id:
                await self._send_cancel_to_node(task.assigned_node, task_id)
            logger.info(f"Task cancelled: {task_id}")
            return True
        return False

    async def get_cluster_status(self) -> Dict[str, Any]:
        """Get overall cluster status."""
        active_nodes = [node for node in self.nodes.values() if node.status == NodeStatus.ACTIVE]
        total_resources = self._aggregate_resources(active_nodes)

        return {
            'coordinator': self.is_coordinator,
            'node_id': self.node_id,
            'active_nodes': len(active_nodes),
            'total_nodes': len(self.nodes),
            'pending_tasks': len([t for t in self.tasks.values() if t.status == TaskStatus.PENDING]),
            'running_tasks': len([t for t in self.tasks.values() if t.status == TaskStatus.RUNNING]),
            'completed_tasks': self.metrics['tasks_completed'],
            'failed_tasks': self.metrics['tasks_failed'],
            'cluster_resources': total_resources,
            'node_details': [asdict(node) for node in active_nodes]
        }

    async def start(self):
        """Start the cluster manager."""
        if not self.http_session:
            self.http_session = aiohttp.ClientSession()

        if self.is_coordinator:
            await self._start_coordinator()
        else:
            await self._start_worker()

    async def stop(self):
        """Stop the cluster manager."""
        if self.http_session:
            await self.http_session.close()

        self.task_executor.shutdown(wait=True)
        self.process_executor.shutdown(wait=True)

        logger.info("Cluster Manager stopped")

    async def _start_coordinator(self):
        """Start coordinator node operations."""
        logger.info("Starting coordinator node...")

        # Register self as coordinator
        self.nodes[self.node_id] = ClusterNode(
            node_id=self.node_id,
            hostname=socket.gethostname(),
            ip_address=self._get_local_ip(),
            port=self.coordinator_port,
            status=NodeStatus.ACTIVE,
            capabilities=self.capabilities,
            resources=self.resources,
            last_heartbeat=datetime.utcnow(),
            tasks_running=0,
            tasks_completed=0,
            uptime=0.0
        )

        # Start background tasks
        asyncio.create_task(self._coordinator_heartbeat())
        asyncio.create_task(self._task_scheduler())
        asyncio.create_task(self._cluster_monitor())

    async def _start_worker(self):
        """Start worker node operations."""
        logger.info("Starting worker node...")

        # Register with coordinator
        await self._register_with_coordinator()

        # Start background tasks
        asyncio.create_task(self._worker_heartbeat())
        asyncio.create_task(self._task_processor())

    async def _schedule_task(self, task: DistributedTask):
        """Schedule a task to an available node."""
        # Find best node for task
        best_node = self._select_best_node(task)

        if best_node:
            task.assigned_node = best_node.node_id
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.utcnow()

            # Send task to node
            await self._send_task_to_node(best_node, task)
            logger.info(f"Task {task.task_id} scheduled to node {best_node.node_id}")
        else:
            # No available nodes, keep in queue
            logger.warning(f"No available nodes for task {task.task_id}")

    def _select_best_node(self, task: DistributedTask) -> Optional[ClusterNode]:
        """Select the best node for a task based on load balancing."""
        available_nodes = [node for node in self.nodes.values() if node.is_available()]

        if not available_nodes:
            return None

        # Score nodes based on task requirements and node capabilities
        node_scores = []
        for node in available_nodes:
            score = self._calculate_node_score(node, task)
            node_scores.append((node, score))

        # Return node with lowest score (best fit)
        best_node, _ = min(node_scores, key=lambda x: x[1])
        return best_node

    def _calculate_node_score(self, node: ClusterNode, task: DistributedTask) -> float:
        """Calculate how well a node fits a task."""
        base_score = node.get_load_score()

        # Adjust for task-specific requirements
        if task.task_type == 'gpu_intensive' and not node.capabilities.get('has_gpu', False):
            base_score += 100  # Heavily penalize

        if task.task_type == 'memory_intensive':
            memory_penalty = node.resources.get('memory_percent', 0) / 100.0
            base_score += memory_penalty * 2

        # Adjust for task priority
        if task.priority == TaskPriority.CRITICAL:
            base_score *= 0.5  # Prefer critical tasks
        elif task.priority == TaskPriority.LOW:
            base_score *= 1.5  # Less preferred for low priority

        return base_score

    async def _process_task_locally(self, task: DistributedTask):
        """Process a task on the local node."""
        try:
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.utcnow()

            # Get task handler
            handler = self.task_handlers.get(task.task_type)
            if not handler:
                raise ValueError(f"No handler registered for task type: {task.task_type}")

            # Execute task
            if task.task_type in ['backtest', 'data_processing']:
                # CPU-intensive tasks use process pool
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self.process_executor,
                    handler,
                    task.payload
                )
            else:
                # Other tasks use thread pool
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self.task_executor,
                    handler,
                    task.payload
                )

            # Update task
            task.status = TaskStatus.COMPLETED
            task.result = result
            task.completed_at = datetime.utcnow()
            task.progress = 1.0

            self.metrics['tasks_completed'] += 1

            logger.info(f"Task completed: {task.task_id}")

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.utcnow()

            self.metrics['tasks_failed'] += 1
            logger.error(f"Task failed: {task.task_id} - {e}")

    async def _coordinator_heartbeat(self):
        """Coordinator heartbeat and health monitoring."""
        while True:
            try:
                # Update own status
                self.nodes[self.node_id].resources = self._get_system_resources()
                self.nodes[self.node_id].last_heartbeat = datetime.utcnow()

                # Check for failed nodes
                failed_nodes = []
                for node_id, node in self.nodes.items():
                    if node_id != self.node_id:
                        time_since_heartbeat = (datetime.utcnow() - node.last_heartbeat).total_seconds()
                        if time_since_heartbeat > self.node_timeout:
                            node.status = NodeStatus.FAILED
                            failed_nodes.append(node_id)
                            self.metrics['node_failures'] += 1

                if failed_nodes:
                    logger.warning(f"Nodes failed: {failed_nodes}")
                    # Reschedule tasks from failed nodes
                    await self._reschedule_failed_tasks(failed_nodes)

                await asyncio.sleep(self.heartbeat_interval)

            except Exception as e:
                logger.error(f"Coordinator heartbeat error: {e}")
                await asyncio.sleep(self.heartbeat_interval)

    async def _worker_heartbeat(self):
        """Worker heartbeat to coordinator."""
        while True:
            try:
                await self._send_heartbeat_to_coordinator()
                await asyncio.sleep(self.heartbeat_interval)
            except Exception as e:
                logger.error(f"Worker heartbeat error: {e}")
                await asyncio.sleep(self.heartbeat_interval)

    async def _task_scheduler(self):
        """Schedule pending tasks to available nodes."""
        while True:
            try:
                # Process task queue
                pending_tasks = [t for t in self.task_queue if t.status == TaskStatus.PENDING]

                for task in pending_tasks:
                    # Check if dependencies are satisfied
                    completed_task_ids = [t.task_id for t in self.tasks.values() if t.status == TaskStatus.COMPLETED]
                    if task.can_start(completed_task_ids):
                        await self._schedule_task(task)
                        self.task_queue.remove(task)

                await asyncio.sleep(5)  # Check every 5 seconds

            except Exception as e:
                logger.error(f"Task scheduler error: {e}")
                await asyncio.sleep(5)

    async def _task_processor(self):
        """Process tasks assigned to this worker node."""
        while True:
            try:
                # Find tasks assigned to this node
                my_tasks = [t for t in self.tasks.values()
                           if t.assigned_node == self.node_id and t.status == TaskStatus.RUNNING]

                for task in my_tasks:
                    if task.is_expired():
                        task.status = TaskStatus.FAILED
                        task.error = "Task timeout"
                        continue

                    # Process task if not already completed
                    if task.status == TaskStatus.RUNNING and task.result is None:
                        await self._process_task_locally(task)

                await asyncio.sleep(1)  # Check frequently

            except Exception as e:
                logger.error(f"Task processor error: {e}")
                await asyncio.sleep(1)

    async def _cluster_monitor(self):
        """Monitor cluster health and performance."""
        while True:
            try:
                status = await self.get_cluster_status()
                logger.info(f"Cluster status: {status['active_nodes']} active nodes, "
                          f"{status['running_tasks']} running tasks")

                # Log performance metrics
                if self.metrics['tasks_submitted'] > 0:
                    success_rate = self.metrics['tasks_completed'] / self.metrics['tasks_submitted']
                    logger.info(f"Task success rate: {success_rate:.2%}")

                await asyncio.sleep(60)  # Log every minute

            except Exception as e:
                logger.error(f"Cluster monitor error: {e}")
                await asyncio.sleep(60)

    # Task Handlers
    def _handle_backtest_task(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle backtesting task."""
        # Placeholder - would integrate with backtest engine
        time.sleep(2)  # Simulate processing time
        return {
            'sharpe_ratio': 1.5,
            'max_drawdown': -0.15,
            'total_return': 0.85,
            'completed': True
        }

    def _handle_signal_task(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle signal generation task."""
        # Placeholder - would integrate with signal engine
        time.sleep(1)
        return {
            'signals_generated': 150,
            'strong_signals': 12,
            'completed': True
        }

    def _handle_optimization_task(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle portfolio optimization task."""
        # Placeholder - would integrate with portfolio optimizer
        time.sleep(3)
        return {
            'optimal_weights': {'AAPL': 0.3, 'MSFT': 0.25, 'GOOGL': 0.45},
            'expected_return': 0.12,
            'expected_volatility': 0.18,
            'completed': True
        }

    def _handle_risk_task(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle risk calculation task."""
        # Placeholder - would integrate with risk manager
        time.sleep(1)
        return {
            'var_95': -0.02,
            'cvar_95': -0.035,
            'stress_test_passed': True,
            'completed': True
        }

    def _handle_data_task(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle data processing task."""
        # Placeholder - would integrate with data router
        time.sleep(2)
        return {
            'records_processed': 10000,
            'data_quality_score': 0.95,
            'completed': True
        }

    # Networking Helpers
    async def _send_task_to_coordinator(self, task: DistributedTask):
        """Send task to coordinator."""
        if not self.http_session:
            return

        url = f"http://{self.coordinator_host}:{self.coordinator_port}/tasks"
        async with self.http_session.post(url, json=asdict(task)) as response:
            if response.status != 200:
                logger.error(f"Failed to send task to coordinator: {response.status}")

    async def _send_task_to_node(self, node: ClusterNode, task: DistributedTask):
        """Send task to a worker node."""
        if not self.http_session:
            return

        url = f"http://{node.ip_address}:{node.port}/tasks"
        async with self.http_session.post(url, json=asdict(task)) as response:
            if response.status != 200:
                logger.error(f"Failed to send task to node {node.node_id}: {response.status}")

    async def _send_heartbeat_to_coordinator(self):
        """Send heartbeat to coordinator."""
        if not self.http_session:
            return

        node_info = asdict(self.nodes.get(self.node_id, self._create_node_info()))
        url = f"http://{self.coordinator_host}:{self.coordinator_port}/heartbeat"
        async with self.http_session.post(url, json=node_info) as response:
            if response.status != 200:
                logger.error(f"Heartbeat failed: {response.status}")

    async def _register_with_coordinator(self):
        """Register worker node with coordinator."""
        if not self.http_session:
            return

        node_info = self._create_node_info()
        url = f"http://{self.coordinator_host}:{self.coordinator_port}/register"
        async with self.http_session.post(url, json=asdict(node_info)) as response:
            if response.status == 200:
                logger.info("Successfully registered with coordinator")
            else:
                logger.error(f"Registration failed: {response.status}")

    def _create_node_info(self) -> ClusterNode:
        """Create node information for registration."""
        return ClusterNode(
            node_id=self.node_id,
            hostname=socket.gethostname(),
            ip_address=self._get_local_ip(),
            port=self.coordinator_port + 1,  # Worker port
            status=NodeStatus.ACTIVE,
            capabilities=self.capabilities,
            resources=self.resources,
            last_heartbeat=datetime.utcnow(),
            tasks_running=0,
            tasks_completed=0,
            uptime=time.time()
        )

    def _get_local_ip(self) -> str:
        """Get local IP address."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"

    async def _reschedule_failed_tasks(self, failed_node_ids: List[str]):
        """Reschedule tasks from failed nodes."""
        failed_tasks = [t for t in self.tasks.values()
                       if t.assigned_node in failed_node_ids and t.status == TaskStatus.RUNNING]

        for task in failed_tasks:
            task.assigned_node = None
            task.status = TaskStatus.PENDING
            self.task_queue.append(task)
            logger.info(f"Rescheduled task {task.task_id} from failed node")

    def _aggregate_resources(self, nodes: List[ClusterNode]) -> Dict[str, float]:
        """Aggregate resources across all nodes."""
        if not nodes:
            return {}

        total_cpu = sum(node.capabilities.get('cpu_count', 0) for node in nodes)
        total_memory = sum(node.capabilities.get('memory_gb', 0) for node in nodes)
        total_gpu = sum(node.capabilities.get('gpu_count', 0) for node in nodes)

        return {
            'total_cpu_cores': total_cpu,
            'total_memory_gb': total_memory,
            'total_gpus': total_gpu,
            'avg_cpu_utilization': sum(node.resources.get('cpu_percent', 0) for node in nodes) / len(nodes),
            'avg_memory_utilization': sum(node.resources.get('memory_percent', 0) for node in nodes) / len(nodes)
        }

    def _initialize_as_coordinator(self):
        """Initialize coordinator-specific components."""
        # Start HTTP server for coordination
        # This would be implemented with aiohttp server
        pass

    def _initialize_as_worker(self):
        """Initialize worker-specific components."""
        # Set up task processing capabilities
        pass
