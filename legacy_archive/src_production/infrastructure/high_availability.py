"""
High Availability Infrastructure - Production Implementation
Provides failover, redundancy, and disaster recovery
"""

import asyncio
import logging
import json
import socket
import subprocess
import time
import os
import uuid
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import aiohttp
import psutil
import docker
from kubernetes import client, config
import redis
import psycopg2
from psycopg2 import pool
import consul
import etcd3

logger = logging.getLogger(__name__)

class NodeStatus(Enum):
    """Node status"""
    ACTIVE = "active"
    STANDBY = "standby"
    FAILED = "failed"
    MAINTENANCE = "maintenance"
    RECOVERING = "recovering"

class ServiceStatus(Enum):
    """Service status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

class FailoverType(Enum):
    """Failover types"""
    AUTOMATIC = "automatic"
    MANUAL = "manual"
    SCHEDULED = "scheduled"
    EMERGENCY = "emergency"

@dataclass
class Node:
    """Node structure"""
    node_id: str
    hostname: str
    ip_address: str
    role: str  # primary, secondary, etc.
    status: NodeStatus
    last_heartbeat: datetime
    services: Dict[str, ServiceStatus]
    resources: Dict[str, float]
    metadata: Dict[str, Any]

@dataclass
class FailoverEvent:
    """Failover event structure"""
    event_id: str
    failover_type: FailoverType
    source_node: str
    target_node: str
    services: List[str]
    timestamp: datetime
    reason: str
    success: bool
    duration: Optional[timedelta] = None

@dataclass
class BackupConfig:
    """Backup configuration"""
    backup_id: str
    source: str
    destination: str
    schedule: str
    retention_days: int
    compression: bool = True
    encryption: bool = True
    last_backup: Optional[datetime] = None
    next_backup: Optional[datetime] = None

class HighAvailabilityManager:
    """Production high availability manager"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.nodes = {}
        self.failover_events = {}
        self.backup_configs = {}
        self.running = False
        self.current_primary = None
        self.consul_client = None
        self.etcd_client = None
        self.redis_client = None
        self.db_pool = None
        self.k8s_client = None
        self.health_checks = {}
        self.metrics = {}

        # Initialize HA components
        self._initialize_service_discovery()
        self._initialize_database_pool()
        self._initialize_monitoring()

    def _initialize_service_discovery(self):
        """Initialize service discovery"""
        try:
            # Consul
            if self.config.get('consul', {}).get('enabled', False):
                consul_config = self.config['consul']
                self.consul_client = consul.Consul(
                    host=consul_config.get('host', 'localhost'),
                    port=consul_config.get('port', 8500)
                )
                logger.info("Consul client initialized")

            # etcd
            if self.config.get('etcd', {}).get('enabled', False):
                etcd_config = self.config['etcd']
                self.etcd_client = etcd3.client(
                    host=etcd_config.get('host', 'localhost'),
                    port=etcd_config.get('port', 2379)
                )
                logger.info("etcd client initialized")

            # Redis
            if self.config.get('redis', {}).get('enabled', False):
                redis_config = self.config['redis']
                self.redis_client = redis.Redis(
                    host=redis_config.get('host', 'localhost'),
                    port=redis_config.get('port', 6379),
                    db=redis_config.get('db', 0),
                    decode_responses=True
                )
                logger.info("Redis client initialized")

        except Exception as e:
            logger.error(f"Failed to initialize service discovery: {e}")

    def _initialize_database_pool(self):
        """Initialize database connection pool"""
        try:
            db_config = self.config.get('database', {})
            if db_config.get('enabled', False):
                self.db_pool = psycopg2.pool.SimpleConnectionPool(
                    minconn=db_config.get('min_connections', 5),
                    maxconn=db_config.get('max_connections', 20),
                    host=db_config.get('host', 'localhost'),
                    port=db_config.get('port', 5432),
                    database=db_config.get('database', 'miniquantfund'),
                    user=db_config.get('user', 'postgres'),
                    password=db_config.get('password', ''),
                    sslmode=db_config.get('sslmode', 'require')
                )
                logger.info("Database connection pool initialized")

        except Exception as e:
            logger.error(f"Failed to initialize database pool: {e}")

    def _initialize_monitoring(self):
        """Initialize monitoring"""
        try:
            # Kubernetes
            if self.config.get('kubernetes', {}).get('enabled', False):
                try:
                    config.load_incluster_config()
                    self.k8s_client = client.CoreV1Api()
                    logger.info("Kubernetes client initialized")
                except Exception as e:
                    logger.warning(f"Failed to initialize Kubernetes client: {e}")

        except Exception as e:
            logger.error(f"Failed to initialize monitoring: {e}")

    async def start(self):
        """Start HA manager"""
        self.running = True

        # Initialize nodes
        await self._discover_nodes()

        # Start monitoring tasks
        asyncio.create_task(self._monitor_nodes())
        asyncio.create_task(self._monitor_services())
        asyncio.create_task(self._monitor_resources())
        asyncio.create_task(self._perform_health_checks())
        asyncio.create_task(self._manage_failover())
        asyncio.create_task(self._manage_backups())
        asyncio.create_task(self._update_metrics())

        logger.info("High availability manager started")

    async def stop(self):
        """Stop HA manager"""
        self.running = False

        if self.db_pool:
            self.db_pool.close()

        logger.info("High availability manager stopped")

    async def _discover_nodes(self):
        """Discover cluster nodes"""
        try:
            # Get current node information
            current_node = await self._get_current_node()
            self.nodes[current_node.node_id] = current_node

            # Discover other nodes from service discovery
            if self.consul_client:
                await self._discover_nodes_consul()
            elif self.etcd_client:
                await self._discover_nodes_etcd()
            elif self.k8s_client:
                await self._discover_nodes_kubernetes()
            else:
                await self._discover_nodes_static()

            logger.info(f"Discovered {len(self.nodes)} nodes")

        except Exception as e:
            logger.error(f"Failed to discover nodes: {e}")

    async def _get_current_node(self) -> Node:
        """Get current node information"""
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)

        # Get system resources
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        return Node(
            node_id=hostname,
            hostname=hostname,
            ip_address=ip_address,
            role=self._determine_node_role(),
            status=NodeStatus.ACTIVE,
            last_heartbeat=datetime.utcnow(),
            services={},
            resources={
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available': memory.available,
                'disk_percent': disk.percent,
                'disk_available': disk.free
            },
            metadata={}
        )

    def _determine_node_role(self) -> str:
        """Determine node role based on configuration"""
        node_roles = self.config.get('node_roles', {})
        hostname = socket.gethostname()

        return node_roles.get(hostname, 'secondary')

    async def _discover_nodes_consul(self):
        """Discover nodes using Consul"""
        try:
            services = self.consul_client.health.service('miniquantfund', passing=True)

            for service in services:
                service_info = service[1]
                node_id = service_info['Service']['ID']

                if node_id not in self.nodes:
                    node = Node(
                        node_id=node_id,
                        hostname=service_info['Node']['Node'],
                        ip_address=service_info['Service']['Address'],
                        role=service_info['Service']['ServiceMeta'].get('role', 'secondary'),
                        status=NodeStatus.ACTIVE,
                        last_heartbeat=datetime.utcnow(),
                        services={},
                        resources={},
                        metadata=service_info['Service']['ServiceMeta']
                    )
                    self.nodes[node_id] = node

        except Exception as e:
            logger.error(f"Failed to discover nodes via Consul: {e}")

    async def _discover_nodes_etcd(self):
        """Discover nodes using etcd"""
        try:
            # Get nodes from etcd
            nodes_key = '/miniquantfund/nodes/'
            nodes = self.etcd_client.get_prefix(nodes_key)

            for node_data in nodes:
                node_info = json.loads(node_data[1].decode())
                node_id = node_info['node_id']

                if node_id not in self.nodes:
                    node = Node(
                        node_id=node_id,
                        hostname=node_info['hostname'],
                        ip_address=node_info['ip_address'],
                        role=node_info['role'],
                        status=NodeStatus.ACTIVE,
                        last_heartbeat=datetime.utcnow(),
                        services={},
                        resources={},
                        metadata=node_info.get('metadata', {})
                    )
                    self.nodes[node_id] = node

        except Exception as e:
            logger.error(f"Failed to discover nodes via etcd: {e}")

    async def _discover_nodes_kubernetes(self):
        """Discover nodes using Kubernetes"""
        try:
            pods = self.k8s_client.list_namespaced_pod(
                namespace=self.config.get('kubernetes', {}).get('namespace', 'default'),
                label_selector='app=miniquantfund'
            )

            for pod in pods.items:
                pod_name = pod.metadata.name
                pod_ip = pod.status.pod_ip

                if pod_ip and pod_name not in self.nodes:
                    node = Node(
                        node_id=pod_name,
                        hostname=pod_name,
                        ip_address=pod_ip,
                        role=pod.metadata.labels.get('role', 'secondary'),
                        status=NodeStatus.ACTIVE,
                        last_heartbeat=datetime.utcnow(),
                        services={},
                        resources={},
                        metadata=pod.metadata.labels or {}
                    )
                    self.nodes[pod_name] = node

        except Exception as e:
            logger.error(f"Failed to discover nodes via Kubernetes: {e}")

    async def _discover_nodes_static(self):
        """Discover nodes using static configuration"""
        static_nodes = self.config.get('static_nodes', [])

        for node_config in static_nodes:
            node_id = node_config['node_id']

            if node_id not in self.nodes:
                node = Node(
                    node_id=node_id,
                    hostname=node_config['hostname'],
                    ip_address=node_config['ip_address'],
                    role=node_config.get('role', 'secondary'),
                    status=NodeStatus.ACTIVE,
                    last_heartbeat=datetime.utcnow(),
                    services={},
                    resources={},
                    metadata=node_config.get('metadata', {})
                )
                self.nodes[node_id] = node

    async def _monitor_nodes(self):
        """Monitor node health and status"""
        while self.running:
            try:
                current_time = datetime.utcnow()

                for node_id, node in list(self.nodes.items()):
                    # Check heartbeat
                    if (current_time - node.last_heartbeat).total_seconds() > 30:
                        if node.status == NodeStatus.ACTIVE:
                            node.status = NodeStatus.FAILED
                            await self._handle_node_failure(node_id)

                    # Update node resources
                    if node_id == socket.gethostname():
                        await self._update_node_resources(node)

                # Update primary node
                await self._update_primary_node()

                await asyncio.sleep(10)  # Check every 10 seconds

            except Exception as e:
                logger.error(f"Error monitoring nodes: {e}")
                await asyncio.sleep(5)

    async def _update_node_resources(self, node: Node):
        """Update node resource information"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            node.resources.update({
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available': memory.available,
                'disk_percent': disk.percent,
                'disk_available': disk.free,
                'load_average': psutil.getloadavg(),
                'network_io': psutil.net_io_counters()._asdict() if psutil.net_io_counters() else {}
            })

        except Exception as e:
            logger.error(f"Failed to update node resources: {e}")

    async def _monitor_services(self):
        """Monitor service health"""
        while self.running:
            try:
                services = self.config.get('services', [])

                for node_id, node in self.nodes.items():
                    for service_config in services:
                        service_name = service_config['name']

                        if node.status == NodeStatus.ACTIVE:
                            status = await self._check_service_health(node, service_config)
                            node.services[service_name] = status

                await asyncio.sleep(30)  # Check every 30 seconds

            except Exception as e:
                logger.error(f"Error monitoring services: {e}")
                await asyncio.sleep(10)

    async def _check_service_health(self, node: Node, service_config: Dict[str, Any]) -> ServiceStatus:
        """Check individual service health"""
        try:
            service_name = service_config['name']
            service_type = service_config['type']
            port = service_config.get('port')
            endpoint = service_config.get('endpoint')

            if service_type == 'http':
                return await self._check_http_service(node, endpoint)
            elif service_type == 'tcp':
                return await self._check_tcp_service(node, port)
            elif service_type == 'database':
                return await self._check_database_service(node, service_config)
            else:
                return ServiceStatus.UNKNOWN

        except Exception as e:
            logger.error(f"Error checking service health: {e}")
            return ServiceStatus.UNHEALTHY

    async def _check_http_service(self, node: Node, endpoint: str) -> ServiceStatus:
        """Check HTTP service health"""
        try:
            url = f"http://{node.ip_address}{endpoint}"

            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        return ServiceStatus.HEALTHY
                    else:
                        return ServiceStatus.DEGRADED

        except Exception:
            return ServiceStatus.UNHEALTHY

    async def _check_tcp_service(self, node: Node, port: int) -> ServiceStatus:
        """Check TCP service health"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((node.ip_address, port))
            sock.close()

            if result == 0:
                return ServiceStatus.HEALTHY
            else:
                return ServiceStatus.UNHEALTHY

        except Exception:
            return ServiceStatus.UNHEALTHY

    async def _check_database_service(self, node: Node, service_config: Dict[str, Any]) -> ServiceStatus:
        """Check database service health"""
        try:
            if not self.db_pool:
                return ServiceStatus.UNKNOWN

            conn = self.db_pool.getconn()
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                cursor.fetchone()
                return ServiceStatus.HEALTHY
            finally:
                self.db_pool.putconn(conn)

        except Exception:
            return ServiceStatus.UNHEALTHY

    async def _monitor_resources(self):
        """Monitor system resources"""
        while self.running:
            try:
                for node_id, node in self.nodes.items():
                    if node.status == NodeStatus.ACTIVE:
                        # Check resource thresholds
                        if node.resources.get('cpu_percent', 0) > 90:
                            await self._handle_resource_warning(node_id, 'cpu', node.resources['cpu_percent'])

                        if node.resources.get('memory_percent', 0) > 90:
                            await self._handle_resource_warning(node_id, 'memory', node.resources['memory_percent'])

                        if node.resources.get('disk_percent', 0) > 90:
                            await self._handle_resource_warning(node_id, 'disk', node.resources['disk_percent'])

                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Error monitoring resources: {e}")
                await asyncio.sleep(30)

    async def _handle_resource_warning(self, node_id: str, resource_type: str, value: float):
        """Handle resource warning"""
        logger.warning(f"Resource warning on {node_id}: {resource_type} at {value}%")

        # Could trigger alerts, auto-scaling, etc.
        if resource_type == 'memory' and value > 95:
            await self._trigger_emergency_failover(node_id, f"Memory usage critical: {value}%")

    async def _perform_health_checks(self):
        """Perform comprehensive health checks"""
        while self.running:
            try:
                # Check overall cluster health
                active_nodes = [n for n in self.nodes.values() if n.status == NodeStatus.ACTIVE]

                if len(active_nodes) == 0:
                    logger.critical("No active nodes in cluster")
                    await self._trigger_emergency_failover(None, "No active nodes")
                elif len(active_nodes) < 2:
                    logger.warning("Cluster has less than 2 active nodes")

                # Check primary node
                if self.current_primary and self.current_primary not in self.nodes:
                    logger.warning("Primary node not found in cluster")
                    await self._elect_new_primary()

                await asyncio.sleep(300)  # Check every 5 minutes

            except Exception as e:
                logger.error(f"Error in health checks: {e}")
                await asyncio.sleep(60)

    async def _manage_failover(self):
        """Manage failover operations"""
        while self.running:
            try:
                # Check for failed nodes
                failed_nodes = [n for n in self.nodes.values() if n.status == NodeStatus.FAILED]

                for failed_node in failed_nodes:
                    if failed_node.role == 'primary':
                        await self._handle_primary_failure(failed_node.node_id)

                await asyncio.sleep(30)  # Check every 30 seconds

            except Exception as e:
                logger.error(f"Error managing failover: {e}")
                await asyncio.sleep(15)

    async def _handle_node_failure(self, node_id: str):
        """Handle node failure"""
        logger.error(f"Node failure detected: {node_id}")

        node = self.nodes[node_id]

        # Create failover event
        failover_event = FailoverEvent(
            event_id=str(uuid.uuid4()),
            failover_type=FailoverType.AUTOMATIC,
            source_node=node_id,
            target_node="",
            services=list(node.services.keys()),
            timestamp=datetime.utcnow(),
            reason=f"Node failure: {node_id}",
            success=False
        )

        self.failover_events[failover_event.event_id] = failover_event

        # If primary node failed, elect new primary
        if node.role == 'primary':
            await self._handle_primary_failure(node_id)

    async def _handle_primary_failure(self, failed_primary: str):
        """Handle primary node failure"""
        logger.critical(f"Primary node failure: {failed_primary}")

        # Elect new primary
        await self._elect_new_primary()

        # Update failover event
        for event in self.failover_events.values():
            if event.source_node == failed_primary and not event.success:
                event.target_node = self.current_primary
                event.success = True
                event.duration = datetime.utcnow() - event.timestamp

    async def _elect_new_primary(self):
        """Elect new primary node"""
        try:
            # Get eligible nodes (active, not failed)
            eligible_nodes = [
                node for node in self.nodes.values()
                if node.status == NodeStatus.ACTIVE and node.role != 'primary'
            ]

            if not eligible_nodes:
                logger.error("No eligible nodes for primary election")
                return

            # Select node with best resources
            best_node = min(eligible_nodes, key=lambda n: n.resources.get('cpu_percent', 100))

            # Update node roles
            if self.current_primary:
                old_primary = self.nodes.get(self.current_primary)
                if old_primary:
                    old_primary.role = 'secondary'

            best_node.role = 'primary'
            self.current_primary = best_node.node_id

            logger.info(f"New primary elected: {best_node.node_id}")

            # Update service discovery
            await self._update_service_discovery()

        except Exception as e:
            logger.error(f"Failed to elect new primary: {e}")

    async def _update_primary_node(self):
        """Update primary node status"""
        if not self.current_primary:
            await self._elect_new_primary()
            return

        primary_node = self.nodes.get(self.current_primary)
        if not primary_node or primary_node.status != NodeStatus.ACTIVE:
            await self._elect_new_primary()

    async def _update_service_discovery(self):
        """Update service discovery with current state"""
        try:
            if self.consul_client:
                await self._update_consul()
            elif self.etcd_client:
                await self._update_etcd()

        except Exception as e:
            logger.error(f"Failed to update service discovery: {e}")

    async def _update_consul(self):
        """Update Consul with current state"""
        try:
            for node_id, node in self.nodes.items():
                # Register service
                self.consul_client.agent.service.register(
                    name='miniquantfund',
                    service_id=node_id,
                    address=node.ip_address,
                    port=8080,
                    tags=[node.role, node.status.value],
                    meta={
                        'hostname': node.hostname,
                        'role': node.role,
                        'status': node.status.value
                    }
                )

                # Update health check
                self.consul_client.agent.check.register(
                    name=f'service:{node_id}',
                    check_id=f'service:{node_id}',
                    service_id=node_id,
                    ttl='30s'
                )

                # Pass health check
                if node.status == NodeStatus.ACTIVE:
                    self.consul_client.agent.check.pass_check(f'service:{node_id}')
                else:
                    self.consul_client.agent.check.fail_check(f'service:{node_id}')

        except Exception as e:
            logger.error(f"Failed to update Consul: {e}")

    async def _update_etcd(self):
        """Update etcd with current state"""
        try:
            for node_id, node in self.nodes.items():
                key = f'/miniquantfund/nodes/{node_id}'
                value = {
                    'node_id': node_id,
                    'hostname': node.hostname,
                    'ip_address': node.ip_address,
                    'role': node.role,
                    'status': node.status.value,
                    'last_heartbeat': node.last_heartbeat.isoformat(),
                    'resources': node.resources
                }

                self.etcd_client.put(key, json.dumps(value))

        except Exception as e:
            logger.error(f"Failed to update etcd: {e}")

    async def _manage_backups(self):
        """Manage backup operations"""
        while self.running:
            try:
                current_time = datetime.utcnow()

                for backup_id, backup_config in self.backup_configs.items():
                    if backup_config.next_backup and current_time >= backup_config.next_backup:
                        await self._perform_backup(backup_config)

                await asyncio.sleep(3600)  # Check every hour

            except Exception as e:
                logger.error(f"Error managing backups: {e}")
                await asyncio.sleep(300)

    async def _perform_backup(self, backup_config: BackupConfig):
        """Perform backup operation"""
        try:
            logger.info(f"Starting backup: {backup_config.backup_id}")

            start_time = datetime.utcnow()

            # Create backup command
            if backup_config.source.startswith('database://'):
                await self._backup_database(backup_config)
            elif backup_config.source.startswith('file://'):
                await self._backup_files(backup_config)

            # Update backup config
            backup_config.last_backup = start_time

            # Schedule next backup
            if backup_config.schedule == 'daily':
                backup_config.next_backup = start_time + timedelta(days=1)
            elif backup_config.schedule == 'weekly':
                backup_config.next_backup = start_time + timedelta(weeks=1)
            elif backup_config.schedule == 'hourly':
                backup_config.next_backup = start_time + timedelta(hours=1)

            logger.info(f"Backup completed: {backup_config.backup_id}")

        except Exception as e:
            logger.error(f"Failed to perform backup {backup_config.backup_id}: {e}")

    async def _backup_database(self, backup_config: BackupConfig):
        """Backup database"""
        try:
            if not self.db_pool:
                return

            # Extract database info
            db_info = backup_config.source.replace('database://', '')

            # Create backup command
            backup_file = f"/backups/{backup_config.backup_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.sql"

            # Use pg_dump for PostgreSQL
            cmd = [
                'pg_dump',
                '-h', self.config['database']['host'],
                '-p', str(self.config['database']['port']),
                '-U', self.config['database']['user'],
                '-d', self.config['database']['database'],
                '-f', backup_file
            ]

            # Set password in environment
            env = os.environ.copy()
            env['PGPASSWORD'] = self.config['database']['password']

            # Execute backup
            result = subprocess.run(cmd, env=env, capture_output=True, text=True)

            if result.returncode != 0:
                raise Exception(f"pg_dump failed: {result.stderr}")

            # Compress if enabled
            if backup_config.compression:
                subprocess.run(['gzip', backup_file])
                backup_file += '.gz'

            # Encrypt if enabled
            if backup_config.encryption:
                # Implement encryption
                pass

            logger.info(f"Database backup completed: {backup_file}")

        except Exception as e:
            logger.error(f"Database backup failed: {e}")
            raise

    async def _backup_files(self, backup_config: BackupConfig):
        """Backup files"""
        try:
            source_path = backup_config.source.replace('file://', '')
            backup_file = f"/backups/{backup_config.backup_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.tar.gz"

            # Create tar archive
            cmd = ['tar', '-czf', backup_file, source_path]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                raise Exception(f"tar failed: {result.stderr}")

            logger.info(f"File backup completed: {backup_file}")

        except Exception as e:
            logger.error(f"File backup failed: {e}")
            raise

    async def _update_metrics(self):
        """Update HA metrics"""
        while self.running:
            try:
                # Calculate cluster metrics
                active_nodes = len([n for n in self.nodes.values() if n.status == NodeStatus.ACTIVE])
                failed_nodes = len([n for n in self.nodes.values() if n.status == NodeStatus.FAILED])
                healthy_services = 0
                total_services = 0

                for node in self.nodes.values():
                    for service_status in node.services.values():
                        total_services += 1
                        if service_status == ServiceStatus.HEALTHY:
                            healthy_services += 1

                service_health_ratio = healthy_services / total_services if total_services > 0 else 0

                self.metrics = {
                    'active_nodes': active_nodes,
                    'failed_nodes': failed_nodes,
                    'total_nodes': len(self.nodes),
                    'healthy_services': healthy_services,
                    'total_services': total_services,
                    'service_health_ratio': service_health_ratio,
                    'current_primary': self.current_primary,
                    'last_failover': max([e.timestamp for e in self.failover_events.values()]) if self.failover_events else None,
                    'cluster_health': 'healthy' if service_health_ratio > 0.8 else 'degraded' if service_health_ratio > 0.5 else 'unhealthy'
                }

                await asyncio.sleep(60)  # Update every minute

            except Exception as e:
                logger.error(f"Error updating metrics: {e}")
                await asyncio.sleep(30)

    async def trigger_manual_failover(self, target_node: Optional[str] = None) -> bool:
        """Trigger manual failover"""
        try:
            logger.info(f"Manual failover triggered, target: {target_node}")

            # Select target node
            if target_node:
                if target_node not in self.nodes:
                    logger.error(f"Target node not found: {target_node}")
                    return False

                new_primary = self.nodes[target_node]
            else:
                # Auto-select best node
                eligible_nodes = [n for n in self.nodes.values() if n.status == NodeStatus.ACTIVE]
                if not eligible_nodes:
                    logger.error("No eligible nodes for failover")
                    return False

                new_primary = min(eligible_nodes, key=lambda n: n.resources.get('cpu_percent', 100))

            # Perform failover
            old_primary = self.current_primary

            # Update roles
            if old_primary and old_primary in self.nodes:
                self.nodes[old_primary].role = 'secondary'

            new_primary.role = 'primary'
            self.current_primary = new_primary.node_id

            # Create failover event
            failover_event = FailoverEvent(
                event_id=str(uuid.uuid4()),
                failover_type=FailoverType.MANUAL,
                source_node=old_primary,
                target_node=new_primary.node_id,
                services=list(new_primary.services.keys()),
                timestamp=datetime.utcnow(),
                reason="Manual failover triggered",
                success=True
            )

            self.failover_events[failover_event.event_id] = failover_event

            # Update service discovery
            await self._update_service_discovery()

            logger.info(f"Manual failover completed: {old_primary} -> {new_primary.node_id}")
            return True

        except Exception as e:
            logger.error(f"Manual failover failed: {e}")
            return False

    async def _trigger_emergency_failover(self, failed_node: Optional[str], reason: str):
        """Trigger emergency failover"""
        logger.critical(f"Emergency failover triggered: {reason}")

        # Perform immediate failover
        await self._handle_primary_failure(failed_node) if failed_node else await self._elect_new_primary()

    def get_cluster_status(self) -> Dict[str, Any]:
        """Get cluster status"""
        return {
            'nodes': {
                node_id: {
                    'hostname': node.hostname,
                    'ip_address': node.ip_address,
                    'role': node.role,
                    'status': node.status.value,
                    'last_heartbeat': node.last_heartbeat.isoformat(),
                    'services': {k: v.value for k, v in node.services.items()},
                    'resources': node.resources
                }
                for node_id, node in self.nodes.items()
            },
            'metrics': self.metrics,
            'failover_events': {
                event_id: {
                    'failover_type': event.failover_type.value,
                    'source_node': event.source_node,
                    'target_node': event.target_node,
                    'services': event.services,
                    'timestamp': event.timestamp.isoformat(),
                    'reason': event.reason,
                    'success': event.success,
                    'duration': event.duration.total_seconds() if event.duration else None
                }
                for event_id, event in self.failover_events.items()
            }
        }
