"""
Real-Time Monitoring - Production Implementation
Comprehensive monitoring and alerting system
"""

import asyncio
import logging
import json
import time
import psutil
import socket
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import aiohttp
import prometheus_client
from prometheus_client import Counter, Gauge, Histogram, CollectorRegistry
import grafana_api
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Metric types"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

class AlertLevel(Enum):
    """Alert levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class SystemComponent(Enum):
    """System components"""
    TRADING_ENGINE = "trading_engine"
    RISK_MANAGER = "risk_manager"
    DATA_FEEDS = "data_feeds"
    BROKER_CONNECTIONS = "broker_connections"
    DATABASE = "database"
    CACHE = "cache"
    NETWORK = "network"
    STORAGE = "storage"

@dataclass
class Metric:
    """Metric structure"""
    name: str
    metric_type: MetricType
    value: float
    labels: Dict[str, str]
    timestamp: datetime
    component: SystemComponent

@dataclass
class Alert:
    """Alert structure"""
    alert_id: str
    level: AlertLevel
    component: SystemComponent
    metric_name: str
    message: str
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None

@dataclass
class Threshold:
    """Threshold structure"""
    metric_name: str
    component: SystemComponent
    warning_threshold: Optional[float]
    critical_threshold: Optional[float]
    operator: str  # >, <, >=, <=, ==, !=
    duration: timedelta  # How long threshold must be exceeded

class RealTimeMonitoring:
    """Production real-time monitoring system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics = {}
        self.alerts = {}
        self.thresholds = {}
        self.running = False
        self.registry = CollectorRegistry()
        
        # Prometheus metrics
        self.prometheus_metrics = {}
        
        # Grafana client
        self.grafana_client = None
        
        # Alert callbacks
        self.alert_callbacks = []
        
        # Initialize monitoring
        self._initialize_prometheus()
        self._initialize_grafana()
        self._initialize_thresholds()
        
    def _initialize_prometheus(self):
        """Initialize Prometheus metrics"""
        try:
            # System metrics
            self.prometheus_metrics['cpu_usage'] = Gauge(
                'cpu_usage_percent',
                'CPU usage percentage',
                ['hostname', 'component'],
                registry=self.registry
            )
            
            self.prometheus_metrics['memory_usage'] = Gauge(
                'memory_usage_percent',
                'Memory usage percentage',
                ['hostname', 'component'],
                registry=self.registry
            )
            
            self.prometheus_metrics['disk_usage'] = Gauge(
                'disk_usage_percent',
                'Disk usage percentage',
                ['hostname', 'mount_point'],
                registry=self.registry
            )
            
            # Trading metrics
            self.prometheus_metrics['orders_submitted'] = Counter(
                'orders_submitted_total',
                'Total number of orders submitted',
                ['symbol', 'side', 'broker'],
                registry=self.registry
            )
            
            self.prometheus_metrics['orders_filled'] = Counter(
                'orders_filled_total',
                'Total number of orders filled',
                ['symbol', 'side', 'broker'],
                registry=self.registry
            )
            
            self.prometheus_metrics['order_latency'] = Histogram(
                'order_latency_seconds',
                'Order execution latency in seconds',
                ['symbol', 'broker'],
                registry=self.registry
            )
            
            # Risk metrics
            self.prometheus_metrics['portfolio_value'] = Gauge(
                'portfolio_value_usd',
                'Portfolio value in USD',
                registry=self.registry
            )
            
            self.prometheus_metrics['risk_metrics'] = Gauge(
                'risk_metrics',
                'Risk metrics',
                ['metric_type', 'confidence_level'],
                registry=self.registry
            )
            
            # Data feed metrics
            self.prometheus_metrics['data_feed_latency'] = Histogram(
                'data_feed_latency_seconds',
                'Data feed latency in seconds',
                ['source', 'symbol'],
                registry=self.registry
            )
            
            self.prometheus_metrics['data_feed_errors'] = Counter(
                'data_feed_errors_total',
                'Total number of data feed errors',
                ['source', 'symbol'],
                registry=self.registry
            )
            
            logger.info("Prometheus metrics initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Prometheus metrics: {e}")
    
    def _initialize_grafana(self):
        """Initialize Grafana client"""
        try:
            grafana_config = self.config.get('grafana', {})
            if grafana_config.get('enabled', False):
                self.grafana_client = grafana_api.GrafanaAPI(
                    auth=grafana_config.get('auth'),
                    host=grafana_config.get('host', 'localhost'),
                    port=grafana_config.get('port', 3000)
                )
                logger.info("Grafana client initialized")
                
        except Exception as e:
            logger.error(f"Failed to initialize Grafana client: {e}")
    
    def _initialize_thresholds(self):
        """Initialize monitoring thresholds"""
        threshold_configs = self.config.get('thresholds', [])
        
        for config in threshold_configs:
            threshold = Threshold(
                metric_name=config['metric_name'],
                component=SystemComponent(config['component']),
                warning_threshold=config.get('warning_threshold'),
                critical_threshold=config.get('critical_threshold'),
                operator=config.get('operator', '>'),
                duration=timedelta(seconds=config.get('duration_seconds', 60))
            )
            
            self.thresholds[f"{threshold.component.value}_{threshold.metric_name}"] = threshold
        
        logger.info(f"Initialized {len(self.thresholds)} thresholds")
    
    async def start(self):
        """Start monitoring system"""
        self.running = True
        
        # Start monitoring tasks
        asyncio.create_task(self._collect_system_metrics())
        asyncio.create_task(self._collect_trading_metrics())
        asyncio.create_task(self._collect_risk_metrics())
        asyncio.create_task(self._collect_data_feed_metrics())
        asyncio.create_task(self._check_thresholds())
        asyncio.create_task(self._update_prometheus())
        asyncio.create_task(self._cleanup_old_metrics())
        
        logger.info("Real-time monitoring started")
    
    async def stop(self):
        """Stop monitoring system"""
        self.running = False
        logger.info("Real-time monitoring stopped")
    
    async def _collect_system_metrics(self):
        """Collect system metrics"""
        while self.running:
            try:
                hostname = socket.gethostname()
                current_time = datetime.utcnow()
                
                # CPU metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                await self._record_metric(
                    'cpu_usage',
                    MetricType.GAUGE,
                    cpu_percent,
                    {'hostname': hostname},
                    current_time,
                    SystemComponent.TRADING_ENGINE
                )
                
                # Memory metrics
                memory = psutil.virtual_memory()
                await self._record_metric(
                    'memory_usage',
                    MetricType.GAUGE,
                    memory.percent,
                    {'hostname': hostname},
                    current_time,
                    SystemComponent.TRADING_ENGINE
                )
                
                # Disk metrics
                disk_partitions = psutil.disk_partitions()
                for partition in disk_partitions:
                    disk_usage = psutil.disk_usage(partition.mountpoint)
                    await self._record_metric(
                        'disk_usage',
                        MetricType.GAUGE,
                        disk_usage.percent,
                        {'hostname': hostname, 'mount_point': partition.mountpoint},
                        current_time,
                        SystemComponent.STORAGE
                    )
                
                # Network metrics
                network_io = psutil.net_io_counters()
                await self._record_metric(
                    'network_bytes_sent',
                    MetricType.COUNTER,
                    network_io.bytes_sent,
                    {'hostname': hostname},
                    current_time,
                    SystemComponent.NETWORK
                )
                
                await self._record_metric(
                    'network_bytes_received',
                    MetricType.COUNTER,
                    network_io.bytes_recv,
                    {'hostname': hostname},
                    current_time,
                    SystemComponent.NETWORK
                )
                
                await asyncio.sleep(30)  # Collect every 30 seconds
                
            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
                await asyncio.sleep(10)
    
    async def _collect_trading_metrics(self):
        """Collect trading metrics"""
        while self.running:
            try:
                # This would integrate with actual trading system
                # For now, we'll simulate some metrics
                
                current_time = datetime.utcnow()
                
                # Order metrics
                await self._record_metric(
                    'orders_per_second',
                    MetricType.GAUGE,
                    np.random.normal(10, 2),  # Simulated
                    {},
                    current_time,
                    SystemComponent.TRADING_ENGINE
                )
                
                # Position metrics
                await self._record_metric(
                    'active_positions',
                    MetricType.GAUGE,
                    np.random.randint(50, 150),  # Simulated
                    {},
                    current_time,
                    SystemComponent.TRADING_ENGINE
                )
                
                # P&L metrics
                await self._record_metric(
                    'daily_pnl',
                    MetricType.GAUGE,
                    np.random.normal(1000, 500),  # Simulated
                    {'currency': 'USD'},
                    current_time,
                    SystemComponent.TRADING_ENGINE
                )
                
                await asyncio.sleep(60)  # Collect every minute
                
            except Exception as e:
                logger.error(f"Error collecting trading metrics: {e}")
                await asyncio.sleep(20)
    
    async def _collect_risk_metrics(self):
        """Collect risk metrics"""
        while self.running:
            try:
                current_time = datetime.utcnow()
                
                # VaR metrics
                await self._record_metric(
                    'portfolio_var_1d',
                    MetricType.GAUGE,
                    np.random.normal(50000, 10000),  # Simulated
                    {'confidence': '95'},
                    current_time,
                    SystemComponent.RISK_MANAGER
                )
                
                await self._record_metric(
                    'portfolio_var_99',
                    MetricType.GAUGE,
                    np.random.normal(75000, 15000),  # Simulated
                    {'confidence': '99'},
                    current_time,
                    SystemComponent.RISK_MANAGER
                )
                
                # Drawdown metrics
                await self._record_metric(
                    'max_drawdown',
                    MetricType.GAUGE,
                    abs(np.random.normal(0.05, 0.02)),  # Simulated
                    {},
                    current_time,
                    SystemComponent.RISK_MANAGER
                )
                
                # Concentration metrics
                await self._record_metric(
                    'concentration_risk',
                    MetricType.GAUGE,
                    np.random.normal(0.3, 0.1),  # Simulated
                    {},
                    current_time,
                    SystemComponent.RISK_MANAGER
                )
                
                await asyncio.sleep(60)  # Collect every minute
                
            except Exception as e:
                logger.error(f"Error collecting risk metrics: {e}")
                await asyncio.sleep(20)
    
    async def _collect_data_feed_metrics(self):
        """Collect data feed metrics"""
        while self.running:
            try:
                current_time = datetime.utcnow()
                
                # Data feed latency
                for source in ['bloomberg', 'reuters', 'alpaca', 'polygon']:
                    await self._record_metric(
                        'data_feed_latency',
                        MetricType.HISTOGRAM,
                        np.random.exponential(0.001),  # Simulated (1ms average)
                        {'source': source},
                        current_time,
                        SystemComponent.DATA_FEEDS
                    )
                
                # Data feed errors
                for source in ['bloomberg', 'reuters', 'alpaca', 'polygon']:
                    await self._record_metric(
                        'data_feed_errors',
                        MetricType.COUNTER,
                        np.random.poisson(0.1),  # Simulated
                        {'source': source},
                        current_time,
                        SystemComponent.DATA_FEEDS
                    )
                
                # Message rate
                await self._record_metric(
                    'messages_per_second',
                    MetricType.GAUGE,
                    np.random.normal(1000, 200),  # Simulated
                    {},
                    current_time,
                    SystemComponent.DATA_FEEDS
                )
                
                await asyncio.sleep(30)  # Collect every 30 seconds
                
            except Exception as e:
                logger.error(f"Error collecting data feed metrics: {e}")
                await asyncio.sleep(10)
    
    async def _record_metric(self, name: str, metric_type: MetricType, value: float,
                            labels: Dict[str, str], timestamp: datetime, 
                            component: SystemComponent):
        """Record a metric"""
        try:
            metric = Metric(
                name=name,
                metric_type=metric_type,
                value=value,
                labels=labels,
                timestamp=timestamp,
                component=component
            )
            
            # Store in memory (with retention limit)
            metric_key = f"{name}_{json.dumps(labels, sort_keys=True)}"
            
            if metric_key not in self.metrics:
                self.metrics[metric_key] = []
            
            self.metrics[metric_key].append(metric)
            
            # Keep only last 1000 data points per metric
            if len(self.metrics[metric_key]) > 1000:
                self.metrics[metric_key] = self.metrics[metric_key][-1000:]
            
        except Exception as e:
            logger.error(f"Failed to record metric {name}: {e}")
    
    async def _check_thresholds(self):
        """Check metric thresholds and generate alerts"""
        while self.running:
            try:
                current_time = datetime.utcnow()
                
                for threshold_key, threshold in self.thresholds.items():
                    # Get recent metrics for this threshold
                    recent_metrics = []
                    
                    for metric_key, metric_list in self.metrics.items():
                        if threshold.metric_name in metric_key:
                            # Filter by component
                            component_match = any(
                                threshold.component.value in label 
                                for label in metric_key.split('_')
                            )
                            
                            if component_match:
                                # Get metrics within threshold duration
                                cutoff_time = current_time - threshold.duration
                                recent_metrics.extend([
                                    m for m in metric_list 
                                    if m.timestamp >= cutoff_time
                                ])
                    
                    if recent_metrics:
                        # Check if threshold is exceeded
                        latest_value = recent_metrics[-1].value
                        
                        warning_exceeded = self._check_threshold_value(
                            latest_value, threshold.warning_threshold, threshold.operator
                        )
                        
                        critical_exceeded = self._check_threshold_value(
                            latest_value, threshold.critical_threshold, threshold.operator
                        )
                        
                        if critical_exceeded:
                            await self._create_alert(
                                AlertLevel.CRITICAL,
                                threshold.component,
                                threshold.metric_name,
                                f"Critical threshold exceeded: {latest_value} {threshold.operator} {threshold.critical_threshold}",
                                latest_value
                            )
                        elif warning_exceeded:
                            await self._create_alert(
                                AlertLevel.WARNING,
                                threshold.component,
                                threshold.metric_name,
                                f"Warning threshold exceeded: {latest_value} {threshold.operator} {threshold.warning_threshold}",
                                latest_value
                            )
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error checking thresholds: {e}")
                await asyncio.sleep(10)
    
    def _check_threshold_value(self, value: float, threshold: Optional[float], operator: str) -> bool:
        """Check if value exceeds threshold"""
        if threshold is None:
            return False
        
        if operator == '>':
            return value > threshold
        elif operator == '<':
            return value < threshold
        elif operator == '>=':
            return value >= threshold
        elif operator == '<=':
            return value <= threshold
        elif operator == '==':
            return value == threshold
        elif operator == '!=':
            return value != threshold
        else:
            return False
    
    async def _create_alert(self, level: AlertLevel, component: SystemComponent,
                          metric_name: str, message: str, value: float):
        """Create alert"""
        try:
            alert = Alert(
                alert_id=f"{component.value}_{metric_name}_{int(time.time())}",
                level=level,
                component=component,
                metric_name=metric_name,
                message=message,
                timestamp=datetime.utcnow()
            )
            
            self.alerts[alert.alert_id] = alert
            
            # Notify callbacks
            for callback in self.alert_callbacks:
                try:
                    await callback(alert)
                except Exception as e:
                    logger.error(f"Error in alert callback: {e}")
            
            # Log alert
            log_level = {
                AlertLevel.INFO: logging.INFO,
                AlertLevel.WARNING: logging.WARNING,
                AlertLevel.ERROR: logging.ERROR,
                AlertLevel.CRITICAL: logging.CRITICAL
            }.get(level, logging.INFO)
            
            logger.log(log_level, f"ALERT: {message}")
            
        except Exception as e:
            logger.error(f"Failed to create alert: {e}")
    
    async def _update_prometheus(self):
        """Update Prometheus metrics"""
        while self.running:
            try:
                # Update Prometheus metrics from collected data
                for metric_key, metric_list in self.metrics.items():
                    if metric_list:
                        latest_metric = metric_list[-1]
                        
                        prometheus_metric = self.prometheus_metrics.get(latest_metric.name)
                        if prometheus_metric:
                            if isinstance(prometheus_metric, Gauge):
                                prometheus_metric.labels(**latest_metric.labels).set(latest_metric.value)
                            elif isinstance(prometheus_metric, Counter):
                                prometheus_metric.labels(**latest_metric.labels).inc(latest_metric.value)
                            elif isinstance(prometheus_metric, Histogram):
                                prometheus_metric.labels(**latest_metric.labels).observe(latest_metric.value)
                
                await asyncio.sleep(15)  # Update every 15 seconds
                
            except Exception as e:
                logger.error(f"Error updating Prometheus metrics: {e}")
                await asyncio.sleep(5)
    
    async def _cleanup_old_metrics(self):
        """Clean up old metrics"""
        while self.running:
            try:
                cutoff_time = datetime.utcnow() - timedelta(hours=24)
                
                for metric_key in list(self.metrics.keys()):
                    metric_list = self.metrics[metric_key]
                    
                    # Remove old metrics
                    self.metrics[metric_key] = [
                        m for m in metric_list 
                        if m.timestamp >= cutoff_time
                    ]
                    
                    # Remove empty metric lists
                    if not self.metrics[metric_key]:
                        del self.metrics[metric_key]
                
                # Clean up old alerts
                for alert_id in list(self.alerts.keys()):
                    alert = self.alerts[alert_id]
                    
                    # Remove resolved alerts older than 24 hours
                    if alert.resolved and alert.resolved_at:
                        if alert.resolved_at < cutoff_time:
                            del self.alerts[alert_id]
                
                await asyncio.sleep(3600)  # Clean up every hour
                
            except Exception as e:
                logger.error(f"Error cleaning up old metrics: {e}")
                await asyncio.sleep(300)
    
    def add_alert_callback(self, callback):
        """Add alert callback"""
        self.alert_callbacks.append(callback)
    
    def get_metric_value(self, name: str, labels: Optional[Dict[str, str]] = None) -> Optional[float]:
        """Get latest metric value"""
        try:
            metric_key = f"{name}_{json.dumps(labels or {}, sort_keys=True)}"
            
            if metric_key in self.metrics and self.metrics[metric_key]:
                return self.metrics[metric_key][-1].value
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get metric value: {e}")
            return None
    
    def get_metrics_history(self, name: str, labels: Optional[Dict[str, str]] = None,
                          start_time: Optional[datetime] = None,
                          end_time: Optional[datetime] = None) -> List[Metric]:
        """Get metrics history"""
        try:
            metric_key = f"{name}_{json.dumps(labels or {}, sort_keys=True)}"
            
            if metric_key not in self.metrics:
                return []
            
            metrics = self.metrics[metric_key]
            
            # Filter by time range
            if start_time or end_time:
                filtered_metrics = []
                for metric in metrics:
                    if start_time and metric.timestamp < start_time:
                        continue
                    if end_time and metric.timestamp > end_time:
                        continue
                    filtered_metrics.append(metric)
                return filtered_metrics
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get metrics history: {e}")
            return []
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve alert"""
        try:
            if alert_id in self.alerts:
                self.alerts[alert_id].resolved = True
                self.alerts[alert_id].resolved_at = datetime.utcnow()
                
                logger.info(f"Alert resolved: {alert_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to resolve alert: {e}")
            return False
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get monitoring summary"""
        try:
            # Count alerts by level
            alerts_by_level = {}
            for alert in self.alerts.values():
                if not alert.resolved:
                    level = alert.level.value
                    alerts_by_level[level] = alerts_by_level.get(level, 0) + 1
            
            # Get system health
            cpu_usage = self.get_metric_value('cpu_usage')
            memory_usage = self.get_metric_value('memory_usage')
            
            # Calculate health score
            health_score = 100
            if cpu_usage and cpu_usage > 80:
                health_score -= 20
            if memory_usage and memory_usage > 80:
                health_score -= 20
            
            if alerts_by_level.get('critical', 0) > 0:
                health_score -= 40
            elif alerts_by_level.get('error', 0) > 0:
                health_score -= 20
            elif alerts_by_level.get('warning', 0) > 0:
                health_score -= 10
            
            health_score = max(0, health_score)
            
            return {
                'total_metrics': len(self.metrics),
                'total_metrics_data_points': sum(len(metrics) for metrics in self.metrics.values()),
                'active_alerts': len([a for a in self.alerts.values() if not a.resolved]),
                'alerts_by_level': alerts_by_level,
                'system_health_score': health_score,
                'system_status': 'healthy' if health_score > 80 else 'warning' if health_score > 60 else 'critical',
                'last_update': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get monitoring summary: {e}")
            return {}
    
    def get_component_status(self, component: SystemComponent) -> Dict[str, Any]:
        """Get component status"""
        try:
            component_metrics = {}
            component_alerts = []
            
            # Get component metrics
            for metric_key, metric_list in self.metrics.items():
                if metric_list:
                    metric = metric_list[-1]
                    if metric.component == component:
                        component_metrics[metric.name] = {
                            'value': metric.value,
                            'timestamp': metric.timestamp.isoformat(),
                            'labels': metric.labels
                        }
            
            # Get component alerts
            for alert in self.alerts.values():
                if alert.component == component and not alert.resolved:
                    component_alerts.append({
                        'alert_id': alert.alert_id,
                        'level': alert.level.value,
                        'message': alert.message,
                        'timestamp': alert.timestamp.isoformat()
                    })
            
            # Calculate component health
            health_score = 100
            if component_alerts:
                for alert in component_alerts:
                    if alert['level'] == 'critical':
                        health_score -= 40
                    elif alert['level'] == 'error':
                        health_score -= 20
                    elif alert['level'] == 'warning':
                        health_score -= 10
            
            health_score = max(0, health_score)
            
            return {
                'component': component.value,
                'health_score': health_score,
                'status': 'healthy' if health_score > 80 else 'warning' if health_score > 60 else 'critical',
                'metrics': component_metrics,
                'alerts': component_alerts,
                'last_update': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get component status: {e}")
            return {}
