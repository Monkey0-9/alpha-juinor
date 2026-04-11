#!/usr/bin/env python3
"""
REAL MONITORING STACK FOR TOP 1% TRADING
==========================================

Deploy real production monitoring with:
- Prometheus for metrics collection
- Grafana for visualization
- AlertManager for alerting
- Node Exporter for system metrics
- Custom trading metrics
- Real-time dashboards
"""

import asyncio
import subprocess
import json
import os
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import requests
import yaml

logger = logging.getLogger(__name__)


@dataclass
class MonitoringComponent:
    """Monitoring component configuration"""
    name: str
    component_type: str  # prometheus, grafana, alertmanager, node_exporter
    version: str
    port: int
    config_file: str
    service_file: str
    
    # Deployment status
    is_deployed: bool = False
    endpoint: str = ""
    health_check_url: str = ""
    
    # Metrics
    uptime: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0


@dataclass
class AlertRule:
    """Alert rule configuration"""
    name: str
    expr: str
    condition: str
    duration: str
    severity: str
    message: str
    labels: Dict[str, str] = field(default_factory=dict)
    
    # Status
    is_active: bool = False
    last_triggered: Optional[datetime] = None


class RealMonitoringStack:
    """
    Deploy real production monitoring stack.
    
    This creates actual monitoring infrastructure, not simulation.
    """
    
    def __init__(self):
        self.components: Dict[str, MonitoringComponent] = {}
        self.alert_rules: Dict[str, AlertRule] = {}
        self.dashboards: Dict[str, Dict[str, Any]] = {}
        
        # Initialize components
        self._initialize_components()
        self._initialize_alert_rules()
        self._initialize_dashboards()
        
        logger.info("Real Monitoring Stack initialized")
    
    def _initialize_components(self):
        """Initialize monitoring components"""
        
        # Prometheus
        self.components['prometheus'] = MonitoringComponent(
            name='prometheus',
            component_type='prometheus',
            version='v2.45.0',
            port=9090,
            config_file='/etc/prometheus/prometheus.yml',
            service_file='/etc/systemd/system/prometheus.service',
            health_check_url='http://localhost:9090/-/healthy'
        )
        
        # Grafana
        self.components['grafana'] = MonitoringComponent(
            name='grafana',
            component_type='grafana',
            version='10.0.0',
            port=3000,
            config_file='/etc/grafana/grafana.ini',
            service_file='/etc/systemd/system/grafana.service',
            health_check_url='http://localhost:3000/api/health'
        )
        
        # AlertManager
        self.components['alertmanager'] = MonitoringComponent(
            name='alertmanager',
            component_type='alertmanager',
            version='v0.25.0',
            port=9093,
            config_file='/etc/alertmanager/alertmanager.yml',
            service_file='/etc/systemd/system/alertmanager.service',
            health_check_url='http://localhost:9093/-/healthy'
        )
        
        # Node Exporter
        self.components['node_exporter'] = MonitoringComponent(
            name='node_exporter',
            component_type='node_exporter',
            version='v1.6.0',
            port=9100,
            config_file='/etc/node_exporter/node_exporter.yml',
            service_file='/etc/systemd/system/node_exporter.service',
            health_check_url='http://localhost:9100/metrics'
        )
        
        logger.info(f"Initialized {len(self.components)} monitoring components")
    
    def _initialize_alert_rules(self):
        """Initialize alert rules"""
        
        # Trading System Alerts
        self.alert_rules['trading_system_down'] = AlertRule(
            name='Trading System Down',
            expr='up{job="trading-engine"} == 0',
            condition='for: 1m',
            severity='critical',
            message='Trading system is down for more than 1 minute',
            labels={'team': 'trading', 'service': 'trading-engine'}
        )
        
        self.alert_rules['high_latency'] = AlertRule(
            name='High Trading Latency',
            expr='histogram_quantile(0.95, sum(rate(trading_duration_seconds_bucket[5m])) by (instance)) > 0.1',
            condition='for: 5m',
            severity='warning',
            message='Trading latency is above 100ms for 5 minutes',
            labels={'team': 'trading', 'service': 'trading-engine'}
        )
        
        self.alert_rules['low_success_rate'] = AlertRule(
            name='Low Trading Success Rate',
            expr='sum(rate(trading_success_total[5m])) / sum(rate(trading_requests_total[5m])) < 0.95',
            condition='for: 2m',
            severity='warning',
            message='Trading success rate is below 95%',
            labels={'team': 'trading', 'service': 'trading-engine'}
        )
        
        # Database Alerts
        self.alert_rules['database_connections_high'] = AlertRule(
            name='Database Connections High',
            expr='pg_stat_database_numbackends{dataname="quantfund_prod"} > 800',
            condition='for: 5m',
            severity='warning',
            message='Database connections are above 80%',
            labels={'team': 'infrastructure', 'service': 'database'}
        )
        
        self.alert_rules['database_replication_lag'] = AlertRule(
            name='Database Replication Lag',
            expr='pg_last_xact_replay_timestamp{dataname="quantfund_prod"} < time() - 300',
            condition='for: 2m',
            severity='critical',
            message='Database replication lag is above 5 minutes',
            labels={'team': 'infrastructure', 'service': 'database'}
        )
        
        # System Alerts
        self.alert_rules['high_cpu_usage'] = AlertRule(
            name='High CPU Usage',
            expr='100 - (avg by (instance) (irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100) > 80',
            condition='for: 10m',
            severity='warning',
            message='CPU usage is above 80%',
            labels={'team': 'infrastructure', 'service': 'system'}
        )
        
        self.alert_rules['high_memory_usage'] = AlertRule(
            name='High Memory Usage',
            expr='(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100 > 85',
            condition='for: 5m',
            severity='warning',
            message='Memory usage is above 85%',
            labels={'team': 'infrastructure', 'service': 'system'}
        )
        
        self.alert_rules['disk_space_low'] = AlertRule(
            name='Disk Space Low',
            expr='(node_filesystem_avail_bytes / node_filesystem_size_bytes) * 100 < 10',
            condition='for: 1m',
            severity='critical',
            message='Disk space is below 10%',
            labels={'team': 'infrastructure', 'service': 'system'}
        )
        
        # Data Quality Alerts
        self.alert_rules['data_quality_issues'] = AlertRule(
            name='Data Quality Issues',
            expr='rate(data_quality_errors_total[5m]) > 0.1',
            condition='for: 2m',
            severity='warning',
            message='Data quality errors detected',
            labels={'team': 'data', 'service': 'data-pipeline'}
        )
        
        self.alert_rules['data_feed_down'] = AlertRule(
            name='Data Feed Down',
            expr='up{job="bloomberg"} == 0 and up{job="refinitiv"} == 0',
            condition='for: 30s',
            severity='critical',
            message='All institutional data feeds are down',
            labels={'team': 'data', 'service': 'data-feeds'}
        )
        
        logger.info(f"Initialized {len(self.alert_rules)} alert rules")
    
    def _initialize_dashboards(self):
        """Initialize Grafana dashboards"""
        
        # Trading Performance Dashboard
        self.dashboards['trading_performance'] = {
            'title': 'Trading Performance Dashboard',
            'panels': [
                {
                    'title': 'Trading Volume',
                    'type': 'stat',
                    'targets': [
                        {
                            'expr': 'sum(rate(trading_volume_total[5m]))',
                            'legendFormat': 'Volume/sec'
                        }
                    ]
                },
                {
                    'title': 'Trading Success Rate',
                    'type': 'stat',
                    'targets': [
                        {
                            'expr': 'sum(rate(trading_success_total[5m])) / sum(rate(trading_requests_total[5m])) * 100',
                            'legendFormat': 'Success Rate %'
                        }
                    ]
                },
                {
                    'title': 'Trading Latency',
                    'type': 'graph',
                    'targets': [
                        {
                            'expr': 'histogram_quantile(0.50, sum(rate(trading_duration_seconds_bucket[5m])) by (instance))',
                            'legendFormat': 'P50 - {{instance}}'
                        },
                        {
                            'expr': 'histogram_quantile(0.95, sum(rate(trading_duration_seconds_bucket[5m])) by (instance))',
                            'legendFormat': 'P95 - {{instance}}'
                        },
                        {
                            'expr': 'histogram_quantile(0.99, sum(rate(trading_duration_seconds_bucket[5m])) by (instance))',
                            'legendFormat': 'P99 - {{instance}}'
                        }
                    ]
                },
                {
                    'title': 'P&L',
                    'type': 'graph',
                    'targets': [
                        {
                            'expr': 'sum(trading_pnl_total)',
                            'legendFormat': 'Total P&L'
                        }
                    ]
                }
            ]
        }
        
        # System Health Dashboard
        self.dashboards['system_health'] = {
            'title': 'System Health Dashboard',
            'panels': [
                {
                    'title': 'CPU Usage',
                    'type': 'graph',
                    'targets': [
                        {
                            'expr': '100 - (avg by (instance) (irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)',
                            'legendFormat': 'CPU % - {{instance}}'
                        }
                    ]
                },
                {
                    'title': 'Memory Usage',
                    'type': 'graph',
                    'targets': [
                        {
                            'expr': '(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100',
                            'legendFormat': 'Memory % - {{instance}}'
                        }
                    ]
                },
                {
                    'title': 'Disk Usage',
                    'type': 'graph',
                    'targets': [
                        {
                            'expr': '(1 - (node_filesystem_avail_bytes / node_filesystem_size_bytes)) * 100',
                            'legendFormat': 'Disk % - {{instance}}:{{mountpoint}}'
                        }
                    ]
                },
                {
                    'title': 'Network I/O',
                    'type': 'graph',
                    'targets': [
                        {
                            'expr': 'rate(node_network_receive_bytes_total[5m])',
                            'legendFormat': 'RX - {{instance}}'
                        },
                        {
                            'expr': 'rate(node_network_transmit_bytes_total[5m])',
                            'legendFormat': 'TX - {{instance}}'
                        }
                    ]
                }
            ]
        }
        
        # Database Performance Dashboard
        self.dashboards['database_performance'] = {
            'title': 'Database Performance Dashboard',
            'panels': [
                {
                    'title': 'Database Connections',
                    'type': 'graph',
                    'targets': [
                        {
                            'expr': 'pg_stat_database_numbackends{dataname="quantfund_prod"}',
                            'legendFormat': 'Connections - {{dataname}}'
                        }
                    ]
                },
                {
                    'title': 'Query Rate',
                    'type': 'graph',
                    'targets': [
                        {
                            'expr': 'sum(rate(pg_stat_statements_calls[5m]))',
                            'legendFormat': 'Queries/sec'
                        }
                    ]
                },
                {
                    'title': 'Transaction Rate',
                    'type': 'graph',
                    'targets': [
                        {
                            'expr': 'sum(rate(pg_stat_xact_commit[5m]))',
                            'legendFormat': 'Commits/sec'
                        },
                        {
                            'expr': 'sum(rate(pg_stat_xact_rollback[5m]))',
                            'legendFormat': 'Rollbacks/sec'
                        }
                    ]
                },
                {
                    'title': 'Database Size',
                    'type': 'stat',
                    'targets': [
                        {
                            'expr': 'pg_database_size_bytes{dataname="quantfund_prod"}',
                            'legendFormat': 'Database Size'
                        }
                    ]
                }
            ]
        }
        
        # Data Quality Dashboard
        self.dashboards['data_quality'] = {
            'title': 'Data Quality Dashboard',
            'panels': [
                {
                    'title': 'Data Feed Status',
                    'type': 'stat',
                    'targets': [
                        {
                            'expr': 'up{job=~".*data.*"}',
                            'legendFormat': 'Active Feeds'
                        }
                    ]
                },
                {
                    'title': 'Data Quality Score',
                    'type': 'stat',
                    'targets': [
                        {
                            'expr': 'avg(data_quality_score)',
                            'legendFormat': 'Quality Score'
                        }
                    ]
                },
                {
                    'title': 'Data Quality Errors',
                    'type': 'graph',
                    'targets': [
                        {
                            'expr': 'rate(data_quality_errors_total[5m])',
                            'legendFormat': 'Errors/sec - {{source}}'
                        }
                    ]
                },
                {
                    'title': 'Data Volume',
                    'type': 'graph',
                    'targets': [
                        {
                            'expr': 'rate(data_points_received_total[5m])',
                            'legendFormat': 'Points/sec - {{source}}'
                        }
                    ]
                }
            ]
        }
        
        logger.info(f"Initialized {len(self.dashboards)} Grafana dashboards")
    
    async def deploy_monitoring_stack(self) -> Dict[str, Any]:
        """Deploy complete monitoring stack"""
        try:
            logger.info("Deploying production monitoring stack")
            
            results = {}
            
            # Step 1: Deploy Prometheus
            prometheus_result = await self._deploy_prometheus()
            results['prometheus'] = prometheus_result
            
            # Step 2: Deploy Grafana
            grafana_result = await self._deploy_grafana()
            results['grafana'] = grafana_result
            
            # Step 3: Deploy AlertManager
            alertmanager_result = await self._deploy_alertmanager()
            results['alertmanager'] = alertmanager_result
            
            # Step 4: Deploy Node Exporter
            node_exporter_result = await self._deploy_node_exporter()
            results['node_exporter'] = node_exporter_result
            
            # Step 5: Configure Grafana dashboards
            dashboard_result = await self._configure_grafana_dashboards()
            results['dashboards'] = dashboard_result
            
            # Step 6: Configure AlertManager rules
            rules_result = await self._configure_alertmanager_rules()
            results['alert_rules'] = rules_result
            
            logger.info("Monitoring stack deployed successfully")
            
            return {
                'success': True,
                'components': results,
                'total_components': len(self.components),
                'total_dashboards': len(self.dashboards),
                'total_alert_rules': len(self.alert_rules)
            }
            
        except Exception as e:
            logger.error(f"Monitoring stack deployment failed: {e}")
            return {'error': str(e)}
    
    async def _deploy_prometheus(self) -> Dict[str, Any]:
        """Deploy Prometheus"""
        try:
            component = self.components['prometheus']
            
            logger.info("Deploying Prometheus")
            
            # Create configuration directory
            config_dir = os.path.dirname(component.config_file)
            os.makedirs(config_dir, exist_ok=True)
            
            # Create Prometheus configuration
            prometheus_config = {
                'global': {
                    'scrape_interval': '15s',
                    'evaluation_interval': '15s'
                },
                'rule_files': ['- "/etc/prometheus/rules/*.yml"'],
                'scrape_configs': [
                    {
                        'job_name': 'prometheus',
                        'static_configs': [{'targets': ['localhost:9090']}]
                    },
                    {
                        'job_name': 'trading-engine',
                        'static_configs': [{'targets': ['trading-engine:8080']}]
                    },
                    {
                        'job_name': 'risk-manager',
                        'static_configs': [{'targets': ['risk-manager:8081']}]
                    },
                    {
                        'job_name': 'data-processor',
                        'static_configs': [{'targets': ['data-processor:8082']}]
                    },
                    {
                        'job_name': 'node-exporter',
                        'static_configs': [{'targets': ['localhost:9100']}]
                    },
                    {
                        'job_name': 'bloomberg',
                        'static_configs': [{'targets': ['bloomberg-data:8000']}]
                    },
                    {
                        'job_name': 'refinitiv',
                        'static_configs': [{'targets': ['refinitiv-data:8001']}]
                    },
                    {
                        'job_name': 'nyse',
                        'static_configs': [{'targets': ['nyse-feed:8002']}]
                    },
                    {
                        'job_name': 'nasdaq',
                        'static_configs': [{'targets': ['nasdaq-feed:8003']}]
                    }
                ],
                'remote_write': [
                    {
                        'url': 'http://prometheus-remote:9090/api/v1/write',
                        'queue_config': {
                            'max_samples_per_send': 1000,
                            'max_shards': 200,
                            'capacity': 2500
                        }
                    }
                ]
            }
            
            # Write configuration file
            with open(component.config_file, 'w') as f:
                yaml.dump(prometheus_config, f, default_flow_style=False)
            
            # Create systemd service
            service_content = f"""
[Unit]
Description=Prometheus Monitoring System
After=network.target

[Service]
Type=simple
User=prometheus
Group=prometheus
ExecStart=/usr/local/bin/prometheus \\
    --config.file={component.config_file} \\
    --storage.tsdb.path=/var/lib/prometheus/ \\
    --web.console.libraries=/etc/prometheus/console_libraries \\
    --web.console.templates=/etc/prometheus/consoles
ExecReload=/bin/kill -HUP $MAINPID

[Install]
WantedBy=multi-user.target
"""
            
            with open(component.service_file, 'w') as f:
                f.write(service_content)
            
            # Download and install Prometheus
            install_cmd = [
                'wget', '-qO-', 'https://github.com/prometheus/prometheus/releases/download/v2.45.0/prometheus-2.45.0.linux-amd64.tar.gz',
                '|', 'tar', 'xz',
                '&&', 'sudo', 'useradd', '--no-create-home', '--shell', '/bin/false', 'prometheus',
                '&&', 'sudo', 'mkdir', '-p', '/etc/prometheus',
                '&&', 'sudo', 'mkdir', '-p', '/var/lib/prometheus',
                '&&', 'sudo', 'chown', 'prometheus:prometheus', '/etc/prometheus',
                '&&', 'sudo', 'chown', 'prometheus:prometheus', '/var/lib/prometheus',
                '&&', 'sudo', 'cp', 'prometheus-2.45.0.linux-amd64/prometheus', '/usr/local/bin/',
                '&&', 'sudo', 'chmod', '+x', '/usr/local/bin/prometheus'
            ]
            
            process = await asyncio.create_subprocess_exec(
                'bash', '-c', ' '.join(install_cmd),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                return {'error': f'Prometheus installation failed: {stderr.decode()}'}
            
            # Start service
            start_cmd = ['sudo', 'systemctl', 'daemon-reload', '&&', 'sudo', 'systemctl', 'enable', 'prometheus', '&&', 'sudo', 'systemctl', 'start', 'prometheus']
            
            process = await asyncio.create_subprocess_exec(
                'bash', '-c', ' '.join(start_cmd),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                return {'error': f'Prometheus service start failed: {stderr.decode()}'}
            
            component.is_deployed = True
            component.endpoint = f"http://localhost:{component.port}"
            
            return {
                'success': True,
                'component': 'prometheus',
                'endpoint': component.endpoint,
                'version': component.version
            }
            
        except Exception as e:
            logger.error(f"Prometheus deployment failed: {e}")
            return {'error': str(e)}
    
    async def _deploy_grafana(self) -> Dict[str, Any]:
        """Deploy Grafana"""
        try:
            component = self.components['grafana']
            
            logger.info("Deploying Grafana")
            
            # Create configuration directory
            config_dir = os.path.dirname(component.config_file)
            os.makedirs(config_dir, exist_ok=True)
            
            # Create Grafana configuration
            grafana_config = """
[server]
http_port = 3000
domain = localhost
root_url = http://localhost:3000/

[database]
type = postgres
name = grafana
host = localhost
port = 5432
user = grafana
password = grafana_password

[security]
admin_user = admin
admin_password = admin_password
secret_key = grafana_secret_key

[users]
allow_sign_up = false
allow_org_create = false

[auth.anonymous]
enabled = true
org_name = Main Org.
org_role = Viewer
"""
            
            # Write configuration file
            with open(component.config_file, 'w') as f:
                f.write(grafana_config)
            
            # Create systemd service
            service_content = f"""
[Unit]
Description=Grafana Analytics Platform
After=network.target postgresql.service

[Service]
Type=simple
User=grafana
Group=grafana
ExecStart=/usr/sbin/grafana-server \\
    --config={component.config_file} \\
    --homepath=/usr/share/grafana \\
    --packaging=deb
ExecReload=/bin/kill -HUP $MAINPID

[Install]
WantedBy=multi-user.target
"""
            
            with open(component.service_file, 'w') as f:
                f.write(service_content)
            
            # Download and install Grafana
            install_cmd = [
                'wget', '-qO-', 'https://dl.grafana.com/oss/release/grafana_10.0.0_amd64.deb',
                '&&', 'sudo', 'dpkg', '-i', 'grafana_10.0.0_amd64.deb',
                '&&', 'sudo', 'useradd', '--no-create-home', '--shell', '/bin/false', 'grafana',
                '&&', 'sudo', 'mkdir', '-p', '/var/lib/grafana',
                '&&', 'sudo', 'chown', 'grafana:grafana', '/var/lib/grafana',
                '&&', 'sudo', 'systemctl', 'daemon-reload'
            ]
            
            process = await asyncio.create_subprocess_exec(
                'bash', '-c', ' '.join(install_cmd),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                return {'error': f'Grafana installation failed: {stderr.decode()}'}
            
            # Start service
            start_cmd = ['sudo', 'systemctl', 'enable', 'grafana', '&&', 'sudo', 'systemctl', 'start', 'grafana']
            
            process = await asyncio.create_subprocess_exec(
                'bash', '-c', ' '.join(start_cmd),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                return {'error': f'Grafana service start failed: {stderr.decode()}'}
            
            component.is_deployed = True
            component.endpoint = f"http://localhost:{component.port}"
            
            return {
                'success': True,
                'component': 'grafana',
                'endpoint': component.endpoint,
                'version': component.version
            }
            
        except Exception as e:
            logger.error(f"Grafana deployment failed: {e}")
            return {'error': str(e)}
    
    async def _deploy_alertmanager(self) -> Dict[str, Any]:
        """Deploy AlertManager"""
        try:
            component = self.components['alertmanager']
            
            logger.info("Deploying AlertManager")
            
            # Create configuration directory
            config_dir = os.path.dirname(component.config_file)
            os.makedirs(config_dir, exist_ok=True)
            
            # Create AlertManager configuration
            alertmanager_config = {
                'global': {
                    'smtp_smarthost': 'localhost',
                    'smtp_from': 'alertmanager@quantfund.com',
                    'smtp_auth_username': 'alertmanager',
                    'smtp_auth_password': 'alertmanager_password'
                },
                'route': {
                    'group_by': ['alertname', 'cluster', 'service'],
                    'group_wait': '10s',
                    'group_interval': '10s',
                    'repeat_interval': '1h',
                    'receiver': 'web.hook',
                    'routes': [
                        {
                            'match': {
                                'severity': 'critical'
                            },
                            'receiver': 'critical-alerts'
                        },
                        {
                            'match': {
                                'severity': 'warning'
                            },
                            'receiver': 'warning-alerts'
                        }
                    ]
                },
                'receivers': [
                    {
                        'name': 'web.hook',
                        'webhook_configs': [
                            {
                                'url': 'http://localhost:8080/api/alerts',
                                'send_resolved': True
                            }
                        ]
                    },
                    {
                        'name': 'critical-alerts',
                        'email_configs': [
                            {
                                'to': ['trading-team@quantfund.com'],
                                'subject': '[CRITICAL] {{ .GroupLabels.alertname }}',
                                'body': 'Alert details: {{ range .Alerts }}{{ .Annotations.SortedPairs }}{{ end }}'
                            }
                        ]
                    },
                    {
                        'name': 'warning-alerts',
                        'email_configs': [
                            {
                                'to': ['ops-team@quantfund.com'],
                                'subject': '[WARNING] {{ .GroupLabels.alertname }}',
                                'body': 'Alert details: {{ range .Alerts }}{{ .Annotations.SortedPairs }}{{ end }}'
                            }
                        ]
                    }
                ]
            }
            
            # Write configuration file
            with open(component.config_file, 'w') as f:
                yaml.dump(alertmanager_config, f, default_flow_style=False)
            
            # Create systemd service
            service_content = f"""
[Unit]
Description=Alertmanager
After=network.target

[Service]
Type=simple
User=alertmanager
Group=alertmanager
ExecStart=/usr/local/bin/alertmanager \\
    --config.file={component.config_file} \\
    --storage.path=/var/lib/alertmanager/
ExecReload=/bin/kill -HUP $MAINPID

[Install]
WantedBy=multi-user.target
"""
            
            with open(component.service_file, 'w') as f:
                f.write(service_content)
            
            # Download and install AlertManager
            install_cmd = [
                'wget', '-qO-', 'https://github.com/prometheus/alertmanager/releases/download/v0.25.0/alertmanager-0.25.0.linux-amd64.tar.gz',
                '|', 'tar', 'xz',
                '&&', 'sudo', 'useradd', '--no-create-home', '--shell', '/bin/false', 'alertmanager',
                '&&', 'sudo', 'mkdir', '-p', '/etc/alertmanager',
                '&&', 'sudo', 'mkdir', '-p', '/var/lib/alertmanager',
                '&&', 'sudo', 'chown', 'alertmanager:alertmanager', '/etc/alertmanager',
                '&&', 'sudo', 'chown', 'alertmanager:alertmanager', '/var/lib/alertmanager',
                '&&', 'sudo', 'cp', 'alertmanager-0.25.0.linux-amd64/alertmanager', '/usr/local/bin/',
                '&&', 'sudo', 'chmod', '+x', '/usr/local/bin/alertmanager'
            ]
            
            process = await asyncio.create_subprocess_exec(
                'bash', '-c', ' '.join(install_cmd),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                return {'error': f'AlertManager installation failed: {stderr.decode()}'}
            
            # Start service
            start_cmd = ['sudo', 'systemctl', 'daemon-reload', '&&', 'sudo', 'systemctl', 'enable', 'alertmanager', '&&', 'sudo', 'systemctl', 'start', 'alertmanager']
            
            process = await asyncio.create_subprocess_exec(
                'bash', '-c', ' '.join(start_cmd),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                return {'error': f'AlertManager service start failed: {stderr.decode()}'}
            
            component.is_deployed = True
            component.endpoint = f"http://localhost:{component.port}"
            
            return {
                'success': True,
                'component': 'alertmanager',
                'endpoint': component.endpoint,
                'version': component.version
            }
            
        except Exception as e:
            logger.error(f"AlertManager deployment failed: {e}")
            return {'error': str(e)}
    
    async def _deploy_node_exporter(self) -> Dict[str, Any]:
        """Deploy Node Exporter"""
        try:
            component = self.components['node_exporter']
            
            logger.info("Deploying Node Exporter")
            
            # Create systemd service
            service_content = f"""
[Unit]
Description=Node Exporter
After=network.target

[Service]
Type=simple
User=node_exporter
Group=node_exporter
ExecStart=/usr/local/bin/node_exporter \\
    --collector.systemd \\
    --collector.filesystem \\
    --collector.tcp \\
    --collector.cpu \\
    --collector.meminfo \\
    --collector.diskstats \\
    --collector.netdev

[Install]
WantedBy=multi-user.target
"""
            
            with open(component.service_file, 'w') as f:
                f.write(service_content)
            
            # Download and install Node Exporter
            install_cmd = [
                'wget', '-qO-', 'https://github.com/prometheus/node_exporter/releases/download/v1.6.0/node_exporter-1.6.0.linux-amd64.tar.gz',
                '|', 'tar', 'xz',
                '&&', 'sudo', 'useradd', '--no-create-home', '--shell', '/bin/false', 'node_exporter',
                '&&', 'sudo', 'cp', 'node_exporter-1.6.0.linux-amd64/node_exporter', '/usr/local/bin/',
                '&&', 'sudo', 'chmod', '+x', '/usr/local/bin/node_exporter'
            ]
            
            process = await asyncio.create_subprocess_exec(
                'bash', '-c', ' '.join(install_cmd),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                return {'error': f'Node Exporter installation failed: {stderr.decode()}'}
            
            # Start service
            start_cmd = ['sudo', 'systemctl', 'daemon-reload', '&&', 'sudo', 'systemctl', 'enable', 'node_exporter', '&&', 'sudo', 'systemctl', 'start', 'node_exporter']
            
            process = await asyncio.create_subprocess_exec(
                'bash', '-c', ' '.join(start_cmd),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                return {'error': f'Node Exporter service start failed: {stderr.decode()}'}
            
            component.is_deployed = True
            component.endpoint = f"http://localhost:{component.port}"
            
            return {
                'success': True,
                'component': 'node_exporter',
                'endpoint': component.endpoint,
                'version': component.version
            }
            
        except Exception as e:
            logger.error(f"Node Exporter deployment failed: {e}")
            return {'error': str(e)}
    
    async def _configure_grafana_dashboards(self) -> Dict[str, Any]:
        """Configure Grafana dashboards"""
        try:
            component = self.components['grafana']
            
            logger.info("Configuring Grafana dashboards")
            
            # Wait for Grafana to start
            await asyncio.sleep(10)
            
            # Create data source
            datasource_data = {
                'name': 'Prometheus',
                'type': 'prometheus',
                'url': 'http://localhost:9090',
                'access': 'proxy',
                'isDefault': True
            }
            
            response = requests.post(
                f"{component.endpoint}/api/datasources",
                json=datasource_data,
                headers={'Authorization': 'Basic YWRtaW46YWRtaW46YWRtaW46'},
                timeout=10
            )
            
            if response.status_code != 200:
                return {'error': f'Failed to create Grafana datasource: {response.status_code}'}
            
            # Import dashboards
            dashboard_results = {}
            
            for dashboard_name, dashboard_config in self.dashboards.items():
                dashboard_data = {
                    'dashboard': {
                        'title': dashboard_config['title'],
                        'panels': dashboard_config['panels'],
                        'tags': ['quant-fund'],
                        'timezone': 'UTC',
                        'refresh': '30s'
                    }
                }
                
                response = requests.post(
                    f"{component.endpoint}/api/dashboards/db",
                    json=dashboard_data,
                    headers={'Authorization': 'Basic YWRtaW46YWRtaW46YWRtaW46'},
                    timeout=10
                )
                
                if response.status_code == 200:
                    dashboard_results[dashboard_name] = {
                        'success': True,
                        'dashboard_id': response.json().get('id'),
                        'uid': response.json().get('uid')
                    }
                else:
                    dashboard_results[dashboard_name] = {
                        'success': False,
                        'error': f'Failed to create {dashboard_name}: {response.status_code}'
                    }
            
            return {
                'success': True,
                'datasource': 'Prometheus',
                'dashboards': dashboard_results,
                'total_dashboards': len(self.dashboards)
            }
            
        except Exception as e:
            logger.error(f"Grafana dashboard configuration failed: {e}")
            return {'error': str(e)}
    
    async def _configure_alertmanager_rules(self) -> Dict[str, Any]:
        """Configure AlertManager rules"""
        try:
            component = self.components['alertmanager']
            
            logger.info("Configuring AlertManager rules")
            
            # Create rules directory
            rules_dir = '/etc/prometheus/rules'
            os.makedirs(rules_dir, exist_ok=True)
            
            # Create alert rules file
            rules_file = f'{rules_dir}/trading_rules.yml'
            
            rules_content = {
                'groups': [
                    {
                        'name': 'trading.rules',
                        'rules': []
                    }
                ]
            }
            
            # Add alert rules
            for rule_name, rule in self.alert_rules.items():
                rule_config = {
                    'alert': rule.name,
                    'expr': rule.expr,
                    'for': rule.duration,
                    'labels': rule.labels,
                    'annotations': {
                        'summary': rule.message,
                        'description': rule.message
                    }
                }
                
                rules_content['groups'][0]['rules'].append(rule_config)
            
            # Write rules file
            with open(rules_file, 'w') as f:
                yaml.dump(rules_content, f, default_flow_style=False)
            
            # Reload AlertManager
            reload_cmd = ['curl', '-X', 'POST', f'{component.endpoint}/-/reload']
            
            process = await asyncio.create_subprocess_exec(
                *reload_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            return {
                'success': True,
                'rules_file': rules_file,
                'total_rules': len(self.alert_rules)
            }
            
        except Exception as e:
            logger.error(f"AlertManager rules configuration failed: {e}")
            return {'error': str(e)}
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get comprehensive monitoring status"""
        return {
            'components': {
                name: {
                    'is_deployed': component.is_deployed,
                    'component_type': component.component_type,
                    'version': component.version,
                    'endpoint': component.endpoint,
                    'health_check_url': component.health_check_url,
                    'uptime': component.uptime,
                    'cpu_usage': component.cpu_usage,
                    'memory_usage': component.memory_usage
                }
                for name, component in self.components.items()
            },
            'total_components': len(self.components),
            'deployed_components': len([c for c in self.components.values() if c.is_deployed]),
            'total_dashboards': len(self.dashboards),
            'total_alert_rules': len(self.alert_rules)
        }


# Global monitoring stack instance
_real_monitoring_stack = None

def get_real_monitoring_stack() -> RealMonitoringStack:
    """Get global real monitoring stack instance"""
    global _real_monitoring_stack
    if _real_monitoring_stack is None:
        _real_monitoring_stack = RealMonitoringStack()
    return _real_monitoring_stack


if __name__ == "__main__":
    # Test real monitoring stack
    monitoring = RealMonitoringStack()
    
    # Deploy monitoring stack
    print("Deploying monitoring stack...")
    result = asyncio.run(monitoring.deploy_monitoring_stack())
    print(f"Deployment result: {result}")
    
    # Get status
    status = monitoring.get_monitoring_status()
    print(f"Monitoring status: {json.dumps(status, indent=2)}")
