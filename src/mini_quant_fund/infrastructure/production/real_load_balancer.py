#!/usr/bin/env python3
"""
REAL LOAD BALANCER AND HIGH AVAILABILITY FOR TOP 1% TRADING
============================================================

Deploy real production load balancer with:
- HAProxy for HTTP/TCP load balancing
- Nginx for web server load balancing
- Keepalived for high availability
- Health checks and failover
- SSL termination
- Real-time traffic monitoring
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
class LoadBalancerNode:
    """Load balancer node configuration"""
    name: str
    ip_address: str
    role: str  # master, backup
    port: int
    weight: int
    health_check_url: str
    
    # Status
    is_healthy: bool = False
    last_health_check: datetime = field(default_factory=datetime.utcnow)
    connection_count: int = 0
    response_time: float = 0.0


@dataclass
class BackendService:
    """Backend service configuration"""
    name: str
    protocol: str  # http, https, tcp
    port: int
    nodes: List[LoadBalancerNode]
    health_check_interval: int = 5
    health_check_timeout: int = 3
    max_retries: int = 3
    
    # Load balancing algorithm
    algorithm: str = "roundrobin"  # roundrobin, leastconn, source
    
    # SSL
    ssl_enabled: bool = False
    ssl_certificate: str = ""
    ssl_certificate_key: str = ""


class RealLoadBalancer:
    """
    Deploy real production load balancer with high availability.
    
    This creates actual load balancing infrastructure, not simulation.
    """
    
    def __init__(self):
        self.load_balancers: Dict[str, Dict[str, Any]] = {}
        self.backend_services: Dict[str, BackendService] = {}
        self.health_checks: Dict[str, asyncio.Task] = {}
        
        # Initialize load balancers and services
        self._initialize_load_balancers()
        self._initialize_backend_services()
        
        logger.info("Real Load Balancer initialized")
    
    def _initialize_load_balancers(self):
        """Initialize load balancer configurations"""
        
        # HAProxy configuration
        self.load_balancers['haproxy'] = {
            'type': 'haproxy',
            'version': '2.8.0',
            'config_file': '/etc/haproxy/haproxy.cfg',
            'stats_port': 8404,
            'admin_port': 8405,
            'max_connections': 10000,
            'timeout_connect': '5s',
            'timeout_client': '50s',
            'timeout_server': '50s',
            'timeout_http_request': '10s'
        }
        
        # Nginx configuration
        self.load_balancers['nginx'] = {
            'type': 'nginx',
            'version': '1.24.0',
            'config_file': '/etc/nginx/nginx.conf',
            'worker_processes': 'auto',
            'worker_connections': 4096,
            'keepalive_timeout': '65',
            'client_max_body_size': '100M'
        }
        
        # Keepalived configuration
        self.load_balancers['keepalived'] = {
            'type': 'keepalived',
            'version': '2.2.8',
            'config_file': '/etc/keepalived/keepalived.conf',
            'virtual_ip': '10.0.1.100',
            'interface': 'eth0',
            'priority': 100,
            'advert_interval': 1
        }
        
        logger.info(f"Initialized {len(self.load_balancers)} load balancers")
    
    def _initialize_backend_services(self):
        """Initialize backend services"""
        
        # Trading Engine Service
        trading_nodes = [
            LoadBalancerNode(
                name='trading-engine-1',
                ip_address='10.0.1.20',
                role='backend',
                port=8080,
                weight=1,
                health_check_url='http://10.0.1.20:8080/health'
            ),
            LoadBalancerNode(
                name='trading-engine-2',
                ip_address='10.0.1.21',
                role='backend',
                port=8080,
                weight=1,
                health_check_url='http://10.0.1.21:8080/health'
            ),
            LoadBalancerNode(
                name='trading-engine-3',
                ip_address='10.0.1.22',
                role='backend',
                port=8080,
                weight=1,
                health_check_url='http://10.0.1.22:8080/health'
            )
        ]
        
        self.backend_services['trading-engine'] = BackendService(
            name='trading-engine',
            protocol='http',
            port=8080,
            nodes=trading_nodes,
            algorithm='leastconn',
            health_check_interval=5,
            health_check_timeout=3,
            max_retries=3
        )
        
        # Risk Manager Service
        risk_nodes = [
            LoadBalancerNode(
                name='risk-manager-1',
                ip_address='10.0.1.30',
                role='backend',
                port=8081,
                weight=1,
                health_check_url='http://10.0.1.30:8081/health'
            ),
            LoadBalancerNode(
                name='risk-manager-2',
                ip_address='10.0.1.31',
                role='backend',
                port=8081,
                weight=1,
                health_check_url='http://10.0.1.31:8081/health'
            )
        ]
        
        self.backend_services['risk-manager'] = BackendService(
            name='risk-manager',
            protocol='http',
            port=8081,
            nodes=risk_nodes,
            algorithm='roundrobin',
            health_check_interval=5,
            health_check_timeout=3,
            max_retries=3
        )
        
        # Data Processor Service
        data_nodes = [
            LoadBalancerNode(
                name='data-processor-1',
                ip_address='10.0.1.40',
                role='backend',
                port=8082,
                weight=1,
                health_check_url='http://10.0.1.40:8082/health'
            ),
            LoadBalancerNode(
                name='data-processor-2',
                ip_address='10.0.1.41',
                role='backend',
                port=8082,
                weight=1,
                health_check_url='http://10.0.1.41:8082/health'
            ),
            LoadBalancerNode(
                name='data-processor-3',
                ip_address='10.0.1.42',
                role='backend',
                port=8082,
                weight=1,
                health_check_url='http://10.0.1.42:8082/health'
            )
        ]
        
        self.backend_services['data-processor'] = BackendService(
            name='data-processor',
            protocol='http',
            port=8082,
            nodes=data_nodes,
            algorithm='roundrobin',
            health_check_interval=5,
            health_check_timeout=3,
            max_retries=3
        )
        
        # Database Service (TCP)
        db_nodes = [
            LoadBalancerNode(
                name='timescaledb-primary',
                ip_address='10.0.1.10',
                role='primary',
                port=5432,
                weight=1,
                health_check_url='tcp://10.0.1.10:5432'
            ),
            LoadBalancerNode(
                name='timescaledb-replica-1',
                ip_address='10.0.1.11',
                role='replica',
                port=5432,
                weight=1,
                health_check_url='tcp://10.0.1.11:5432'
            ),
            LoadBalancerNode(
                name='timescaledb-replica-2',
                ip_address='10.0.1.12',
                role='replica',
                port=5432,
                weight=1,
                health_check_url='tcp://10.0.1.12:5432'
            )
        ]
        
        self.backend_services['database'] = BackendService(
            name='database',
            protocol='tcp',
            port=5432,
            nodes=db_nodes,
            algorithm='roundrobin',
            health_check_interval=10,
            health_check_timeout=5,
            max_retries=3
        )
        
        logger.info(f"Initialized {len(self.backend_services)} backend services")
    
    async def deploy_load_balancer_stack(self) -> Dict[str, Any]:
        """Deploy complete load balancer stack"""
        try:
            logger.info("Deploying production load balancer stack")
            
            results = {}
            
            # Step 1: Deploy HAProxy
            haproxy_result = await self._deploy_haproxy()
            results['haproxy'] = haproxy_result
            
            # Step 2: Deploy Nginx
            nginx_result = await self._deploy_nginx()
            results['nginx'] = nginx_result
            
            # Step 3: Deploy Keepalived
            keepalived_result = await self._deploy_keepalived()
            results['keepalived'] = keepalived_result
            
            # Step 4: Start health checks
            health_check_result = await self._start_health_checks()
            results['health_checks'] = health_check_result
            
            # Step 5: Configure SSL certificates
            ssl_result = await self._configure_ssl_certificates()
            results['ssl'] = ssl_result
            
            logger.info("Load balancer stack deployed successfully")
            
            return {
                'success': True,
                'components': results,
                'total_components': len(self.load_balancers),
                'total_services': len(self.backend_services)
            }
            
        except Exception as e:
            logger.error(f"Load balancer stack deployment failed: {e}")
            return {'error': str(e)}
    
    async def _deploy_haproxy(self) -> Dict[str, Any]:
        """Deploy HAProxy load balancer"""
        try:
            config = self.load_balancers['haproxy']
            
            logger.info("Deploying HAProxy")
            
            # Create configuration directory
            config_dir = os.path.dirname(config['config_file'])
            os.makedirs(config_dir, exist_ok=True)
            
            # Generate HAProxy configuration
            haproxy_config = f"""
global
    daemon
    maxconn {config['max_connections']}
    user haproxy
    group haproxy
    log /dev/log local0
    log /dev/log local1 notice

defaults
    log global
    mode http
    timeout connect {config['timeout_connect']}
    timeout client {config['timeout_client']}
    timeout server {config['timeout_server']}
    timeout http-request {config['timeout_http_request']}
    option httplog
    option dontlognull
    option redispatch
    retries 3
    maxconn 2000

# Statistics page
stats enable
stats uri /stats
stats realm HAProxy Statistics
stats auth admin:admin_password
stats refresh 30s
stats show-node

# Frontend for HTTP traffic
frontend http_frontend
    bind *:80
    bind *:443 ssl crt /etc/ssl/certs/quantfund.pem
    mode http
    option httplog
    option forwardfor
    option http-server-close
    stats uri /haproxy?stats
    default_backend trading_engine_pool

# Frontend for TCP traffic
frontend tcp_frontend
    bind *:3306
    mode tcp
    default_backend database_pool

# Trading Engine Backend
backend trading_engine_pool
    mode http
    balance {self.backend_services['trading-engine'].algorithm}
    option httpchk GET /health
    server trading-engine-1 10.0.1.20:8080 weight 1 check inter 5000 rise 2 fall 3
    server trading-engine-2 10.0.1.21:8080 weight 1 check inter 5000 rise 2 fall 3
    server trading-engine-3 10.0.1.22:8080 weight 1 check inter 5000 rise 2 fall 3

# Risk Manager Backend
backend risk_manager_pool
    mode http
    balance {self.backend_services['risk-manager'].algorithm}
    option httpchk GET /health
    server risk-manager-1 10.0.1.30:8081 weight 1 check inter 5000 rise 2 fall 3
    server risk-manager-2 10.0.1.31:8081 weight 1 check inter 5000 rise 2 fall 3

# Data Processor Backend
backend data_processor_pool
    mode http
    balance {self.backend_services['data-processor'].algorithm}
    option httpchk GET /health
    server data-processor-1 10.0.1.40:8082 weight 1 check inter 5000 rise 2 fall 3
    server data-processor-2 10.0.1.41:8082 weight 1 check inter 5000 rise 2 fall 3
    server data-processor-3 10.0.1.42:8082 weight 1 check inter 5000 rise 2 fall 3

# Database Backend (TCP)
backend database_pool
    mode tcp
    balance roundrobin
    option tcp-check
    server timescaledb-primary 10.0.1.10:5432 weight 1 check inter 10000 rise 2 fall 3
    server timescaledb-replica-1 10.0.1.11:5432 weight 1 check inter 10000 rise 2 fall 3
    server timescaledb-replica-2 10.0.1.12:5432 weight 1 check inter 10000 rise 2 fall 3

# Admin interface
listen admin_interface
    bind *:{config['admin_port']}
    mode http
    stats enable
    stats uri /
    stats realm HAProxy Statistics
    stats auth admin:admin_password
"""
            
            # Write configuration file
            with open(config['config_file'], 'w') as f:
                f.write(haproxy_config)
            
            # Install HAProxy
            install_cmd = [
                'sudo', 'apt-get', 'update',
                '&&', 'sudo', 'apt-get', 'install', '-y', 'haproxy',
                '&&', 'sudo', 'useradd', '--system', '--no-create-home', 'haproxy',
                '&&', 'sudo', 'mkdir', '-p', '/var/lib/haproxy',
                '&&', 'sudo', 'chown', 'haproxy:haproxy', '/var/lib/haproxy'
            ]
            
            process = await asyncio.create_subprocess_exec(
                'bash', '-c', ' '.join(install_cmd),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                return {'error': f'HAProxy installation failed: {stderr.decode()}'}
            
            # Test configuration
            test_cmd = ['sudo', 'haproxy', '-f', config['config_file'], '-c']
            
            process = await asyncio.create_subprocess_exec(
                *test_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                return {'error': f'HAProxy configuration test failed: {stderr.decode()}'}
            
            # Start HAProxy
            start_cmd = ['sudo', 'systemctl', 'enable', 'haproxy', '&&', 'sudo', 'systemctl', 'start', 'haproxy']
            
            process = await asyncio.create_subprocess_exec(
                'bash', '-c', ' '.join(start_cmd),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                return {'error': f'HAProxy service start failed: {stderr.decode()}'}
            
            return {
                'success': True,
                'component': 'haproxy',
                'version': config['version'],
                'stats_port': config['stats_port'],
                'admin_port': config['admin_port'],
                'config_file': config['config_file']
            }
            
        except Exception as e:
            logger.error(f"HAProxy deployment failed: {e}")
            return {'error': str(e)}
    
    async def _deploy_nginx(self) -> Dict[str, Any]:
        """Deploy Nginx web server"""
        try:
            config = self.load_balancers['nginx']
            
            logger.info("Deploying Nginx")
            
            # Create configuration directory
            config_dir = os.path.dirname(config['config_file'])
            os.makedirs(config_dir, exist_ok=True)
            
            # Generate Nginx configuration
            nginx_config = f"""
user www-data;
worker_processes {config['worker_processes']};
pid /run/nginx.pid;
include /etc/nginx/modules-enabled/*.conf;

events {{
    worker_connections {config['worker_connections']};
    multi_accept on;
}}

http {{
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout {config['keepalive_timeout']};
    types_hash_max_size 2048;
    client_max_body_size {config['client_max_body_size']};

    include /etc/nginx/mime.types;
    default_type application/octet-stream;

    # Logging
    access_log /var/log/nginx/access.log;
    error_log /var/log/nginx/error.log;

    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_proxied any;
    gzip_comp_level 6;
    gzip_types text/plain text/css text/xml text/javascript application/json application/javascript application/xml+rss;

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=trading:10m rate=100r/s;

    # Upstream for trading engine
    upstream trading_engine {{
        least_conn;
        server 10.0.1.20:8080 weight=1 max_fails=3 fail_timeout=30s;
        server 10.0.1.21:8080 weight=1 max_fails=3 fail_timeout=30s;
        server 10.0.1.22:8080 weight=1 max_fails=3 fail_timeout=30s;
    }}

    # Upstream for risk manager
    upstream risk_manager {{
        server 10.0.1.30:8081 weight=1 max_fails=3 fail_timeout=30s;
        server 10.0.1.31:8081 weight=1 max_fails=3 fail_timeout=30s;
    }}

    # Upstream for data processor
    upstream data_processor {{
        server 10.0.1.40:8082 weight=1 max_fails=3 fail_timeout=30s;
        server 10.0.1.41:8082 weight=1 max_fails=3 fail_timeout=30s;
        server 10.0.1.42:8082 weight=1 max_fails=3 fail_timeout=30s;
    }}

    # Main server block
    server {{
        listen 80;
        listen [::]:80;
        server_name quantfund.com www.quantfund.com;
        return 301 https://$server_name$request_uri;
    }}

    # HTTPS server block
    server {{
        listen 443 ssl http2;
        listen [::]:443 ssl http2;
        server_name quantfund.com www.quantfund.com;

        # SSL configuration
        ssl_certificate /etc/ssl/certs/quantfund.pem;
        ssl_certificate_key /etc/ssl/private/quantfund.key;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
        ssl_prefer_server_ciphers off;
        ssl_session_cache shared:SSL:10m;
        ssl_session_timeout 10m;

        # Security headers
        add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload";
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection "1; mode=block";

        # Trading API endpoints
        location /api/trading/ {{
            limit_req zone=trading burst=20 nodelay;
            proxy_pass http://trading_engine;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_connect_timeout 5s;
            proxy_send_timeout 10s;
            proxy_read_timeout 10s;
        }}

        # Risk management API endpoints
        location /api/risk/ {{
            limit_req zone=api burst=10 nodelay;
            proxy_pass http://risk_manager;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }}

        # Data processing API endpoints
        location /api/data/ {{
            limit_req zone=api burst=10 nodelay;
            proxy_pass http://data_processor;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }}

        # Health check endpoint
        location /health {{
            access_log off;
            return 200 "healthy\\n";
            add_header Content-Type text/plain;
        }}

        # Static files
        location /static/ {{
            alias /var/www/static/;
            expires 1y;
            add_header Cache-Control "public, immutable";
        }}
    }}
}}
"""
            
            # Write configuration file
            with open(config['config_file'], 'w') as f:
                f.write(nginx_config)
            
            # Install Nginx
            install_cmd = [
                'sudo', 'apt-get', 'update',
                '&&', 'sudo', 'apt-get', 'install', '-y', 'nginx',
                '&&', 'sudo', 'mkdir', '-p', '/var/www/static',
                '&&', 'sudo', 'chown', 'www-data:www-data', '/var/www/static'
            ]
            
            process = await asyncio.create_subprocess_exec(
                'bash', '-c', ' '.join(install_cmd),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                return {'error': f'Nginx installation failed: {stderr.decode()}'}
            
            # Test configuration
            test_cmd = ['sudo', 'nginx', '-t']
            
            process = await asyncio.create_subprocess_exec(
                *test_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                return {'error': f'Nginx configuration test failed: {stderr.decode()}'}
            
            # Start Nginx
            start_cmd = ['sudo', 'systemctl', 'enable', 'nginx', '&&', 'sudo', 'systemctl', 'start', 'nginx']
            
            process = await asyncio.create_subprocess_exec(
                'bash', '-c', ' '.join(start_cmd),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                return {'error': f'Nginx service start failed: {stderr.decode()}'}
            
            return {
                'success': True,
                'component': 'nginx',
                'version': config['version'],
                'worker_processes': config['worker_processes'],
                'worker_connections': config['worker_connections'],
                'config_file': config['config_file']
            }
            
        except Exception as e:
            logger.error(f"Nginx deployment failed: {e}")
            return {'error': str(e)}
    
    async def _deploy_keepalived(self) -> Dict[str, Any]:
        """Deploy Keepalived for high availability"""
        try:
            config = self.load_balancers['keepalived']
            
            logger.info("Deploying Keepalived")
            
            # Create configuration directory
            config_dir = os.path.dirname(config['config_file'])
            os.makedirs(config_dir, exist_ok=True)
            
            # Generate Keepalived configuration
            keepalived_config = f"""
global_defs {{
    router_id LB_MASTER
    enable_script_security
}}

vrrp_script chk_haproxy {{
    script "/usr/local/bin/check_haproxy.sh"
    interval 2
    weight -2
    fall 2
    rise 2
}}

vrrp_instance VI_1 {{
    state MASTER
    interface {config['interface']}
    virtual_router_id 51
    priority {config['priority']}
    advert_int {config['advert_interval']}
    authentication {{
        auth_type PASS
        auth_pass quantfund_ha
    }}
    virtual_ipaddress {{
        {config['virtual_ip']}
    }}
    track_script {{
        chk_haproxy
    }}
    notify_master "/usr/local/bin/keepalived_notify.sh master"
    notify_backup "/usr/local/bin/keepalived_notify.sh backup"
    notify_fault "/usr/local/bin/keepalived_notify.sh fault"
}}
"""
            
            # Write configuration file
            with open(config['config_file'], 'w') as f:
                f.write(keepalived_config)
            
            # Create health check script
            health_check_script = """#!/bin/bash
# HAProxy health check script
if systemctl is-active --quiet haproxy; then
    exit 0
else
    exit 1
fi
"""
            
            with open('/usr/local/bin/check_haproxy.sh', 'w') as f:
                f.write(health_check_script)
            
            # Create notification script
            notify_script = """#!/bin/bash
# Keepalived notification script
STATE=$1
logger "Keepalived state changed to: $STATE"

# Send notification to monitoring system
curl -X POST http://localhost:8080/api/keepalived/notify \\
    -H "Content-Type: application/json" \\
    -d '{{"state": "'$STATE'", "timestamp": "'$(date -u +%Y-%m-%dT%H:%M:%SZ)'"}}'
"""
            
            with open('/usr/local/bin/keepalived_notify.sh', 'w') as f:
                f.write(notify_script)
            
            # Make scripts executable
            chmod_cmd = [
                'sudo', 'chmod', '+x', '/usr/local/bin/check_haproxy.sh',
                '&&', 'sudo', 'chmod', '+x', '/usr/local/bin/keepalived_notify.sh'
            ]
            
            process = await asyncio.create_subprocess_exec(
                *chmod_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            # Install Keepalived
            install_cmd = [
                'sudo', 'apt-get', 'update',
                '&&', 'sudo', 'apt-get', 'install', '-y', 'keepalived'
            ]
            
            process = await asyncio.create_subprocess_exec(
                'bash', '-c', ' '.join(install_cmd),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                return {'error': f'Keepalived installation failed: {stderr.decode()}'}
            
            # Start Keepalived
            start_cmd = ['sudo', 'systemctl', 'enable', 'keepalived', '&&', 'sudo', 'systemctl', 'start', 'keepalived']
            
            process = await asyncio.create_subprocess_exec(
                'bash', '-c', ' '.join(start_cmd),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                return {'error': f'Keepalived service start failed: {stderr.decode()}'}
            
            return {
                'success': True,
                'component': 'keepalived',
                'version': config['version'],
                'virtual_ip': config['virtual_ip'],
                'interface': config['interface'],
                'priority': config['priority']
            }
            
        except Exception as e:
            logger.error(f"Keepalived deployment failed: {e}")
            return {'error': str(e)}
    
    async def _start_health_checks(self) -> Dict[str, Any]:
        """Start health checks for all backend services"""
        try:
            logger.info("Starting health checks")
            
            for service_name, service in self.backend_services.items():
                # Create health check task for each service
                task = asyncio.create_task(
                    self._health_check_loop(service_name, service)
                )
                self.health_checks[service_name] = task
            
            return {
                'success': True,
                'total_health_checks': len(self.health_checks)
            }
            
        except Exception as e:
            logger.error(f"Health checks start failed: {e}")
            return {'error': str(e)}
    
    async def _health_check_loop(self, service_name: str, service: BackendService):
        """Health check loop for a service"""
        try:
            while True:
                for node in service.nodes:
                    try:
                        # Perform health check
                        start_time = time.time()
                        
                        if service.protocol == 'http':
                            response = requests.get(
                                node.health_check_url,
                                timeout=service.health_check_timeout
                            )
                            is_healthy = response.status_code == 200
                        else:  # TCP
                            import socket
                            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                            sock.settimeout(service.health_check_timeout)
                            result = sock.connect_ex((node.ip_address, node.port))
                            is_healthy = result == 0
                            sock.close()
                        
                        end_time = time.time()
                        response_time = (end_time - start_time) * 1000  # Convert to ms
                        
                        # Update node status
                        node.is_healthy = is_healthy
                        node.last_health_check = datetime.utcnow()
                        node.response_time = response_time
                        
                        if not is_healthy:
                            logger.warning(f"Node {node.name} health check failed")
                        
                    except Exception as e:
                        logger.error(f"Health check error for {node.name}: {e}")
                        node.is_healthy = False
                        node.last_health_check = datetime.utcnow()
                
                # Wait for next check
                await asyncio.sleep(service.health_check_interval)
                
        except Exception as e:
            logger.error(f"Health check loop error for {service_name}: {e}")
    
    async def _configure_ssl_certificates(self) -> Dict[str, Any]:
        """Configure SSL certificates"""
        try:
            logger.info("Configuring SSL certificates")
            
            # Create SSL directory
            ssl_dir = '/etc/ssl/certs'
            os.makedirs(ssl_dir, exist_ok=True)
            
            # Generate self-signed certificate (in production, use Let's Encrypt or commercial cert)
            cert_cmd = [
                'sudo', 'openssl', 'req', '-x509', '-nodes', '-days', '365',
                '-newkey', 'rsa:4096',
                '-keyout', '/etc/ssl/private/quantfund.key',
                '-out', '/etc/ssl/certs/quantfund.pem',
                '-subj', '/C=US/ST=NY/L=New York/O=Quant Fund/CN=quantfund.com'
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cert_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                return {'error': f'SSL certificate generation failed: {stderr.decode()}'}
            
            # Set proper permissions
            chmod_cmd = [
                'sudo', 'chmod', '600', '/etc/ssl/private/quantfund.key',
                '&&', 'sudo', 'chmod', '644', '/etc/ssl/certs/quantfund.pem'
            ]
            
            process = await asyncio.create_subprocess_exec(
                'bash', '-c', ' '.join(chmod_cmd),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            return {
                'success': True,
                'certificate_file': '/etc/ssl/certs/quantfund.pem',
                'key_file': '/etc/ssl/private/quantfund.key'
            }
            
        except Exception as e:
            logger.error(f"SSL certificate configuration failed: {e}")
            return {'error': str(e)}
    
    def get_load_balancer_status(self) -> Dict[str, Any]:
        """Get comprehensive load balancer status"""
        return {
            'load_balancers': {
                name: {
                    'type': config['type'],
                    'version': config['version'],
                    'config_file': config['config_file']
                }
                for name, config in self.load_balancers.items()
            },
            'backend_services': {
                name: {
                    'protocol': service.protocol,
                    'port': service.port,
                    'algorithm': service.algorithm,
                    'total_nodes': len(service.nodes),
                    'healthy_nodes': len([n for n in service.nodes if n.is_healthy]),
                    'nodes': {
                        node.name: {
                            'ip_address': node.ip_address,
                            'port': node.port,
                            'is_healthy': node.is_healthy,
                            'last_health_check': node.last_health_check.isoformat(),
                            'response_time': node.response_time
                        }
                        for node in service.nodes
                    }
                }
                for name, service in self.backend_services.items()
            },
            'total_load_balancers': len(self.load_balancers),
            'total_services': len(self.backend_services),
            'health_checks_running': len(self.health_checks)
        }


# Global load balancer instance
_real_load_balancer = None

def get_real_load_balancer() -> RealLoadBalancer:
    """Get global real load balancer instance"""
    global _real_load_balancer
    if _real_load_balancer is None:
        _real_load_balancer = RealLoadBalancer()
    return _real_load_balancer


if __name__ == "__main__":
    # Test real load balancer
    load_balancer = RealLoadBalancer()
    
    # Deploy load balancer stack
    print("Deploying load balancer stack...")
    result = asyncio.run(load_balancer.deploy_load_balancer_stack())
    print(f"Deployment result: {result}")
    
    # Get status
    status = load_balancer.get_load_balancer_status()
    print(f"Load balancer status: {json.dumps(status, indent=2)}")
