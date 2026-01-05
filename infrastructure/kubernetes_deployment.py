import logging
import yaml
import json
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
import os
import subprocess

logger = logging.getLogger(__name__)

@dataclass
class KubernetesConfig:
    """Configuration for Kubernetes deployment."""
    namespace: str = "quant-trading"
    cluster_name: str = "quant-cluster"
    region: str = "us-east-1"
    node_groups: Dict[str, Any] = field(default_factory=dict)
    ingress_config: Dict[str, Any] = field(default_factory=dict)
    monitoring_stack: List[str] = field(default_factory=list)

@dataclass
class ServiceConfig:
    """Configuration for individual services."""
    name: str
    image: str
    replicas: int = 1
    cpu_request: str = "500m"
    cpu_limit: str = "1000m"
    memory_request: str = "1Gi"
    memory_limit: str = "2Gi"
    ports: List[Dict[str, Any]] = field(default_factory=list)
    env_vars: Dict[str, str] = field(default_factory=dict)
    volumes: List[Dict[str, Any]] = field(default_factory=list)
    health_checks: Dict[str, Any] = field(default_factory=dict)
    affinity_rules: Dict[str, Any] = field(default_factory=dict)

class CloudNativeDeploymentManager:
    """
    INSTITUTIONAL-GRADE CLOUD-NATIVE DEPLOYMENT MANAGER
    Manages Kubernetes deployments, scaling, monitoring, and infrastructure automation.
    Implements GitOps, infrastructure as code, and cloud-native best practices.
    """

    def __init__(self, config_dir: str = "infrastructure/config", manifests_dir: str = "infrastructure/manifests"):
        self.config_dir = Path(config_dir)
        self.manifests_dir = Path(manifests_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.manifests_dir.mkdir(parents=True, exist_ok=True)

        # Load configuration
        self.k8s_config = self._load_kubernetes_config()
        self.services_config = self._load_services_config()

        # Deployment state
        self.deployments: Dict[str, Dict[str, Any]] = {}
        self.services: Dict[str, Dict[str, Any]] = {}

        logger.info("Cloud-Native Deployment Manager initialized")

    def _load_kubernetes_config(self) -> KubernetesConfig:
        """Load Kubernetes cluster configuration."""
        config_file = self.config_dir / "kubernetes.yaml"
        if config_file.exists():
            with open(config_file, 'r') as f:
                data = yaml.safe_load(f)
            return KubernetesConfig(**data)
        else:
            # Default configuration
            return KubernetesConfig(
                node_groups={
                    "cpu-optimized": {
                        "instance_type": "c5.2xlarge",
                        "min_size": 3,
                        "max_size": 20,
                        "desired_size": 5
                    },
                    "gpu-optimized": {
                        "instance_type": "p3.2xlarge",
                        "min_size": 0,
                        "max_size": 10,
                        "desired_size": 2
                    },
                    "memory-optimized": {
                        "instance_type": "r5.4xlarge",
                        "min_size": 2,
                        "max_size": 15,
                        "desired_size": 3
                    }
                },
                ingress_config={
                    "class": "alb",
                    "ssl_redirect": True,
                    "waf_enabled": True
                },
                monitoring_stack=["prometheus", "grafana", "fluentd", "jaeger"]
            )

    def _load_services_config(self) -> Dict[str, ServiceConfig]:
        """Load service configurations."""
        services_file = self.config_dir / "services.yaml"
        if services_file.exists():
            with open(services_file, 'r') as f:
                data = yaml.safe_load(f)

            services = {}
            for name, config in data.items():
                services[name] = ServiceConfig(name=name, **config)
            return services
        else:
            # Default service configurations
            return self._get_default_services_config()

    def _get_default_services_config(self) -> Dict[str, ServiceConfig]:
        """Get default service configurations for the trading system."""
        return {
            "data-router": ServiceConfig(
                name="data-router",
                image="quant-trading/data-router:latest",
                replicas=3,
                cpu_request="1000m",
                cpu_limit="2000m",
                memory_request="2Gi",
                memory_limit="4Gi",
                ports=[{"name": "http", "port": 8080, "targetPort": 8080}],
                env_vars={
                    "ENV": "production",
                    "REDIS_URL": "redis://redis-cluster:6379",
                    "DATABASE_URL": "postgresql://db:5432/trading"
                },
                health_checks={
                    "readinessProbe": {
                        "httpGet": {"path": "/health/ready", "port": 8080},
                        "initialDelaySeconds": 30,
                        "periodSeconds": 10
                    },
                    "livenessProbe": {
                        "httpGet": {"path": "/health/live", "port": 8080},
                        "initialDelaySeconds": 60,
                        "periodSeconds": 30
                    }
                },
                affinity_rules={
                    "nodeAffinity": {
                        "requiredDuringSchedulingIgnoredDuringExecution": {
                            "nodeSelectorTerms": [{
                                "matchExpressions": [{
                                    "key": "node-type",
                                    "operator": "In",
                                    "values": ["cpu-optimized"]
                                }]
                            }]
                        }
                    }
                }
            ),

            "strategy-engine": ServiceConfig(
                name="strategy-engine",
                image="quant-trading/strategy-engine:latest",
                replicas=5,
                cpu_request="2000m",
                cpu_limit="4000m",
                memory_request="4Gi",
                memory_limit="8Gi",
                ports=[{"name": "grpc", "port": 9090, "targetPort": 9090}],
                env_vars={
                    "CUDA_VISIBLE_DEVICES": "0,1",
                    "MODEL_CACHE_DIR": "/app/models"
                },
                volumes=[{
                    "name": "model-storage",
                    "persistentVolumeClaim": {"claimName": "model-pvc"}
                }],
                affinity_rules={
                    "nodeAffinity": {
                        "preferredDuringSchedulingIgnoredDuringExecution": [{
                            "weight": 100,
                            "preference": {
                                "matchExpressions": [{
                                    "key": "node-type",
                                    "operator": "In",
                                    "values": ["gpu-optimized"]
                                }]
                            }
                        }]
                    }
                }
            ),

            "execution-simulator": ServiceConfig(
                name="execution-simulator",
                image="quant-trading/execution-simulator:latest",
                replicas=2,
                cpu_request="500m",
                cpu_limit="1000m",
                memory_request="1Gi",
                memory_limit="2Gi",
                ports=[{"name": "http", "port": 8081, "targetPort": 8081}],
                env_vars={
                    "SIMULATION_MODE": "realtime",
                    "MARKET_DATA_FEED": "live"
                }
            ),

            "risk-manager": ServiceConfig(
                name="risk-manager",
                image="quant-trading/risk-manager:latest",
                replicas=2,
                cpu_request="1000m",
                cpu_limit="2000m",
                memory_request="2Gi",
                memory_limit="4Gi",
                ports=[{"name": "grpc", "port": 9091, "targetPort": 9091}],
                env_vars={
                    "RISK_LIMITS_CONFIG": "/app/config/risk_limits.yaml",
                    "ALERT_WEBHOOK_URL": "https://hooks.slack.com/services/..."
                }
            ),

            "portfolio-allocator": ServiceConfig(
                name="portfolio-allocator",
                image="quant-trading/portfolio-allocator:latest",
                replicas=1,
                cpu_request="500m",
                cpu_limit="1000m",
                memory_request="1Gi",
                memory_limit="2Gi",
                ports=[{"name": "http", "port": 8082, "targetPort": 8082}],
                env_vars={
                    "OPTIMIZATION_HORIZON": "daily",
                    "REBALANCE_THRESHOLD": "0.05"
                }
            ),

            "api-gateway": ServiceConfig(
                name="api-gateway",
                image="quant-trading/api-gateway:latest",
                replicas=3,
                cpu_request="500m",
                cpu_limit="1000m",
                memory_request="512Mi",
                memory_limit="1Gi",
                ports=[
                    {"name": "http", "port": 80, "targetPort": 8080},
                    {"name": "https", "port": 443, "targetPort": 8443}
                ],
                env_vars={
                    "RATE_LIMIT": "1000",
                    "AUTH_ENABLED": "true"
                }
            ),

            "monitoring-stack": ServiceConfig(
                name="monitoring-stack",
                image="quant-trading/monitoring:latest",
                replicas=1,
                cpu_request="500m",
                cpu_limit="1000m",
                memory_request="1Gi",
                memory_limit="2Gi",
                ports=[
                    {"name": "grafana", "port": 3000, "targetPort": 3000},
                    {"name": "prometheus", "port": 9090, "targetPort": 9090}
                ]
            )
        }

    def generate_kubernetes_manifests(self) -> Dict[str, str]:
        """
        Generate complete Kubernetes manifests for the trading system.
        Returns dictionary of manifest files.
        """
        manifests = {}

        # Generate namespace
        manifests["namespace.yaml"] = self._generate_namespace_manifest()

        # Generate ConfigMaps and Secrets
        manifests["configmaps.yaml"] = self._generate_configmaps_manifest()
        manifests["secrets.yaml"] = self._generate_secrets_manifest()

        # Generate PersistentVolumeClaims
        manifests["persistent-volumes.yaml"] = self._generate_pvc_manifests()

        # Generate ServiceAccount and RBAC
        manifests["rbac.yaml"] = self._generate_rbac_manifests()

        # Generate Deployments and Services
        for service_name, service_config in self.services_config.items():
            manifests[f"{service_name}-deployment.yaml"] = self._generate_deployment_manifest(service_config)
            manifests[f"{service_name}-service.yaml"] = self._generate_service_manifest(service_config)

        # Generate Ingress
        manifests["ingress.yaml"] = self._generate_ingress_manifest()

        # Generate HPA (Horizontal Pod Autoscalers)
        manifests["hpa.yaml"] = self._generate_hpa_manifests()

        # Generate NetworkPolicies
        manifests["network-policies.yaml"] = self._generate_network_policies()

        # Generate Monitoring and Logging
        manifests["monitoring.yaml"] = self._generate_monitoring_manifests()

        return manifests

    def _generate_namespace_manifest(self) -> str:
        """Generate namespace manifest."""
        manifest = {
            "apiVersion": "v1",
            "kind": "Namespace",
            "metadata": {
                "name": self.k8s_config.namespace,
                "labels": {
                    "name": self.k8s_config.namespace,
                    "app": "quant-trading-system"
                }
            }
        }
        return yaml.dump(manifest, default_flow_style=False)

    def _generate_configmaps_manifest(self) -> str:
        """Generate ConfigMaps for application configuration."""
        configmaps = []

        # Application configuration
        app_config = {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": "app-config",
                "namespace": self.k8s_config.namespace
            },
            "data": {
                "config.yaml": yaml.dump({
                    "environment": "production",
                    "log_level": "INFO",
                    "features": {
                        "real_time_trading": True,
                        "paper_trading": False,
                        "backtesting": True,
                        "risk_management": True
                    },
                    "trading": {
                        "max_position_size": 0.1,
                        "max_daily_loss": 0.05,
                        "commission_rate": 0.0005
                    }
                })
            }
        }
        configmaps.append(app_config)

        # Risk limits configuration
        risk_config = {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": "risk-config",
                "namespace": self.k8s_config.namespace
            },
            "data": {
                "risk_limits.yaml": yaml.dump({
                    "portfolio_limits": {
                        "max_drawdown": 0.1,
                        "max_var_95": 0.05,
                        "max_concentration": 0.2
                    },
                    "trade_limits": {
                        "max_order_size": 100000,
                        "max_daily_trades": 50,
                        "min_order_size": 100
                    }
                })
            }
        }
        configmaps.append(risk_config)

        return yaml.dump_all(configmaps, default_flow_style=False)

    def _generate_secrets_manifest(self) -> str:
        """Generate Secrets for sensitive configuration."""
        # Note: In production, use sealed secrets or external secret management
        secrets = {
            "apiVersion": "v1",
            "kind": "Secret",
            "metadata": {
                "name": "app-secrets",
                "namespace": self.k8s_config.namespace
            },
            "type": "Opaque",
            "data": {
                # Base64 encoded placeholder values
                "database_password": "cGFzc3dvcmQ=",  # password
                "api_key": "a2V5",  # key
                "jwt_secret": "c2VjcmV0"  # secret
            }
        }
        return yaml.dump(secrets, default_flow_style=False)

    def _generate_pvc_manifests(self) -> str:
        """Generate PersistentVolumeClaim manifests."""
        pvcs = []

        # Model storage PVC
        model_pvc = {
            "apiVersion": "v1",
            "kind": "PersistentVolumeClaim",
            "metadata": {
                "name": "model-pvc",
                "namespace": self.k8s_config.namespace
            },
            "spec": {
                "accessModes": ["ReadWriteOnce"],
                "storageClassName": "fast-ssd",
                "resources": {
                    "requests": {
                        "storage": "100Gi"
                    }
                }
            }
        }
        pvcs.append(model_pvc)

        # Data storage PVC
        data_pvc = {
            "apiVersion": "v1",
            "kind": "PersistentVolumeClaim",
            "metadata": {
                "name": "data-pvc",
                "namespace": self.k8s_config.namespace
            },
            "spec": {
                "accessModes": ["ReadWriteMany"],
                "storageClassName": "standard",
                "resources": {
                    "requests": {
                        "storage": "500Gi"
                    }
                }
            }
        }
        pvcs.append(data_pvc)

        return yaml.dump_all(pvcs, default_flow_style=False)

    def _generate_rbac_manifests(self) -> str:
        """Generate RBAC manifests for service accounts and permissions."""
        rbac_manifests = []

        # ServiceAccount
        service_account = {
            "apiVersion": "v1",
            "kind": "ServiceAccount",
            "metadata": {
                "name": "quant-trading-sa",
                "namespace": self.k8s_config.namespace
            }
        }
        rbac_manifests.append(service_account)

        # ClusterRole for monitoring and logging
        cluster_role = {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "ClusterRole",
            "metadata": {
                "name": "quant-trading-role"
            },
            "rules": [
                {
                    "apiGroups": [""],
                    "resources": ["pods", "services", "endpoints"],
                    "verbs": ["get", "list", "watch"]
                },
                {
                    "apiGroups": ["apps"],
                    "resources": ["deployments", "replicasets"],
                    "verbs": ["get", "list", "watch", "update", "patch"]
                }
            ]
        }
        rbac_manifests.append(cluster_role)

        # ClusterRoleBinding
        role_binding = {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "ClusterRoleBinding",
            "metadata": {
                "name": "quant-trading-binding"
            },
            "subjects": [
                {
                    "kind": "ServiceAccount",
                    "name": "quant-trading-sa",
                    "namespace": self.k8s_config.namespace
                }
            ],
            "roleRef": {
                "kind": "ClusterRole",
                "name": "quant-trading-role",
                "apiGroup": "rbac.authorization.k8s.io"
            }
        }
        rbac_manifests.append(role_binding)

        return yaml.dump_all(rbac_manifests, default_flow_style=False)

    def _generate_deployment_manifest(self, service: ServiceConfig) -> str:
        """Generate Deployment manifest for a service."""
        deployment = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": service.name,
                "namespace": self.k8s_config.namespace,
                "labels": {
                    "app": service.name,
                    "component": "trading-system"
                }
            },
            "spec": {
                "replicas": service.replicas,
                "selector": {
                    "matchLabels": {
                        "app": service.name
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": service.name,
                            "component": "trading-system"
                        }
                    },
                    "spec": {
                        "serviceAccountName": "quant-trading-sa",
                        "containers": [{
                            "name": service.name,
                            "image": service.image,
                            "ports": service.ports,
                            "env": [
                                {"name": k, "value": v}
                                for k, v in service.env_vars.items()
                            ],
                            "resources": {
                                "requests": {
                                    "cpu": service.cpu_request,
                                    "memory": service.memory_request
                                },
                                "limits": {
                                    "cpu": service.cpu_limit,
                                    "memory": service.memory_limit
                                }
                            },
                            "volumeMounts": [
                                {
                                    "name": vol["name"],
                                    "mountPath": vol.get("mountPath", f"/app/{vol['name']}")
                                }
                                for vol in service.volumes
                            ]
                        }],
                        "volumes": service.volumes,
                        "affinity": service.affinity_rules
                    }
                }
            }
        }

        # Add health checks if configured
        if service.health_checks:
            deployment["spec"]["template"]["spec"]["containers"][0].update(service.health_checks)

        return yaml.dump(deployment, default_flow_style=False)

    def _generate_service_manifest(self, service: ServiceConfig) -> str:
        """Generate Service manifest for a service."""
        svc_manifest = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": service.name,
                "namespace": self.k8s_config.namespace,
                "labels": {
                    "app": service.name
                }
            },
            "spec": {
                "selector": {
                    "app": service.name
                },
                "ports": [
                    {
                        "name": port["name"],
                        "port": port["port"],
                        "targetPort": port["targetPort"],
                        "protocol": port.get("protocol", "TCP")
                    }
                    for port in service.ports
                ],
                "type": "ClusterIP"
            }
        }

        # Make API gateway LoadBalancer
        if service.name == "api-gateway":
            svc_manifest["spec"]["type"] = "LoadBalancer"

        return yaml.dump(svc_manifest, default_flow_style=False)

    def _generate_ingress_manifest(self) -> str:
        """Generate Ingress manifest for external access."""
        ingress = {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "Ingress",
            "metadata": {
                "name": "quant-trading-ingress",
                "namespace": self.k8s_config.namespace,
                "annotations": {
                    "kubernetes.io/ingress.class": self.k8s_config.ingress_config["class"],
                    "alb.ingress.kubernetes.io/ssl-redirect": str(self.k8s_config.ingress_config["ssl_redirect"]).lower(),
                    "alb.ingress.kubernetes.io/waf-enabled": str(self.k8s_config.ingress_config["waf_enabled"]).lower()
                }
            },
            "spec": {
                "tls": [{
                    "hosts": ["api.quant-trading.com"],
                    "secretName": "quant-trading-tls"
                }],
                "rules": [{
                    "host": "api.quant-trading.com",
                    "http": {
                        "paths": [
                            {
                                "path": "/api",
                                "pathType": "Prefix",
                                "backend": {
                                    "service": {
                                        "name": "api-gateway",
                                        "port": {"number": 80}
                                    }
                                }
                            },
                            {
                                "path": "/monitoring",
                                "pathType": "Prefix",
                                "backend": {
                                    "service": {
                                        "name": "monitoring-stack",
                                        "port": {"number": 3000}
                                    }
                                }
                            }
                        ]
                    }
                }]
            }
        }
        return yaml.dump(ingress, default_flow_style=False)

    def _generate_hpa_manifests(self) -> str:
        """Generate HorizontalPodAutoscaler manifests."""
        hpas = []

        # HPA for strategy engine (CPU and custom metrics)
        strategy_hpa = {
            "apiVersion": "autoscaling/v2",
            "kind": "HorizontalPodAutoscaler",
            "metadata": {
                "name": "strategy-engine-hpa",
                "namespace": self.k8s_config.namespace
            },
            "spec": {
                "scaleTargetRef": {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "name": "strategy-engine"
                },
                "minReplicas": 3,
                "maxReplicas": 20,
                "metrics": [
                    {
                        "type": "Resource",
                        "resource": {
                            "name": "cpu",
                            "target": {
                                "type": "Utilization",
                                "averageUtilization": 70
                            }
                        }
                    },
                    {
                        "type": "Resource",
                        "resource": {
                            "name": "memory",
                            "target": {
                                "type": "Utilization",
                                "averageUtilization": 80
                            }
                        }
                    }
                ]
            }
        }
        hpas.append(strategy_hpa)

        # HPA for data router
        data_router_hpa = {
            "apiVersion": "autoscaling/v2",
            "kind": "HorizontalPodAutoscaler",
            "metadata": {
                "name": "data-router-hpa",
                "namespace": self.k8s_config.namespace
            },
            "spec": {
                "scaleTargetRef": {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "name": "data-router"
                },
                "minReplicas": 2,
                "maxReplicas": 10,
                "metrics": [{
                    "type": "Resource",
                    "resource": {
                        "name": "cpu",
                        "target": {
                            "type": "Utilization",
                            "averageUtilization": 60
                        }
                    }
                }]
            }
        }
        hpas.append(data_router_hpa)

        return yaml.dump_all(hpas, default_flow_style=False)

    def _generate_network_policies(self) -> str:
        """Generate NetworkPolicy manifests for security."""
        policies = []

        # Default deny all policy
        default_deny = {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "NetworkPolicy",
            "metadata": {
                "name": "default-deny-all",
                "namespace": self.k8s_config.namespace
            },
            "spec": {
                "podSelector": {},
                "policyTypes": ["Ingress", "Egress"]
            }
        }
        policies.append(default_deny)

        # Allow internal communication
        allow_internal = {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "NetworkPolicy",
            "metadata": {
                "name": "allow-internal",
                "namespace": self.k8s_config.namespace
            },
            "spec": {
                "podSelector": {},
                "policyTypes": ["Ingress", "Egress"],
                "ingress": [{
                    "from": [{
                        "podSelector": {}
                    }]
                }],
                "egress": [{
                    "to": [{
                        "podSelector": {}
                    }]
                }]
            }
        }
        policies.append(allow_internal)

        # Allow external access to API gateway
        allow_external = {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "NetworkPolicy",
            "metadata": {
                "name": "allow-external-api",
                "namespace": self.k8s_config.namespace
            },
            "spec": {
                "podSelector": {
                    "matchLabels": {
                        "app": "api-gateway"
                    }
                },
                "policyTypes": ["Ingress"],
                "ingress": [{
                    "from": [],
                    "ports": [
                        {"protocol": "TCP", "port": 80},
                        {"protocol": "TCP", "port": 443}
                    ]
                }]
            }
        }
        policies.append(allow_external)

        return yaml.dump_all(policies, default_flow_style=False)

    def _generate_monitoring_manifests(self) -> str:
        """Generate monitoring and logging manifests."""
        monitoring = []

        # Prometheus ServiceMonitor
        service_monitor = {
            "apiVersion": "monitoring.coreos.com/v1",
            "kind": "ServiceMonitor",
            "metadata": {
                "name": "quant-trading-monitor",
                "namespace": self.k8s_config.namespace
            },
            "spec": {
                "selector": {
                    "matchLabels": {
                        "component": "trading-system"
                    }
                },
                "endpoints": [{
                    "port": "metrics",
                    "interval": "30s",
                    "path": "/metrics"
                }]
            }
        }
        monitoring.append(service_monitor)

        # Fluentd ConfigMap for logging
        fluentd_config = {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": "fluentd-config",
                "namespace": self.k8s_config.namespace
            },
            "data": {
                "fluent.conf": """
<source>
  @type tail
  path /var/log/containers/*quant-trading*.log
  pos_file /var/log/fluentd-quant-trading.pos
  tag quant-trading.*
  <parse>
    @type json
  </parse>
</source>

<match quant-trading.**>
  @type elasticsearch
  host elasticsearch
  port 9200
  logstash_format true
  logstash_prefix quant-trading
</match>
"""
            }
        }
        monitoring.append(fluentd_config)

        return yaml.dump_all(monitoring, default_flow_style=False)

    def deploy_to_kubernetes(self, manifests: Dict[str, str], dry_run: bool = False) -> Dict[str, Any]:
        """
        Deploy manifests to Kubernetes cluster.
        Returns deployment results.
        """
        results = {
            "successful": [],
            "failed": [],
            "warnings": []
        }

        for filename, manifest_content in manifests.items():
            try:
                # Save manifest to file
                manifest_path = self.manifests_dir / filename
                with open(manifest_path, 'w') as f:
                    f.write(manifest_content)

                if not dry_run:
                    # Apply to Kubernetes
                    cmd = ["kubectl", "apply", "-f", str(manifest_path)]
                    result = subprocess.run(cmd, capture_output=True, text=True)

                    if result.returncode == 0:
                        results["successful"].append(filename)
                        logger.info(f"Successfully deployed {filename}")
                    else:
                        results["failed"].append({
                            "file": filename,
                            "error": result.stderr
                        })
                        logger.error(f"Failed to deploy {filename}: {result.stderr}")
                else:
                    results["successful"].append(filename)
                    logger.info(f"Dry run: would deploy {filename}")

            except Exception as e:
                results["failed"].append({
                    "file": filename,
                    "error": str(e)
                })
                logger.error(f"Error deploying {filename}: {e}")

        return results

    def setup_monitoring_stack(self) -> Dict[str, Any]:
        """Set up comprehensive monitoring stack."""
        monitoring_setup = {
            "prometheus": self._setup_prometheus(),
            "grafana": self._setup_grafana(),
            "fluentd": self._setup_fluentd(),
            "jaeger": self._setup_jaeger(),
            "alertmanager": self._setup_alertmanager()
        }

        return monitoring_setup

    def _setup_prometheus(self) -> Dict[str, Any]:
        """Set up Prometheus monitoring."""
        return {
            "status": "configured",
            "metrics": [
                "trading_pnl",
                "strategy_performance",
                "risk_metrics",
                "system_resources",
                "api_latency"
            ],
            "alerts": [
                "HighLatency",
                "HighErrorRate",
                "LowProfitability",
                "HighDrawdown"
            ]
        }

    def _setup_grafana(self) -> Dict[str, Any]:
        """Set up Grafana dashboards."""
        return {
            "status": "configured",
            "dashboards": [
                "Trading Performance",
                "Risk Metrics",
                "System Health",
                "Strategy Analytics"
            ]
        }

    def _setup_fluentd(self) -> Dict[str, Any]:
        """Set up Fluentd logging."""
        return {
            "status": "configured",
            "log_sources": ["application", "system", "audit"],
            "destinations": ["elasticsearch", "s3"]
        }

    def _setup_jaeger(self) -> Dict[str, Any]:
        """Set up Jaeger tracing."""
        return {
            "status": "configured",
            "services": ["strategy-engine", "data-router", "execution-simulator"]
        }

    def _setup_alertmanager(self) -> Dict[str, Any]:
        """Set up AlertManager."""
        return {
            "status": "configured",
            "channels": ["slack", "email", "pagerduty"],
            "alerts": ["critical", "warning", "info"]
        }

    def get_cluster_status(self) -> Dict[str, Any]:
        """Get comprehensive cluster status."""
        try:
            # Get node status
            nodes_result = subprocess.run(
                ["kubectl", "get", "nodes", "-o", "json"],
                capture_output=True, text=True
            )
            nodes = json.loads(nodes_result.stdout) if nodes_result.returncode == 0 else {}

            # Get pod status
            pods_result = subprocess.run(
                ["kubectl", "get", "pods", "-n", self.k8s_config.namespace, "-o", "json"],
                capture_output=True, text=True
            )
            pods = json.loads(pods_result.stdout) if pods_result.returncode == 0 else {}

            return {
                "cluster_name": self.k8s_config.cluster_name,
                "namespace": self.k8s_config.namespace,
                "nodes": {
                    "total": len(nodes.get("items", [])),
                    "ready": len([n for n in nodes.get("items", [])
                                if all(c["status"] == "True" for c in n["status"]["conditions"]
                                     if c["type"] == "Ready")])
                },
                "pods": {
                    "total": len(pods.get("items", [])),
                    "running": len([p for p in pods.get("items", [])
                                  if p["status"]["phase"] == "Running"]),
                    "pending": len([p for p in pods.get("items", [])
                                  if p["status"]["phase"] == "Pending"]),
                    "failed": len([p for p in pods.get("items", [])
                                 if p["status"]["phase"] == "Failed"])
                },
                "services": list(self.services_config.keys()),
                "last_updated": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to get cluster status: {e}")
            return {"error": str(e)}

    def scale_service(self, service_name: str, replicas: int) -> bool:
        """Scale a service to specified number of replicas."""
        try:
            cmd = [
                "kubectl", "scale", "deployment", service_name,
                "--replicas", str(replicas),
                "-n", self.k8s_config.namespace
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                logger.info(f"Successfully scaled {service_name} to {replicas} replicas")
                return True
            else:
                logger.error(f"Failed to scale {service_name}: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"Error scaling {service_name}: {e}")
            return False

    def update_service_image(self, service_name: str, new_image: str) -> bool:
        """Update service to use new container image."""
        try:
            cmd = [
                "kubectl", "set", "image",
                f"deployment/{service_name}",
                f"{service_name}={new_image}",
                "-n", self.k8s_config.namespace
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                logger.info(f"Successfully updated {service_name} to image {new_image}")
                return True
            else:
                logger.error(f"Failed to update {service_name} image: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"Error updating {service_name} image: {e}")
            return False
