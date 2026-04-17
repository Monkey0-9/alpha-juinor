# Production Deployment Guide - MiniQuantFund v4.0.0

## Overview

This guide provides comprehensive instructions for deploying MiniQuantFund v4.0.0 in a production environment for live trading with real money.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Infrastructure Requirements](#infrastructure-requirements)
3. [Security Setup](#security-setup)
4. [Configuration](#configuration)
5. [Deployment Steps](#deployment-steps)
6. [Monitoring and Maintenance](#monitoring-and-maintenance)
7. [Troubleshooting](#troubleshooting)
8. [Emergency Procedures](#emergency-procedures)

## Prerequisites

### System Requirements

- **Operating System**: Ubuntu 20.04 LTS or CentOS 8
- **CPU**: Minimum 16 cores, recommended 32 cores
- **Memory**: Minimum 64GB RAM, recommended 128GB RAM
- **Storage**: Minimum 1TB SSD, recommended 2TB NVMe SSD
- **Network**: 10Gbps connection to exchange colocation
- **Power**: Dual power supplies with UPS backup

### Software Requirements

- **Docker**: 20.10+
- **Kubernetes**: 1.21+
- **Helm**: 3.7+
- **Python**: 3.11+
- **PostgreSQL**: 14+
- **Redis**: 6.2+
- **Prometheus**: 2.30+
- **Grafana**: 8.0+

### Regulatory Requirements

- FINRA Broker-Dealer Registration
- SEC Registration
- MiFID II Compliance (EU)
- AML Program Implementation
- Cybersecurity Framework Compliance

## Infrastructure Requirements

### Network Architecture

```
[Internet] -> [Firewall] -> [DMZ] -> [Internal Network]
                                   |
                                   v
                              [Load Balancer]
                                   |
                    +--------------+--------------+
                    |              |              |
                [Web Tier]   [App Tier]   [Data Tier]
                    |              |              |
                [K8s Cluster] [K8s Cluster] [Database]
```

### Kubernetes Cluster Configuration

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: miniquantfund-production
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: production-config
  namespace: miniquantfund-production
data:
  ENVIRONMENT: "production"
  LOG_LEVEL: "INFO"
  MAX_CONCURRENT_ORDERS: "1000"
  RISK_CHECK_INTERVAL: "1"
```

### Storage Configuration

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: logs-pvc
  namespace: miniquantfund-production
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 500Gi
  storageClassName: fast-ssd
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: data-pvc
  namespace: miniquantfund-production
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 1Ti
  storageClassName: nvme-ssd
```

## Security Setup

### SSL/TLS Configuration

1. Generate SSL certificates:
```bash
openssl req -x509 -nodes -days 365 -newkey rsa:4096 \
  -keyout /etc/ssl/private/miniquantfund.key \
  -out /etc/ssl/certs/miniquantfund.crt \
  -subj "/C=US/ST=NY/L=New York/O=MiniQuantFund/CN=miniquantfund.com"
```

2. Configure Kubernetes secrets:
```bash
kubectl create secret tls miniquantfund-tls \
  --cert=/etc/ssl/certs/miniquantfund.crt \
  --key=/etc/ssl/private/miniquantfund.key \
  --namespace=miniquantfund-production
```

### Database Security

1. Enable PostgreSQL SSL:
```sql
-- postgresql.conf
ssl = on
ssl_cert_file = '/etc/ssl/certs/server.crt'
ssl_key_file = '/etc/ssl/private/server.key'
ssl_ca_file = '/etc/ssl/certs/ca.crt'
```

2. Create database users with limited privileges:
```sql
CREATE USER trading_app WITH PASSWORD 'secure_password';
GRANT CONNECT ON DATABASE miniquantfund TO trading_app;
GRANT USAGE ON SCHEMA public TO trading_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO trading_app;
```

### Network Security

1. Configure firewall rules:
```bash
# Allow only necessary ports
ufw allow 443/tcp    # HTTPS
ufw allow 8443/tcp   # Trading API
ufw allow 9090/tcp   # Monitoring
ufw deny 22/tcp      # SSH (restricted to management network)
ufw enable
```

2. Setup network policies:
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: miniquantfund-network-policy
  namespace: miniquantfund-production
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: miniquantfund-production
    ports:
    - protocol: TCP
      port: 8080
    - protocol: TCP
      port: 9090
```

## Configuration

### Environment Variables

Create a secure environment file:

```bash
# .env.production
export ALPACA_API_KEY="your_alpaca_api_key"
export ALPACA_SECRET_KEY="your_alpaca_secret_key"
export BLOOMBERG_API_KEY="your_bloomberg_api_key"
export REUTERS_API_KEY="your_reuters_api_key"
export POLYGON_API_KEY="your_polygon_api_key"
export DB_PASSWORD="secure_db_password"
export SECURITY_MASTER_PASSWORD="secure_master_password"
export COMPLIANCE_ENCRYPTION_PASSWORD="secure_compliance_password"
export GRAFANA_AUTH="admin:secure_grafana_password"
```

### Production Configuration

Update `config/production.json` with your specific settings:

```json
{
  "market_data": {
    "feeds": [
      {
        "source": "alpaca",
        "enabled": true,
        "priority": 1,
        "api_key": "${ALPACA_API_KEY}",
        "secret_key": "${ALPACA_SECRET_KEY}",
        "base_url": "https://api.alpaca.markets"
      }
    ]
  },
  "risk_management": {
    "position_limits": [
      {
        "symbol": "AAPL",
        "asset_class": "equity",
        "max_position": 10000,
        "max_notional": 1500000,
        "max_percentage": 0.10
      }
    ],
    "circuit_breakers": [
      {
        "name": "portfolio_drawdown",
        "type": "drawdown_limit",
        "threshold": 100000,
        "action": "stop_trading",
        "auto_reset": true,
        "reset_minutes": 30
      }
    ]
  }
}
```

## Deployment Steps

### Step 1: Prepare Infrastructure

1. Install Kubernetes cluster:
```bash
# For production, use managed Kubernetes service
# AWS EKS, Google GKE, or Azure AKS recommended
```

2. Install required operators:
```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Install Prometheus
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring --create-namespace

# Install Grafana
helm install grafana prometheus-community/grafana \
  --namespace monitoring --create-namespace
```

### Step 2: Deploy Database

1. Deploy PostgreSQL:
```bash
helm install postgres bitnami/postgresql \
  --namespace miniquantfund-production \
  --set auth.postgresPassword="${DB_PASSWORD}" \
  --set auth.database="miniquantfund" \
  --set primary.persistence.size="1Ti" \
  --set primary.resources.requests.memory="8Gi" \
  --set primary.resources.requests.cpu="4"
```

2. Deploy Redis:
```bash
helm install redis bitnami/redis \
  --namespace miniquantfund-production \
  --set auth.password="${REDIS_PASSWORD}" \
  --set master.persistence.size="100Gi" \
  --set replica.replicaCount="3"
```

### Step 3: Deploy Application

1. Create deployment manifests:
```yaml
# trading-engine-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: trading-engine
  namespace: miniquantfund-production
spec:
  replicas: 3
  selector:
    matchLabels:
      app: trading-engine
  template:
    metadata:
      labels:
        app: trading-engine
    spec:
      containers:
      - name: trading-engine
        image: miniquantfund/trading-engine:4.0.0
        ports:
        - containerPort: 8080
        env:
        - name: ENV
          value: "production"
        - name: DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: password
        resources:
          requests:
            cpu: 500m
            memory: 512Mi
          limits:
            cpu: 1000m
            memory: 1Gi
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
```

2. Deploy services:
```bash
kubectl apply -f trading-engine-deployment.yaml
kubectl apply -f risk-manager-deployment.yaml
kubectl apply -f monitoring-deployment.yaml
```

### Step 4: Configure Monitoring

1. Setup Prometheus scraping:
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
  namespace: monitoring
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
    scrape_configs:
    - job_name: 'miniquantfund'
      kubernetes_sd_configs:
      - role: pod
        namespaces:
          names:
          - miniquantfund-production
      relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
```

2. Setup Grafana dashboards:
```bash
# Import pre-built dashboards
kubectl create configmap grafana-dashboards \
  --from-file=dashboard-configs/ \
  --namespace=monitoring
```

### Step 5: Verify Deployment

1. Check pod status:
```bash
kubectl get pods -n miniquantfund-production
```

2. Check services:
```bash
kubectl get services -n miniquantfund-production
```

3. Test health endpoints:
```bash
curl -k https://trading-engine.miniquantfund.com/health
```

## Monitoring and Maintenance

### Key Metrics to Monitor

1. **System Metrics**
   - CPU usage (threshold: 80% warning, 90% critical)
   - Memory usage (threshold: 80% warning, 90% critical)
   - Disk usage (threshold: 85% warning, 95% critical)
   - Network latency (threshold: 100ms warning, 500ms critical)

2. **Trading Metrics**
   - Order execution rate
   - Order latency (threshold: 100ms warning, 500ms critical)
   - Fill rate (threshold: 95% warning, 90% critical)
   - Error rate (threshold: 1% warning, 5% critical)

3. **Risk Metrics**
   - Portfolio VaR (threshold: $80k warning, $100k critical)
   - Drawdown (threshold: 5% warning, 10% critical)
   - Position limits utilization
   - Leverage ratio

### Alert Configuration

Set up alerts for critical conditions:

```yaml
# prometheus-rules.yaml
groups:
- name: miniquantfund.rules
  rules:
  - alert: HighCPUUsage
    expr: cpu_usage_percent > 90
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "High CPU usage detected"
      description: "CPU usage is above 90% for more than 5 minutes"
  
  - alert: HighLatency
    expr: order_latency_seconds > 0.5
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "High order latency detected"
      description: "Order latency is above 500ms for more than 2 minutes"
```

### Backup Procedures

1. **Database Backup** (daily):
```bash
#!/bin/bash
# backup-db.sh
DATE=$(date +%Y%m%d_%H%M%S)
kubectl exec -n miniquantfund-production postgres-0 -- \
  pg_dump -U postgres miniquantfund | \
  gzip > /backups/miniquantfund_${DATE}.sql.gz

# Upload to secure storage
aws s3 cp /backups/miniquantfund_${DATE}.sql.gz \
  s3://miniquantfund-backups/database/
```

2. **Configuration Backup** (weekly):
```bash
#!/bin/bash
# backup-config.sh
DATE=$(date +%Y%m%d_%H%M%S)
kubectl get all -n miniquantfund-production -o yaml > \
  /backups/miniquantfund_config_${DATE}.yaml

aws s3 cp /backups/miniquantfund_config_${DATE}.yaml \
  s3://miniquantfund-backups/config/
```

## Troubleshooting

### Common Issues

1. **High Latency**
   - Check network connectivity to exchanges
   - Verify order routing configuration
   - Monitor system resource utilization

2. **Order Rejections**
   - Verify account balances
   - Check position limits
   - Review broker API status

3. **Database Connection Issues**
   - Check PostgreSQL pod status
   - Verify connection pool settings
   - Review network policies

### Debug Commands

```bash
# Check pod logs
kubectl logs -n miniquantfund-production trading-engine-xxx

# Check resource usage
kubectl top pods -n miniquantfund-production

# Debug pod connectivity
kubectl exec -n miniquantfund-production trading-engine-xxx -- \
  ping postgres-service

# Check events
kubectl get events -n miniquantfund-production --sort-by='.lastTimestamp'
```

### Performance Tuning

1. **Database Optimization**
```sql
-- Add indexes for frequently queried columns
CREATE INDEX idx_trades_symbol_timestamp ON trades(symbol, timestamp);
CREATE INDEX idx_positions_symbol ON positions(symbol);

-- Update statistics
ANALYZE;

-- Check slow queries
SELECT query, mean_time, calls 
FROM pg_stat_statements 
ORDER BY mean_time DESC 
LIMIT 10;
```

2. **Application Tuning**
```python
# Increase connection pool size
DATABASE_POOL_SIZE = 20
DATABASE_MAX_OVERFLOW = 30

# Optimize asyncio event loop
UVLOOP_ENABLED = True
```

## Emergency Procedures

### Trading Halt

1. **Manual Halt**:
```bash
# Scale down trading engine
kubectl scale deployment trading-engine \
  --replicas=0 -n miniquantfund-production
```

2. **Emergency Halt** (circuit breaker triggered):
```bash
# Stop all trading pods
kubectl delete pod -l app=trading-engine \
  -n miniquantfund-production
```

### Disaster Recovery

1. **Restore from Backup**:
```bash
# Restore database
gunzip -c /backups/miniquantfund_20240301_120000.sql.gz | \
kubectl exec -i -n miniquantfund-production postgres-0 -- \
  psql -U postgres miniquantfund
```

2. **Failover to DR Site**:
```bash
# Update DNS to point to DR site
# Update load balancer configuration
# Verify all services are running
```

### Security Incident Response

1. **Immediate Actions**:
   - Isolate affected systems
   - Preserve forensic evidence
   - Notify security team

2. **Investigation**:
   - Review audit logs
   - Analyze system logs
   - Identify root cause

3. **Recovery**:
   - Patch vulnerabilities
   - Update security policies
   - Restore from clean backups

## Compliance Requirements

### Regulatory Reporting

1. **Trade Reporting** (FINRA TRACE):
   - Submit within 15 minutes of execution
   - Include all required fields
   - Maintain audit trail

2. **Position Reporting** (Form 13F):
   - Quarterly filing required
   - Report all positions > $10M
   - File within 45 days of quarter end

### Data Retention

- **Trade Records**: 7 years
- **Communication Records**: 6 years
- **Risk Management Records**: 5 years
- **Audit Logs**: 7 years

### Security Compliance

- **SOC 2 Type II**: Annual assessment
- **PCI DSS**: If handling card payments
- **GDPR**: For EU client data
- **CCPA**: For California residents

## Support Contacts

- **Technical Support**: tech-support@miniquantfund.com
- **Security Team**: security@miniquantfund.com
- **Compliance Officer**: compliance@miniquantfund.com
- **Emergency Hotline**: +1-800-QUANT-HELP

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 4.0.0 | 2026-04-17 | Initial production release |
| 4.0.1 | 2026-04-18 | Bug fixes and performance improvements |
| 4.1.0 | 2026-05-01 | Additional broker integrations |

---

**Document Version**: 1.0  
**Last Updated**: 2026-04-17  
**Next Review**: 2026-07-17
