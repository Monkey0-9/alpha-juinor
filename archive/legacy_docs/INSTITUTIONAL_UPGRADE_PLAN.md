# Institutional Upgrade Implementation Plan
**Version**: 1.0.0
**Date**: 2024
**Status**: PLANNED - AWAITING APPROVAL

---

## Executive Summary

This plan transforms the mini-quant-fund from a research prototype into an institutional-grade quantitative trading platform. The implementation follows the priority order specified in the task: Top Priorities (A-F) first, then Medium Priorities (G-J), with clear acceptance criteria for each.

---

## Phase 1: Database Upgrade (PR #1) - TOP PRIORITY

### Objective
Replace SQLite with PostgreSQL + TimescaleDB for production time-series & state management.

### Files to Create
```
database/
├── adapters/
│   ├── __init__.py
│   └── postgres_manager.py      # PostgreSQL manager with Timescale support
├── migrations/
│   ├── __init__.py
│   ├── env.py                   # Alembic environment
│   ├── script.py.mako           # Migration template
│   └── versions/
│       ├── 001_initial_schema.py
│       └── 002_add_timescale.py
└── base.py                      # SQLAlchemy base classes

scripts/
├── migrate_sqlite_to_pg.py      # Data migration script
└── verify_db_status_pg.py       # Verification script

tests/
├── test_db_migrations.py        # Migration tests
└── test_postgres_integration.py # Integration tests
```

### Files to Modify
- `database/manager.py` - Add Postgres adapter, keep SQLite for dev mode
- `Dockerfile` - Add PostgreSQL service
- `pyproject.toml` - Add psycopg2, alembic, sqlalchemy
- `docker-compose.yml` - Add Postgres + Timescale services

### Implementation Steps

#### Step 1.1: Create Postgres Manager Adapter
```python
# database/adapters/postgres_manager.py
- Connection pooling with SQLAlchemy
- Timescale hypertable creation for price_history
- Daily partitioning support
- Full API compatibility with SQLite manager
```

#### Step 1.2: Set Up Alembic Migrations
```python
# database/migrations/versions/001_initial_schema.py
- Create all tables from current schema
- Add schema version tracking
```

```python
# database/migrations/versions/002_add_timescale.py
- Convert price_history to hypertable
- Add daily partitioning
```

#### Step 1.3: Create Migration Script
```python
# scripts/migrate_sqlite_to_pg.py
- Export data from SQLite
- Transform schema for PostgreSQL
- Import into PostgreSQL
- Verify data integrity
```

#### Step 1.4: Update Docker Compose
```yaml
# Add to docker-compose.yml
services:
  postgres:
    image: timescale/timescaledb:latest-pg15
    environment:
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_DB: mini_quant
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

volumes:
  postgres_data:
```

#### Step 1.5: Environment Configuration
```bash
# .env
DB_ENGINE=postgres  # or sqlite for dev
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=mini_quant
POSTGRES_USER=mini_quant
POSTGRES_PASSWORD=<from vault>
```

### Acceptance Criteria
- [ ] `python verify_db_status.py` returns success for Postgres
- [ ] Backtest path runs unchanged with `DB_ENGINE=postgres`
- [ ] Live path runs unchanged with `DB_ENGINE=postgres`
- [ ] Timescale hypertable created for price_history
- [ ] Migration script transfers all data correctly

---

## Phase 2: Kafka Streaming Pipeline (PR #2) - TOP PRIORITY

### Objective
Implement robust market-data ingestion pipeline with streaming + durable storage.

### Files to Create
```
data/ingest_streaming/
├── __init__.py
├── producer.py                  # Kafka producer for market data
├── consumer.py                  # Kafka consumer for feature generation
├── schemas/
│   ├── __init__.py
│   ├── quote_schema.avsc       # Avro schema for quotes
│   ├── trade_schema.avsc       # Avro schema for trades
│   └── fundamentals_schema.avsc
├── schema_registry.py           # Schema registry client
└── replay_service.py            # Replay facility for regulatory compliance

scripts/
├── ingest_live_kafka_smoke.py   # Smoke test for Kafka
└── replay_raw.py                # Replay raw feed from S3/GCS

docker-compose.kafka.yml         # Kafka + Zookeeper + Schema Registry
```

### Files to Modify
- `data/collectors/data_router.py` - Wire to Kafka producer
- `docker-compose.yml` - Include Kafka services
- `pyproject.toml` - Add confluent-kafka, avro-python3

### Implementation Steps

#### Step 2.1: Create Avro Schemas
```json
// data/ingest_streaming/schemas/quote_schema.avsc
{
  "type": "record",
  "name": "Quote",
  "fields": [
    {"name": "symbol", "type": "string"},
    {"name": "bid", "type": "double"},
    {"name": "ask", "type": "double"},
    {"name": "timestamp", "type": "long", "logicalType": "timestamp-millis"},
    {"name": "source", "type": "string"}
  ]
}
```

#### Step 2.2: Create Kafka Producer
```python
# data/ingest_streaming/producer.py
- Topic management per instrument type
- Avro serialization
- Idempotent produce
- Error handling and retries
```

#### Step 2.3: Create Kafka Consumer
```python
# data/ingest_streaming/consumer.py
- Consumer groups for scaling
- Offset management (commit to Kafka)
- At-least-once semantics
- Idempotent feature computation
```

#### Step 2.4: Wire Data Router to Producer
```python
# In data_router.py - modify get_price_history
- After fetching data, produce to Kafka topic
- Add data quality metadata
- Handle producer errors gracefully
```

#### Step 2.5: Create Replay Service
```python
# data/ingest_streaming/replay_service.py
- Read from cold storage (S3/GCS)
- Replay to Kafka topics
- Support date range selection
```

### Acceptance Criteria
- [ ] `python scripts/ingest_live_kafka_smoke.py` runs against local Kafka
- [ ] Valid Avro records written to Kafka topics
- [ ] Consumer processes quotes/trades and persists features
- [ ] Replay facility works for regulatory compliance
- [ ] Backfill automatically triggered on freshness breach

---

## Phase 3: Vault Secrets Management (PR #3) - TOP PRIORITY

### Objective
Harden secrets and credentials management using HashiCorp Vault.

### Files to Create
```
utils/
├── secrets_vault.py             # Vault client wrapper
└── secrets_loader.py            # Environment-based secrets loader

scripts/
├── bootstrap_vault_policies.py  # Create Vault policies
└── init_vault_secrets.py        # Initialize secrets in Vault

tests/
└── test_secrets.py              # Secrets verification tests
```

### Files to Modify
- `brokers/alpaca_broker.py` - Fetch credentials from Vault
- `brokers/ccxt_broker.py` - Fetch credentials from Vault
- `configs/config_manager.py` - Remove hardcoded credentials
- `docker-compose.yml` - Add Vault service
- `.env.example` - Remove actual credentials, use placeholders

### Implementation Steps

#### Step 3.1: Create Vault Client Wrapper
```python
# utils/secrets_vault.py
- HashiCorp Vault integration
- Token-based authentication
- Caching layer
- Development mode with local secrets
```

#### Step 3.2: Create Secrets Loader
```python
# utils/secrets_loader.py
- Load secrets from Vault or environment
- Fallback to environment variables
- Validation and health checks
```

#### Step 3.3: Create Vault Bootstrap Script
```python
# scripts/bootstrap_vault_policies.py
- Create policies for different services
- Set up KV v2 secrets engine
- Configure ACLs
```

#### Step 3.4: Refactor Brokers
```python
# brokers/alpaca_broker.py
# Before:
api_key = "actual_key"
secret_key = "actual_secret"

# After:
from utils.secrets_vault import get_secret
api_key = get_secret("alpaca/api_key")
secret_key = get_secret("alpaca/secret_key")
```

#### Step 3.5: Add .env.example
```bash
# .env.example (committed to repo)
DB_ENGINE=sqlite
ALPACA_API_KEY=${ALPACA_API_KEY}
ALPACA_SECRET_KEY=${ALPACA_SECRET_KEY}
# ... other secrets loaded from Vault
```

### Acceptance Criteria
- [ ] `python -m verify_secrets` returns green check
- [ ] No secrets in repo (scan with detect-secrets)
- [ ] Local dev mode works with dev Vault
- [ ] CI injects secrets from Vault/CI secret store
- [ ] Broker connections work with Vault-sourced credentials

---

## Phase 4: OMS & Execution Layer (PR #4) - TOP PRIORITY

### Objective
Add Order Management System with pre-trade risk, market impact models, and execution simulation.

### Files to Create
```
execution/
├── oms.py                       # Order Management System
├── order.py                     # Order class with lifecycle states
├── market_impact.py             # Almgren-Chriss implementation
├── execution_simulator.py       # Fill simulation with slippage
├── pretrade_checks.py           # Pre-trade risk checks
└── tca.py                       # Transaction Cost Analysis

tests/
├── test_order_lifecycle.py      # OMS state transition tests
└── test_market_impact.py        # Market impact model tests
```

### Files to Modify
- `brokers/alpaca_broker.py` - Register via OMS
- `brokers/mock_broker.py` - Adapt for OMS interface
- `execution/impact.py` - Integrate with market impact model
- `risk/engine.py` - Integrate pre-trade checks

### Implementation Steps

#### Step 4.1: Create Order Class with Lifecycle
```python
# execution/order.py
class Order:
    NEW = "NEW"
    ACK = "ACK"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"

    def __init__(self, symbol, side, quantity, order_type="market"):
        self.order_id = str(uuid.uuid4())
        self.status = self.NEW
        # ... other fields
```

#### Step 4.2: Create OMS
```python
# execution/oms.py
class OrderManagementSystem:
    def __init__(self, broker_adapter):
        self.orders = {}
        self.broker = broker_adapter
        self.risk_checker = PreTradeRiskChecker()

    def place_order(self, order):
        # Pre-trade risk check
        # Create order with lifecycle tracking
        # Submit to broker
        # Track fills
        # Return order with TCA metrics
```

#### Step 4.3: Implement Almgren-Chriss Model
```python
# execution/market_impact.py
class AlmgrenChrissImpact:
    def __init__(self, sigma, lambda_param, eta, gamma):
        self.sigma = sigma  # Volatility
        self.lambda_param = lambda_param  # Risk aversion
        self.eta = eta  # Temporary impact coefficient
        self.gamma = gamma  # Permanent impact coefficient

    def calculate_impact(self, quantity, avg_daily_volume, price):
        # Implementation of Almgren-Chriss optimal execution
```

#### Step 4.4: Create Execution Simulator
```python
# execution/execution_simulator.py
class ExecutionSimulator:
    def __init__(self, market_impact_model, slippage_model):
        self.impact = market_impact_model

    def simulate_fill(self, order, market_conditions):
        # Calculate expected slippage
        # Calculate market impact
        # Return fill price and metrics
```

#### Step 4.5: Integrate with Existing Code
```python
# In backtest or live trading code
from execution.oms import OrderManagementSystem

oms = OrderManagementSystem(broker)
order = oms.place_order(
    symbol="AAPL",
    side="buy",
    quantity=100,
    order_type="market"
)
```

### Acceptance Criteria
- [ ] `pytest tests/test_order_lifecycle.py` passes
- [ ] Backtest uses OMS with execution simulator
- [ ] TCA metrics (slippage, fill rate) produced
- [ ] Pre-trade risk checks prevent excessive exposure
- [ ] Market impact model parameters configurable

---

## Phase 5: MLOps & Model Registry (PR #5) - TOP PRIORITY

### Objective
Implement deterministic backtests, model versioning, and automated validation.

### Files to Create
```
models/
├── __init__.py
├── registry.py                  # Model registry
├── metadata.py                  # Model metadata schemas
├── artifact_store.py            # S3/GCS artifact storage
└── validation/
    ├── __init__.py
    ├── backtest_validator.py    # Deterministic backtest validation
    └── model_drift.py           # Drift detection

scripts/
├── train_ml_alpha.py            # Model training with registry
└── inspect_model.py             # Model artifact inspection

tests/
├── test_model_registry.py       # Registry tests
├── test_backtest_regression.py  # Backtest regression tests
└── test_model_drift.py          # Drift detection tests
```

### Files to Modify
- `alpha_families/ml_alpha.py` - Register models with registry
- `pyproject.toml` - Add mlflow or wandb

### Implementation Steps

#### Step 5.1: Create Model Registry
```python
# models/registry.py
class ModelRegistry:
    def __init__(self, artifact_store):
        self.store = artifact_store
        self.db = DatabaseManager()

    def register_model(self, model_id, model_artifact, metadata):
        # Store artifact in S3/GCS
        # Record metadata in database
        # Generate model hash
        # Return model_version

    def get_model(self, model_id, version=None):
        # Retrieve model from artifact store
        # Validate hash
        # Return model
```

#### Step 5.2: Create Model Metadata Schema
```python
# models/metadata.py
@dataclass
class ModelMetadata:
    model_id: str
    model_type: str  # 'ml_alpha', 'volatility', 'return'
    training_data_hash: str
    train_params: Dict[str, Any]
    version: str
    metrics: Dict[str, float]
    created_at: datetime
    created_by: str
    git_commit: str
```

#### Step 5.3: Implement Deterministic Backtest Validation
```python
# models/validation/backtest_validator.py
class BacktestValidator:
    def __init__(self, reference_outputs_path):
        self.reference_path = reference_outputs_path

    def validate(self, current_output, tolerance=0.01):
        # Compare with stored snapshot
        # Check RNG seeds
        # Return validation result with drift metrics
```

#### Step 5.4: Implement Model Drift Detection
```python
# models/validation/model_drift.py
class ModelDriftDetector:
    def __init__(self, baseline_distribution):
        self.baseline = baseline_distribution

    def detect_drift(self, current_predictions):
        # Statistical test (KS test, PSI)
        # Return drift score and alert
```

#### Step 5.5: Update Training Script
```python
# scripts/train_ml_alpha.py
from models.registry import ModelRegistry

def train_alpha_model(ticker, data):
    # Train model
    # Compute metrics
    # Register with model registry
    # Push artifact to S3
```

### Acceptance Criteria
- [ ] Model artifact listing accessible via registry
- [ ] `python inspect_model.py --id <model_id>` works
- [ ] `pytest tests/test_backtest_regression.py` compares with snapshot
- [ ] Drift detection triggers alerts
- [ ] Training script pushes to registry

---

## Phase 6: Observability & CI/CD (PR #6) - MEDIUM PRIORITY

### Objective
Add Prometheus metrics, distributed tracing, central logs, and CI/CD pipelines.

### Files to Create
```
monitoring/
├── prometheus_exporter.py       # Prometheus metrics exporter
├── tracing.py                   # Distributed tracing setup
└── dashboards/
    ├── ingestion.json           # Ingestion dashboard
    ├── execution.json           # Execution dashboard
    └── system.json              # System metrics dashboard

.github/
└── workflows/
    ├── ci.yml                   # Main CI pipeline
    ├── integration.yml          # Integration tests
    └── nightly_benchmarks.yml   # Performance benchmarks

tests/
└── test_observability.py        # Observability tests
```

### Files to Modify
- `monitoring/metrics.py` - Add Prometheus metrics
- `monitoring/alerts.py` - Integrate with alerting
- `main.py` - Add instrumentation
- `docker-compose.yml` - Add Prometheus, Grafana, Jaeger services

### Implementation Steps

#### Step 6.1: Create Prometheus Exporter
```python
# monitoring/prometheus_exporter.py
from prometheus_client import Counter, Histogram, Gauge

INGESTION_LATENCY = Histogram('ingestion_latency_seconds', 'Ingestion latency')
FEATURE_COMPUTE_TIME = Histogram('feature_compute_seconds', 'Feature compute time')
ORDER_LATENCY = Histogram('order_latency_seconds', 'Order placement latency')
ACTIVE_POSITIONS = Gauge('active_positions', 'Number of active positions')
```

#### Step 6.2: Add Tracing
```python
# monitoring/tracing.py
import jaeger_client
from opentracing import global_tracer

def init_tracing(service_name):
    config = jaeger_client.Config(
        config={
            'sampler': {'type': 'const', 'param': 1},
            'logging': True,
        },
        service_name=service_name,
    )
    global_tracer().init_tracer(config)
```

#### Step 6.3: Instrument Core Modules
```python
# In data/collectors/data_router.py
from monitoring.tracing import traced

@traced("data_router_fetch")
def get_price_history(self, ticker, start_date, end_date):
    # Existing code with tracing
```

#### Step 6.4: Create CI Pipeline
```yaml
# .github/workflows/ci.yml
jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:15
      kafka:
        image: confluentinc/cp-kafka
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
      - name: Run tests
        run: pytest tests/ -v
```

#### Step 6.5: Create Grafana Dashboards
```json
// monitoring/dashboards/ingestion.json
{
  "dashboard": {
    "title": "Ingestion Metrics",
    "panels": [
      {"type": "graph", "title": "Ingestion Latency"},
      {"type": "graph", "title": "Data Quality Score"}
    ]
  }
}
```

### Acceptance Criteria
- [ ] Dashboards show live metrics examples
- [ ] Alerts trigger when ingestion lag > threshold
- [ ] CI passes on PRs
- [ ] Integration tests run on main branch
- [ ] Benchmark job stores results

---

## Phase 7: Security & Compliance (PR #7) - MEDIUM PRIORITY

### Objective
Implement audit logging, RBAC, and compliance hooks.

### Files to Create
```
audit/
├── decision_log.py              # Immutable decision logging
├── tamper_evidence.py           # Cryptographic hashing
└── export.py                    # Export to cold storage

tests/
└── test_audit_tamper.py         # Tamper evidence tests
```

### Files to Modify
- `audit/decision_log.py` - Add cryptographic hashing
- `governance/` - Add RBAC checks

### Implementation Steps

#### Step 7.1: Create Tamper-Evident Logging
```python
# audit/tamper_evidence.py
import hashlib
import hmac

class TamperEvidence:
    @staticmethod
    def hash_record(record, previous_hash):
        """Create cryptographic hash chain"""
        data = json.dumps(record, sort_keys=True) + previous_hash
        return hashlib.sha256(data.encode()).hexdigest()

    @staticmethod
    def verify_chain(records):
        """Verify integrity of log chain"""
        # Verify hash chain
        # Return validity status
```

#### Step 7.2: Update Decision Log
```python
# audit/decision_log.py
class DecisionLogger:
    def log_decision(self, decision):
        # Create immutable record
        # Add cryptographic hash
        # Append to audit log
        # Export to cold storage
```

### Acceptance Criteria
- [ ] `python verify_decision_layer.py` returns valid
- [ ] Cryptographic hashes verified
- [ ] Audit logs export to S3/GCS
- [ ] RBAC enforced in dashboards

---

## Phase 8: Documentation & Runbooks

### Files to Create
```
docs/
├── runbook_db_migration.md
├── runbook_kafka_ingestion.md
├── runbook_vault_setup.md
├── runbook_oms_testing.md
└── runbook_model_validation.md
```

---

## Dependencies & Tech Stack

| Component | Technology |
|-----------|------------|
| Time-series DB | PostgreSQL + TimescaleDB |
| Message Bus | Apache Kafka |
| Secret Store | HashiCorp Vault |
| Model Registry | MLflow or Weights & Biases |
| Metrics | Prometheus |
| Tracing | Jaeger |
| Logs | Loki or ELK |
| CI/CD | GitHub Actions |
| Container Orchestration | Docker Compose / Kubernetes |

---

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| Migration downtime | Zero-downtime migration with dual-write |
| Kafka complexity | Start with single-node, scale as needed |
| Vault learning curve | Dev mode for local development |
| Performance regression | Benchmark suite and monitoring |
| Model drift | Automated validation and alerts |

---

## Timeline Estimate

| Phase | Estimated Effort |
|-------|------------------|
| Phase 1: Database | 3-4 days |
| Phase 2: Kafka | 4-5 days |
| Phase 3: Vault | 2 days |
| Phase 4: OMS | 4-5 days |
| Phase 5: MLOps | 3-4 days |
| Phase 6: Observability | 3 days |
| Phase 7: Security | 2 days |
| **Total** | **21-25 days** |

---

## Verification Commands

```bash
# Database
python scripts/verify_db_status_pg.py

# Kafka
python scripts/ingest_live_kafka_smoke.py

# Secrets
python -m utils.secrets_vault verify

# OMS
pytest tests/test_order_lifecycle.py -v

# Model Registry
python scripts/inspect_model.py --list

# Observability
curl http://localhost:9090/metrics
```

---

## Next Steps

1. **Approve this plan** - Review and approve the implementation plan
2. **Start Phase 1** - Begin with Database Upgrade (foundational)
3. **Iterative PRs** - Each phase = 1 PR with tests and runbook
4. **Continuous Verification** - Run acceptance criteria after each phase

---

*This plan follows institutional-grade quant platform best practices and maps directly to the acceptance criteria in the task specification.*

