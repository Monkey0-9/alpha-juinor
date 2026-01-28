# Institutional Upgrade - Master TODO List
**Status**: IN PROGRESS
**Start Date**: 2024

---

## Phase 1: Database Upgrade (PostgreSQL + TimescaleDB) ✅ IN PROGRESS
- [x] Create database/adapters/postgres_manager.py
- [x] Create database/adapters/__init__.py
- [x] Update database/manager.py with dual-engine support
- [x] Create database/__init__.py (Unified Database Factory)
- [x] Create scripts/migrate_sqlite_to_pg.py
- [x] Create scripts/verify_db_status_pg.py
- [x] Update pyproject.toml with PostgreSQL dependencies
- [x] Update docker-compose.yml with PostgreSQL service
- [x] Create tests/test_db_migrations.py
- [ ] Verify Phase 1: Run migrations and tests

## Phase 2: Kafka Streaming Pipeline
- [ ] Create data/ingest_streaming/ directory structure
- [ ] Create data/ingest_streaming/schemas/quote_schema.avsc
- [ ] Create data/ingest_streaming/schemas/trade_schema.avsc
- [ ] Create data/ingest_streaming/schema_registry.py
- [ ] Create data/ingest_streaming/producer.py
- [ ] Create data/ingest_streaming/consumer.py
- [ ] Create data/ingest_streaming/replay_service.py
- [ ] Update data/collectors/data_router.py to use producer
- [ ] Create docker-compose.kafka.yml
- [ ] Create scripts/ingest_live_kafka_smoke.py
- [ ] Create tests/test_kafka_ingest_consume.py

## Phase 3: Vault Secrets Management
- [ ] Create utils/secrets_vault.py
- [ ] Create utils/secrets_loader.py
- [ ] Create scripts/bootstrap_vault_policies.py
- [ ] Refactor brokers/alpaca_broker.py to use secrets_vault
- [ ] Refactor brokers/ccxt_broker.py to use secrets_vault
- [ ] Update configs/config_manager.py to use secrets
- [ ] Create .env.example with placeholders
- [ ] Create tests/test_secrets.py

## Phase 4: OMS & Execution Layer
- [ ] Create execution/order.py (Order class with lifecycle)
- [ ] Create execution/oms.py (Order Management System)
- [ ] Create execution/market_impact.py (Almgren-Chriss)
- [ ] Create execution/execution_simulator.py
- [ ] Create execution/pretrade_checks.py
- [ ] Create execution/tca.py (Transaction Cost Analysis)
- [ ] Refactor brokers/alpaca_broker.py for OMS interface
- [ ] Create tests/test_order_lifecycle.py
- [ ] Create tests/test_market_impact.py

## Phase 5: MLOps & Model Registry
- [ ] Create models/registry.py
- [ ] Create models/metadata.py
- [ ] Create models/artifact_store.py
- [ ] Create models/validation/backtest_validator.py
- [ ] Create models/validation/model_drift.py
- [ ] Update alpha_families/ml_alpha.py to use registry
- [ ] Create scripts/train_ml_alpha.py
- [ ] Create scripts/inspect_model.py
- [ ] Create tests/test_model_registry.py
- [ ] Create tests/test_backtest_regression.py

## Phase 6: Observability & CI/CD
- [ ] Create monitoring/prometheus_exporter.py
- [ ] Create monitoring/tracing.py
- [ ] Instrument core modules (data_router, features, execution)
- [ ] Create monitoring/dashboards/ingestion.json
- [ ] Create monitoring/dashboards/execution.json
- [ ] Create .github/workflows/ci.yml
- [ ] Create .github/workflows/integration.yml
- [ ] Create .github/workflows/nightly_benchmarks.yml
- [ ] Create tests/test_observability.py

## Phase 7: Security & Compliance
- [ ] Create audit/tamper_evidence.py
- [ ] Update audit/decision_log.py with cryptographic hashing
- [ ] Create audit/export.py for cold storage export
- [ ] Add RBAC checks to governance modules
- [ ] Create tests/test_audit_tamper.py
- [ ] Create docs/runbook_audit_compliance.md

---

## Daily Progress Log

### Day 1: Phase 1 Database Upgrade
- Created database/adapters/postgres_manager.py - PostgreSQL manager with TimescaleDB
- Created database/adapters/__init__.py
- Created database/__init__.py - Unified Database Factory
- Created scripts/migrate_sqlite_to_pg.py - Migration script
- Created scripts/verify_db_status_pg.py - Verification script
- Updated docker-compose.yml with PostgreSQL + Prometheus + Grafana
- Updated pyproject.toml with PostgreSQL dependencies (SQLAlchemy, psycopg2)
- Created tests/test_db_migrations.py

### Phase 1 Files Created:
1. database/adapters/__init__.py
2. database/adapters/postgres_manager.py
3. database/__init__.py
4. scripts/migrate_sqlite_to_pg.py
5. scripts/verify_db_status_pg.py
6. tests/test_db_migrations.py

### Modified Files:
1. docker-compose.yml - Added PostgreSQL, Prometheus, Grafana services
2. pyproject.toml - Added PostgreSQL, SQLAlchemy, monitoring dependencies

### Next Steps:
1. Run tests to verify Phase 1 implementation
2. Start Phase 2: Kafka Streaming Pipeline
