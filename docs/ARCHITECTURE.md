# MiniQuantFund Architecture Documentation

## Overview

MiniQuantFund is an institutional-grade quantitative trading system built with enterprise software engineering practices. This document describes the modular architecture, component interactions, and operational guidelines.

## System Architecture

### High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MINIQUANTFUND ARCHITECTURE                         │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   MARKET    │────▶│   DATA      │────▶│ STRATEGY    │────▶│ EXECUTION   │
│   DATA      │     │   ROUTER    │     │   ENGINE    │     │   ENGINE    │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
       │                   │                   │                   │
       │                   │                   │                   │
       ▼                   ▼                   ▼                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           GOVERNANCE LAYER                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   CIRCUIT    │  │     RISK     │  │   SAFETY     │  │  LIFECYCLE   │     │
│  │  BREAKERS    │  │   MANAGER    │  │   KILL       │  │   MANAGER    │     │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────────────────────┘
       │                   │                   │                   │
       ▼                   ▼                   ▼                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         INFRASTRUCTURE LAYER                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   DATABASE   │  │   MONITORING │  │   SECURITY   │  │   RESILIENCE │     │
│  │  (SQLite/    │  │  (SLOs/      │  │  (Secrets/   │  │  (Circuit    │     │
│  │   Postgres)  │  │   Alerts)    │  │   Rotation)  │  │   Breakers)  │     │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Module Structure

### Core Modules (`src/mini_quant_fund/`)

#### 1. Core Engine (`core/`)
- **Orchestrator** (`engine/orchestrator.py`): Main system coordinator
- **Trading Loop** (`engine/loop.py`): 1-second decision cycle
- **Configuration** (`production_config.py`): Environment-aware settings
- **Session Tracking** (`global_session_tracker.py`): Market session management

#### 2. Data Layer (`data/`)
- **Data Router** (`collectors/data_router.py`): Multi-provider data aggregation
  - Alpaca, Yahoo Finance, Polygon, Bloomberg integration
  - Automatic failover between providers
  - Data quality validation
- **Provider Governance** (`governance/provider_router.py`): Smart provider selection
- **Confidence Manager** (`intelligence/confidence_manager.py`): Data quality scoring

#### 3. Execution Layer (`execution/`)
- **Alpaca Handler** (`alpaca_handler.py`): Live trading via Alpaca API
  - Circuit breaker integration
  - Order execution with retry logic
- **Advanced Execution** (`advanced_execution.py`): TWAP, VWAP algorithms
- **Multi-Prime Brokerage** (`brokers/multi_prime_brokerage.py`): Institutional broker failover
  - Goldman Sachs, Morgan Stanley, JP Morgan integration
  - Capital efficiency optimization
  - Counterparty risk monitoring

#### 4. Strategy Layer (`strategies/`)
- **Institutional Strategy** (`institutional_strategy.py`): Core signal generation
- **Strategy Factory** (`factory.py`): Dynamic strategy instantiation
- **13 Trading Types**: Scalping, Day Trading, Swing Trading, Momentum, etc.

#### 5. Risk & Governance (`risk/`, `governance/`, `safety/`)
- **Circuit Breaker** (`safety/circuit_breaker.py`): Automatic trading halt on losses
  - Daily/weekly loss limits
  - Per-trade loss limits
- **Lifecycle Manager** (`governance/lifecycle_manager.py`): Symbol state management
- **Risk Engine** (`risk/engine.py`): Portfolio risk calculations

#### 6. Infrastructure (`infra/`, `monitoring/`, `security/`)
- **Resilience Framework** (`infra/resilience_framework.py`):
  - Distributed circuit breakers
  - Exponential backoff retry
  - Bulkhead isolation
  - Timeout enforcement
- **Production Monitor** (`monitoring/production_monitor.py`):
  - Real-time SLO tracking
  - Multi-channel alerting (Slack, PagerDuty)
  - Performance regression detection
- **Secret Manager** (`security/runtime_secret_manager.py`):
  - Automatic secret rotation
  - Vault/AWS/Azure integration
  - Audit logging

## Data Flow

### Trading Cycle (1-second loop)

```
1. Market Session Check
   └─▶ Verify market is open (NYSE)

2. Data Fetch
   └─▶ DataRouter.get_panel_parallel(tickers, 5-day lookback)
   └─▶ Validate data quality (score > 0.6)
   └─▶ Fallback to Yahoo Finance if primary fails

3. Signal Generation
   └─▶ InstitutionalStrategy.generate_signals(market_data)
   └─▶ 13 parallel strategy types
   └─▶ Conviction scoring (0.0 - 1.0)

4. Risk & Governance
   └─▶ LifecycleManager.check_gates()
   └─▶ CircuitBreaker.is_halted()
   └─▶ KillSwitch not engaged

5. Execution Planning
   └─▶ ExecutionEngine.create_execution_plan()
   └─▶ Algorithm selection (MARKET, TWAP, VWAP)
   └─▶ Order slicing for large orders

6. Order Execution
   └─▶ AlpacaExecutionHandler.submit_order()
   └─▶ MultiPrimeBrokerage.allocate_order() (for failover)
   └─▶ Record fill to circuit breaker

7. Position Update
   └─▶ Update portfolio exposure
   └─▶ Calculate P&L
   └─▶ Check stop-loss/take-profit

8. Audit & Logging
   └─▶ DatabaseManager.log_trade()
   └─▶ InstitutionalLogger.log_decision()
   └─▶ Update dashboard state
```

## Configuration

### Environment Variables

```bash
# Required for Live Trading
ALPACA_API_KEY=your_api_key
ALPACA_SECRET_KEY=your_secret_key

# Optional
ALPACA_BASE_URL=https://paper-api.alpaca.markets  # or https://api.alpaca.markets
TRADING_MODE=paper  # or "live"
EXECUTE_TRADES=false  # Set to "true" to enable real trading

# Database
DATABASE_URL=postgresql://user:pass@localhost/mini_quant_fund

# Monitoring
SLACK_WEBHOOK_URL=https://hooks.slack.com/...
PAGERDUTY_INTEGRATION_KEY=your_key

# Secrets Management
VAULT_ADDR=https://vault.example.com
VAULT_TOKEN=your_token
SECRET_MASTER_KEY=encryption_key

# Circuit Breaker
CIRCUIT_STATE_PATH=runs/circuit_state.json
```

### Configuration Files

- `configs/golden_config.yaml`: Production configuration
- `configs/safety_config.yaml`: Circuit breaker limits
- `configs/universe.json`: Trading universe definition

## Operational Guidelines

### Before Trading

1. **Run Diagnostics**
   ```bash
   python diagnose_trading_execution.py --fix
   ```

2. **Verify System Status**
   - Check all ✅ in diagnostic output
   - Verify EXECUTE_TRADES flag
   - Confirm kill switch not engaged

3. **Start Paper Trading**
   ```bash
   python main.py --mode paper
   ```

4. **Monitor for 1-2 Hours**
   - Verify all 13 strategies generating signals
   - Confirm P&L calculation working
   - Check no errors in logs

### Live Trading (Only After Paper Success)

1. **Set Live Credentials**
   ```bash
   export ALPACA_API_KEY=live_key
   export ALPACA_SECRET_KEY=live_secret
   export TRADING_MODE=live
   export EXECUTE_TRADES=true
   ```

2. **Start with Kill Switch Ready**
   ```bash
   touch runtime/KILL_SWITCH  # Pause immediately if needed
   python main.py --mode live
   ```

3. **Active Monitoring**
   - Watch dashboard every 5 minutes
   - Verify leverage < 0.5x
   - Confirm daily loss < 1% of account

### Emergency Procedures

```bash
# Immediate stop
touch runtime/KILL_SWITCH

# Graceful stop
Ctrl+C

# Circuit breaker reset (if needed)
python -c "from mini_quant_fund.infra.resilience_framework import resilience; resilience.reset_circuit('alpaca_api')"
```

## Testing

### Unit Tests
```bash
pytest tests/ -v --tb=short
```

### Integration Tests
```bash
pytest tests/integration/ -v
```

### Chaos Engineering
```bash
python tests/chaos/chaos_test_suite.py
```

### Performance Benchmarks
```bash
python benchmarks/throughput_test.py --duration 60 --rps 1000
```

## Deployment

### CI/CD Pipeline

1. **Lint & Security Scan**
2. **Unit Tests**
3. **Integration Tests**
4. **Performance Benchmarks**
5. **Chaos Tests**
6. **Load Tests**
7. **Production Readiness Check**
8. **Canary Deployment**
9. **Full Production Rollout**

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mini-quant-fund
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: trading-engine
        image: ghcr.io/user/mini-quant-fund:latest
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        env:
        - name: TRADING_MODE
          value: "paper"
        - name: ALPACA_API_KEY
          valueFrom:
            secretKeyRef:
              name: trading-secrets
              key: alpaca-api-key
```

## Monitoring & Alerting

### Service Level Objectives (SLOs)

| SLO | Target | Measurement |
|-----|--------|-------------|
| Availability | 99.9% | Uptime over 24h |
| Latency (P99) | < 10ms | End-to-end cycle time |
| Throughput | 1000 RPS | Requests per second |
| Error Rate | < 0.1% | Failed operations |
| Data Freshness | < 5s | Age of market data |

### Alert Channels

- **Slack**: Warnings and notifications
- **PagerDuty**: Critical and emergency alerts
- **Email**: Daily summaries
- **Logs**: All events with structured logging

## Security

### Secret Management

1. **Never commit secrets to git**
2. **Use SecretManager for runtime access**
3. **Enable automatic rotation** (90-day default)
4. **Audit all secret access**

### Access Control

- API keys stored in environment or Vault
- Database credentials rotated monthly
- Broker credentials encrypted at rest

## Troubleshooting

### Common Issues

**No trades executing**
```bash
python diagnose_trading_execution.py
# Check EXECUTE_TRADES, kill switch, circuit breaker
```

**High latency**
```bash
python benchmarks/throughput_test.py --quick
# Check CPU/memory, network latency to brokers
```

**Data quality issues**
```bash
python -c "from mini_quant_fund.data.collectors.data_router import DataRouter; dr = DataRouter(); print(dr._validate_data_quality(df))"
```

## Development Guidelines

### Code Style
- Black formatter (88 char line length)
- isort for imports
- flake8 for linting
- Type hints required

### Testing Requirements
- >80% code coverage
- Unit tests for all strategies
- Integration tests for data/execution pipeline
- Chaos tests for fault tolerance

### Documentation
- Docstrings for all public methods
- Architecture Decision Records (ADRs)
- Operational runbooks

## Contact & Support

- **Issues**: GitHub Issues
- **Documentation**: This file and `/docs`
- **Emergency**: On-call rotation via PagerDuty

---

**Mandate: Survival first. Audit everything. No silent failures.**
