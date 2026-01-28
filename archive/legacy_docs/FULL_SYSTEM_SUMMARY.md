# Mini-Quant-Fund: Complete Software System Summary

## Executive Overview

**Mini-Quant-Fund** is an institutional-grade, autonomous trading engine designed for live institutional execution. It implements a deterministic, survival-first approach with zero tolerance for silent failures. The system operates 24/7 with an 11-layer governance stack and comprehensive risk management.

### Core Design Philosophy
- **Survival First**: Aggressive risk controls and circuit breakers
- **Audit Everything**: Every decision produces a 15-field JSON audit record
- **No Silent Failures**: Halt-on-failure for critical governance violations
- **Deterministic**: Reproducible decision-making with full provenance

---

## System Architecture

### High-Level Components

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        MINI-QUANT-FUND TRADING SYSTEM                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                     ORCHESTRATION LAYER                               │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │   │
│  │  │ trading_daemon  │  │ main.py         │  │ live_trading_daemon │  │   │
│  │  │ (24/7 Loop)     │  │ (Single Cycle)  │  │ (Institutional)     │  │   │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                     INTELLIGENCE LAYER                                │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐ │   │
│  │  │                    Meta-Brain                                   │ │   │
│  │  │  • Ensemble Aggregation    • Confidence-Weighted               │ │   │
│  │  │  • Disagreement Penalty    • Fractional Kelly Sizing           │ │   │
│  │  │  • CVaR-First Decisions    • Z-Score Normalization             │ │   │
│  │  └─────────────────────────────────────────────────────────────────┘ │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐ │   │
│  │  │                    Agent Orchestra                              │ │   │
│  │  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐  │ │   │
│  │  │  │Technical│ │Sentiment│ │Valuation│ │Fundament│ │  Risk   │  │ │   │
│  │  │  │ Agent   │ │ Agent   │ │ Agent   │ │ al Agent│ │  Agent  │  │ │   │
│  │  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘  │ │   │
│  │  └─────────────────────────────────────────────────────────────────┘ │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                     DECISION & OPTIMIZATION                          │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │   │
│  │  │ Institutional   │  │  Risk Manager   │  │  Portfolio          │  │   │
│  │  │  Allocator      │  │  (Pre/Post)     │  │  Allocator          │  │   │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                     EXECUTION LAYER                                   │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │   │
│  │  │ Alpaca Broker   │  │  Mock Broker    │  │  OMS (Order Mgmt)   │  │   │
│  │  │ (Live/Paper)    │  │  (Testing)      │  │  System             │  │   │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                     DATA LAYER                                        │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │   │
│  │  │ Data Router     │  │  Governance     │  │  Database           │  │   │
│  │  │ (Multi-Provider)│  │  Agent          │  │  Manager            │  │   │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                     MONITORING & ALERTING                            │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │   │
│  │  │ Alert Manager   │  │  Logging Config │  │  Metrics            │  │   │
│  │  │ (Telegram/Slack)│  │                 │  │                     │  │   │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### 1. Orchestration Layer

#### `main.py` - Single Cycle Entry Point
**Purpose**: Run a single trading cycle for testing or paper trading

**Key Features**:
- Initializes ConfigManager for institutional configuration
- Loads 252-day historical data from database
- Creates InstitutionalStrategy via StrategyFactory
- Generates signals and makes allocation decisions
- Executes orders through broker
- Logs all decisions with full audit trail

**Workflow**:
```
1. Load Configuration (Golden Config with SHA256 hash verification)
2. Initialize DatabaseManager (SQLite/PostgreSQL)
3. Fetch active symbols from governance table
4. Load 252-day price history for all symbols
5. Generate signals via Technical, Sentiment, Valuation, Fundamental agents
6. Meta-Brain aggregates signals with disagreement penalty
7. RiskManager applies pre-trade checks
8. Allocator computes target weights
9. OMS creates and submits orders
10. Persist all decisions and orders to database
```

#### `trading_daemon.py` - 24/7 Trading Loop
**Purpose**: Continuous trading operation with configurable triggers

**Key Features**:
- Runs indefinitely until kill switch or error
- Configurable cycle triggers (time-based, signal-based)
- Automatic recovery from transient failures
- Heartbeat monitoring
- Graceful shutdown on SIGINT/SIGTERM

#### `live_trading_daemon.py` - Institutional Live Trading
**Purpose**: Production-ready 24/7 institutional trading

**Key Governance Requirements**:
- **Phase 0**: 1260 rows (5 years) mandatory history check
- **Phase 1**: Data quality validation (score ≥ 0.6)
- **Phase 2**: Signal generation validation
- **Phase 3**: Risk validation (VaR/CVaR limits)
- **Phase 4**: Order generation and execution

**Components**:
- `InstitutionalLiveAgent`: Main agent class
- `check_1260_rows_requirement()`: Mandatory pre-trade gate
- `governance_halt()`: Halt system on violation
- `heartbeat_worker()`: Daemon thread for monitoring

---

### 2. Intelligence Layer

#### Meta-Brain (`agents/meta_brain.py`)
**Purpose**: Aggregate all agent outputs into final trading decisions

**Core Algorithm**:
1. **Collect Agent Outputs**: Gather predictions from all analytical agents
2. **Confidence-Weighted Ensemble**: μ̂ᵢ = Σₖ wₖ · αᵢ,ₖ
3. **Disagreement Penalty**: μ̂ᵢ = μ̂ᵢ × exp(-β × Var(μₖ))
4. **Robust Z-Score**: zᵢ = (μ̂ᵢ − median) / MAD
5. **Conviction Score**: Sᵢ = μ̂ᵢ / σ̂ᵢ
6. **Kelly Sizing**: fᵢ = γ × μ̂ᵢ / σ̂ᵢ²
7. **Risk Rules Application**: CVaR-first decision logic
8. **Final Decision**: EXECUTE_BUY, EXECUTE_SELL, HOLD, or REJECT

**Configuration**:
```python
DEFAULT_BETA = 0.5        # Disagreement penalty strength
DEFAULT_GAMMA = 0.3       # Kelly fractional factor
RISK_FREE_RATE = 0.02     # Annual risk-free rate
min_confidence_threshold = 0.3
min_data_quality_threshold = 0.6
max_position_size = 0.10   # 10% max position
cvar_limit = 0.05         # 5% portfolio CVaR limit
```

**Decision Rules (CVaR-First)**:
1. If risk_override → REJECT
2. If marginal CVaR > limit → REJECT
3. If leverage limit exceeded → REJECT
4. If position reduction recommended → SELL
5. If position increase recommended → BUY
6. Otherwise → HOLD

#### Agent Orchestra (`agents/orchestrator.py`)
**Purpose**: Coordinate analytical agents into investment committee

**Analytical Agents**:
1. **TechnicalAgent**: Momentum, RSI, MACD, Bollinger Bands signals
2. **SentimentAgent**: News/social media sentiment
3. **ValuationAgent**: P/E, P/B, dividend yield analysis
4. **FundamentalAgent**: EPS, revenue, cash flow analysis

**Control Agents**:
1. **RiskAgent**: Pre-trade risk validation
2. **PortfolioAgent**: Portfolio sizing and allocation

**Workflow**:
```
Analyzers → Consensus Signal → Risk Scaling → Portfolio Sizing → Final Weight
```

---

### 3. Risk Management (`risk/engine.py`)

#### Risk Regimes
| Regime | Market Condition | Risk Action |
|--------|-----------------|-------------|
| BULL_QUIET | Low Vol, Uptrend | Risk-On (1.0x) |
| BULL_VOLATILE | High Vol, Uptrend | Caution (0.7x) |
| BEAR_QUIET | Low Vol, Downtrend | Risk-Off (0.5x) |
| BEAR_CRISIS | High Vol, Downtrend | Block (0.0x) |

#### Risk Limits
| Limit Type | Value | Action |
|------------|-------|--------|
| Max Gross Leverage | 1.0 | SCALE |
| Max Drawdown | 18% | REJECT |
| VaR (95%) | 4% | SCALE |
| CVaR (95%) | 6% | SCALE |
| Stress Loss | 25% | REJECT |
| Sector Exposure | 15% | SCALE |
| Correlation Shock | 0.70 | SCALE |

#### Risk Decisions
- **ALLOW**: Pass all checks, execute as-is
- **SCALE**: Reduce position size based on violations
- **REJECT**: Block the trade entirely
- **FREEZE**: Stop all trading activities
- **LIQUIDATE**: Close all positions immediately
- **RECOVERY**: Gradual re-entry after freeze

#### Circuit Breakers (3 Tiers)
- **Tier 1**: VaR slightly high → SCALE
- **Tier 2**: VaR > 1.5x limit OR vol > 2x target → Defensive (25% sizing)
- **Tier 3**: Drawdown > limit AND high vol → FREEZE

#### Advanced Risk Features
- **EVT (Extreme Value Theory)**: Fat-tail detection with GPD fitting
- **HMM Regime Detection**: Hidden Markov Model for regime classification
- **Stress Testing**: Black Monday, 2008 Crisis, COVID, Inflation scenarios
- **Drawdown Adaptation**: Exponential decay scaling (λ=5)
- **Recovery Protocol**: 5-tier ramp (20% → 100%) after freeze

---

### 4. Portfolio Management (`portfolio/allocator.py`)

#### Capital Auction Engine
**Purpose**: Deterministic capital allocation through competitive bidding

**Method**: Fractional Kelly with constraints

**Constraints**:
- Gross leverage ≤ 1.0
- Gross long ≤ 0.6
- Gross short ≤ 0.4
- Sector cap ≤ 0.15
- Turnover ≤ 0.20
- Single name ≤ 0.10

#### Hedging Overlays
1. **Sector Hedging**: Equal-weight sector balancing
2. **Beta Neutralization**: Market beta hedging
3. **Dynamic Stop-Loss**: Volatility-based stops

---

### 5. Execution Layer

#### Order Management System (`execution/oms.py`)

**Order Lifecycle**:
```
PENDING → SUBMITTED → ACKNOWLEDGED → PARTIAL/FILLED
                                    ↓
              CANCELLED/REJECTED/EXPIRED ←───
```

**Order States**:
- PENDING, SUBMITTED, ACKNOWLEDGED, PARTIAL, FILLED
- CANCELLED, REJECTED, EXPIRED

**Order Types**:
- MARKET (default)
- LIMIT (with price)
- STOP (with trigger)
- STOP_LIMIT (combined)

**Time in Force**:
- DAY (default)
- GTC (Good Till Cancel)
- IOC (Immediate or Cancel)
- FOK (Fill or Kill)

**Pre-Trade Risk Checks**:
- Order value ≤ $1,000,000
- Position concentration ≤ 10%
- Open orders per symbol ≤ 5

#### Alpaca Broker (`brokers/alpaca_broker.py`)

**Features**:
- REST API with retry logic (3 retries, exponential backoff)
- Rate limit handling (429 responses)
- Idempotency via UUID client_order_id
- Fractional shares (4 decimal places)

**Order Submission**:
```python
payload = {
    "symbol": ticker,
    "qty": str(round(qty, 4)),
    "side": "buy"/"sell",
    "type": "market",
    "time_in_force": "day",
    "client_order_id": str(uuid.uuid4())
}
```

---

### 6. Data Layer

#### Data Router (`data/collectors/data_router.py`)

**Multi-Provider Architecture**:
| Provider | Stocks | FX | Crypto | Max History | Entitlement |
|----------|--------|-----|--------|-------------|-------------|
| Yahoo | ✓ | ✓ | ✓ | 5000 days | No |
| Polygon | ✓ | ✓ | ✓ | 5000 days | Yes |
| Alpaca | ✓ | ✗ | ✓ | 730 days | Yes |
| Binance | ✗ | ✗ | ✓ | 1000 days | No |

**Provider Priority**: Yahoo → Polygon → Alpaca

**Phase 0 Guards**:
- No multi-year history in live loop (max 5 days)
- No retry on entitlement failures (400/403)
- Data quality validation (score ≥ 0.6)

**Data Quality Formula**:
```
score = 1.0 - (
    missing_dates_pct × 0.3 +
    duplicate_pct × 0.2 +
    zero_negative_flag × 0.2 +
    extreme_spike_flag × 0.05
)
```

#### Symbol Governor (`data/governance/governance_agent.py`)

**Classification Rules**:
| State | Row Count | Quality | Action |
|-------|-----------|---------|--------|
| ACTIVE | ≥ 1260 | ≥ 0.6 | ALLOW TRADING |
| DEGRADED | 1000-1259 | Any | BLOCK TRADING |
| QUARANTINED | < 1000 | Any | BLOCK TRADING |

---

### 7. Database Layer

#### Database Schema

**Core Tables**:
- `price_history`: Daily OHLCV data
- `decisions`: Trading decisions with full audit
- `orders`: Order lifecycle tracking
- `positions`: Current portfolio positions
- `audit_log`: System audit trail
- `symbol_governance`: Symbol classification
- `data_quality`: Quality scores per symbol
- `model_outputs`: Agent predictions
- `features`: Computed features

**Decision Record (15 Fields)**:
1. cycle_id
2. symbol
3. timestamp
4. final_decision
5. position_size
6. stop_loss
7. reason_codes (JSON)
8. mu_hat
9. sigma_hat
10. conviction
11. data_quality_score
12. provider_confidence
13. agent_results (JSON)
14. risk_checks (JSON)
15. metadata (JSON)

#### Database Adapters
- **SQLiteAdapter**: Development and testing
- **PostgresAdapter**: Production with TimescaleDB

---

### 8. Configuration Management (`configs/config_manager.py`)

**Golden Config**: `configs/golden_config.yaml`

**Features**:
- SHA256 hash verification for immutability
- Secret injection from Vault
- Singleton pattern for system-wide access
- Runtime integrity validation

**Secret Mapping**:
```python
{
    'alpaca_api_key': ('quant-fund/api-keys', 'ALPACA_API_KEY'),
    'alpaca_secret_key': ('quant-fund/api-keys', 'ALPACA_SECRET_KEY'),
    'postgres_password': ('quant-fund/database', 'POSTGRES_PASSWORD')
}
```

---

### 9. Monitoring & Alerting (`monitoring/alerts.py`)

#### Alert Channels
- **Telegram**: Direct messaging
- **Slack**: Team channel integration
- **Discord**: Community integration

#### Alert Severity Levels
- **LOW**: Informational
- **MEDIUM**: Warning
- **HIGH**: Action required
- **CRITICAL**: Immediate attention

#### Alert Categories
- RISK, PERFORMANCE, SYSTEM
- MARKET_DATA, TRADING, DATA_QUALITY

#### Key Alerts
- Data quality failure (score < 0.6)
- Missing trading days
- Quality threshold breach
- CVaR breach (3 consecutive → KILL_SWITCH)
- Heartbeat (configurable interval)

#### Deduplication
- 10-minute default window
- Configurable via environment variable

---

## System Workflow

### Complete Trading Cycle

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRADING CYCLE FLOW                           │
└─────────────────────────────────────────────────────────────────┘

┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   START      │───→│   PHASE 0    │───→│   PHASE 1    │
│   CYCLE      │    │   Check      │    │   Data Load  │
└──────────────┘    │   Kill Switch│    └──────────────┘
                    │   1260 Rows  │            │
                    └──────────────┘            ▼
                                      ┌──────────────┐
                                      │   PHASE 2    │
                                      │   Signals    │
                                      └──────────────┘
                                              │
                                              ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   EXECUTE    │←───│   PHASE 5    │←───│   PHASE 4    │
│   ORDERS     │    │   OMS        │    │   Risk Check │
└──────────────┘    │   Submit     │    └──────────────┘
                    └──────────────┘            │
                           │                   ▼
                           │           ┌──────────────┐
                           │           │   PHASE 3    │
                           │           │   Allocation │
                           │           └──────────────┘
                           │                   │
                           └───────────────────┘
                                   (REJECT path)
```

### Phase Details

**Phase 0: Pre-Flight Checks**
- Check kill switch file
- Verify symbol governance (ACTIVE state)
- Verify 1260 rows of history
- Halt on failure

**Phase 1: Data Loading**
- Fetch 252-day price history
- Validate data quality (score ≥ 0.6)
- Archive raw responses

**Phase 2: Signal Generation**
- Technical analysis (RSI, MACD, BB)
- Sentiment analysis
- Valuation metrics
- Fundamental analysis

**Phase 3: Allocation**
- Meta-Brain ensemble aggregation
- Kelly sizing calculation
- Constraint application

**Phase 4: Risk Validation**
- VaR/CVaR limits
- Drawdown limits
- Sector exposure limits
- Circuit breaker checks

**Phase 5: Execution**
- OMS order creation
- Broker submission
- Fill processing
- Audit persistence

---

## Governance & Safety

### 11-Layer Governance Stack

1. **Kill Switch**: Manual binary halt
2. **Data Governance**: Symbol classification
3. **History Requirement**: 1260 rows minimum
4. **Data Quality**: Score ≥ 0.6 threshold
5. **Signal Validation**: At least one valid agent
6. **Risk Limits**: VaR/CVaR constraints
7. **Circuit Breakers**: 3-tier freeze system
8. **CVaR Persistence**: 3 breaches = KILL_SWITCH
9. **Recovery Protocol**: 10-day cooldown
10. **Audit Trail**: 15-field records
11. **Alerting**: Multi-channel notifications

### Emergency Procedures

**Kill Switch Activation**:
```powershell
New-Item -Path "runtime/KILL_SWITCH" -ItemType File
```

**Resume Trading**:
```powershell
Remove-Item "runtime/KILL_SWITCH"
```

---

## Configuration Reference

### Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| DB_ENGINE | sqlite | Database backend |
| POSTGRES_PASSWORD | - | Database password |
| ALPACA_API_KEY | - | Alpaca API key |
| ALPACA_SECRET_KEY | - | Alpaca secret |
| TELEGRAM_BOT_TOKEN | - | Telegram notification |
| TELEGRAM_CHAT_ID | - | Telegram chat ID |
| SLACK_WEBHOOK_URL | - | Slack webhook |
| DISCORD_WEBHOOK_URL | - | Discord webhook |
| MONITOR_NOTIFY_LEVEL | INFO | Alert threshold |
| HEARTBEAT_INTERVAL | 3600 | Heartbeat seconds |
| ALERT_DEDUP_WINDOW | 600 | Deduplication window |

### Key Constants

| Constant | Value | Purpose |
|----------|-------|---------|
| REQUIRED_HISTORY_ROWS | 1260 | 5 years daily data |
| MAX_LIVE_HISTORY_DAYS | 5 | Live loop max history |
| MIN_DATA_QUALITY | 0.6 | Minimum quality score |
| DEFAULT_BETA | 0.5 | Disagreement penalty |
| DEFAULT_GAMMA | 0.3 | Kelly fraction |
| RISK_FREE_RATE | 0.02 | Annual risk-free rate |
| MAX_POSITION_PCT | 0.10 | 10% max position |
| MAX_LEVERAGE | 1.0 | Gross leverage limit |
| MAX_DRAWDOWN | 0.18 | Drawdown limit |
| VAR_LIMIT | 0.04 | VaR limit (4%) |
| CVAR_LIMIT | 0.06 | CVaR limit (6%) |
| SECTOR_LIMIT | 0.15 | Sector exposure limit |
| KILL_SWITCH_TRIGGER | 0.25 | 25% loss = halt |

---

## File Structure

```
mini-quant-fund/
├── agents/                      # AI Agent Layer
│   ├── base_agent.py
│   ├── meta_brain.py            # Ensemble aggregation
│   ├── technical_agent.py
│   ├── sentiment_agent.py
│   ├── valuation_agent.py
│   ├── fundamental_agent.py
│   ├── risk_agent.py
│   ├── portfolio_agent.py
│   └── orchestrator.py          # Agent coordination
│
├── alpha_agents/                # Specialized Alpha
├── alpha_families/              # Alpha Strategies
├── analytics/                   # Metrics
├── audit/                       # Audit System
├── backtest/                    # Backtesting
├── brokers/                     # Execution
│   ├── alpaca_broker.py
│   └── mock_broker.py
├── compliance/                  # Compliance
├── configs/                     # Configuration
│   ├── config_manager.py
│   └── golden_config.yaml
├── data/                        # Data Layer
│   ├── collectors/
│   │   └── data_router.py       # Multi-provider router
│   ├── governance/
│   │   └── governance_agent.py  # Symbol classification
│   └── providers/
├── database/                    # Database
│   ├── manager.py
│   └── adapters/
├── execution/                   # Execution
│   └── oms.py                   # Order Management
├── monitoring/                  # Monitoring
│   └── alerts.py                # Alerting System
├── portfolio/                   # Portfolio
│   └── allocator.py
├── risk/                        # Risk Management
│   ├── engine.py
│   ├── cvar.py
│   ├── tail_risk.py
│   └── ...
├── strategies/                  # Strategies
│   └── factory.py
├── utils/                       # Utilities
│   ├── logging_config.py
│   └── metrics.py
│
├── main.py                      # Single cycle entry
├── trading_daemon.py            # 24/7 daemon
├── live_trading_daemon.py       # Institutional agent
├── run_cycle.py
├── dashboard.py
│
├── docker-compose.yml           # Docker orchestration
├── requirements.txt             # Dependencies
├── pyproject.toml               # Project config
│
└── README.md                    # Documentation
```

---

## Command Reference

| Command | Purpose |
|---------|---------|
| `python main.py --run-once --mode paper` | Single paper cycle |
| `python trading_daemon.py --trigger 5` | Run 24/7 daemon |
| `python live_trading_daemon.py` | Institutional trading |
| `python ingest_history.py` | Historical backfill |
| `python dashboard.py` | Streamlit dashboard |

---

## Verification Checklist

- [x] System architecture documented
- [x] All core components implemented
- [x] Risk management rules coded
- [x] Database schema deployed
- [x] Broker integration tested
- [x] Backtest engine operational
- [x] Monitoring system active
- [x] Documentation complete

---

**Document Version**: 1.0
**Last Updated**: 2026-01-19
**Status**: PRODUCTION READY

