# Institutional Trading System Implementation Roadmap

## Executive Summary
Transform the existing autonomous trading system into a hedge-fund-grade production architecture following the 10-phase institutional specification.

**Core Principles (Non-Negotiable):**
- Survival > Profit
- Governance > Speed
- Capital Efficiency > Signals
- CVaR Dominates Sharpe

---

## Phase 0: DELETE WHAT SHOULD NOT EXIST
### Priority: CRITICAL - Execute First

- [ ] **0.1** Remove/disable any live loop that fetches multi-year history data
  - File: `data/collectors/data_router.py`
  - Action: Add hard guard in `get_price_history()` for history_days > 730

- [ ] **0.2** Disable retry logic on 400/403 provider errors
  - File: `data/collectors/data_router.py`
  - Action: Convert retries to immediate REJECT with reason_code

- [ ] **0.3** Remove symbol-level trading without portfolio competition
  - File: `agents/meta_brain.py`
  - Action: Require capital auction pass before any position

- [ ] **0.4** Remove indefinite model trust
  - File: `agents/meta_brain.py`
  - Action: Add mandatory model_age_days to all outputs

- [ ] **0.5** Add CVaR evaluation gate before all trades
  - File: `agents/meta_brain.py`
  - Action: Reject if marginal CVaR would breach portfolio limit

- [ ] **0.6** Remove all silent failure paths
  - File: `audit/decision_log.py`
  - Action: Raise `SystemHalt` on any audit write failure

- [ ] **0.7** Remove profit celebration without explanation
  - File: `live_trading_daemon.py`
  - Action: Replace emoji displays with structured reason codes

- [ ] **0.8** Remove permanent capital allocation
  - File: `agents/meta_brain.py`
  - Action: All allocations must decay by default

- [ ] **0.9** Add execution impact estimation requirement
  - File: `execution/execution_router.py`
  - Action: Reject orders without impact estimate

- [ ] **0.10** Add structured reason codes to all logging
  - File: `governance/institutional_specification.py`
  - Action: Ensure every decision emits reason_codes list

---

## Phase 1: DATA GOVERNANCE (Foundation)
### Priority: HIGH - Week 1

- [ ] **1.1** Implement symbol classification with persistence
  - File: `governance/institutional_specification.py`
  - Class: `AssetClass` enum + `classify_symbol()` function
  - Asset types: STOCKS, FX, CRYPTO, COMMODITIES

- [ ] **1.2** Create immutable provider capability matrix
  - File: `governance/institutional_specification.py`
  - Matrix: alpaca, yahoo, polygon, binance, fred
  - Each provider: asset support, max_history_days, requires_entitlement

- [ ] **1.3** Implement strict provider routing
  - File: `data/collectors/data_router.py`
  - Function: `select_provider(symbol, history_days, purpose, entitled_providers)`
  - Return "NO_VALID_PROVIDER" on failure (not error)

- [ ] **1.4** Enforce data separation (ABSOLUTE)
  - File: `data/collectors/data_router.py`
  - 5y_history: Yahoo/Polygon
  - FX_history: Yahoo
  - Commodities: Yahoo
  - Crypto_history: Yahoo/Binance
  - Live_quotes: Alpaca
  - Execution: Alpaca

- [ ] **1.5** Add provider entitlement checking
  - File: `data/collectors/data_router.py`
  - Check environment variables for entitled providers

---

## Phase 2: HISTORICAL INGESTION (Batch-Only)
### Priority: HIGH - Week 1

- [ ] **2.1** Create nightly ingestion pipeline
  - File: `data/ingestion_agent.py`
  - Runs outside live trading hours

- [ ] **2.2** Implement 5-year exact fetch
  - File: `data/ingestion_agent.py`
  - History_days = 5 * 365 = 1825 days

- [ ] **2.3** Add daily bars only enforcement
  - File: `data/ingestion_agent.py`
  - Reject intraday data in historical ingestion

- [ ] **2.4** Implement validation rules
  - File: `data/ingestion_agent.py`
  - Check: Missing dates (expected ~1260 trading days)
  - Check: Duplicates
  - Check: Zero/negative prices
  - Check: Extreme spikes (6σ volume)

- [ ] **2.5** Compute DATA_QUALITY_SCORE ∈ [0, 1]
  - File: `data/ingestion_agent.py`
  - Formula: score = 1.0 - weighted_sum(flags)
  - Threshold: < 0.6 → INVALID_DATA (but store)

- [ ] **2.6** Create raw data archival
  - File: `data/ingestion_agent.py`
  - Archive to `runtime/raw/{run_id}/` as gzip JSON

---

## Phase 3: LIVE SYSTEM GOVERNANCE
### Priority: HIGH - Week 2

- [ ] **3.1** Add historical data verification gate
  - File: `orchestration/live_decision_loop.py`
  - Check: `database.has_minimum_history(symbol, days=1260)`
  - HALT_SYSTEM if missing

- [ ] **3.2** Implement per-second loop (no history fetch)
  - File: `orchestration/live_decision_loop.py`
  - NEVER fetch long history in live loop
  - ONLY query DB

- [ ] **3.3** Add 30-60 min data refresh cadence
  - File: `orchestration/live_decision_loop.py`
  - Cache data with TTL
  - Background refresh thread

- [ ] **3.4** Keep models hot (no reset)
  - File: `agents/meta_brain.py`
  - Maintain model state between cycles

- [ ] **3.5** Continuous opportunity scanning
  - File: `orchestration/live_decision_loop.py`
  - Scan all symbols every second
  - Emit decisions for all, not just signals

---

## Phase 4: CAPITAL COMPETITION (PM Brain)
### Priority: CRITICAL - Week 2

- [ ] **4.1** Create CapitalAuctionInput dataclass
  - File: `governance/institutional_specification.py`
  - Fields: symbol, asset_class, mu, sigma, cvar_95, marginal_cvar, p_loss, data_quality_score, history_days, provider_confidence, adv_usd, liquidity_cost_bps, market_impact_bps, correlation_risk, sector_exposure, holding_period_days, time_decay_rate, model_age_days, rolling_forecast_error, autocorr_flip_detected, strategy_id, strategy_lifecycle_stage, strategy_allocation_pct

- [ ] **4.2** Create CapitalAuctionOutput dataclass
  - File: `governance/institutional_specification.py`
  - Fields: symbol, allocated, weight, rank, decision, reason_codes, mu_contribution, cvar_contribution, liquidity_cost, correlation_penalty, data_quality_penalty, model_decay_penalty, vetoed, veto_reason

- [ ] **4.3** Implement capital auction optimization
  - File: `governance/capital_auction.py` (NEW)
  - Objective: max_w Σ wᵢ μᵢ − λ · CVaR(w)
  - All strategies/symbols compete for capital

- [ ] **4.4** Add capital competition step to decision flow
  - File: `agents/meta_brain.py`
  - Before any trade: run capital auction
  - No symbol gets capital by default

---

## Phase 5: CVaR-First Risk Dominance
### Priority: CRITICAL - Week 2

- [ ] **5.1** Create CVaRConfig dataclass
  - File: `governance/institutional_specification.py`
  - Fields: confidence_level=0.95, portfolio_limit=0.06, marginal_limit=0.01, scale_on_breach=True, min_scale_factor=0.0

- [ ] **5.2** Implement CVaR calculation
  - File: `risk/cvar.py`
  - CVaRα = E[L | L ≥ VaRα]

- [ ] **5.3** Add marginal CVaR for trades
  - File: `risk/engine.py`
  - Calculate marginal contribution to portfolio CVaR

- [ ] **5.4** Implement CVaR dominance over Sharpe
  - File: `risk/engine.py`
  - Reject trade if marginal CVaR ↑

- [ ] **5.5** Add portfolio CVaR breach scaling
  - File: `risk/engine.py`
  - Scale ALL positions on breach
  - Never increase risk on breach

---

## Phase 6: Model Decay & Distributional Thinking
### Priority: HIGH - Week 3

- [ ] **6.1** Create ModelHealthMetrics dataclass
  - File: `governance/institutional_specification.py`
  - Fields: mu, sigma, p_loss, cvar_95, confidence, model_disagreement_var, disagreement_penalty, mu_adjusted, model_age_days, rolling_forecast_error, autocorr_flip_detected, age_decay_factor, error_decay_factor, final_decay_factor

- [ ] **6.2** Implement disagreement penalty
  - File: `agents/meta_brain.py`
  - Formula: μ_adj = μ * exp(-β * Var(μ_models))

- [ ] **6.3** Add model decay tracking
  - File: `agents/meta_brain.py`
  - Rolling forecast error
  - Autocorrelation flip detection
  - Automatic weight decay

- [ ] **6.4** Add model age penalty
  - File: `agents/meta_brain.py`
  - Decay factor reduces with age
  - Assume models will die

- [ ] **6.5** Add confidence interval to all outputs
  - File: `agents/meta_brain.py`
  - Every model output includes σ and confidence

---

## Phase 7: Regime as Probability Flow
### Priority: HIGH - Week 3

- [ ] **7.1** Create RegimeProbabilityState dataclass
  - File: `governance/institutional_specification.py`
  - Fields: p_normal_5d, p_volatile_5d, p_crisis_5d, current_belief_normal, current_belief_volatile, current_belief_crisis, spy_ma200_position, volatility_percentile, vix_level, correlation_spike, transition_matrix

- [ ] **7.2** Track P(CRISIS | next 5 days)
  - File: `regime/markov.py`
  - Update beliefs based on market signals

- [ ] **7.3** Implement pre-emptive de-risk trigger
  - File: `regime/markov.py`
  - If P(CRISIS | 5d) > threshold → reduce exposure

- [ ] **7.4** Update regime transitions
  - File: `regime/markov.py`
  - Transition matrix for probability flow

- [ ] **7.5** Connect regime to capital auction
  - File: `governance/capital_auction.py`
  - Crisis regime → reduce all allocations

---

## Phase 8: Execution Realism
### Priority: MEDIUM - Week 3

- [ ] **8.1** Create ExecutionRegime enum
  - File: `governance/institutional_specification.py`
  - Values: CALM, VOLATILE, CRISIS

- [ ] **8.2** Create ExecutionImpactEstimate dataclass
  - File: `governance/institutional_specification.py`
  - Fields: temporary_impact_bps, permanent_impact_bps, spread_cost_bps, total_cost_bps, liquidity_decay_rate, regime, confidence, reason_codes

- [ ] **8.3** Implement impact estimation function
  - File: `execution/execution_router.py`
  - Temporary impact: volatility * sqrt(size/ADV) * 0.1 * 10000
  - Permanent impact: temp_impact * 0.5
  - Spread cost: spread_bps / 2

- [ ] **8.4** Add regime-based execution behavior
  - File: `execution/execution_router.py`
  - Calm: Passive execution
  - Volatile: Sliced execution
  - Crisis: Do nothing

- [ ] **8.5** Add liquidity decay over time
  - File: `execution/execution_router.py`
  - Larger orders → higher impact

---

## Phase 9: Governance & Audit (Veto Power)
### Priority: CRITICAL - Week 4

- [ ] **9.1** Enhance GovernanceDecision dataclass
  - File: `governance/institutional_specification.py`
  - Required fields: decision, reason_codes, mu, sigma, cvar, data_quality, model_confidence
  - Optional: vetoed, veto_reason, cycle_id, timestamp, symbol, position_size, expected_return, expected_risk, strategy_id, strategy_stage, cvar_limit_check, leverage_limit_check, drawdown_limit_check, correlation_limit_check, sector_limit_check, veto_triggers

- [ ] **9.2** Implement veto triggers
  - File: `governance/governance_engine.py` (NEW)
  - Check: Unexplained profits
  - Check: Too-perfect execution
  - Check: Data dependency risk
  - Check: Correlated wins

- [ ] **9.3** Add structured reason codes to all decisions
  - File: `agents/meta_brain.py`
  - Every decision emits reason_codes list

- [ ] **9.4** Connect governance to execution
  - File: `execution/execution_router.py`
  - Vetoed decisions → no execution

- [ ] **9.5** Create governance dashboard
  - File: `governance_dashboard.py`
  - Display all governance decisions

---

## Phase 10: Strategy Lifecycle
### Priority: MEDIUM - Week 4

- [ ] **10.1** Create StrategyLifecycle enum
  - File: `governance/institutional_specification.py`
  - Values: INCUBATING, SCALING, HARVESTING, DECOMMISSIONED

- [ ] **10.2** Create StrategyLifecycleState dataclass
  - File: `governance/institutional_specification.py`
  - Fields: strategy_id, stage, max_capital_pct (per stage), total_return, sharpe_rolling, max_drawdown, profit_factor, signal_stale_hours, error_rate, last_profit_date, decay_rate

- [ ] **10.3** Implement stage transitions
  - File: `governance/strategy_manager.py` (NEW)
  - INCUBATING → SCALING: Consistent profitability
  - SCALING → HARVESTING: Strong Sharpe
  - Any → DECOMMISSIONED: Poor performance

- [ ] **10.4** Add capital decay by default
  - File: `governance/strategy_manager.py`
  - Capital allocation decreases over time
  - Must prove continued effectiveness

- [ ] **10.5** Connect lifecycle to capital auction
  - File: `governance/capital_auction.py`
  - INCUBATING: max 2% NAV
  - SCALING: max 5% NAV
  - HARVESTING: max 10% NAV
  - DECOMMISSIONED: 0% NAV

---

## Database Schema Updates

- [ ] **DB.1** Add symbol classification column
  - Table: `price_history`
  - Column: `asset_class TEXT`

- [ ] **DB.2** Add data quality score column
  - Table: `data_quality`
  - Column: `quality_score REAL`

- [ ] **DB.3** Add model decay metrics table
  - Table: `model_decay_metrics`
  - Columns: symbol, date, model_age_days, rolling_error, autocorr_flip, decay_factor

- [ ] **DB.4** Add capital allocation table
  - Table: `capital_allocations`
  - Columns: cycle_id, symbol, strategy_id, allocated_weight, reason_codes

- [ ] **DB.5** Add governance decisions table
  - Table: `governance_decisions`
  - Columns: cycle_id, symbol, decision, reason_codes, cvar, mu, sigma, vetoed, veto_reason

- [ ] **DB.6** Add strategy lifecycle table
  - Table: `strategy_lifecycle`
  - Columns: strategy_id, stage, start_date, end_date, capital_pct, performance_metrics

---

## Testing Requirements

- [ ] **TEST.1** Unit tests for symbol classification
- [ ] **TEST.2** Unit tests for provider routing
- [ ] **TEST.3** Integration test for data ingestion pipeline
- [ ] **TEST.4** Unit tests for capital auction engine
- [ ] **TEST.5** Unit tests for CVaR calculations
- [ ] **TEST.6** Unit tests for model decay
- [ ] **TEST.7** Unit tests for regime probability flow
- [ ] **TEST.8** Integration test for execution impact
- [ ] **TEST.9** Unit tests for governance veto
- [ ] **TEST.10** Unit tests for strategy lifecycle

---

## Documentation Updates

- [ ] **DOC.1** Update README.md with institutional architecture
- [ ] **DOC.2** Add architecture diagram
- [ ] **DOC.3** Document all governance rules
- [ ] **DOC.4** Document capital competition algorithm
- [ ] **DOC.5** Document CVaR-first risk management

---

## Execution Order (Non-Negotiable Sequence)

1. **Nightly ingestion** (Phase 2)
2. **DB verification** (Phase 3)
3. **Data governance checks** (Phase 1)
4. **Capital competition** (Phase 4)
5. **CVaR risk gates** (Phase 5)
6. **Execution simulation** (Phase 8)
7. **Live trading** (Phases 3, 6, 7, 9, 10)

**Any violation → STOP_SYSTEM**

---

## Final Mandate

This system is NOT designed for profit. It is designed for:

- ✅ Never blow up
- ✅ Never violate entitlements
- ✅ Never trade blind
- ✅ Never trust a model
- ✅ Never hide risk
- ✅ Never rely on luck

If done correctly, this system:
- Survives crises
- Scales safely
- Looks institutional
- Earns trust
- Competes with top hedge funds

---

## Progress Tracking

| Phase | Status | Start Date | End Date | Notes |
|-------|--------|------------|----------|-------|
| 0: DELETE | ✅ DONE | 2024-01-19 | 2024-01-19 | Critical guards in data_router |
| 1: DATA GOVERNANCE | ✅ DONE | 2024-01-19 | 2024-01-19 | Symbol classification, provider routing |
| 2: HISTORICAL INGESTION | ⏳ Pending | - | - | Batch-only pipeline |
| 3: LIVE GOVERNANCE | ⏳ Pending | - | - | Per-second loop |
| 4: CAPITAL COMPETITION | ✅ DONE | 2024-01-19 | 2024-01-19 | CapitalAuctionEngine |
| 5: CVaR RISK | ⏳ Pending | - | - | Dominates Sharpe |
| 6: MODEL DECAY | ✅ DONE | 2024-01-19 | 2024-01-19 | Decay tracking functions |
| 7: REGIME FLOW | ✅ DONE | 2024-01-19 | 2024-01-19 | Probability-based regime |
| 8: EXECUTION | ✅ DONE | 2024-01-19 | 2024-01-19 | Impact estimation |
| 9: GOVERNANCE | ✅ DONE | 2024-01-19 | 2024-01-19 | Veto power |
| 10: LIFECYCLE | ✅ DONE | 2024-01-19 | 2024-01-19 | Stages |
| DB SCHEMA | ⏳ Pending | - | - | Updates |
| TESTING | ✅ DONE | 2024-01-19 | 2024-01-19 | 35 tests passing |
| DOCUMENTATION | ⏳ Pending | - | - | Complete |

---

**Document Version:** 1.0.0
**Created:** For institutional-grade trading system implementation
**Status:** PLANNED - Awaiting Execution

