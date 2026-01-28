# INSTITUTIONAL ARCHITECTURE SPECIFICATION v2.0
# Mini Quant Fund - Autonomous Trading Platform Upgrade

**Document Version:** 2.0.0
**Date:** 2024
**Classification:** Internal Use Only
**Status:** Specification

---

## EXECUTIVE SUMMARY

This document specifies a fully integrated institutional upgrade for the Mini Quant Fund autonomous trading platform. The upgrade encompasses five critical domains designed to achieve institutional-grade performance, explainability, and adaptability:

1. **Factor Attribution & PnL Explainability** - Full transparency into portfolio PnL decomposition
2. **Online Agent Weight Learning** - Adaptive, regime-aware agent weighting
3. **Reinforcement Learning Execution Engine** - Intelligent execution with minimal slippage
4. **Portfolio-Level Optimization Layer** - Global portfolio optimization
5. **Live Slippage Feedback Loop** - Continuous execution improvement

---

## PART 1: FACTOR ATTRIBUTION & PnL EXPLAINABILITY

### 1.1 Objective

Provide complete transparency into portfolio performance by decomposing PnL into attributable components.

### 1.2 Mathematical Formulation

#### Return Decomposition

For each symbol `i` on date `t`:

$$R_{i,t}^{total} = R_{i,t}^{market} + R_{i,t}^{factors} + R_{i,t}^{\alpha} + \epsilon_{i,t}$$

**Mandatory Factors:**
| Factor | Construction | Weight |
|--------|-------------|--------|
| Momentum | 12-month return, skip 1 month | Long top 30%, Short bottom 30% |
| Value | Book-to-Market or E/P proxy | Long high BV/M, Short low BV/M |
| Volatility | Realized vol rank | Long low vol, Short high vol |
| Size | Market cap rank | Long small caps, Short large caps |
| Liquidity | Volume/ADV ratio | Long liquid, Short illiquid |
| Sector | GICS sector dummies | Long/short within sector |

**Regression Method:**
- Window: 252 trading days
- Update: Daily
- Robustness: HAC (Newey-West) standard errors

### 1.3 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    ATTRIBUTION ENGINE                           │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │ Price Data   │    │ Factor Data  │    │ Model Outputs│      │
│  │ (OHLCV)      │    │ (Benchmarks) │    │ (Agent μ,σ)  │      │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘      │
│         │                   │                   │               │
│         ▼                   ▼                   ▼               │
│  ┌─────────────────────────────────────────────────────┐        │
│  │              ATTRIBUTION COMPUTER                    │        │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │        │
│  │  │ Factor      │  │ Regression  │  │ Attribution │ │        │
│  │  │ Loader      │  │ Engine      │  │ Calculator  │ │        │
│  │  └─────────────┘  └─────────────┘  └─────────────┘ │        │
│  └─────────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────┘
```

### 1.4 Persistence Schemas

```sql
-- Factor Returns
CREATE TABLE IF NOT EXISTS factor_returns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    factor_name TEXT NOT NULL,
    factor_return REAL NOT NULL,
    t_stat REAL,
    p_value REAL,
    n_observations INTEGER,
    r_squared REAL,
    UNIQUE(date, factor_name)
);

-- Security Factor Exposures
CREATE TABLE IF NOT EXISTS security_factor_exposures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    date TEXT NOT NULL,
    beta_market REAL,
    beta_momentum REAL,
    beta_value REAL,
    beta_volatility REAL,
    beta_size REAL,
    beta_liquidity REAL,
    alpha REAL,
    residual_std REAL,
    r_squared REAL,
    UNIQUE(symbol, date)
);

-- Agent P&L Attribution
CREATE TABLE IF NOT EXISTS agent_pnl (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    agent_name TEXT NOT NULL,
    mu_hat REAL,
    sigma_hat REAL,
    confidence REAL,
    agent_weight REAL,
    realized_return REAL,
    pnl_contribution REAL,
    UNIQUE(cycle_id, symbol, agent_name)
);

-- Portfolio Daily Attribution
CREATE TABLE IF NOT EXISTS portfolio_attribution (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    cycle_id TEXT,
    total_portfolio_return REAL,
    market_contribution REAL,
    momentum_contribution REAL,
    value_contribution REAL,
    volatility_contribution REAL,
    size_contribution REAL,
    liquidity_contribution REAL,
    sector_contribution REAL,
    alpha_contribution REAL,
    idiosyncratic_contribution REAL,
    portfolio_volatility REAL,
    portfolio_beta REAL,
    tracking_error REAL,
    information_ratio REAL,
    top_contributor_symbol TEXT,
    top_contributor_pnl REAL,
    worst_detractor_symbol TEXT,
    worst_detractor_pnl REAL,
    UNIQUE(date)
);
```

---

## PART 2: ONLINE AGENT WEIGHT LEARNING

### 2.1 Objective

Replace static confidence weighting with adaptive online learning. Agents that perform well recently and in current regimes gain influence.

### 2.2 Mathematical Formulation

#### State Definition

$$s_t = [Sharpe_{agent,i}^{rolling}, HitRate_{agent,i}^{recent}, Regime_t, VolRegime_t, Breadth_t]$$

#### Reward Function

$$r_t = R_{realized} - \lambda \cdot Risk$$

### 2.3 Algorithm Selection

#### Primary: Exponentiated Gradient (EG)

**Why EG?**
- Robust to non-stationary environments
- Natural rebalancing toward better performers
- Built-in exploration through softmax

**Update Rule:**

$$w_{t+1,i} = \frac{w_{t,i} \exp(\eta \cdot g_{t,i})}{\sum_j w_{t,j} \exp(\eta \cdot g_{t,j})}$$

**Parameters:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Learning rate η | 0.1 | Balance adaptation vs. stability |
| Minimum weight | 0.05 | Prevent complete elimination |
| Maximum weight | 0.40 | Diversification constraint |

#### Secondary: Thompson Sampling (TS)

For regime-aware switching with beta distributions.

### 2.4 Update Cadence & Constraints

| Setting | Value |
|---------|-------|
| Update frequency | Daily |
| Lookback window | 21 days |
| Weight floor | 5% |
| Weight cap | 40% |
| Decay half-life | 10 days |

### 2.5 Persistence Schema

```sql
-- Agent Weight History
CREATE TABLE IF NOT EXISTS agent_weights (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    cycle_id TEXT NOT NULL,
    agent_name TEXT NOT NULL,
    weight REAL NOT NULL,
    rolling_sharpe REAL,
    recent_hit_rate REAL,
    regime TEXT,
    volatility_regime REAL,
    algorithm_used TEXT,
    UNIQUE(cycle_id, agent_name)
);

-- Performance Tracking
CREATE TABLE IF NOT EXISTS agent_performance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_name TEXT NOT NULL,
    date TEXT NOT NULL,
    predicted_return REAL,
    actual_return REAL,
    hit INTEGER,
    pnl_contribution REAL,
    factor_adjusted_return REAL,
    UNIQUE(agent_name, date)
);

-- Regime Classification
CREATE TABLE IF NOT EXISTS regime_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    regime TEXT NOT NULL,
    regime_prob REAL,
    spy_ma200_position REAL,
    volatility_percentile REAL,
    vix_level REAL,
    UNIQUE(date)
);
```

---

## PART 3: REINFORCEMENT LEARNING EXECUTION ENGINE

### 3.1 Objective

Minimize execution costs through intelligent order placement:
- Slippage reduction
- Market impact minimization
- Delay cost balancing

### 3.2 RL Environment Specification

#### State Space

$$s_t = [size/ADV, \sigma_{daily}, spread_{bps}, imbalance, time\_remaining, vol\_regime]$$

| State Variable | Range | Description |
|----------------|-------|-------------|
| size/ADV | [0, 0.5] | Order size relative to ADV |
| σ_daily | [0, 0.10] | Daily volatility |
| spread_bps | [0, 100] | Bid-ask spread in bps |
| imbalance | [-1, 1] | Order book imbalance |
| time_remaining | [0, 1] | Fraction of trading day remaining |
| vol_regime | [0, 1] | Vol regime percentile |

#### Action Space

| Action | Description |
|--------|-------------|
| MARKET | Immediate execution |
| LIMIT | Set limit price offset (bps) |
| TWAP | Divide into equal slices |
| POV | Percentage of volume |
| VWAP | Volume-weighted execution |
| WAIT | Delay execution |

#### Reward Function

$$r_t = -(\lambda_{slippage} \cdot Slippage + \lambda_{impact} \cdot Impact + \lambda_{delay} \cdot Delay)$$

### 3.3 Algorithm Selection

#### Primary: Soft Actor-Critic (SAC)

**Why SAC?**
- Sample-efficient off-policy algorithm
- Automatic entropy tuning
- Stable learning for continuous actions

**Hyperparameters:**

| Parameter | Value |
|-----------|-------|
| Learning rate (actor) | 3e-4 |
| Learning rate (critic) | 3e-4 |
| Buffer size | 1,000,000 |
| Batch size | 256 |
| Target update τ | 0.005 |
| Discount γ | 0.99 |

### 3.4 Safe Exploration & Deployment Boundaries

```python
class SafeExecutionRL:
    def __init__(self):
        self.kill_switch = False
        self.max_slippage_bps = 50
        self.max_impact_bps = 30

    def select_action(self, state: State, explore: bool = True) -> Action:
        if self.kill_switch:
            return Action.MARKET

        action = self.model.predict(state, deterministic=not explore)

        if state.size_adv > 0.25 and action == Action.MARKET:
            action = Action.TWAP

        if self._estimate_slippage(state, action) > self.max_slippage_bps:
            action = Action.LIMIT

        return action
```

#### Deployment Boundaries

| Condition | Action | Threshold |
|-----------|--------|-----------|
| Slippage spike | Kill switch | >100 bps |
| Model uncertainty | Fallback | σ > 0.3 |
| Latency > 1s | Switch to TWAP | 1000 ms |
| Price > 5% away | Cancel | 5% |
| Volume anomaly | Abort | Vol > 3x ADV |

### 3.5 Persistence Schema

```sql
-- Execution Model Versions
CREATE TABLE IF NOT EXISTS execution_model_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    version_id TEXT NOT NULL UNIQUE,
    algorithm TEXT NOT NULL,
    model_path TEXT,
    trained_at TEXT,
    metrics_json TEXT,
    hyperparameters_json TEXT,
    status TEXT
);

-- Execution Decisions
CREATE TABLE IF NOT EXISTS execution_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id TEXT NOT NULL,
    cycle_id TEXT,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    quantity REAL NOT NULL,
    size_adv REAL,
    volatility REAL,
    spread_bps REAL,
    imbalance REAL,
    time_remaining REAL,
    action TEXT NOT NULL,
    confidence REAL,
    q_values_json TEXT,
    used_fallback BOOLEAN DEFAULT 0,
    fallback_reason TEXT,
    UNIQUE(order_id)
);

-- Execution Outcomes
CREATE TABLE IF NOT EXISTS execution_outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id TEXT NOT NULL,
    expected_price REAL,
    fill_price REAL,
    fill_time TEXT,
    slippage_bps REAL,
    market_impact_bps REAL,
    commission REAL,
    total_cost_bps REAL,
    adv REAL,
    adv_dollar REAL,
    vol_regime REAL,
    bid_ask_spread REAL,
    model_estimated_cost REAL,
    cost_error_bps REAL,
    UNIQUE(order_id)
);
```

---

## PART 4: PORTFOLIO-LEVEL OPTIMIZATION LAYER

### 4.1 Objective

Convert symbol-level signals into a globally optimal portfolio maximizing risk-adjusted returns.

### 4.2 Mathematical Formulation

#### Optimization Problem

$$\max_w \; \mu^T w - \lambda w^T \Sigma w$$

**Subject to:**

| Constraint | Form | Value |
|------------|------|-------|
| Gross leverage | ∑|wᵢ| ≤ 1.0 | 1.0 |
| Gross long | ∑wᵢ>₀ wᵢ ≤ 0.6 | 60% |
| Gross short | ∑wᵢ<₀ |wᵢ| ≤ 0.4 | 40% |
| Sector cap | ∑ᵢ₌ₛₑcₜₒᵣ |wᵢ| ≤ 0.15 | 15% |
| Turnover | ∑|wᵢ - wᵢᵖʳᵉᵛ| ≤ 0.20 | 20% |
| Liquidity | wᵢ ≤ 0.02·ADVᵢ/NAV | 2% ADV |
| Single name | |wᵢ| ≤ 0.10 | 10% |

### 4.3 Architecture Change

**Before:**
```
Meta-Brain → Trades
```

**After:**
```
Meta-Brain → Target Weights → Portfolio Optimizer → Trades
```

### 4.4 Optimization Methods

| Method | Use Case | Pros | Cons |
|--------|----------|------|------|
| Quadratic (CVXPY) | Full optimization | Exact solution | Slower |
| Sequential | Large universes | Fast | Approximate |
| Greedy | Emergency | Instant | Suboptimal |

### 4.5 Rebalance Frequency

| Horizon | Frequency | Trigger |
|---------|-----------|---------|
| Strategic | Weekly | Friday 16:00 |
| Tactical | Daily | End of cycle |
| Emergency | Intraday | Risk breach |

### 4.6 Persistence Schema

```sql
-- Portfolio Targets
CREATE TABLE IF NOT EXISTS portfolio_targets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    target_weight REAL NOT NULL,
    current_weight REAL,
    weight_change REAL,
    marginal_risk REAL,
    expected_return REAL,
    sharpe_contribution REAL,
    turnover_cost REAL,
    liquidation_priority INTEGER,
    UNIQUE(cycle_id, symbol)
);

-- Portfolio Constraints
CREATE TABLE IF NOT EXISTS portfolio_constraints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    constraint_type TEXT NOT NULL,
    limit_value REAL,
    current_value REAL,
    utilization_pct REAL,
    binding INTEGER,
    UNIQUE(date, constraint_type)
);

-- Optimization Results
CREATE TABLE IF NOT EXISTS optimization_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_id TEXT NOT NULL,
    optimizer_status TEXT,
    solve_time_ms REAL,
    objective_value REAL,
    constraints_violated TEXT,
    portfolio_return REAL,
    portfolio_volatility REAL,
    portfolio_sharpe REAL,
    UNIQUE(cycle_id)
);
```

### 4.7 Integration with Risk Engine

```python
def optimize_portfolio(
    signals: Dict[str, float],
    cov_matrix: np.ndarray,
    current_weights: Dict[str, float],
    risk_limits: Dict[str, float]
) -> Dict[str, float]:

    # 1. Apply risk limits to signals
    constrained_signals = apply_risk_limits(signals, risk_limits)

    # 2. Run optimization
    result = cvxpy.optimize(
        objective=constrained_signals.T @ w - lambda * w.T @ cov_matrix @ w,
        constraints=build_constraints(w, current_weights, risk_limits)
    )

    # 3. Validate with risk engine
    risk_check = risk_manager.check_pre_trade(result.weights, ...)

    if not risk_check.ok:
        # Scale to meet risk constraints
        result.weights = scale_weights(result.weights, risk_check.scale_factor)

    return result.weights
```

---

## PART 5: LIVE SLIPPAGE FEEDBACK LOOP

### 5.1 Objective

Ensure real execution costs continuously correct the system through closed-loop feedback.

### 5.2 Feedback Loop Architecture

```
Signal → Order → Fill → Slippage → Model Update
```

### 5.3 Slippage Model

$$\text{slippage} = f(size/ADV, volatility, spread)$$

**Model Form:**

$$Slippage_{bps} = \alpha_0 + \alpha_1 \cdot \frac{Q}{ADV} + \alpha_2 \cdot \sigma_{daily} + \alpha_3 \cdot spread_{bps} + \alpha_4 \cdot \frac{Q}{ADV} \cdot \sigma_{daily}$$

### 5.4 Persistence Schema

```sql
-- Execution Feedback
CREATE TABLE IF NOT EXISTS execution_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id TEXT NOT NULL UNIQUE,
    expected_price REAL,
    fill_price REAL,
    slippage_bps REAL,
    market_impact_bps REAL,
    commission REAL,
    total_cost_bps REAL,
    market_conditions_json TEXT,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    quantity REAL,
    adv REAL,
    adv_dollar REAL,
    volatility REAL,
    spread_bps REAL,
    time_of_day TEXT,
    regime TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Slippage Model Coefficients
CREATE TABLE IF NOT EXISTS slippage_model_coeffs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    estimation_date TEXT NOT NULL,
    alpha_0 REAL,
    alpha_1 REAL,
    alpha_2 REAL,
    alpha_3 REAL,
    alpha_4 REAL,
    r_squared REAL,
    n_observations INTEGER,
    status TEXT,
    UNIQUE(symbol, estimation_date)
);

-- Slippage Prediction Errors
CREATE TABLE IF NOT EXISTS slippage_prediction_errors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    symbol TEXT NOT NULL,
    predicted_slippage REAL,
    actual_slippage REAL,
    error_bps REAL,
    error_pct REAL,
    market_condition TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
```

### 5.5 Feedback Actions

```python
class SlippageFeedback:
    def __init__(self):
        self.slippage_tracker = SlippageTracker()
        self.kelly_adjuster = KellyAdjuster()
        self.execution_selector = ExecutionSelector()

    def process_execution(self, order: Order, fill: Fill):
        # 1. Calculate slippage
        slippage = self.calculate_slippage(order, fill)

        # 2. Update model
        self.update_slippage_model(order.symbol, slippage)

        # 3. Apply feedback actions
        if slippage > HIGH_SLIPPAGE_THRESHOLD:
            # Penalize symbol in portfolio construction
            self.adjust_symbol_score(order.symbol, penalty=0.8)

            # Adjust Kelly sizing
            self.kelly_adjuster.reduce_position(order.symbol, factor=0.7)

            # Update execution strategy selection
            self.execution_selector.set_preferred_strategy(
                order.symbol,
                Strategy.POV if slippage > 0.05 else Strategy.TWAP
            )

        # 4. Improve backtest realism
        self.update_backtest_slippage_estimate(order.symbol, slippage)
```

### 5.6 Regime Sensitivity

```python
class RegimeSensitiveSlippage:
    REGIME_ADJUSTMENTS = {
        'BULL_QUIET': {'multiplier': 1.0, 'confidence': 0.9},
        'BULL_VOLATILE': {'multiplier': 1.3, 'confidence': 0.7},
        'BEAR_QUIET': {'multiplier': 1.2, 'confidence': 0.8},
        'BEAR_CRISIS': {'multiplier': 2.0, 'confidence': 0.5}
    }

    def estimate_slippage(self, symbol: str, size_adv: float,
                          volatility: float, spread: float, regime: str) -> float:
        base_slippage = self.base_model.predict(size_adv, volatility, spread)

        adjustment = self.REGIME_ADJUSTMENTS.get(regime, {'multiplier': 1.0})
        confidence = adjustment['confidence']

        # Apply regime multiplier
        adjusted = base_slippage * adjustment['multiplier']

        # Adjust confidence interval based on regime
        uncertainty = adjusted * (1 - confidence)

        return {
            'estimate': adjusted,
            'lower': adjusted - uncertainty,
            'upper': adjusted + uncertainty,
            'regime': regime,
            'confidence': confidence
        }
```

---

## IMPLEMENTATION ROADMAP

### Phase 1: Foundation (Week 1-2)
- [ ] Extend database schema with new tables
- [ ] Implement factor attribution engine
- [ ] Create baseline persistence layer

### Phase 2: Online Learning (Week 3-4)
- [ ] Implement Exponentiated Gradient weight learner
- [ ] Add Thompson Sampling for regime switching
- [ ] Build monitoring and alerting

### Phase 3: RL Execution (Week 5-8)
- [ ] Set up training environment
- [ ] Train SAC agent on historical data
- [ ] Paper trading validation

### Phase 4: Portfolio Optimization (Week 9-10)
- [ ] Implement portfolio optimizer
- [ ] Integrate with risk engine
- [ ] Backtest validation

### Phase 5: Slippage Feedback (Week 11-12)
- [ ] Implement slippage tracking
- [ ] Build feedback loop mechanisms
- [ ] End-to-end testing

---

## SUCCESS CRITERIA

| Domain | Metric | Target |
|--------|--------|--------|
| Attribution | R² (factor model) | > 0.6 |
| Attribution | Top contributor accuracy | > 80% |
| Weight Learning | Hit rate improvement | +10% |
| RL Execution | Slippage reduction | -20% vs baseline |
| Portfolio Opt | Sharpe ratio improvement | +15% |
| Slippage Feedback | Prediction accuracy | R² > 0.5 |

---

## APPENDIX A: COMPLETE SCHEMA EXTENSION

```sql
-- ============================================================================
-- COMPLETE DATABASE SCHEMA EXTENSION FOR INSTITUTIONAL ARCHITECTURE
-- ============================================================================

-- Factor Attribution Tables
CREATE TABLE IF NOT EXISTS factor_returns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    factor_name TEXT NOT NULL,
    factor_return REAL NOT NULL,
    t_stat REAL,
    p_value REAL,
    n_observations INTEGER,
    r_squared REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(date, factor_name)
);

CREATE TABLE IF NOT EXISTS security_factor_exposures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    date TEXT NOT NULL,
    beta_market REAL,
    beta_momentum REAL,
    beta_value REAL,
    beta_volatility REAL,
    beta_size REAL,
    beta_liquidity REAL,
    alpha REAL,
    residual_std REAL,
    r_squared REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, date)
);

CREATE TABLE IF NOT EXISTS agent_pnl (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    agent_name TEXT NOT NULL,
    mu_hat REAL,
    sigma_hat REAL,
    confidence REAL,
    agent_weight REAL,
    realized_return REAL,
    pnl_contribution REAL,
    factor_adjusted_return REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(cycle_id, symbol, agent_name)
);

CREATE TABLE IF NOT EXISTS portfolio_attribution (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    cycle_id TEXT,
    total_portfolio_return REAL,
    market_contribution REAL,
    momentum_contribution REAL,
    value_contribution REAL,
    volatility_contribution REAL,
    size_contribution REAL,
    liquidity_contribution REAL,
    sector_contribution REAL,
    alpha_contribution REAL,
    idiosyncratic_contribution REAL,
    portfolio_volatility REAL,
    portfolio_beta REAL,
    tracking_error REAL,
    information_ratio REAL,
    top_contributor_symbol TEXT,
    top_contributor_pnl REAL,
    worst_detractor_symbol TEXT,
    worst_detractor_pnl REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(date)
);

-- Agent Weight Learning Tables
CREATE TABLE IF NOT EXISTS agent_weights (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    cycle_id TEXT NOT NULL,
    agent_name TEXT NOT NULL,
    weight REAL NOT NULL,
    rolling_sharpe REAL,
    recent_hit_rate REAL,
    regime TEXT,
    volatility_regime REAL,
    algorithm_used TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(cycle_id, agent_name)
);

CREATE TABLE IF NOT EXISTS agent_performance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_name TEXT NOT NULL,
    date TEXT NOT NULL,
    predicted_return REAL,
    actual_return REAL,
    hit INTEGER,
    pnl_contribution REAL,
    factor_adjusted_return REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(agent_name, date)
);

CREATE TABLE IF NOT EXISTS regime_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    regime TEXT NOT NULL,
    regime_prob REAL,
    spy_ma200_position REAL,
    volatility_percentile REAL,
    vix_level REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(date)
);

-- Execution RL Tables
CREATE TABLE IF NOT EXISTS execution_model_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    version_id TEXT NOT NULL UNIQUE,
    algorithm TEXT NOT NULL,
    model_path TEXT,
    trained_at TEXT,
    metrics_json TEXT,
    hyperparameters_json TEXT,
    status TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS execution_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id TEXT NOT NULL,
    cycle_id TEXT,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    quantity REAL NOT NULL,
    size_adv REAL,
    volatility REAL,
    spread_bps REAL,
    imbalance REAL,
    time_remaining REAL,
    action TEXT NOT NULL,
    confidence REAL,
    q_values_json TEXT,
    used_fallback BOOLEAN DEFAULT 0,
    fallback_reason TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(order_id)
);

CREATE TABLE IF NOT EXISTS execution_outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id TEXT NOT NULL,
    expected_price REAL,
    fill_price REAL,
    fill_time TEXT,
    slippage_bps REAL,
    market_impact_bps REAL,
    commission REAL,
    total_cost_bps REAL,
    adv REAL,
    adv_dollar REAL,
    vol_regime REAL,
    bid_ask_spread REAL,
    model_estimated_cost REAL,
    cost_error_bps REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(order_id)
);

-- Portfolio Optimization Tables
CREATE TABLE IF NOT EXISTS portfolio_targets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    target_weight REAL NOT NULL,
    current_weight REAL,
    weight_change REAL,
    marginal_risk REAL,
    expected_return REAL,
    sharpe_contribution REAL,
    turnover_cost REAL,
    liquidation_priority INTEGER,
    UNIQUE(cycle_id, symbol)
);

CREATE TABLE IF NOT EXISTS portfolio_constraints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    constraint_type TEXT NOT NULL,
    limit_value REAL,
    current_value REAL,
    utilization_pct REAL,
    binding INTEGER,
    UNIQUE(date, constraint_type)
);

CREATE TABLE IF NOT EXISTS optimization_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_id TEXT NOT NULL,
    optimizer_status TEXT,
    solve_time_ms REAL,
    objective_value REAL,
    constraints_violated TEXT,
    portfolio_return REAL,
    portfolio_volatility REAL,
    portfolio_sharpe REAL,
    UNIQUE(cycle_id)
);

-- Slippage Feedback Tables
CREATE TABLE IF NOT EXISTS execution_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id TEXT NOT NULL UNIQUE,
    expected_price REAL,
    fill_price REAL,
    slippage_bps REAL,
    market_impact_bps REAL,
    commission REAL,
    total_cost_bps REAL,
    market_conditions_json TEXT,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    quantity REAL,
    adv REAL,
    adv_dollar REAL,
    volatility REAL,
    spread_bps REAL,
    time_of_day TEXT,
    regime TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS slippage_model_coeffs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    estimation_date TEXT NOT NULL,
    alpha_0 REAL,
    alpha_1 REAL,
    alpha_2 REAL,
    alpha_3 REAL,
    alpha_4 REAL,
    r_squared REAL,
    n_observations INTEGER,
    status TEXT,
    UNIQUE(symbol, estimation_date)
);

CREATE TABLE IF NOT EXISTS slippage_prediction_errors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    symbol TEXT NOT NULL,
    predicted_slippage REAL,
    actual_slippage REAL,
    error_bps REAL,
    error_pct REAL,
    market_condition TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Create Indexes
CREATE INDEX IF NOT EXISTS idx_factor_returns_date ON factor_returns(date);
CREATE INDEX IF NOT EXISTS idx_sec_exposure_symbol ON security_factor_exposures(symbol);
CREATE INDEX IF NOT EXISTS idx_agent_pnl_cycle ON agent_pnl(cycle_id);
CREATE INDEX IF NOT EXISTS idx_agent_weights_date ON agent_weights(date);
CREATE INDEX IF NOT EXISTS idx_exec_decision_order ON execution_decisions(order_id);
CREATE INDEX IF NOT EXISTS idx_portfolio_targets_cycle ON portfolio_targets(cycle_id);
CREATE INDEX IF NOT EXISTS idx_execution_feedback_symbol ON execution_feedback(symbol);
```

---

## APPENDIX B: INTEGRATION CHECKLIST

### Database Integration
- [ ] Create new tables
- [ ] Add indexes
- [ ] Write migration scripts
- [ ] Test data integrity

### Meta-Brain Integration
- [ ] Update constructor to accept agent weights
- [ ] Modify aggregation to use learned weights
- [ ] Add fallback to static weights

### Cycle Orchestrator Integration
- [ ] Add attribution computation step
- [ ] Add weight learning step
- [ ] Add portfolio optimization step
- [ ] Add execution strategy selection

### Risk Engine Integration
- [ ] Share factor exposures
- [ ] Coordinate constraint handling
- [ ] Integrate CVaR calculations

### Monitoring Integration
- [ ] Create attribution dashboard
- [ ] Create weight trajectory plots
- [ ] Create execution cost analysis
- [ ] Create portfolio optimization metrics

---

**Document End**

*This specification provides the complete architectural blueprint for the Mini Quant Fund institutional upgrade. All components are designed for production deployment with proper fail-safes, monitoring, and documentation.*

