# Decision Authority Boundaries

> **Rule**: No component may both decide AND execute. Violations = Incident.

## Authority Matrix

| Component | Allowed Actions | Forbidden Actions |
|-----------|-----------------|-------------------|
| **Alpha Agents** | μ, σ, CVaR estimates | weights, trades, execution |
| **PM Brain** | Portfolio weights, allocation | Direct execution, order routing |
| **Risk Enforcer** | Block trades, scale positions | Trade direction, alpha signals |
| **Execution Layer** | Order routing, venue selection | Position sizing, trade decisions |
| **LLM/AI Advisor** | Explanation, analysis, research | BUY/SELL decisions, any trades |

---

## Boundary Enforcement

### 1. Alpha Agents → PM Brain

Alpha agents emit **signals only**:

```python
@dataclass
class AlphaOutput:
    symbol: str
    mu: float           # Expected return
    sigma: float        # Volatility estimate
    confidence: float   # Signal confidence
    cvar_95: float      # Conditional VaR
    # ❌ NO: weight, quantity, trade_direction
```

### 2. PM Brain → Risk Layer

PM Brain emits **weights only**:

```python
@dataclass
class AllocationResult:
    weights: Dict[str, float]
    rejected_assets: List[RejectedAsset]  # With "why not" reasons
    # ❌ NO: orders, execution_venue, timing
```

### 3. Risk Layer → Execution

Risk layer emits **approved plan only**:

```python
@dataclass
class ApprovedPlan:
    weights: Dict[str, float]   # Potentially scaled
    blocked: List[str]          # Blocked symbols with reasons
    risk_multiplier: float      # Portfolio-level scaling
    # ❌ NO: route_via, broker_selection
```

### 4. Execution Layer → Broker

Execution emits **orders only**:

```python
@dataclass
class OrderInstruction:
    symbol: str
    side: str           # BUY or SELL (from approved plan)
    quantity: float     # Computed from weight + NAV
    order_type: str
    venue: str          # Routing decision
    # ❌ NO: override_risk, ignore_blocks
```

---

## LLM/AI Restrictions

> [!CAUTION]
> LLMs are **READ-ONLY** advisors. They may NEVER:
>
> - Emit BUY/SELL signals
> - Modify risk parameters
> - Override governance decisions
> - Access execution APIs

**Allowed LLM Actions:**

- Explain past decisions
- Analyze market conditions
- Generate research summaries
- Propose hypothetical scenarios (clearly marked)

---

## CI Enforcement

### Import Guards

```python
# execution/order_router.py
from alpha_families import *  # ❌ FORBIDDEN - CI will fail
from portfolio import *        # ❌ FORBIDDEN - CI will fail
```

### Interface Type Checks

```python
# Execution layer must NOT accept alpha types
def route_order(order: OrderInstruction) -> OrderResult:
    assert not isinstance(order, AlphaOutput)  # Hard fail
    assert not isinstance(order, AllocationResult)  # Hard fail
```

### CI Rule

```yaml
# .github/workflows/governance.yml
- name: Check Decision Boundaries
  run: |
    python scripts/check_decision_boundaries.py
    # Fails if execution imports alpha
    # Fails if alpha imports execution
```

---

## Violation Handling

1. **Detection**: Automated via CI and runtime guards
2. **Logging**: All boundary violations logged to audit DB
3. **Response**:
   - Development: Build fails
   - Production: Trading halts
4. **Resolution**: Requires incident postmortem and explicit fix

---

## Diagram

```
┌─────────────────┐
│  Alpha Agents   │
│  (μ, σ, CVaR)   │
└────────┬────────┘
         │ AlphaOutput
         ▼
┌─────────────────┐
│    PM Brain     │
│   (weights)     │──────────────────┐
└────────┬────────┘                  │
         │ AllocationResult          │ RejectedAssets
         ▼                           │ (why not)
┌─────────────────┐                  │
│  Risk Enforcer  │                  │
│ (block/scale)   │                  │
└────────┬────────┘                  │
         │ ApprovedPlan              ▼
         ▼                    ┌─────────────┐
┌─────────────────┐           │ Audit Trail │
│   Execution     │           └─────────────┘
│ (routing only)  │
└────────┬────────┘
         │ OrderInstruction
         ▼
┌─────────────────┐
│     Broker      │
└─────────────────┘
```

---

## Ownership

| Boundary | Owner | Reviewer |
|----------|-------|----------|
| Alpha → PM | Quant Lead | Risk Officer |
| PM → Risk | Portfolio Manager | Risk Officer |
| Risk → Execution | Risk Officer | Systems Engineer |
| Execution → Broker | Systems Engineer | Compliance |
