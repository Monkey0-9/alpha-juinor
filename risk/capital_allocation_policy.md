# Capital Allocation Policy (Elite Platform)

**Effective Date**: 2026-02-01
**Owner**: Chief Risk Officer (CRO)
**Version**: 1.0 (Production)
**Governance**: Risk Committee Approval Required for changes.

## 1. Objective

To deploy capital systematically to the High-Frequency Trading (HFT) strategies while protecting the firm's long-term viability. This policy governs the "Phase 4" scaling process, enforcing a mathematical approach to sizing and strict "hard decks" for risk management.

## 2. Capital Pools

| Pool Name | Allocation (% of AUM) | Purpose | Restrictions |
|-----------|-----------------------|---------|--------------|
| **Incubator (Seed)** | **2%** | Initial Live Testing (Phase 4.1). MVS strategies only. | Max 100k exposure per symbol. |
| **Growth** | **10%** | Strategies with >3 months profitable track record. | Max 500k exposure per symbol. |
| **Core** | **40%** | Proven, reliable alpha streams (Sharpe > 2.0, >1yr). | Full algorithm sizing allowed. |
| **Reserve** | **48%** | Unallocated cash for opportunities or drawdown buffer. | Treasury bills / Money Market only. |

## 3. Allocation & Scaling Rules

*No strategy receives capital without passing the **R2P Pipeline** gates.*

### 3.1 Graduation Criteria (Incubator -> Growth) using Bayesian Confidence

- **Duration**: Minimum 30 trading days live.
- **Performance**: Positive PnL net of all costs (including coloc fees).
- **Statistical Significance**: T-stat > 2.0 on daily returns.
- **Stability**: Zero "Kill Switch" triggers caused by logic errors.
- **Risk**: Max Drawdown < 5%.

### 3.2 Graduation Criteria (Growth -> Core)

- **Duration**: Minimum 6 months live.
- **Scalability**: Demonstrated ability to handle 5x current volume without alpha decay.
- **Correlation**: Correlation to existing Core strategies < 0.5.
- **Stress Test**: Survival in "Chaos Monkey" simulations with 100% pass rate.

### 3.3 Position Sizing: Modified Kelly Criterion

For **Growth** and **Core** strategies, position size $f^*$ is calculated as:
$$ f^* = \text{Half-Kelly} = 0.5 \times \frac{\mu}{\sigma^2} $$
Where:

- $\mu$: Expected excess return (emailed weekly).
- $\sigma^2$: Variance of returns.
- **Constraint**: $\sum |positions| \le \text{Gross Leverage Limit} (2.0x)$

## 4. Drawdown & Stop-Loss Protocols (The "Hard Deck")

| Threshold | Automated Action | Review Requirement | Recovery Procedure |
|-----------|------------------|--------------------|--------------------|
| **-3% Daily** | **Pause Trading** | Risk Manager review required to resume. | Check for data anomalies. Verify execution latency. |
| **-10% Peak-to-Trough** | **Reduce Allocation 50%** | Full code & logic audit required. | Strategy demoted to "Incubator". Must re-qualify. |
| **-20% Peak-to-Trough** | **Decommission Strategy** | Strategy moved to "Retired" state. Post-mortem required. | Codebase frozen. "Lesson Learned" doc required. |

## 5. Governance & Review Process

- **Real-time**: `risk_agent.py` monitors triggers continuously.
- **Daily**: Automated report on utilization and limit breaches (Slack/Email).
- **Monthly**: Allocation Committee meeting to adjust Pool sizes.
- **Emergency**: Any "Kill Switch" event triggers immediate Allocation Committee review.

## 6. Recovery Mode

If the firm-wide NAV drops > 15%:

1. All "Incubator" and "Growth" strategies are **Disabled**.
2. "Core" strategies reduced to 50% sizing.
3. "Reserve" capital locked.
4. "War Room" convened to determine systemic cause.
