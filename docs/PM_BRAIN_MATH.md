# PM Brain Mathematical Specification

> **Status**: FROZEN
> **Enforcement**: strict (code must implement exactly)

## 1. Objective Function

Maximize the risk-adjusted utility subject to institutional constraints.

$$
\max_{w} \left( \mu^\top w - \lambda w^\top \Sigma w - \gamma \text{CVaR}_\alpha(w) - \kappa \|w - w_{\text{prev}}\|_1 - \eta \text{Impact}(w) \right)
$$

### Terms

| Symbol | Description | Implementation |
|--------|-------------|----------------|
| $\mu$ | Expected Returns Vector | `mu = alpha_mu * confidence * regime_compatibility` |
| $\Sigma$ | Covariance Matrix | Ledoit-Wolf shrinkage on 252-day window |
| $\lambda$ | Risk Aversion | Tunable scalar (default: 1.0) |
| $\gamma$ | CVaR Penalty | Tunable scalar (default: 0.5) |
| $\kappa$ | Transaction Cost | Linear cost factor (e.g., spread + comms) |
| $\eta$ | Market Impact | Quadratic impact factor |
| $\alpha$ | CVaR Confidence | 0.95 (95%) |

### Market Impact Model

$$
\text{Impact}(w) = \sum_i \left( 0.1 \cdot \sigma_i \cdot \frac{|\Delta w_i| \cdot \text{NAV}}{\text{ADV}_i} \right)^{3/2}
$$
*Note: In QP solver, approximated as quadratic term.*

---

## 2. Constraints

1. **Budget Constraint**:
    $$ \sum w_i = 1.0 $$

2. **Gross Leverage**:
    $$ \sum |w_i| \le L_{\text{gross}} $$
    *(Default $L=1.0$ for Long Only, $L=1.5$ for L/S)*

3. **Position Limits**:
    $$ 0 \le w_i \le w_{\text{max}, i} $$
    *(Default $w_{\text{max}} = 0.20$)*

4. **Sector Exposure**:
    $$ \sum_{j \in \text{Sector}_k} w_j \le S_k $$
    *(Default $S_k = 0.40$)*

5. **CVaR Limit**:
    $$ \text{CVaR}_{0.95}(w) \le \text{Limit}_{\text{risk}} $$

---

## 3. Regime Integration

Every asset $i$ has a regime capability score $r_i \in [0, 1]$.

$$
\hat{\mu}_i = \mu_i \cdot \text{confidence}_i \cdot r_i(t)
$$

Where Regime State $p(t)$ evolves as:
$$ p(t+1) = U(t)p(t) $$
$$ r_i(t) = \sum_{k=1}^K p_k(t) \cdot C_{i,k} $$

---

## 4. Entanglement Gating

Before optimization, update $w_{\text{max}, i}$:

If $\text{Analysis}_{\text{entanglement}} > \text{Threshold}$:
$$ w_{\text{max}, i} \leftarrow w_{\text{max}, i} \cdot (1 - \beta \cdot \text{Entanglement}_i) $$

---

## 5. Path Integral Stress Test

Post-optimization, evaluate allocation $w^*$:

$$
\text{StressedCVaR} = \frac{1}{\alpha} \sum_{j: L(\tau_j) \ge \text{VaR}} \tilde{w}_j L(\tau_j)
$$

Where $\tilde{w}_j$ are importance sampling weights from macro-shock biased paths.

If $\text{StressedCVaR} > \text{HardLimit}$:

- Re-run optimization with tighter $\gamma$ or reduced $L_{\text{gross}}$.
