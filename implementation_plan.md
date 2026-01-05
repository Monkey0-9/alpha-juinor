# Implementation Plan - Advanced Institutional Mathematics

## Goal
Transform the system from "Functional" to "Steel-Hard" using 9 concrete institutional theorems.

## 1. Risk Engine Upgrade (`risk/`)
- **`risk/tail_risk.py`**:
    - **Math**: Extreme Value Theory (EVT) using Generalized Pareto Distribution (GPD) on tail losses.
    - **Logic**: Calculate 1-day 99% CVaR. If `CVaR > Threshold`, freeze trading.
- **`risk/sizing.py`**:
    - **Math**: `f* = (mu - r)/sigma^2 * (1 - gamma*vol_penalty)`.
    - **Logic**: Continuous sizing adjustment based on realized vol.

## 2. Regime Engine (`regime/`)
- **`regime/markov.py`**:
    - **Math**: 3-State Markov Chain (Bull, Bear, Panic).
    - **Input**: Rolling returns, VIX.
    - **Output**: `P(Next=Panic)`. If > 30%, hard stop.

## 3. Strategy Enhancements (`strategies/`)
- **Expected Value Gate**:
    - Compute `WinRate * AvgWin - LossRate * AvgLoss`.
    - Require `EV > 0` and `EV > TransactionCosts`.
- **Multi-Horizon Consensus**:
    - Signal = `w_d * Sig_D + w_h * Sig_H`.

## 4. Execution & Hardening
- **Liquidity Guard**:
    - Reject orders where `Size > 1% ADV`.
- **Drawdown Brake**:
    - `TargetRisk *= exp(-5 * CurrentDrawdown)`.

## Execution Order
1.  **Tail Risk (EVT/CVaR)** - Critical Safety.
2.  **Markov Regime** - Critical filter.
3.  **Kelly Sizing** - Returns optimization.
4.  **Strategy Filters** - Precision.
