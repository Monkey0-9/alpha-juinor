# Research: RL Controller V2 - Advanced State Representation

## Objective

Enhance the `RLMetaController` state space to capture market fragility and cross-asset stress, enabling smarter defensive rotation.

## Current State (V1)

- **Inputs**: [VIX, MarketTrend, PortfolioVol, CashRatio, DayOfWeek]
- **Limitation**: Reacts to volatility *after* it appears. Lacks predictive macro inputs.

## V2 Proposed Features

1. **Yield Curve Slope**: `10Y - 2Y` Treasury Yield. (Leading indicator of recession/liquidity).
2. **Sector Rotation Momentum**: Relative strength of `XLF` (Financials) vs `XLU` (Utilities). Risk-on vs. Risk-off.
3. **Options Skew**: `Put/Call Ratio` or Skew Index. (Market fear gauge).
4. **Liquidity Proxy**: Bid-Ask spread average of the S&P 500 universe.

## Implementation Plan

1. **Data Source**: Add `FredData` or `YieldCollector` to `data/collectors/`.
2. **State Vector**: Expand dimension from 5 to 9.
3. **Training**: Retrain PPO agent on 10 years of this richer history.

## Hypothesis

V2 will switch to "Defensive" mode *before* a crash by detecting yield curve inversion and rising options skew, reducing Max Drawdown by an estimated further 5-8%.
