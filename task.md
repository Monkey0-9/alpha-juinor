# Task: Integrate Free Market Data Sources

Measurements of success:
- [x] All requested providers implemented in `data/providers/`.
- [x] Fallback mechanism works (Yahoo -> Stooq -> etc.).
- [x] Caching (Parquet) implemented for all downloads.
- [x] Integration test passes fetching data from each source.

## Plan
- [x] **Planning**
    - [x] Create implementation plan.
    - [x] Update `task.md`.
- [x] **Macro & Fundamental Providers**
    - [x] Implement `data/providers/fred.py` (Regime/Risk).
    - [x] Implement `data/providers/alpha_vantage.py` (Macro/Backup).
- [x] **Crypto Providers**
    - [x] Implement `data/providers/binance.py` (High Freq/Vol).
    - [x] Implement `data/providers/coingecko.py` (Fundamentals).
- [x] **Equity Backups**
    - [x] Implement `data/providers/stooq.py` (No API key backup).
    - [x] Implement `data/providers/polygon.py` (EOD Validation).
- [x] **Core Integration**
    - [x] Create `data/collectors/data_router.py` to handle `get_data(ticker)` with fallback logic.
    - [x] Update `main.py` or `universe_manager.py` to use the router.
- [x] **Verification**
    - [x] Create `tests/test_data_providers.py`.
# Task: Multi-Source Logic Integration
- [x] **Data Router Upgrade**
    - [x] Implement `get_macro_context()` in `DataRouter` (aggregates FRED VIX, Yields).
    - [x] Add `validate_price_integrity()` (Implemented as `cross_check_quote`).
- [x] **Strategy Upgrade**
    - [x] Update `InstitutionalStrategy.generate_signals` to accept `macro_data`.
    - [x] Implement `MacroRiskGate` (e.g. VIX > 30 -> Block Longs).
- [x] **Pipeline Integraiton**
    - [x] Update `main.py` loop to fetch macro data and pass to strategy.
# Task: Institutional System Hardening
- [x] **1. Live-Trading Readiness Checklist**
    - [x] Implement `ops/checklists.py` (Pre-flight checks).
    - [x] Integrate into `main.py` startup.
- [x] **2. Capital Preservation Mode**
    - [x] Update `risk/engine.py` with `check_capital_preservation`.
    - [x] Enforce reduced leverage during drawdown.
- [x] **3. Crash Survival Switch**
    - [x] Implement `CrashSwitch` in `engine/live_engine.py`.
    - [x] Add `SAFE_MODE` logic to cancel/flatten.
- [x] **4. Auto Universe Rotation**
    - [x] Implement `rotate_universe` in `data/universe_manager.py`.
    - [x] Filter non-tradables and cap size.
- [x] **5. Final Polish**
    - [x] Remove emojis from logs.
    - [x] Verify 24/7 loop stability.

# Task: Advanced Institutional Mathematics (Steel-Hardening)
- [x] **1. Tail-Risk Protection (EVT)**
    - [x] Create `risk/tail_risk.py` implementing Peaks Over Threshold (POT) & GPD.
    - [x] Add `compute_cvar` logic.
    - [x] Integrate into `RiskManager.check_portfolio`.
- [x] **2. Markov Regime Switching**
    - [x] Create `regime/markov.py` with transition matrix logic.
    - [x] Integrate `P(Panic)` check into `RiskManager`.
- [x] **3. Dynamic Kelly Sizing**
    - [x] Implement Volatility-Penalized Kelly formula in `risk/sizing.py`.
    - [x] Replace fixed sizing in `Allocator`.
- [x] **4. Trade Quality Filter (EV)**
    - [x] Implement `ExpectedValue = P(Win)*Gain - P(Loss)*Loss`.
    - [x] Add gate to `InstitutionalStrategy`.
- [x] **5. Multi-Horizon Stability**
    - [x] Implement `MultiTimeframeConsensus` check (Weekly/Daily/Hourly).
- [x] **6. Loss Shape Control**
    - [x] Implement Volatility-Based Stops (ATR-adaptive).
- [x] **7. Drawdown Adaptation**
    - [x] Implement `Risk = Risk0 * exp(-lambda * DD)`.
- [x] **8. Liquidity Impact**
    - [x] Implement Impact Model `eta * (Order/ADV)`.
- [x] **9. Strategy Decay**
    - [x] Implement `StabilityRatio = SharpeOOS / SharpeIS`.
