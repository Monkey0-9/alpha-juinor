# Institutional System Architecture

This document maps the implemented system components to the 10-Point Institutional Architecture Specification.

## 1. Market Listener (Ultra-Low Latency Sentinel)
- **Status**: ✅ Implemented
- **Component**: `engine/market_listener.py`
- **Logic**: 
    - Real-time adaptive polling (10s Crypto, 60s Equities).
    - Detects `FLASH_CRASH` (>3% drop) and `VOLATILITY_SPIKE` (>2%).
    - Zero logic, zero ML, pure speed.

## 2. Event Engine (Decision Brain)
- **Status**: ✅ Implemented
- **Component**: `main.py` (Real-Time Event Loop)
- **Logic**:
    - Wakes `LiveEngine` *only* on Events or Scheduled Heartbeats.
    - Sleeps in 1s intervals to conserve resources.

## 3. Dual-Speed Intelligence
- **Status**: ✅ Implemented
- **Fast Path**: `main.py` detects `FLASH_CRASH` -> Routes directly to `LiveEngine.enter_safe_mode()` (Bypasses strategy).
- **Slow Path**: `main.py` runs `LiveEngine.run_once()` for standard rebalancing and volatility adjustments.

## 4. Strategy Layer (Multi-Brain)
- **Status**: ✅ Implemented
- **Component**: `strategies/institutional_strategy.py`
- **Logic**: Uses `MacroRiskGate` to switch regimes (Risk On/Off) based on VIX/Yields.

## 5. Capital Preservation Engine
- **Status**: ✅ Implemented
- **Component**: `risk/engine.py` (`RiskManager`)
- **Logic**: `check_capital_preservation` enforces leverage reduction during drawdowns or high VIX.

## 6. Auto-Universe Rotation
- **Status**: ✅ Implemented
- **Component**: `data/universe_manager.py`
- **Logic**: `rotate_universe` filters illiquid/blacklisted assets dynamically.

## 7. Execution Intelligence
- **Status**: ✅ Implemented
- **Component**: `backtest/execution.py` (`RealisticExecutionHandler`)
- **Logic**: Validates volume participation and slippage constraints before submittal.

## 8. Monitoring & Self-Awareness
- **Status**: ✅ Implemented
- **Component**: `monitoring/metrics.py`, `monitoring/alerts.py`, `LiveEngine.broadcast_summary`
- **Logic**: Real-time Telegram alerts, risk tier diagnosis, and daily performance reports.

## 9. Crash-Proof Engineering
- **Status**: ✅ Implemented
- **Component**: `engine/live_engine.py` (`CrashSwitch`), `main.py` (Hardened Loop)
- **Logic**: 
    - `enter_safe_mode`: Cancels all orders, halts trading.
    - `main.py`: Catches all exceptions, 30s cooldown, infinite retry.
    - Optional API Keys: System runs gracefully without FRED/AlphaVantage.

## 10. Performance & Stability
- **Status**: ✅ Implemented
- **Component**: `data/collectors/data_router.py`
- **Logic**: 
    - Parquet Caching ("Cache Everything").
    - Timezone Normalization (UTC Enforcement).
    - Lightweight `get_latest_price` for listener.

---
**Verdict**: The system now strictly adheres to the Institutional Philosophy: "Survival through all regimes."
