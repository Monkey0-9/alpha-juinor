# Institutional System Walkthrough

This document serves as the final handover for the **Institutional-Grade Quantitative Trading System**.

## 1. System Status
- **Architecture**: Real-Time Event Driven (Listener + Engine).
- **Stability**: Hardened against crashes, API failures, and market shocks.
- **Risk**: Capital preservation, volatility scaling, and circuit breakers active.
- **Data**: Centralized Router (Yahoo/Alpaca/FRED) with fallback and UTC normalization.

## 2. Data Integrity & Sources (Verified)
To ensure **Institutional-Grade Reliability** and prevent losses from bad data:
*   **Primary Equity Source**: **Yahoo Finance** ("Yoho") - The most widely used retail data source.
*   **Primary Crypto Source**: **Binance** - High-fidelity real-time execution data.
*   **Macro Source**: **FRED (Federal Reserve)** - Validated government economic data.
*   **Fallback Safety**: If Yahoo fails, we revert to **Stooq** or **AlphaVantage** (Top-tier backups) to prevent "blind spots". We NEVER use unverified sources.

## 3. Key Components
1.  **Main Loop (`main.py`)**:
    - Runs 24/7.
    - Adaptive polling (Listener) instead of sleep.
    - "Fast Path" for Flash Crashes.
    - "Slow Path" for Rebalancing.
2.  **Market Listener (`engine/market_listener.py`)**:
    - Watches price/volume in real-time.
    - Triggers events transparently.
3.  **Risk Manager (`risk/engine.py`)**:
    - The "Gatekeeper" ensuring no bad trades pass.
4.  **Data Router (`data/collectors/data_router.py`)**:
    - The "Supply Chain" ensuring clean, normalized data.

## 3. Operations Guide
### Start the System
```bash
python main.py
```
*That's it. It runs forever.*

### Monitor Performance
- **Logs**: streamed to console and files.
- **Telegram**: Real-time alerts (if configured).
- **Daily Reports**: Generated at 17:30.

### Emergency Procedures
- **Stop**: `Ctrl+C` (Clean shutdown).
- **Crash**: System auto-restarts after 30s cooldown (unless in `SAFE_MODE`).
- **Safe Mode**: Requires manual intervention to reset `crash_mode` or restart process.

## 4. Verification
- **Tests**: `python -m pytest tests/test_institutional_full.py` (ALL GREEN).
- **Architecture**: See `INSTITUTIONAL_ARCHITECTURE.md`.

---
**Handover Status**: RELEASED.
