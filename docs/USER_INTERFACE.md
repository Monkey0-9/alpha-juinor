# Mini-Quant Fund: User Interface Documentation

This document outlines the various UI components available for interacting with the Mini-Quant Fund trading system.

## 1. Mobile Terminal (React Native)
**Path:** `src/mini_quant_fund/ui/mobile/App.js`

A React Native scaffold designed for institutional traders on the go.
- **Real-time Monitoring:** View P&L, balance, and active positions.
- **Execution:** Quick buy/sell buttons for rapid response.
- **Alerts:** Displays real-time strategy alerts (e.g., gamma squeeze detections).

## 2. Social Copy Trading
**Path:** `src/mini_quant_fund/ui/social/copy_trading.py`

Logic for managing a "Social Fund" where followers can mirror the actions of a lead trader.
- **Proportional Sizing:** Automatically adjusts trade sizes based on follower capital.
- **Risk Multipliers:** Allows followers to scale risk relative to the lead trader.
- **Active Management:** Track follower counts and historical mirrored trades.

## 3. Voice NLP Engine
**Path:** `src/mini_quant_fund/ui/voice/nlp_engine.py`

A speech-to-intent engine for hands-free trading operations.
- **Natural Language Parsing:** Supports commands like "Buy 100 shares of Apple".
- **Ticker Resolution:** Resolves common company names (e.g., "Microsoft") to tickers ("MSFT").
- **Portfolio Queries:** Responds to status requests like "How am I doing?".

## 4. Desktop Dashboards (Existing)
- **Streamlit App:** `src/mini_quant_fund/ui/streamlit_app.py` - High-level analytics and visualization.
- **Terminal Dashboard:** `src/mini_quant_fund/ui/terminal_dashboard.py` - CLI-based real-time monitoring.
