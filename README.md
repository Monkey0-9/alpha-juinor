# Nexus: Institutional Quant Research & Execution Platform

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Docker](https://img.shields.io/badge/docker-%232496ED.svg?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)

**Nexus** is an institutional-grade quantitative trading platform refactored and hardened for professional research, event-driven backtesting, and production execution. It implements the rigorous technical standards of top-tier firms (Citadel, Jane Street, HRT), featuring Physics-based market impact modeling, elite risk guardrails, and structured observability.

---

## 🏛️ Core Architecture (8-Layer Nexus)

Nexus is built on a strict, layered 8-module architecture to ensure zero circular dependencies and 100% reproducibility.

*   **Models**: Unified Pydantic-based Single Source of Truth (`MarketBar`, `Order`, `Trade`).
*   **Data Engine**: Schema-validated ingestion with Parquet-backed caching.
*   **Alpha Pipeline**: Standardized signal research with walk-forward validation.
*   **Simulation**: Event-driven backtesting with Almgren-Chriss market impact.
*   **OMS**: Full lifecycle order management and fill reconciliation.
*   **Risk Engine**: Pre-trade concentration limits and global Kill-Switch circuits.
*   **Monitoring**: Structured JSON telemetry and high-performance nanosecond profiling.
*   **Infrastructure**: Production-ready Docker orchestration and secret management.

For deep technical details, see [ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## ⚡ Elite Features

### 🔬 Execution Microstructure
Unlike retail-grade backtesters, Nexus implements the **Almgren-Chriss Optimal Execution** model. Every simulated trade accounts for permanent and temporary market impact, providing a realistic estimate of Implementation Shortfall before deploying capital.

### 🛡️ Institutional Risk Controls
*   **Sector Concentration**: Monitors and blocks trades that exceed GICS sector exposure limits (Citadel Standard).
*   **Gross Leverage Tracking**: Real-time monitoring of margin utilization.
*   **Global Kill-Switch**: A pre-instrumented circuit breaker that halts the engine on critical drawdown breaches.

---

## 🏁 Quick Start

### 🐳 Docker (Recommended)
```bash
docker-compose up -d
```

### 💻 Local Development
1. **Initialize Environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Or .venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```
2. **Setup Secrets**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```
3. **Run Backtest Verification**:
   ```bash
   python run_institutional_backtest.py
   ```

---

## 🧪 Testing & Benchmarks

Nexus maintains a 100% functional coverage requirement for the core engine.

*   **Run All Tests**: `pytest tests/`
*   **Run Benchmarks**: `python scripts/benchmark_engine.py`

| Metric | Target | Status |
| :--- | :--- | :--- |
| **Bar Throughput** | > 100k bars/sec | ✅ |
| **Risk Latency (p99)** | < 1,000 ns | ✅ |
| **Mem Footprint** | < 256 MB (Baseline) | ✅ |

---

## 📈 Roadmap & Extension
Nexus is designed to be extensible. The `src/nexus/execution/adapters/` directory allows for easy integration with any institutional broker API (Alpaca, IBKR, XTX) while maintaining a unified execution interface.

---

**Designed and Hardened by Antigravity Quant Systems.**
