
# Institutional Quant Fund API Summary

## Core Components

### 1. Execution Engine (`execution/`)
- **`ExecutionEngine`**: Orchestrates trade decisions and routes them to brokers.
- **`UltimateExecutor`**: Handles precise order execution, slippage prevention, and split-order management.
- **`OMS` (Order Management System)**: Tracks order lifecycle, persists trade history, and enforces pre-trade risk checks.

### 2. Brokers (`brokers/`)
- **`AlpacaExecutionHandler`**: Real/Paper trading integration with Alpaca Markets REST API.
- **`MockBroker`**: Simulated broker for paper trading and backtesting.
- **`CCXTBroker`**: Multi-exchange support for crypto assets.

### 3. Intelligence & Signals (`intelligence/`, `agents/`)
- **`MetaBrain`**: High-level decision engine aggregating multiple alpha signals.
- **`MomentumAgent`, `MeanReversionAgent`, `VolatilityAgent`**: Specialized alpha generation agents.
- **`LiveDecisionLoop`**: Per-second decision tick for real-time operation.

### 4. Risk Management (`risk/`)
- **`PreTradeRiskManager`**: Enforces notional limits, ADV percentage caps, and fat-finger checks.
- **`MarketImpactModel`**: Estimates slippage and temporary/permanent price impact.
- **`GlobalKillSwitch`**: Emergency halt mechanism.

### 5. Infrastructure (`infrastructure/`)
- **`FreeCloudDeployment`**: (Simulated) Cloud-agnostic deployment logic.
- **`deployment.yaml`**: Kubernetes production manifest for core services.
- **`docker-compose.yml`**: Local stack including TimescaleDB, Prometheus, and Grafana.

## Trading Modes
- **Backtest**: Historical simulation using CSV/Database data.
- **Paper**: Live market data with simulated execution.
- **Live**: Real capital execution (requires `TRADING_MODE=live`).
