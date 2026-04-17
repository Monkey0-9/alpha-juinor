import importlib
import sys

dependencies = [
    "hvac", "alpaca_trade_api", "ccxt", "redis", "influxdb_client",
    "sqlalchemy", "alembic", "pydantic", "fastapi", "uvicorn",
    "transformers", "torch", "pennylane", "qiskit", "statsmodels",
    "sklearn", "pandas_market_calendars", "loguru", "prometheus_client"
]

missing = []
for dep in dependencies:
    try:
        importlib.import_module(dep.split('-')[0].replace('-', '_'))
    except ImportError:
        missing.append(dep)

if missing:
    print("--- MISSING START ---")
    for d in missing:
        print(d)
    print("--- MISSING END ---")
    sys.exit(1)
else:
    print("ALL_DEPS_FOUND")
    sys.exit(0)
