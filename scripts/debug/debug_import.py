import sys
import os
sys.path.insert(0, os.getcwd())
print(f"CWD: {os.getcwd()}")
print(f"Path: {sys.path[:3]}")

try:
    import portfolio
    print(f"Imported portfolio: {portfolio}")
    print(f"Portfolio file: {getattr(portfolio, '__file__', 'No file')}")
    print(f"Portfolio path: {getattr(portfolio, '__path__', 'No path')}")

    import portfolio.opportunity_cost
    print("Success importing portfolio.opportunity_cost")

except Exception as e:
    print(f"Error: {e}")

try:
    import backtest.portfolio
    print(f"Imported backtest.portfolio: {backtest.portfolio}")
except Exception as e:
    print(f"Import backtest error: {e}")
