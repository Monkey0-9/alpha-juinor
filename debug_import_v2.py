import sys
import os
sys.path.insert(0, os.getcwd())
print(f"Sys Path: {sys.path}")
try:
    import portfolio
    print(f"Portfolio: {portfolio}")
    print(f"Portfolio file: {getattr(portfolio, '__file__', 'None')}")
    print(f"Portfolio path: {getattr(portfolio, '__path__', 'None')}")
except Exception as e:
    print(f"Import failed: {e}")
