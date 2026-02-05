# scripts/preflight_env.py
import sys

REQUIRED = [
    "yaml",
    "dotenv",
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "statsmodels",
]

missing = []
for m in REQUIRED:
    try:
        __import__(m)
    except ImportError:
        missing.append(m)

if missing:
    print(f"ERROR: Missing Python packages: {missing}")
    print("Run: python -m pip install -r requirements.txt")
    sys.exit(1)

print("[OK] environment dependencies satisfied")
