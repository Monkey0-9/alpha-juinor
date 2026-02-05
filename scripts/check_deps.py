import importlib
import sys
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("CheckDeps")

REQUIRED_MODULES = [
    "networkx",
    "optuna",
    "xgboost",
    "lightgbm",
    "pandas",
    "numpy",
    "statsmodels",
    "transformers",
    # Core internal modules (verify path is correct)
    "strategies.base",
    "core.global_session_tracker",
    "alpha.network_alpha"
]

def check_dependencies():
    logger.info("Starting Pre-Flight Dependency Check...")
    missing = []

    for module in REQUIRED_MODULES:
        try:
            importlib.import_module(module)
            logger.info(f"✅ Found: {module}")
        except ImportError as e:
            logger.error(f"❌ MISSING: {module} ({e})")
            missing.append(module)
        except Exception as e:
             logger.error(f"❌ ERROR loading {module}: {e}")
             missing.append(module)

    if missing:
        logger.critical(f"FATAL: {len(missing)} dependencies missing. Aborting launch.")
        logger.critical(f"Run: pip install {' '.join([m for m in missing if '.' not in m])}")
        sys.exit(1)
    else:
        logger.info("All systems GO. Dependency check passed.")
        sys.exit(0)

if __name__ == "__main__":
    check_dependencies()
