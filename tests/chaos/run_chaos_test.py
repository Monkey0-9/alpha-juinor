"""
Chaos Monkey Test Runner.
Demonstrates the Chaos Monkey injecting faults into a simulated process.
"""

import time
import logging
import sys
import os

# Add root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from tests.chaos.chaos_monkey import ChaosMonkey, inject_chaos, configure_chaos

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Verify Chaos Loggers show up
logging.getLogger("tests.chaos.chaos_monkey").setLevel(logging.INFO)

# Setup Chaos
config = {
    'enabled': True,
    'latency_prob': 0.5,     # High probability for demo
    'exception_prob': 0.1,   # Low probability for crash
    'max_latency_ms': 500
}
monkey = ChaosMonkey(config)
configure_chaos(config) # set global

@inject_chaos
def sensitive_operation(n):
    """Simulate a database query or network call"""
    # Simulate work
    time.sleep(0.05)
    return n * n

def run_chaos_simulation():
    monkey.unleash()
    logger.info("Starting Chaos Simulation Loop (10 iterations)...")

    success = 0
    failures = 0

    for i in range(10):
        try:
            start = time.time()
            result = sensitive_operation(i)
            duration = time.time() - start
            logger.info(f"Iter {i}: Success (Duration: {duration:.3f}s)")
            success += 1
        except RuntimeError as e:
            logger.error(f"Iter {i}: CAUGHT EXCEPTION: {e}")
            failures += 1
        except Exception as e:
            logger.error(f"Iter {i}: Unexpected: {e}")
            failures += 1

    logger.info(f"Simulation Complete. Success: {success}, Failures: {failures}")
    monkey.cage()

if __name__ == "__main__":
    run_chaos_simulation()
