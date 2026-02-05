"""
Chaos Monkey Framework for Mini Quant Fund.
"The best way to avoid failure is to fail constantly."

Usage:
    monkey = ChaosMonkey(config)
    monkey.unleash()
"""

import time
import random
import logging
import threading
from typing import Dict, Any, List, Callable

logger = logging.getLogger(__name__)

class ChaosMonkey:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.enabled = self.config.get('enabled', False)
        self.latency_prob = self.config.get('latency_prob', 0.1)
        self.exception_prob = self.config.get('exception_prob', 0.01)
        self.max_latency_ms = self.config.get('max_latency_ms', 2000)
        self.targets = self.config.get('targets', [])  # list of function names or components to target
        self._active = False

    def unleash(self):
        """Enable chaos"""
        self._active = True
        logger.warning("🐒 CHAOS MONKEY UNLEASHED! Expect turbulence.")

    def cage(self):
        """Disable chaos"""
        self._active = False
        logger.info("🐒 Chaos Monkey caged.")

    def maybe_inject_latency(self, context_name: str):
        """
        Call this in critical paths to simulate network jitter.
        """
        if not self.enabled or not self._active:
            return

        if random.random() < self.latency_prob:
            delay = random.uniform(0.1, self.max_latency_ms / 1000.0)
            logger.warning(f"🐒 Chaos: Injecting {delay:.2f}s latency into {context_name}")
            time.sleep(delay)

    def maybe_raise_exception(self, context_name: str):
        """
        Call this to simulate random crashes/errors.
        """
        if not self.enabled or not self._active:
            return

        if random.random() < self.exception_prob:
            logger.critical(f"🐒 Chaos: Injecting CRITICAL EXCEPTION into {context_name}")
            raise RuntimeError(f"Chaos Monkey struck {context_name}!")

    def corrupt_data(self, data: Any) -> Any:
        """
        Simulate data corruption (NaNs, Infinite, Zeroes).
        """
        if not self.enabled or not self._active:
            return data

        # Example for dict or list
        if random.random() < 0.05: # 5% chance to corrupt
            logger.warning("🐒 Chaos: Data Corruption event!")
            return None # Return None or garbage

        return data

# Singleton instance for global access during testing
_chaos_instance = ChaosMonkey()

def get_chaos_monkey() -> ChaosMonkey:
    return _chaos_instance

def configure_chaos(config: Dict[str, Any]):
    global _chaos_instance
    _chaos_instance = ChaosMonkey(config)

# Decorator for easy injection
def inject_chaos(func):
    """
    Decorator to wrap functions with chaos injection.
    """
    def wrapper(*args, **kwargs):
        monkey = get_chaos_monkey()
        monkey.maybe_inject_latency(func.__name__)
        monkey.maybe_raise_exception(func.__name__)
        return func(*args, **kwargs)
    return wrapper
