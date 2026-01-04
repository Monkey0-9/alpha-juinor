# tests/chaos/failure_injector.py
import random
import time
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class FailureInjector:
    """
    Chaos Engineering Harness for Quant Systems.
    Intercepts data/execution calls and injects realistic institutional failures.
    """
    
    def __init__(self, target_provider):
        self.provider = target_provider
        self.enabled = False
        self.scenarios = {
            "missing_bars": 0.0,
            "api_timeouts": 0.0,
            "zero_volume": 0.0,
            "price_gaps": 0.0,
            "stale_prices": 0.0
        }

    def configure(self, scenarios: Dict[str, float]):
        """Set probabilities for various failures (0.0 to 1.0)."""
        self.scenarios.update(scenarios)
        self.enabled = True
        logger.info(f"Chaos Harness enabled with scenarios: {self.scenarios}")

    def fetch_ohlcv(self, *args, **kwargs) -> pd.DataFrame:
        """Wrapped data fetch with failure injection."""
        # 1. Simulate API Timeout
        if self.enabled and random.random() < self.scenarios.get("api_timeouts", 0):
            delay = random.uniform(2.0, 10.0)
            logger.warning(f"CHAOS: Simulating API Timeout ({delay:.1f}s delay)")
            time.sleep(delay)
            # In some cases, actually raise exception
            if random.random() < 0.3:
                raise ConnectionError("CHAOS: Simulated API Connection Failure")

        data = self.provider.fetch_ohlcv(*args, **kwargs)
        
        if not self.enabled or data.empty:
            return data
            
        # 2. Simulate Missing Bars (Dropping random rows)
        if random.random() < self.scenarios.get("missing_bars", 0):
            n_to_drop = max(1, int(len(data) * 0.1))
            indices = random.sample(range(len(data)), n_to_drop)
            logger.warning(f"CHAOS: Injecting {n_to_drop} missing bars")
            data = data.drop(data.index[indices])
            
        # 3. Simulate Zero Volume / Stale Prices
        if random.random() < self.scenarios.get("zero_volume", 0):
            idx = random.randint(0, len(data) - 1)
            logger.warning(f"CHAOS: Overwriting bar at {data.index[idx]} with zero volume")
            data.iloc[idx, data.columns.get_loc('Volume')] = 0.0
            
        # 4. Simulate Price Gaps (Flash Crash Scenario)
        if random.random() < self.scenarios.get("price_gaps", 0):
            idx = random.randint(0, len(data) - 1)
            gap_pct = random.choice([-0.10, -0.05, 0.05, 0.10])
            logger.warning(f"CHAOS: Injecting {gap_pct:+.1%} price gap at {data.index[idx]}")
            for col in ['Open', 'High', 'Low', 'Close']:
                data.iloc[idx, data.columns.get_loc(col)] *= (1 + gap_pct)
                
        return data

    def __getattr__(self, name):
        """Pass through other methods to the underlying provider."""
        return getattr(self.provider, name)
