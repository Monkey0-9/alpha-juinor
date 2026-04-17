import time
import functools
import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict
from ..core.enterprise_logger import get_enterprise_logger

logger = get_enterprise_logger("profiler")

class LatencyProfiler:
    """
    High-resolution nanosecond profiler for institutional trading hot-paths.
    Tracks p50, p95, p99 latencies for specific components.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LatencyProfiler, cls).__new__(cls)
            cls._instance.metrics = defaultdict(list)
        return cls._instance

    def record(self, component: str, duration_ns: int):
        self.metrics[component].append(duration_ns)

    def get_stats(self, component: str) -> Dict[str, float]:
        durations = self.metrics.get(component, [])
        if not durations:
            return {"mean": 0, "p50": 0, "p95": 0, "p99": 0, "count": 0}
        
        durations_ms = [d / 1_000_000 for d in durations]
        return {
            "mean": float(np.mean(durations_ms)),
            "p50": float(np.percentile(durations_ms, 50)),
            "p95": float(np.percentile(durations_ms, 95)),
            "p99": float(np.percentile(durations_ms, 99)),
            "count": len(durations)
        }

    def reset(self):
        self.metrics.clear()

        logger.info("--- Institutional Latency Report (ms) ---")
        for component in sorted(self.metrics.keys()):
            stats = self.get_stats(component)
            logger.info(
                f"LtncyReport: {component}",
                component=component,
                count=stats['count'],
                p50_ms=round(stats['p50'], 4),
                p95_ms=round(stats['p95'], 4),
                p99_ms=round(stats['p99'], 4)
            )

def profile_ns(component: str):
    """Decorator for profiling function execution in nanoseconds."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter_ns()
            result = func(*args, **kwargs)
            duration = time.perf_counter_ns() - start
            LatencyProfiler().record(component, duration)
            return result
        return wrapper
    return decorator

class SectionTimer:
    """Context manager for timing sections of code."""
    def __init__(self, component: str):
        self.component = component
        self.start_ns = 0

    def __enter__(self):
        self.start_ns = time.perf_counter_ns()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.perf_counter_ns() - self.start_ns
        LatencyProfiler().record(self.component, duration)
