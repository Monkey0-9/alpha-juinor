"""
Performance Profiling Utilities

Timing decorators and profiling tools for identifying bottlenecks.
"""
import time
import functools
import logging
from typing import Callable, Any
from collections import defaultdict
import json

logger = logging.getLogger(__name__)

# Global timing statistics
_timing_stats = defaultdict(lambda: {"count": 0, "total_time": 0.0, "min_time": float('inf'), "max_time": 0.0})


def timeit(func: Callable) -> Callable:
    """
    Decorator to measure function execution time.
    
    Usage:
        @timeit
        def expensive_function():
            pass
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            elapsed = time.perf_counter() - start
            func_name = f"{func.__module__}.{func.__name__}"
            
            # Update statistics
            stats = _timing_stats[func_name]
            stats["count"] += 1
            stats["total_time"] += elapsed
            stats["min_time"] = min(stats["min_time"], elapsed)
            stats["max_time"] = max(stats["max_time"], elapsed)
            
            # Log if significant
            if elapsed > 1.0:
                logger.info(f"⏱️  {func_name}: {elapsed:.3f}s")
    
    return wrapper


def get_timing_report() -> dict:
    """
    Get comprehensive timing statistics.
    
    Returns:
        Dict with timing data for all profiled functions
    """
    report = {}
    for func_name, stats in _timing_stats.items():
        if stats["count"] > 0:
            report[func_name] = {
                "calls": stats["count"],
                "total_time": round(stats["total_time"], 3),
                "avg_time": round(stats["total_time"] / stats["count"], 3),
                "min_time": round(stats["min_time"], 3),
                "max_time": round(stats["max_time"], 3)
            }
    
    # Sort by total time
    report = dict(sorted(report.items(), key=lambda x: x[1]["total_time"], reverse=True))
    return report


def print_timing_report():
    """Print formatted timing report."""
    report = get_timing_report()
    
    if not report:
        print("No timing data collected")
        return
    
    print("\n" + "="*80)
    print("PERFORMANCE PROFILING REPORT")
    print("="*80)
    print(f"{'Function':<50} {'Calls':>8} {'Total(s)':>10} {'Avg(s)':>10} {'Min(s)':>10} {'Max(s)':>10}")
    print("-"*80)
    
    for func_name, stats in report.items():
        # Truncate long function names
        display_name = func_name if len(func_name) <= 50 else "..." + func_name[-47:]
        print(f"{display_name:<50} {stats['calls']:>8} {stats['total_time']:>10.3f} {stats['avg_time']:>10.3f} "
              f"{stats['min_time']:>10.3f} {stats['max_time']:>10.3f}")
    
    print("="*80)
    
    # Top 3 bottlenecks
    top_3 = list(report.items())[:3]
    if top_3:
        print("\n🔍 TOP 3 BOTTLENECKS:")
        for i, (func, stats) in enumerate(top_3, 1):
            print(f"  {i}. {func}: {stats['total_time']:.3f}s total ({stats['calls']} calls)")
    print()


def save_timing_report(filepath: str = "profiling_report.json"):
    """Save timing report to JSON file."""
    report = get_timing_report()
    with open(filepath, 'w') as f:
        json.dump(report, f, indent=2)
    logger.info(f"Timing report saved to {filepath}")


def reset_timing_stats():
    """Clear all timing statistics."""
    global _timing_stats
    _timing_stats.clear()
