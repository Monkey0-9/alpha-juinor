"""
Bottleneck Discovery: Software Stack Profiler.
Instruments the LiveDecisionLoop to identify high-latency components.
"""

import sys
import os
sys.path.append(os.getcwd())

import cProfile
import pstats
import io
from orchestration.live_decision_loop import LiveDecisionLoop
import yaml

def profile_decision_loop():
    print("Initializing LiveDecisionLoop for Profiling...")

    # Load minimal config
    with open("configs/kill_switch_config.yaml", "r") as f:
        config = yaml.safe_load(f) or {}

    # Initialize loop (Paper mode implied)
    # Mocking components usually required, but LiveDecisionLoop handles graceful degradation
    loop = LiveDecisionLoop(symbols=["AAPL", "GOOG"], tick_interval=0.01)

    # Monkeypatch router to avoid network I/O during profiling
    loop.router.get_latest_price = lambda symbol: 150.0
    loop.router.get_latest_prices_parallel = lambda symbols: {s: 150.0 for s in symbols}

    loop.running = True # Allow one cycle

    profiler = cProfile.Profile()
    print("Starting Profiler for 5 cycles...")

    profiler.enable()
    # Run critical path N times
    for _ in range(5):
        loop._run_decision_tick()
    profiler.disable()

    print("Profiling Complete. Analyzing...")

    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumtime')
    ps.print_stats(20) # Top 20 functions by cumulative time

    print("\n=== BOTTLENECK ANALYSIS ===")
    print(s.getvalue())

    # Save to file
    ps.dump_stats("tests/profiling/decision_loop.prof")
    print("Stats saved to tests/profiling/decision_loop.prof")

if __name__ == "__main__":
    profile_decision_loop()
