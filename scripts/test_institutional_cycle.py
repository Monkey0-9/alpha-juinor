
import sys
import logging
import time

# Ensure root is in path
sys.path.append(".")

from orchestration.cycle_orchestrator import CycleOrchestrator

logging.basicConfig(level=logging.INFO)

def main():
    print("Initializing Institutional Cycle Orchestrator...")
    engine = CycleOrchestrator(mode="paper")

    print("Running Cycle...")
    start = time.time()
    results = engine.run_cycle()
    elapsed = time.time() - start

    print(f"Cycle Finished in {elapsed:.2f}s")
    print(f"Decisions Count: {len(results)}")

    if len(results) > 0:
        print("First Decision Sample:", results[0].to_audit_record())
    else:
        print("ERROR: No decisions produced!")
        sys.exit(1)

if __name__ == "__main__":
    main()
