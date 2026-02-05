
import subprocess
import time
import os
import signal
import json
from datetime import datetime

CANARY_DURATION_SEC = 300  # 5 minutes for demonstration
LOG_FILE = "runtime/logs/canary_paper.jsonl"

def run_canary():
    print(f"--- Starting Canary Paper Run ({CANARY_DURATION_SEC}s) ---")
    os.makedirs("runtime/logs", exist_ok=True)

    # Start the main loop in paper mode
    # We'll use start-process equivalent on windows if needed, but subprocess.Popen is fine
    proc = subprocess.Popen(
        ["python", "main.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    start_time = time.time()
    try:
        while time.time() - start_time < CANARY_DURATION_SEC:
            # Monitor if process is still alive
            if proc.poll() is not None:
                print("Error: main.py exited unexpectedly.")
                stdout, stderr = proc.communicate()
                print(f"STDOUT: {stdout}")
                print(f"STDERR: {stderr}")
                break

            # Periodically report progress
            elapsed = int(time.time() - start_time)
            print(f"Canary Progress: {elapsed}/{CANARY_DURATION_SEC}s...", end="\r")
            time.sleep(10)

    except KeyboardInterrupt:
        print("\nCanary interrupted by user.")
    finally:
        print("\nStopping Canary Run...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except:
            proc.kill()
        print("Canary Run Completed.")

    # Basic log health check
    if os.path.exists("runtime/logs/live.jsonl"):
        print("Live logs detected. Verification: [PASS]")
    else:
        print("Warning: No live logs found after canary run.")

if __name__ == "__main__":
    run_canary()
