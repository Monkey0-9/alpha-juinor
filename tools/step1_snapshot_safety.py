import os
import shutil
from datetime import datetime

base_dir = r"C:\mini-quant-fund"
runtime_dir = os.path.join(base_dir, "runtime")
logs_dir = os.path.join(base_dir, "logs")
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = os.path.join(runtime_dir, "agent_results", ts)

def run_step1():
    print(f"Creating results directory: {results_dir}")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, "logs"), exist_ok=True)

    # Snapshot DBs
    db_files = ["institutional_trading.db", "audit.db"]
    for db in db_files:
        src = os.path.join(runtime_dir, db)
        if os.path.exists(src):
            dst = os.path.join(results_dir, f"{db}.bak")
            print(f"Backing up {db} to {dst}")
            shutil.copy2(src, dst)
        else:
            print(f"Warning: {src} not found")

    # Snapshot logs
    if os.path.exists(logs_dir):
        print(f"Backing up logs to {os.path.join(results_dir, 'logs')}")
        for item in os.listdir(logs_dir):
            s = os.path.join(logs_dir, item)
            d = os.path.join(results_dir, "logs", item)
            if os.path.isfile(s):
                shutil.copy2(s, d)

    # Create kill switch
    kill_switch_path = os.path.join(runtime_dir, "KILL_SWITCH")
    print(f"Creating kill switch at: {kill_switch_path}")
    with open(kill_switch_path, "w") as f:
        f.write("System halted by automation agent for readiness check.")

    print(f"Step 1 Complete. TS: {ts}")
    # Write TS to a file for subsequent steps to read if needed
    with open(os.path.join(runtime_dir, "agent_results", "latest_ts.txt"), "w") as f:
        f.write(ts)

if __name__ == "__main__":
    run_step1()
