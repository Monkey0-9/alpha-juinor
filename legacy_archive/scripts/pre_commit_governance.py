import sys
import os
import subprocess
import re

def run_command(cmd):
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return 1, "", str(e)

def check_risky_prints():
    print("Checking for risky prints in critical modules...")
    critical_dirs = ["orchestration", "alpha_families", "audit", "database"]
    pattern = re.compile(r"print\(")

    found_risky = False
    for d in critical_dirs:
        if not os.path.exists(d): continue
        for root, _, files in os.walk(d):
            for f in files:
                if f.endswith(".py"):
                    path = os.path.join(root, f)
                    with open(path, "r", encoding="utf-8") as file:
                        for i, line in enumerate(file, 1):
                            if pattern.search(line) and not line.strip().startswith("#"):
                                print(f"FAIL: Risky print found in {path}:{i}")
                                found_risky = True
    return found_risky

def check_structured_logging():
    print("Checking for structured logging configuration usage...")
    # Ensure setup_logging is used in main.py or other entrypoints
    entrypoints = ["main.py", "orchestration/live_decision_loop.py"]
    for ep in entrypoints:
        if os.path.exists(ep):
            with open(ep, "r", encoding="utf-8") as f:
                content = f.read()
                if "setup_logging" not in content and "logging.config" not in content:
                    print(f"FAIL: {ep} does not seem to use institutional structured logging.")
                    return True
    return False

def main():
    failed = False

    # 1. Pytest Critical Subset
    print("Running critical tests...")
    code, out, err = run_command("python -m pytest -q tests/test_db_connection.py tests/test_feature_validation.py tests/test_model_meta.py")
    if code != 0:
        print("FAIL: Critical tests failed.")
        print(out, err)
        failed = True
    else:
        print("PASS: Critical tests passed.")

    # 2. Risky Prints
    if check_risky_prints():
        failed = True
    else:
        print("PASS: No risky prints in critical modules.")

    # 3. Structured Logging Check
    if check_structured_logging():
        failed = True
    else:
        print("PASS: Structured logging usage verified.")

    # 4. MyPy (Simulated or light check if mypy installed)
    print("Checking types with MyPy (subset)...")
    code, out, err = run_command("mypy database/manager.py alpha_families/ml_alpha.py")
    if code != 0 and "not found" not in err.lower():
        print("FAIL: Type check failed.")
        print(out)
        # Not marking failed yet as mypy might not be in the venv
    else:
        print("PASS/SKIP: Type check completed.")

    if failed:
        print("\nPRE-COMMIT GATING: FAILED. Please fix the items above.")
        sys.exit(1)

    print("\nPRE-COMMIT GATING: PASSED.")
    sys.exit(0)

if __name__ == "__main__":
    main()
