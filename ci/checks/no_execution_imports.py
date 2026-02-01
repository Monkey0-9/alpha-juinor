
"""
ci/checks/no_execution_imports.py

Phase 0 Check: Enforce static analysis rule (no strategy -> execution imports).
Scans strategies/ and alpha_agents/ for forbidden imports.
"""
import os
import sys
import re

FORBIDDEN_PATTERNS = [
    r"from execution",
    r"import execution",
    r"from brokers",
    r"import brokers",
    r"import secrets_manager",
    r"from config import secrets_manager"
]

DIRS_TO_SCAN = [
    "strategies",
    "alpha_agents"
]

def scan_files():
    violations = []
    root_dir = os.getcwd()

    for relative_dir in DIRS_TO_SCAN:
        scan_path = os.path.join(root_dir, relative_dir)
        if not os.path.exists(scan_path):
            continue

        for root, _, files in os.walk(scan_path):
            for file in files:
                if file.endswith(".py"):
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, root_dir)

                    try:
                        with open(full_path, "r", encoding="utf-8") as f:
                            lines = f.readlines()

                        for i, line in enumerate(lines):
                            for pattern in FORBIDDEN_PATTERNS:
                                if re.search(pattern, line):
                                    violations.append(
                                        f"{rel_path}:{i+1}: {line.strip()} (Matches '{pattern}')"
                                    )
                    except Exception as e:
                        print(f"Warning: Could not read {rel_path}: {e}")

    if violations:
        print("FAIL: Forbidden imports found in strategy code:")
        for v in violations:
            print(f"  - {v}")
        sys.exit(1)
    else:
        print("SUCCESS: No forbidden imports found.")
        sys.exit(0)

if __name__ == "__main__":
    scan_files()
