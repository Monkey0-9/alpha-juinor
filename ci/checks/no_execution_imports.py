import ast
import os
import sys

DIRS_TO_SCAN = ["strategies", "alpha_agents"]
FORBIDDEN = {"execution", "brokers"}

def scan_files():
    violations = []
    root_dir = os.getcwd()
    for relative_dir in DIRS_TO_SCAN:
        scan_path = os.path.join(root_dir, relative_dir)
        if not os.path.exists(scan_path):
            continue
        for root, _, files in os.walk(scan_path):
            for file in files:
                if not file.endswith(".py"): continue
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, root_dir)
                try:
                    with open(full_path, "r", encoding="utf-8") as f:
                        tree = ast.parse(f.read(), filename=rel_path)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for alias in node.names:
                                if any(alias.name.split(".")[0] == m for m in FORBIDDEN):
                                    violations.append(f"{rel_path}:{getattr(node, 'lineno', 1)}: import {alias.name}")
                        elif isinstance(node, ast.ImportFrom) and node.module:
                            if any(node.module.split(".")[0] == m for m in FORBIDDEN):
                                violations.append(f"{rel_path}:{getattr(node, 'lineno', 1)}: from {node.module} import ...")
                except Exception as e:
                    print(f"Warning: Could not analyze {rel_path}: {e}")
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
