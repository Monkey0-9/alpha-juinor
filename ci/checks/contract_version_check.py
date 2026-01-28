"""
ci/checks/contract_version_check.py

Validates that all target JSON outputs or Python dataclasses
contain contract_version and schema_hash fields.
"""

import os
import ast
import sys

def check_contracts():
    """
    Checks python files in contracts/ and risk/quantum/contracts.py
    Ensures every dataclass definition has version fields.
    """
    targets = [
        "contracts/allocation.py",
        "contracts/alpha.py", # If exists
        "risk/contracts.py",  # If exists
        "risk/quantum/contracts.py"
    ]

    required_fields = {"contract_version", "schema_hash"}
    violations = []

    for rel_path in targets:
        if not os.path.exists(rel_path):
            continue

        with open(rel_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=rel_path)

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Check if it's a dataclass (heuristic: decorator)
                is_dataclass = any(
                    (isinstance(d, ast.Name) and d.id == "dataclass") or
                    (isinstance(d, ast.Call) and isinstance(d.func, ast.Name) and d.func.id == "dataclass")
                    for d in node.decorator_list
                )

                if not is_dataclass:
                    continue

                # Check fields
                fields = set()
                for item in node.body:
                    if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                        fields.add(item.target.id)

                missing = required_fields - fields
                if missing:
                    violations.append(f"{rel_path}: Class '{node.name}' missing {missing}")

    if violations:
        print("[CI] CONTRACT VERSION VIOLATIONS:")
        for v in violations:
            print(f"  - {v}")
        sys.exit(1)
    else:
        print("[CI] All contracts versioned correctly.")
        sys.exit(0)

if __name__ == "__main__":
    check_contracts()
