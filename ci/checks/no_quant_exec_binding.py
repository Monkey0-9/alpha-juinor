"""
ci/checks/no_quant_exec_binding.py

Static analysis check to forbid quantum module usage in execution layer.
Master Rule: Execution must not depend on Quantum (Physics) modules directly.
"""

import os
import ast
import sys

def check_quantum_binding():
    violations = []

    # Execution scope
    target_scope = "execution"
    forbidden_module = "risk.quantum"

    root_dir = os.getcwd()

    for root, dirs, files in os.walk(root_dir):
        rel_root = os.path.relpath(root, root_dir)

        # Only check execution directory
        if "execution" not in rel_root.split(os.sep):
            continue

        for file in files:
            if not file.endswith(".py"):
                continue

            path = os.path.join(root, file)
            with open(path, "r", encoding="utf-8") as f:
                try:
                    tree = ast.parse(f.read(), filename=path)
                except Exception:
                    continue

                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            if forbidden_module in alias.name:
                                violations.append(f"{path}: Import '{alias.name}' forbidden. Exec cannot touch Quantum.")
                    elif isinstance(node, ast.ImportFrom):
                        if node.module and forbidden_module in node.module:
                             violations.append(f"{path}: From-Import '{node.module}' forbidden. Exec cannot touch Quantum.")

    if violations:
        print("[CI] QUANTUM-EXECUTION BINDING DETECTED (FORBIDDEN):")
        for v in violations:
            print(f"  - {v}")
        sys.exit(1)
    else:
        print("[CI] Execution layer is free of Quantum entanglements.")
        sys.exit(0)

if __name__ == "__main__":
    check_quantum_binding()
