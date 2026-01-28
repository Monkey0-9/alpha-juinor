"""
ci/checks/no_execution_imports.py

Static analysis check to forbid execution layer imports in forbidden zones.
Enforces Master Rule #1: No component may both decide and execute.
"""

import os
import ast
import sys

def check_imports():
    """
    Scans codebase for forbidden imports.
    """
    violations = []

    # Rules: file_pattern -> forbidden_modules
    rules = {
        "portfolio": ["execution", "alpaca.trading"],
        "alpha_families": ["execution", "portfolio", "risk"], # Alphas should be pure signal
        "risk": ["execution"],
        "strategies": ["execution"]
    }

    root_dir = os.getcwd()

    for root, dirs, files in os.walk(root_dir):
        # Skip venv, git, etc
        if "venv" in root or ".git" in root or "__pycache__" in root:
            continue

        rel_root = os.path.relpath(root, root_dir)

        # Check if current dir matches a rule scope
        forbidden = []
        for scope, bad_mods in rules.items():
            if scope in rel_root.split(os.sep):
                forbidden = bad_mods
                break

        if not forbidden:
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
                            for bad in forbidden:
                                if alias.name.startswith(bad):
                                    violations.append(f"{path}: Import '{alias.name}' forbidden in scope '{rel_root}'")
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            for bad in forbidden:
                                if node.module.startswith(bad):
                                    violations.append(f"{path}: From-Import '{node.module}' forbidden in scope '{rel_root}'")

    if violations:
        print("[CI] IMPORT BOUNDARY VIOLATIONS FOUND:")
        for v in violations:
            print(f"  - {v}")
        sys.exit(1)
    else:
        print("[CI] Import boundaries clean.")
        sys.exit(0)

if __name__ == "__main__":
    check_imports()
