"""
scripts/fix_db_api.py
Automated script to replace _get_connection() with get_connection() context manager
"""
import os
import re
from pathlib import Path

# Files to fix (excluding adapters and tests which legitimately use _get_connection)
TARGET_PATTERNS = [
    "data/governance/*.py",
    "data/collectors/*.py",
    "data_intelligence/*.py",
    "tools/*.py",
    "scripts/*.py",
    "*.py"  # Root level scripts
]

EXCLUDE_PATTERNS = [
    "database/adapters/",
    "tests/",
    "fix_db_api.py",  # Don't fix ourselves
]

def should_process_file(filepath: str) -> bool:
    """Check if file should be processed."""
    for exclude in EXCLUDE_PATTERNS:
        if exclude in filepath:
            return False
    return filepath.endswith(".py")

def fix_file(filepath: Path) -> tuple[bool, int]:
    """
    Fix _get_connection() usage in a file.

    Returns:
        (changed, num_replacements)
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return False, 0

    original_content = content
    replacements = 0

    # Pattern 1: conn = self.db._get_connection()
    # Replace with: with self.db.get_connection() as conn:
    pattern1 = r'(\s+)conn\s*=\s*self\.db\._get_connection\(\)'
    if re.search(pattern1, content):
        print(f"Warning: {filepath} needs manual review for context manager conversion")
        # For now, just replace the call itself
        content = re.sub(r'self\.db\._get_connection\(\)', 'self.db.get_connection()', content)
        replacements += len(re.findall(pattern1, original_content))

    # Pattern 2: conn = db._get_connection()
    pattern2 = r'(\s+)conn\s*=\s*db\._get_connection\(\)'
    if re.search(pattern2, content):
        content = re.sub(r'db\._get_connection\(\)', 'db.get_connection()', content)
        replacements += len(re.findall(pattern2, original_content))

    # Pattern 3: Direct usage like db._get_connection().execute(...)
    pattern3 = r'\.db\._get_connection\(\)\.execute\('
    if re.search(pattern3, content):
        print(f"Warning: {filepath} has inline _get_connection().execute() - needs manual fix")

    pattern4 = r'db\._get_connection\(\)\.execute\('
    if re.search(pattern4, content):
        print(f"Warning: {filepath} has inline db._get_connection().execute() - needs manual fix")

    if content != original_content:
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return True, replacements
        except Exception as e:
            print(f"Error writing {filepath}: {e}")
            return False, 0

    return False, 0

def main():
    root = Path("c:/mini-quant-fund")
    files_changed = 0
    total_replacements = 0

    print("Scanning for _get_connection() usage...")

    # Find all Python files
    for pyfile in root.rglob("*.py"):
        if not should_process_file(str(pyfile)):
            continue

        changed, count = fix_file(pyfile)
        if changed:
            files_changed += 1
            total_replacements += count
            print(f"✓ Fixed: {pyfile.relative_to(root)} ({count} replacements)")

    print(f"\nSummary:")
    print(f"  Files changed: {files_changed}")
    print(f"  Total replacements: {total_replacements}")
    print(f"\nNOTE: Files with inline .execute() calls need manual review!")

if __name__ == "__main__":
    main()
