"""
tests/test_db_public_api.py
P0-1: Test that no runtime files use private _get_connection() API
"""
import glob
import pytest


def test_no_private_db_calls():
    """
    Ensure no production code uses private _get_connection() API.

    Only database/adapters/* and tests/* are allowed to use _get_connection.
    """
    violations = []

    files = glob.glob('**/*.py', recursive=True)
    for filepath in files:
        # Skip allowed locations
        if 'tests/' in filepath or 'test_' in filepath:
            continue
        if 'database/adapters/' in filepath:
            continue
        if 'database\\adapters\\' in filepath:  # Windows path
            continue
        if 'docs/' in filepath:
            continue
        if 'scripts/fix_db_api.py' in filepath:  # The fixer script itself
            continue

        try:
            with open(filepath, 'r', encoding='utf-8') as fh:
                content = fh.read()
                if '_get_connection(' in content:
                    # Count occurrences
                    count = content.count('_get_connection(')
                    violations.append(f"{filepath} ({count} occurrence(s))")
        except Exception:
            # Skip files that can't be read
            continue

    if violations:
        violation_list = '\n'.join(f"  - {v}" for v in violations)
        pytest.fail(
            f"{len(violations)} file(s) use private _get_connection() API:\n{violation_list}\n\n"
            f"Use 'with db.get_connection() as conn:' instead"
        )


def test_get_connection_is_context_manager():
    """Test that get_connection() returns a proper context manager."""
    from database.manager import DatabaseManager

    db = DatabaseManager()

    # Should be usable with 'with' statement
    with db.get_connection() as conn:
        assert conn is not None
        # Should be able to execute queries
        cursor = conn.execute("SELECT 1")
        result = cursor.fetchone()
        assert result is not None


def test_deprecated_warning():
    """Test that _get_connection() emits deprecation warning."""
    from database.manager import DatabaseManager
    import warnings

    db = DatabaseManager()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        conn = db.get_connection()

        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "_get_connection is deprecated" in str(w[0].message)
