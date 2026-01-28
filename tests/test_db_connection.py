import pytest

from database.manager import DatabaseManager


def test_get_connection():
    db = DatabaseManager()

    # Check if get_connection returns a context manager or connection
    # Since existing adapter is SQLite/Postgres, it should work
    try:
        conn = db.get_connection()
        # Verify it has execute method or similar
        assert hasattr(conn, 'execute') or hasattr(conn, 'cursor')
    except NotImplementedError:
        pytest.fail("get_connection() raised NotImplementedError")
    except Exception as e:
        # It might fail if no DB file, but manager usually creates one
        pytest.fail(f"get_connection() failed: {e}")

def test_get_connection_context_manager():
    db = DatabaseManager()
    # Check context manager usage
    # Note: SQLite connection is a context manager, but the Adapter might return a wrapped one or raw
    # We test if it supports __enter__
    conn = db.get_connection()
    assert hasattr(conn, '__enter__')
    assert hasattr(conn, '__exit__')
