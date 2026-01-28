"""
database/adapters/context_manager.py
Context manager wrapper for database connections
"""
import contextlib
from typing import Any


@contextlib.contextmanager
def connection_context(get_conn_func):
    """
    Context manager for database connections.

    Args:
        get_conn_func: Function that returns a connection object

    Yields:
        Connection object

    Example:
        with connection_context(adapter._get_connection) as conn:
            conn.execute("SELECT * FROM table")
    """
    conn = None
    try:
        conn = get_conn_func()
        yield conn
    finally:
        if conn is not None:
            # SQLite connections in this codebase don't need explicit close
            # as they're managed by the adapter pool
            pass
