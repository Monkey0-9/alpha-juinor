"""
Unified Database Factory.

This module provides a unified interface for database operations that can
switch between SQLite (development) and PostgreSQL (production) based on
the DB_ENGINE environment variable.

Usage:
    from database import get_db

    db = get_db()  # Returns the appropriate database manager
    db.upsert_daily_price(record)
    df = db.get_daily_prices("AAPL")
"""

import os
import logging
from typing import Any, Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Environment variable for database engine selection
DB_ENGINE = os.getenv("DB_ENGINE", "sqlite").lower()

# Import database managers
from .manager import DatabaseManager as SQLiteManager

# Try to import PostgresManager, handle case where it's not fully implemented
try:
    from .adapters.postgres_manager import PostgresManager
except (ImportError, NotImplementedError):
    # Create a stub class if PostgreSQL is not available
    class PostgresManager:
        def __init__(self):
            raise NotImplementedError(
                "PostgreSQL adapter not available. "
                "Ensure DB_ENGINE=sqlite or configure PostgreSQL connection."
            )

# Singleton instances
_sqlite_instance = None
_postgres_instance = None


class DatabaseFactory:
    """
    Factory for creating database manager instances.

    Supports:
    - SQLite: For development and testing
    - PostgreSQL: For production with TimescaleDB
    """

    @staticmethod
    def get_sqlite_manager() -> SQLiteManager:
        """Get SQLite database manager singleton"""
        global _sqlite_instance
        if _sqlite_instance is None:
            _sqlite_instance = SQLiteManager()
        return _sqlite_instance

    @staticmethod
    def get_postgres_manager() -> PostgresManager:
        """Get PostgreSQL database manager singleton"""
        global _postgres_instance
        if _postgres_instance is None:
            _postgres_instance = PostgresManager()
        return _postgres_instance

    @staticmethod
    def get_engine() -> str:
        """Get the current database engine"""
        return DB_ENGINE

    @staticmethod
    def set_engine(engine: str) -> None:
        """Set the database engine (sqlite or postgres)"""
        global DB_ENGINE
        DB_ENGINE = engine.lower()
        logger.info(f"Database engine set to: {DB_ENGINE}")

    @staticmethod
    def is_postgres() -> bool:
        """Check if PostgreSQL is the active engine"""
        return DB_ENGINE == "postgres"

    @staticmethod
    def is_sqlite() -> bool:
        """Check if SQLite is the active engine"""
        return DB_ENGINE == "sqlite"


class UnifiedDatabaseManager:
    """
    Unified database manager that delegates to the appropriate backend.

    This class provides a consistent interface regardless of the underlying
    database engine (SQLite or PostgreSQL).
    """

    def __init__(self):
        """Initialize the unified database manager"""
        self._sqlite = None
        self._postgres = None
        self._engine = DB_ENGINE

    @property
    def sqlite(self) -> SQLiteManager:
        """Get SQLite manager"""
        if self._sqlite is None:
            self._sqlite = DatabaseFactory.get_sqlite_manager()
        return self._sqlite

    @property
    def postgres(self) -> PostgresManager:
        """Get PostgreSQL manager"""
        if self._postgres is None:
            self._postgres = DatabaseFactory.get_postgres_manager()
        return self._postgres

    @property
    def current(self) -> Any:
        """Get the current database manager based on DB_ENGINE"""
        if self._engine == "postgres":
            return self.postgres
        return self.sqlite

    @property
    def engine(self) -> str:
        """Get the current engine name"""
        return self._engine

    def health_check(self) -> Dict[str, Any]:
        """Health check for current database"""
        return self.current.health_check()

    # =========================================================================
    # PRICE HISTORY OPERATIONS
    # =========================================================================

    def upsert_daily_price(self, record: Dict) -> bool:
        """Insert or update daily price record"""
        return self.current.upsert_daily_price(record)

    def upsert_daily_prices_batch(self, records: List[Dict]) -> int:
        """Batch upsert daily prices"""
        return self.current.upsert_daily_prices_batch(records)

    def get_daily_prices(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: Optional[int] = None
    ):
        """Get daily price history for a symbol"""
        return self.current.get_daily_prices(symbol, start_date, end_date, limit)

    def get_daily_prices_batch(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str
    ) -> Dict[str, Any]:
        """Get daily prices for multiple symbols"""
        return self.current.get_daily_prices_batch(symbols, start_date, end_date)

    # =========================================================================
    # INTRADAY PRICE OPERATIONS
    # =========================================================================

    def upsert_intraday_price(self, record: Dict) -> bool:
        """Insert or update intraday price record"""
        return self.current.upsert_intraday_price(record)

    def get_intraday_prices(self, symbol: str, date: Optional[str] = None):
        """Get intraday prices for a symbol"""
        return self.current.get_intraday_prices(symbol, date)

    # =========================================================================
    # FEATURE OPERATIONS
    # =========================================================================

    def upsert_features(self, records: List[Dict]) -> int:
        """Insert or update feature records"""
        return self.current.upsert_features(records)

    def get_latest_features(self, symbols: List[str]) -> Dict[str, Dict]:
        """Get latest features for a list of symbols"""
        return self.current.get_latest_features(symbols)

    # =========================================================================
    # GOVERNANCE OPERATIONS
    # =========================================================================

    def upsert_symbol_governance(self, record: Dict) -> bool:
        """Upsert symbol governance record"""
        return self.current.upsert_symbol_governance(record)

    def get_active_symbols(self) -> List[str]:
        """Get all symbols with ACTIVE state"""
        return self.current.get_active_symbols()

    def get_symbol_governance(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get governance state for a specific symbol"""
        return self.current.get_symbol_governance(symbol)

    # =========================================================================
    # ORDER OPERATIONS
    # =========================================================================

    def insert_orders(self, orders: List[Dict]) -> int:
        """Insert order records"""
        return self.current.insert_orders(orders)

    # =========================================================================
    # POSITION OPERATIONS
    # =========================================================================

    def upsert_position(self, position: Dict) -> bool:
        """Insert or update position"""
        return self.current.upsert_position(position)

    def get_positions(self) -> List[Dict]:
        """Get all current positions"""
        return self.current.get_positions()

    # =========================================================================
    # AUDIT OPERATIONS
    # =========================================================================

    def log_audit(self, entry: Dict) -> int:
        """Log audit entry"""
        return self.current.log_audit(entry)

    # =========================================================================
    # UTILITY OPERATIONS
    # =========================================================================

    def get_symbol_coverage(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """Get data coverage summary"""
        return self.current.get_symbol_coverage(start_date, end_date)

    def close(self):
        """Close database connections"""
        if self._sqlite:
            self._sqlite.close()
        if self._postgres:
            self._postgres.close()


# Singleton instance
_unified_instance = None


def get_db() -> UnifiedDatabaseManager:
    """
    Get the unified database manager singleton.

    Returns:
        UnifiedDatabaseManager: A unified interface to the database

    Example:
        >>> from database import get_db
        >>> db = get_db()
        >>> prices = db.get_daily_prices("AAPL", "2024-01-01", "2024-12-31")
    """
    global _unified_instance
    if _unified_instance is None:
        _unified_instance = UnifiedDatabaseManager()
    return _unified_instance


def get_sqlite_db() -> SQLiteManager:
    """Get SQLite database manager (for development)"""
    return DatabaseFactory.get_sqlite_manager()


def get_postgres_db() -> PostgresManager:
    """Get PostgreSQL database manager (for production)"""
    return DatabaseFactory.get_postgres_manager()


# Convenience imports for common operations
__all__ = [
    'get_db',
    'get_sqlite_db',
    'get_postgres_db',
    'DatabaseFactory',
    'UnifiedDatabaseManager',
    'DB_ENGINE',
]
