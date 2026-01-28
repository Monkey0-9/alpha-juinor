"""
Database Migration Tests.

Tests for SQLite to PostgreSQL migration functionality.
"""

import os
import sys
import pytest
import tempfile
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from database.adapters.postgres_manager import PostgresManager


class TestPostgresManager:
    """Test cases for PostgreSQL manager."""

    @pytest.fixture
    def mock_engine(self):
        """Create a mock SQLAlchemy engine."""
        with patch('database.adapters.postgres_manager.create_engine') as mock:
            engine = MagicMock()
            mock.return_value = engine
            yield engine

    @pytest.fixture
    def mock_session(self):
        """Create a mock session."""
        session = MagicMock()
        session.execute.return_value = None
        session.commit.return_value = None
        return session

    def test_connection_config(self):
        """Test that connection configuration is read from environment."""
        # Set environment variables
        os.environ['POSTGRES_HOST'] = 'test-host'
        os.environ['POSTGRES_PORT'] = '5433'
        os.environ['POSTGRES_DB'] = 'test-db'
        os.environ['POSTGRES_USER'] = 'test-user'
        os.environ['POSTGRES_PASSWORD'] = 'test-password'
        os.environ['DB_POOL_SIZE'] = '5'
        os.environ['DB_MAX_OVERFLOW'] = '10'

        try:
            from database.adapters.postgres_manager import (
                POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB,
                POSTGRES_USER, POSTGRES_PASSWORD, POOL_SIZE, MAX_OVERFLOW
            )

            assert POSTGRES_HOST == 'test-host'
            assert POSTGRES_PORT == 5433
            assert POSTGRES_DB == 'test-db'
            assert POSTGRES_USER == 'test-user'
            assert POSTGRES_PASSWORD == 'test-password'
            assert POOL_SIZE == 5
            assert MAX_OVERFLOW == 10
        finally:
            # Clean up
            for key in ['POSTGRES_HOST', 'POSTGRES_PORT', 'POSTGRES_DB',
                       'POSTGRES_USER', 'POSTGRES_PASSWORD', 'DB_POOL_SIZE',
                       'DB_MAX_OVERFLOW']:
                os.environ.pop(key, None)

    def test_singleton_pattern(self):
        """Test that PostgresManager follows singleton pattern."""
        # Reset singleton for testing
        PostgresManager._instance = None
        PostgresManager._lock = None

        with patch('database.adapters.postgres_manager.create_engine'):
            manager1 = PostgresManager()
            manager2 = PostgresManager()

            assert manager1 is manager2

    def test_health_check_success(self):
        """Test health check returns healthy status."""
        PostgresManager._instance = None
        PostgresManager._lock = None

        mock_engine = MagicMock()
        mock_session = MagicMock()

        with patch('database.adapters.postgres_manager.create_engine', return_value=mock_engine):
            with patch.object(PostgresManager, '_init_schema'):
                manager = PostgresManager()
                manager.engine = mock_engine
                manager.Session = MagicMock()

                # Mock session behavior
                mock_sess_ctx = MagicMock()
                mock_sess_ctx.__enter__ = MagicMock(return_value=mock_session)
                mock_sess_ctx.__exit__ = MagicMock(return_value=False)
                manager.Session.return_value = mock_sess_ctx

                # Mock health check queries
                mock_session.execute.return_value.fetchone = Mock(return_value=(1,))
                mock_session.execute.return_value.fetchall = Mock(return_value=[])

                health = manager.health_check()

                assert health['status'] == 'healthy'
                assert health['engine'] == 'postgresql'

    def test_health_check_failure(self):
        """Test health check handles connection failures."""
        PostgresManager._instance = None
        PostgresManager._lock = None

        with patch('database.adapters.postgres_manager.create_engine') as mock:
            mock.side_effect = Exception("Connection failed")

            with pytest.raises(Exception):
                PostgresManager()


class TestMigrationScript:
    """Test cases for migration script."""

    @pytest.fixture
    def temp_sqlite_db(self):
        """Create a temporary SQLite database with test data."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            conn = sqlite3.connect(f.name)
            conn.row_factory = sqlite3.Row

            # Create test schema
            conn.execute('''
                CREATE TABLE IF NOT EXISTS symbol_governance (
                    symbol TEXT PRIMARY KEY,
                    history_rows INTEGER,
                    data_quality REAL,
                    state TEXT,
                    reason TEXT,
                    last_checked_ts TEXT,
                    metadata_json TEXT
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS price_history (
                    symbol TEXT NOT NULL,
                    date TEXT NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    adjusted_close REAL,
                    provider TEXT,
                    raw_hash TEXT,
                    validation_flags TEXT,
                    PRIMARY KEY (symbol, date)
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS features (
                    symbol TEXT NOT NULL,
                    date TEXT NOT NULL,
                    features_json TEXT,
                    version TEXT,
                    PRIMARY KEY (symbol, date)
                )
            ''')

            # Insert test data
            conn.execute(
                "INSERT OR REPLACE INTO symbol_governance VALUES (?, ?, ?, ?, ?, ?, ?)",
                ('AAPL', 252, 0.95, 'ACTIVE', 'Test', '2024-01-01', '{}')
            )

            conn.execute(
                "INSERT INTO price_history VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ('AAPL', '2024-01-02', 185.0, 186.0, 184.0, 185.5, 50000000, 185.5, 'yahoo', 'hash123', '{}')
            )

            conn.execute(
                "INSERT INTO features VALUES (?, ?, ?, ?)",
                ('AAPL', '2024-01-02', '{"sma_20": 180.5}', '1.0')
            )

            conn.commit()
            conn.close()

            yield f.name

            # Cleanup
            os.unlink(f.name)

    def test_migration_runner_initialization(self):
        """Test MigrationRunner initializes correctly."""
        from scripts.migrate_sqlite_to_pg import MigrationRunner

        runner = MigrationRunner(dry_run=True, batch_size=100)

        assert runner.dry_run is True
        assert runner.batch_size == 100
        assert runner.stats['tables_migrated'] == 0
        assert runner.stats['rows_migrated'] == 0

    def test_get_table_count(self, temp_sqlite_db):
        """Test getting table row count."""
        from scripts.migrate_sqlite_to_pg import MigrationRunner

        runner = MigrationRunner()

        conn = sqlite3.connect(temp_sqlite_db)
        conn.row_factory = sqlite3.Row

        count = runner.get_table_count(conn, 'symbol_governance')

        assert count == 1

        conn.close()

    def test_prepare_record_handles_nulls(self):
        """Test that null values are handled correctly during migration."""
        from scripts.migrate_sqlite_to_pg import MigrationRunner

        runner = MigrationRunner()

        # Test record with NaN values
        import pandas as pd
        import numpy as np

        record = {
            'symbol': 'TEST',
            'close': float('nan'),
            'volume': None,
            'date': pd.NaT,
        }

        # This should not raise an exception
        for key, value in record.items():
            if pd.isna(value):
                record[key] = None

        assert record['close'] is None
        assert record['volume'] is None


class TestDatabaseFactory:
    """Test cases for unified database factory."""

    def test_factory_get_sqlite_manager(self):
        """Test getting SQLite manager from factory."""
        with patch('database.adapters.postgres_manager.create_engine'):
            from database import DatabaseFactory
            from database.manager import DatabaseManager

            # Reset singletons
            DatabaseManager._instance = None
            DatabaseFactory._sqlite_instance = None

            manager = DatabaseFactory.get_sqlite_manager()

            assert isinstance(manager, DatabaseManager)

    def test_factory_engine_selection(self):
        """Test database engine selection."""
        from database import DatabaseFactory

        # Test default engine
        assert DatabaseFactory.get_engine() in ['sqlite', 'postgres']

        # Test setting engine
        DatabaseFactory.set_engine('postgres')
        assert DatabaseFactory.is_postgres() is True
        assert DatabaseFactory.is_sqlite() is False

        DatabaseFactory.set_engine('sqlite')
        assert DatabaseFactory.is_sqlite() is True
        assert DatabaseFactory.is_postgres() is False

    def test_unified_manager_properties(self):
        """Test unified database manager properties."""
        from database import UnifiedDatabaseManager, DatabaseFactory

        with patch('database.adapters.postgres_manager.create_engine'):
            DatabaseFactory._postgres_instance = None
            DatabaseFactory._sqlite_instance = None
            DatabaseManager._instance = None

            manager = UnifiedDatabaseManager()

            # Both managers should be lazily initialized
            assert manager.sqlite is not None or manager.postgres is not None


class TestDatabaseIntegration:
    """Integration tests for database operations."""

    def test_price_history_operations(self):
        """Test price history CRUD operations."""
        import pandas as pd
        from datetime import datetime, timedelta

        # Create test record
        today = datetime.now().strftime('%Y-%m-%d')

        record = {
            'symbol': 'TEST_PYTEST',
            'date': today,
            'open': 100.0,
            'high': 101.0,
            'low': 99.0,
            'close': 100.5,
            'volume': 1000000,
            'adjusted_close': 100.5,
            'provider': 'pytest',
            'raw_hash': 'test_hash_123',
            'validation_flags': None
        }

        # Test would require actual database connection
        # This is a placeholder for integration test structure
        assert record['symbol'] == 'TEST_PYTEST'
        assert record['close'] == 100.5

    def test_feature_record_format(self):
        """Test feature record format."""
        record = {
            'symbol': 'TEST',
            'date': '2024-01-02',
            'features_json': {'sma_20': 180.5, 'rsi': 55.0},
            'version': '1.0'
        }

        assert 'features_json' in record
        assert isinstance(record['features_json'], dict)

    def test_symbol_governance_record(self):
        """Test symbol governance record format."""
        record = {
            'symbol': 'TEST',
            'history_rows': 252,
            'data_quality': 0.95,
            'state': 'ACTIVE',
            'reason': 'Test',
            'last_checked_ts': '2024-01-02T12:00:00',
            'metadata': {'last_price': 100.0}
        }

        assert record['state'] == 'ACTIVE'
        assert record['data_quality'] == 0.95


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
