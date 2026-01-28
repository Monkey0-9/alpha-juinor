"""
Unit tests for market data loader.
"""

import pytest
import sqlite3
import pandas as pd
import numpy as np
import os
from pathlib import Path


class TestMarketDataLoader:
    """Test market data loading functionality."""

    @pytest.fixture
    def test_db_path(self, tmp_path):
        """Create a temporary test database."""
        db_path = tmp_path / "test_trading.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Create price_history table
        cursor.execute("""
            CREATE TABLE price_history (
                symbol TEXT,
                date TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER
            )
        """)

        # Insert test data for AAPL (300 bars)
        dates = pd.date_range('2023-01-01', periods=300, freq='D')
        for i, date in enumerate(dates):
            cursor.execute("""
                INSERT INTO price_history (symbol, date, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, ('AAPL', date.strftime('%Y-%m-%d'), 150+i*0.1, 151+i*0.1, 149+i*0.1, 150.5+i*0.1, 1000000))

        conn.commit()
        conn.close()

        return str(db_path)

    def test_load_252_bars_for_symbol(self, test_db_path):
        """Test loading exactly 252 recent bars for a symbol."""
        conn = sqlite3.connect(test_db_path)

        query = """
            SELECT date, close
            FROM price_history
            WHERE symbol = 'AAPL'
            ORDER BY date DESC
            LIMIT 252
        """
        df = pd.read_sql_query(query, conn)
        conn.close()

        assert len(df) == 252
        assert 'close' in df.columns
        assert 'date' in df.columns

    def test_insufficient_bars_detection(self, test_db_path):
        """Test detection of symbols with insufficient bars."""
        conn = sqlite3.connect(test_db_path)
        cursor = conn.cursor()

        # Add symbol with only 100 bars
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        for i, date in enumerate(dates):
            cursor.execute("""
                INSERT INTO price_history (symbol, date, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, ('TSLA', date.strftime('%Y-%m-%d'), 200+i*0.1, 201+i*0.1, 199+i*0.1, 200.5+i*0.1, 500000))

        conn.commit()

        # Check TSLA has < 252 bars
        query = """
            SELECT COUNT(*) as count
            FROM price_history
            WHERE symbol = 'TSLA'
        """
        cursor.execute(query)
        count = cursor.fetchone()[0]

        conn.close()

        assert count == 100
        assert count < 252

    def test_market_data_not_empty(self, test_db_path):
        """Test that market data is not empty for valid symbols."""
        conn = sqlite3.connect(test_db_path)

        query = """
            SELECT date, close
            FROM price_history
            WHERE symbol = 'AAPL'
            ORDER BY date DESC
            LIMIT 252
        """
        df = pd.read_sql_query(query, conn)
        conn.close()

        assert not df.empty
        assert df['close'].notnull().all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
