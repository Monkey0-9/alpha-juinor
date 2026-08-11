import pytest
import pandas as pd
from datetime import datetime, timezone
import tempfile
from pathlib import Path

from nexus.data.ingestion.base import DataSource
from nexus.data.ingestion.yahoo import YahooFinanceSource
from nexus.data.validation.pipeline import ValidationPipeline
from nexus.data.storage.parquet_writer import ParquetWriter
from nexus.data.storage.duckdb_helper import DuckDBHelper
from unittest.mock import patch

def test_validation_pipeline():
    pipeline = ValidationPipeline()
    
    # Create mock data with duplicates and out-of-order timestamps
    data = {
        'timestamp': [
            datetime(2023, 1, 1, tzinfo=timezone.utc),
            datetime(2023, 1, 2, tzinfo=timezone.utc),
            datetime(2023, 1, 2, tzinfo=timezone.utc), # Duplicate
            datetime(2023, 1, 1, 12, tzinfo=timezone.utc), # Out of order
            datetime(2023, 1, 3, tzinfo=timezone.utc)
        ],
        'symbol': ['AAPL'] * 5,
        'open': [100.0, 101.0, 101.0, 100.5, 102.0],
        'high': [101.0, 102.0, 102.0, 101.5, 103.0],
        'low': [99.0, 100.0, 100.0, 99.5, -1.0], # Negative price
        'close': [100.5, 101.5, 101.5, 101.0, 102.5],
        'volume': [1000, 1100, 1100, 1050, 1200]
    }
    df = pd.DataFrame(data)
    
    passed, clean_df = pipeline.validate_batch(df)
    
    assert not passed # Should fail due to duplicates, order, and negative price
    assert len(clean_df) == 2 # Only day 1 and day 2 (first occurance) survive cleanly
    assert (clean_df['low'] >= 0).all()
    assert clean_df['timestamp'].is_monotonic_increasing

def test_storage_and_retrieval():
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = ParquetWriter(base_path=tmpdir)
        
        data = {
            'timestamp': [datetime(2023, 1, 1, tzinfo=timezone.utc)],
            'symbol': ['MSFT'],
            'open': [200.0], 'high': [205.0], 'low': [199.0], 'close': [202.0], 'volume': [5000]
        }
        df = pd.DataFrame(data)
        
        writer.write_ohlcv(df, source="test_source")
        
        # Check DuckDB
        db = DuckDBHelper(db_path=":memory:", lake_base_path=tmpdir)
        result = db.query("SELECT * FROM market_data WHERE symbol='MSFT'").df()
        
        assert len(result) == 1
        assert result.iloc[0]['close'] == 202.0
        
        db.close()

@patch('nexus.data.ingestion.yahoo.yf.download')
def test_yahoo_ingestion(mock_download):
    # Setup mock dataframe from yfinance
    mock_df = pd.DataFrame({
        'Open': [150.0], 'High': [155.0], 'Low': [149.0], 'Close': [152.0], 
        'Adj Close': [152.0], 'Volume': [1000000]
    }, index=pd.DatetimeIndex([datetime(2023, 1, 1)], name='Date'))
    # yfinance returns MultiIndex columns when group_by='ticker' and multiple tickers, 
    # but for single ticker it might just return standard columns. We'll simulate single ticker.
    mock_download.return_value = mock_df

    source = YahooFinanceSource()
    df = source.fetch_historical_ohlcv(['AAPL'], datetime(2023, 1, 1), datetime(2023, 1, 2))
    
    assert not df.empty
    assert 'symbol' in df.columns
    assert df.iloc[0]['symbol'] == 'AAPL'
    assert df.iloc[0]['close'] == 152.0
    assert df.iloc[0]['timestamp'].tzinfo is not None # Should be timezone aware (UTC)
