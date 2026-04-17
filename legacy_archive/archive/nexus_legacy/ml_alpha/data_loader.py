"""
Data Loader for ML Alpha
Handles loading of historical data for training, replacing mock data.
"""
import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)

def load_training_data(data_path: str) -> list[pd.DataFrame]:
    """
    Loads historical data from CSVs in the specified directory.
    Expects standard OHLCV format with 'timestamp' or 'date' column.
    """
    if not os.path.exists(data_path):
        logger.warning(f"Data path {data_path} does not exist.")
        return []

    dfs = []
    files = [f for f in os.listdir(data_path) if f.endswith('.csv')]

    if not files:
        logger.warning(f"No CSV files found in {data_path}")
        return []

    logger.info(f"Found {len(files)} data files. Loading...")

    for f in files:
        try:
            file_path = os.path.join(data_path, f)
            df = pd.read_csv(file_path)

            # Standardize columns
            df.columns = [c.lower() for c in df.columns]

            # Ensure required columns exist
            required = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required):
                logger.debug(f"Skipping {f}: Missing required columns.")
                continue

            dfs.append(df)
        except Exception as e:
            logger.error(f"Error loading {f}: {e}")

    logger.info(f"Successfully loaded {len(dfs)} dataframes.")
    return dfs
