import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ParquetWriter:
    """
    Writes market data to the data lake in partitioned Parquet format.
    """

    def __init__(self, base_path: str = "data/lake"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def write_ohlcv(self, df: pd.DataFrame, source: str) -> None:
        """
        Writes a DataFrame of OHLCV data to the data lake.
        Partitions by source, symbol, and year-month.

        Args:
            df: DataFrame containing at least ['timestamp', 'symbol'] and OHLCV cols.
            source: Source identifier (e.g., 'yahoo', 'alpaca').
        """
        if df is None or df.empty:
            logger.warning("Empty DataFrame passed to ParquetWriter. Skipping.")
            return

        required_cols = {
            "timestamp",
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "volume",
        }
        if not required_cols.issubset(df.columns):
            raise ValueError(
                f"DataFrame is missing required columns: {required_cols - set(df.columns)}"
            )

        # Ensure timestamp is datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # We partition by Year-Month for efficient time-series querying
        df["year_month"] = df["timestamp"].dt.strftime("%Y_%m")

        # Create output path: data/lake/source={source}/
        source_path = self.base_path / f"source={source}"

        symbols = df["symbol"].unique()
        for symbol in symbols:
            sym_df = df[df["symbol"] == symbol]

            for ym in sym_df["year_month"].unique():
                ym_df = sym_df[sym_df["year_month"] == ym].copy()
                ym_df = ym_df.drop(columns=["year_month"])

                # Partition path: data/lake/source={source}/symbol={symbol}/{year_month}.parquet
                symbol_path = source_path / f"symbol={symbol}"
                symbol_path.mkdir(parents=True, exist_ok=True)

                file_path = symbol_path / f"{ym}.parquet"

                # If file exists, we append and deduplicate
                if file_path.exists():
                    existing_df = pd.read_parquet(file_path)
                    combined_df = pd.concat([existing_df, ym_df])
                    # Deduplicate by timestamp, keeping the last (newest) record
                    combined_df = combined_df.drop_duplicates(
                        subset=["timestamp"], keep="last"
                    )
                    combined_df = combined_df.sort_values("timestamp").reset_index(
                        drop=True
                    )
                    combined_df.to_parquet(file_path, engine="pyarrow", index=False)
                    logger.debug(f"Updated {file_path}")
                else:
                    ym_df.sort_values("timestamp").to_parquet(
                        file_path, engine="pyarrow", index=False
                    )
                    logger.debug(f"Created {file_path}")
