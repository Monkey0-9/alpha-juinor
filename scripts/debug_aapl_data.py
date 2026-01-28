from database.manager import DatabaseManager
import pandas as pd
from data.processors.features import compute_features_for_symbol

db = DatabaseManager()
df = db.get_daily_prices("AAPL")
print(f"Total rows for AAPL: {len(df)}")
if not df.empty:
    print(f"NaN counts per column BEFORE rename:\n{df.isna().sum()}")

    # Canonical Mapping
    df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)

    # Prepare target
    df['target'] = df['Close'].pct_change(fill_method=None).shift(-1)
    print(f"NaNs in target: {df['target'].isna().sum()}")

    # Check all columns for NaNs
    print(f"NaN counts per column AFTER target:\n{df.isna().sum()}")

    # Dropna only on target first
    df_clean = df.dropna(subset=['target'])
    print(f"Rows after dropping NaNs in target: {len(df_clean)}")

    # Check if there are columns that are ALL NaN
    all_nan_cols = [col for col in df_clean.columns if df_clean[col].isna().all()]
    print(f"Columns that are ALL NaN: {all_nan_cols}")
