
import yfinance as yf
import pandas as pd
from datetime import datetime

with open("result.txt", "w") as f:
    f.write("Testing yfinance download...\n")
    start_date = "2020-01-01"
    end_date = "2026-01-20"
    symbol = "AAPL"

    f.write(f"Requesting {symbol} from {start_date} to {end_date}\n")
    df = yf.download(symbol, start=start_date, end=end_date, progress=False, auto_adjust=True)

    f.write(f"Rows returned: {len(df)}\n")
    if not df.empty:
        f.write(f"Min Date: {df.index.min()}\n")
        f.write(f"Max Date: {df.index.max()}\n")
    else:
        f.write("Empty DataFrame returned.\n")
