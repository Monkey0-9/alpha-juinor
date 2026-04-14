
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

ticker = "AAPL"
start_date = (datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d')
end_date = datetime.now().strftime('%Y-%m-%d')

print(f"Fetching {ticker} from {start_date} to {end_date}")
df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)

print(f"Columns: {df.columns}")
print(f"Index: {df.index}")
print(f"Empty: {df.empty}")
if not df.empty:
    print(df.head())
    print(f"Is MultiIndex: {isinstance(df.columns, pd.MultiIndex)}")
