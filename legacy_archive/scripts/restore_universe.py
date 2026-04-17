
import json
import logging

# Top 249 Assets (Approx)
# Mix of S&P 500, Nasdaq 100, Major ETFs, and Crypto
tickers = [
    # --- INDICES / ETFs (Macro) ---
    "SPY", "QQQ", "IWM", "DIA", "TLT", "GLD", "SLV", "USO", "UNG", "HYG", "LQD", "EEM", "EFA", "XLE", "XLF", "XLK", "XLV", "XLY", "XLP", "XLU", "XLI", "XLB", "IYR", "VNQ", "SMH",

    # --- MEGA CAP TECH ---
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO", "COST", "PEP", "CSCO", "TMUS", "CMCSA", "INTC", "AMD", "QCOM", "NFLX", "TXN", "HON", "AMGN", "INTU", "SBUX", "GILD", "MDLZ", "BKNG", "ADI", "ISRG", "ADP", "VRTX", "REGN", "PYPL", "KLAC", "LRCX", "PANW", "SNPS", "CDNS", "MELI", "MNST", "MAR", "ORLY", "CTAS", "NXPI", "FTNT", "KDP", "ADSK", "XEL", "PAYX", "PCAR", "ROST", "MRVL", "ODFL", "MCHP", "CPRT", "KHC", "IDXX", "AEP", "CTSH", "FAST", "EXC", "BKR", "EA", "CSX", "VRSK", "GEHC", "ALGN", "WBD", "ANSS", "DLTR", "EBAY", "SIRI",

    # --- FINANCE / BANKS ---
    "JPM", "BAC", "WFC", "C", "GS", "MS", "BLK", "SCHW", "AXP", "V", "MA", "COF", "USB", "PNC", "TFC", "BK", "STT", "FITB", "RF", "HBAN", "KEY", "CFG",

    # --- INDUSTRIAL / DEFENSE ---
    "BA", "LMT", "RTX", "GD", "NOC", "GE", "HON", "MMM", "CAT", "DE", "UNP", "UPS", "FDX", "EMR", "ETN", "ITW", "WM", "RSG",

    # --- ENERGY ---
    "XOM", "CVX", "COP", "SLB", "EOG", "PXD", "MPC", "PSX", "VLO", "OXY", "HES", "KMI", "WMB", "HAL",

    # --- HEALTHCARE ---
    "JNJ", "UNH", "LLY", "PFE", "ABBV", "MRK", "TMO", "DHR", "ABT", "BMY", "CVS", "CI", "ELV", "HCA", "SYK", "EW", "ZTS", "BDX",

    # --- CONSUMER ---
    "WMT", "PG", "KO", "HD", "MCD", "NKE", "LOW", "TGT", "TJX", "DG", "CL", "EL", "KMB", "GIS", "SYY", "KR",

    # --- CRYPTO (Major + DeFi) ---
    "BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "XRP-USD", "DOT-USD", "DOGE-USD", "AVAX-USD", "LINK-USD", "MATIC-USD", "UNI-USD", "LTC-USD",

    # --- HIGH GROWTH / MOMENTUM ---
    "PLTR", "AI", "SOFI", "DKNG", "HOOD", "COIN", "MSTR", "RKA", "CVNA", "UPST", "AFRM", "PATH", "U", "RBLX", "NET", "CRWD", "DDOG", "ZS",

    # --- FOREX (In Ticker Format if supported by Yahoo as Ticker=X) ---
    "EURUSD=X", "JPY=X", "GBPUSD=X", "AUDUSD=X", "CAD=X", "CHF=X",

    # --- COMMODITIES ---
    "GC=F", "SI=F", "CL=F", "NG=F", "HG=F", "KC=F", "CC=F", "SB=F"
]

# Dedup and Sort
unique_tickers = sorted(list(set(tickers)))

print(f"Generated {len(unique_tickers)} tickers.")

config = {
    "active_tickers": unique_tickers,
    "last_updated": "2026-01-17 14:00:00",
    "description": "Expanded Institutional Universe (249+ Assets)"
}

with open("configs/universe.json", "w") as f:
    json.dump(config, f, indent=4)
