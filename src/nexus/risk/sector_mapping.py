from typing import Dict, List

# risk/sector_mapping.py

# Mapping of Institutional Symbols to Sectors and their respective Hedging ETFs
# Based on holdings in golden_config.yaml

SECTOR_MAP = {
    # Technology (XLK)
    "AAPL": "Technology", "ADSK": "Technology", "ADI": "Technology", "AMD": "Technology",
    "ANSS": "Technology", "AVGO": "Technology", "CDNS": "Technology", "CSCO": "Technology",
    "FTNT": "Technology", "INTC": "Technology", "KLAC": "Technology", "LRCX": "Technology",
    "MCHP": "Technology", "MSFT": "Technology", "NVDA": "Technology", "NXPI": "Technology",
    "ORCL": "Technology", "PANW": "Technology", "QCOM": "Technology", "SNPS": "Technology",
    "TXN": "Technology", "ZS": "Technology",

    # Financials (XLF)
    "AXP": "Financials", "BAC": "Financials", "BK": "Financials", "BLK": "Financials",
    "C": "Financials", "COF": "Financials", "GS": "Financials", "JPM": "Financials",
    "MA": "Financials", "MS": "Financials", "PNC": "Financials", "SCHW": "Financials",
    "USB": "Financials", "V": "Financials", "WFC": "Financials",

    # Healthcare (XLV)
    "ABBV": "Healthcare", "ABT": "Healthcare", "AMGN": "Healthcare", "BDX": "Healthcare",
    "BMY": "Healthcare", "CI": "Healthcare", "CVS": "Healthcare", "DHR": "Healthcare",
    "ELV": "Healthcare", "GILD": "Healthcare", "JNJ": "Healthcare", "LLY": "Healthcare",
    "MRK": "Healthcare", "PFE": "Healthcare", "REGN": "Healthcare", "TMO": "Healthcare",
    "UNH": "Healthcare", "VRTX": "Healthcare", "ZTS": "Healthcare",

    # Energy (XLE)
    "BKR": "Energy", "COP": "Energy", "CVX": "Energy", "EOG": "Energy", "HAL": "Energy",
    "HES": "Energy", "MPC": "Energy", "OXY": "Energy", "PSX": "Energy", "SLB": "Energy",
    "XOM": "Energy",

    # Consumer Discretionary (XLY)
    "AMZN": "Consumer Discretionary", "BKNG": "Consumer Discretionary", "COST": "Consumer Discretionary",
    "HD": "Consumer Discretionary", "LOW": "Consumer Discretionary", "MCD": "Consumer Discretionary",
    "NKE": "Consumer Discretionary", "ORLY": "Consumer Discretionary", "SBUX": "Consumer Discretionary",
    "TGT": "Consumer Discretionary", "TJX": "Consumer Discretionary", "TSLA": "Consumer Discretionary",

    # Comm Services (XLC)
    "GOOGL": "Comm Services", "META": "Comm Services", "NFLX": "Comm Services", "TMUS": "Comm Services",
    "WBD": "Comm Services",

    # Crypto (BITO / Proxy)
    "BTC-USD": "Crypto", "ETH-USD": "Crypto", "SOL-USD": "Crypto", "ADA-USD": "Crypto",
    "AVAX-USD": "Crypto", "DOGE-USD": "Crypto", "DOT-USD": "Crypto", "XRP-USD": "Crypto",
    "LINK-USD": "Crypto", "MATIC-USD": "Crypto", "LTC-USD": "Crypto", "UNI-USD": "Crypto",
}

HEDGE_ETFS = {
    "Technology": "XLK",
    "Financials": "XLF",
    "Healthcare": "XLV",
    "Energy": "XLE",
    "Consumer Discretionary": "XLY",
    "Comm Services": "XLC",
    "Crypto": "BITO", # Bitcoin ETF Proxy
}

def get_sector(symbol: str) -> str:
    return SECTOR_MAP.get(symbol, "Unknown")

def get_hedge_etf(sector: str) -> str:
    return HEDGE_ETFS.get(sector, "SPY") # Default to SPY if unknown
