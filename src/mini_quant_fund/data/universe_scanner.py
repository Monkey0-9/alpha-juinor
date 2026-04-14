"""
Full Market Universe Scanner - Trade All Stocks.

Features:
- Scan entire market (8000+ US stocks)
- Filter by liquidity, market cap, sector
- Daily universe refresh
- Alpaca/IEX integration for full list
"""

import logging
import os
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from datetime import datetime
import time

logger = logging.getLogger(__name__)


@dataclass
class Asset:
    """Tradable asset definition."""
    symbol: str
    name: str
    exchange: str
    asset_class: str
    tradable: bool
    shortable: bool
    marginable: bool
    market_cap: Optional[float] = None
    sector: Optional[str] = None
    avg_volume: Optional[float] = None


class UniverseScanner:
    """
    Scan entire market for tradable assets.

    Sources:
    - Alpaca API (all assets)
    - IEX Cloud (fundamentals)
    - Yahoo Finance (backup)
    """

    def __init__(
        self,
        min_price: float = 5.0,
        max_price: float = 10000.0,
        min_volume: float = 100000,
        min_market_cap: float = 100_000_000,
        exclude_otc: bool = True
    ):
        self.min_price = min_price
        self.max_price = max_price
        self.min_volume = min_volume
        self.min_market_cap = min_market_cap
        self.exclude_otc = exclude_otc

        # Cached universe
        self.universe: Dict[str, Asset] = {}
        self.last_refresh: Optional[datetime] = None

        # Sector weights for diversification
        self.sector_caps = {
            "Technology": 0.25,
            "Healthcare": 0.15,
            "Financials": 0.15,
            "Consumer Discretionary": 0.10,
            "Communication Services": 0.10,
            "Industrials": 0.08,
            "Consumer Staples": 0.05,
            "Energy": 0.05,
            "Materials": 0.03,
            "Utilities": 0.02,
            "Real Estate": 0.02
        }

    def refresh_universe(self) -> int:
        """Refresh entire market universe from Alpaca."""
        try:
            import alpaca_trade_api as tradeapi

            api_key = os.getenv("ALPACA_API_KEY")
            api_secret = os.getenv("ALPACA_SECRET_KEY")
            base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

            if not api_key or not api_secret:
                logger.warning("Alpaca credentials not found, using sample universe")
                return self._load_sample_universe()

            api = tradeapi.REST(api_key, api_secret, base_url)

            # Get ALL assets
            assets = api.list_assets(status="active")

            count = 0
            for asset in assets:
                # Filter by exchange
                if self.exclude_otc and asset.exchange == "OTC":
                    continue

                # Only US equities
                if asset.asset_class != "us_equity":
                    continue

                # Must be tradable
                if not asset.tradable:
                    continue

                self.universe[asset.symbol] = Asset(
                    symbol=asset.symbol,
                    name=asset.name,
                    exchange=asset.exchange,
                    asset_class=asset.asset_class,
                    tradable=asset.tradable,
                    shortable=asset.shortable,
                    marginable=asset.marginable
                )
                count += 1

            self.last_refresh = datetime.utcnow()
            logger.info(f"Loaded {count} assets from Alpaca")

            return count

        except Exception as e:
            logger.error(f"Failed to refresh universe: {e}")
            return self._load_sample_universe()

    def _load_sample_universe(self) -> int:
        """Load sample universe if API fails."""
        # Major US stocks - comprehensive list
        sample_symbols = [
            # FAANG + Big Tech
            "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NVDA", "TSLA",
            "NFLX", "AMD", "INTC", "CRM", "ORCL", "ADBE", "CSCO", "AVGO",
            "QCOM", "TXN", "MU", "AMAT", "LRCX", "KLAC", "SNPS", "CDNS",

            # Financials
            "JPM", "BAC", "WFC", "C", "GS", "MS", "BLK", "SCHW", "AXP",
            "V", "MA", "PYPL", "SQ", "COIN", "HOOD", "USB", "PNC",

            # Healthcare
            "JNJ", "UNH", "PFE", "ABBV", "MRK", "LLY", "BMY", "AMGN",
            "GILD", "BIIB", "REGN", "VRTX", "MRNA", "CVS", "CI", "HUM",

            # Consumer
            "WMT", "COST", "TGT", "HD", "LOW", "NKE", "SBUX", "MCD",
            "KO", "PEP", "PG", "CL", "KMB", "MDLZ", "KHC", "GIS",

            # Industrials
            "CAT", "DE", "BA", "RTX", "LMT", "GD", "NOC", "HON",
            "MMM", "GE", "UPS", "FDX", "UNP", "CSX", "NSC",

            # Energy
            "XOM", "CVX", "COP", "SLB", "EOG", "PXD", "OXY", "VLO",
            "MPC", "PSX", "HAL", "BKR", "DVN", "FANG",

            # Communication
            "T", "VZ", "CMCSA", "DIS", "NFLX", "TMUS", "CHTR",

            # Materials
            "LIN", "APD", "SHW", "ECL", "DD", "NEM", "FCX",

            # Utilities
            "NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE",

            # Real Estate REITs
            "PLD", "AMT", "CCI", "EQIX", "PSA", "SPG", "O",

            # More Tech
            "NOW", "SNOW", "DDOG", "ZS", "NET", "CRWD", "PANW",
            "OKTA", "WDAY", "TEAM", "DOCU", "ZM", "U", "PLTR",
            "PATH", "MDB", "ESTC", "CFLT", "S", "BILL",

            # EV & Clean Energy
            "RIVN", "LCID", "NIO", "XPEV", "LI", "ENPH", "SEDG",
            "FSLR", "RUN", "PLUG", "CHPT",

            # Retail & E-commerce
            "SHOP", "ETSY", "EBAY", "W", "CHWY", "PTON",

            # Gaming & Entertainment
            "ATVI", "EA", "TTWO", "RBLX", "DKNG", "PENN",

            # Airlines & Travel
            "DAL", "UAL", "AAL", "LUV", "ABNB", "BKNG", "EXPE",

            # Biotech
            "ILMN", "DXCM", "ISRG", "TMO", "DHR", "A", "BIO",

            # More Financials
            "TFC", "FITB", "KEY", "CFG", "RF", "HBAN", "ZION",
            "CMA", "FRC", "SIVB", "WAL", "PACW",

            # Crypto-related
            "MARA", "RIOT", "MSTR", "SI", "SBNY",

            # SPACs / Recent IPOs
            "SOFI", "UPST", "AFRM", "HOOD", "DOCS", "YOU",
        ]

        for symbol in sample_symbols:
            self.universe[symbol] = Asset(
                symbol=symbol,
                name=symbol,
                exchange="NASDAQ",
                asset_class="us_equity",
                tradable=True,
                shortable=True,
                marginable=True
            )

        self.last_refresh = datetime.utcnow()
        logger.info(f"Loaded {len(sample_symbols)} sample assets")

        return len(sample_symbols)

    def get_tradable_symbols(self) -> List[str]:
        """Get all tradable symbols."""
        if not self.universe:
            self.refresh_universe()
        return list(self.universe.keys())

    def filter_by_criteria(
        self,
        prices: Dict[str, float],
        volumes: Dict[str, float],
        market_caps: Optional[Dict[str, float]] = None
    ) -> List[str]:
        """Filter universe by trading criteria."""
        filtered = []

        for symbol, asset in self.universe.items():
            price = prices.get(symbol, 0)
            volume = volumes.get(symbol, 0)
            market_cap = (market_caps or {}).get(symbol, self.min_market_cap)

            # Apply filters
            if price < self.min_price or price > self.max_price:
                continue
            if volume < self.min_volume:
                continue
            if market_cap < self.min_market_cap:
                continue

            filtered.append(symbol)

        return filtered

    def get_sector_allocation(
        self,
        symbols: List[str],
        sectors: Dict[str, str]
    ) -> Dict[str, List[str]]:
        """Group symbols by sector."""
        allocation = {}

        for symbol in symbols:
            sector = sectors.get(symbol, "Unknown")
            if sector not in allocation:
                allocation[sector] = []
            allocation[sector].append(symbol)

        return allocation


# Global singleton
_scanner: Optional[UniverseScanner] = None


def get_universe_scanner() -> UniverseScanner:
    """Get or create global universe scanner."""
    global _scanner
    if _scanner is None:
        _scanner = UniverseScanner()
    return _scanner
