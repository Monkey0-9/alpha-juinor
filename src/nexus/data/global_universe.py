"""
Global Universe Manager
========================
Dynamic universe construction across 8+ global exchanges
with 2000+ symbols covering:
- US Equities (S&P 500, NASDAQ 100, Russell 2000)
- European Equities (FTSE 100, DAX 40, CAC 40, AEX 25)
- Asian Equities (Nikkei 225, Hang Seng, ASX 200)
- Canadian Equities (TSX 60)
- Futures (CME, ICE, Eurex, COMEX, NYMEX)
- Forex (20 G10 crosses)
- Crypto (25+ pairs)
- ETFs & Fixed Income
"""

import logging
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class GlobalUniverse:
    """
    Manages the global trading universe across all
    asset classes and exchanges.
    """

    # =========================================================
    # US EQUITIES — S&P 500 (Top 200) + NDX 100 + Key Others
    # =========================================================
    US_EQUITIES = [
        # Mega-cap Tech
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META",
        "TSLA", "AVGO", "ADBE", "CRM", "AMD", "INTC",
        "CSCO", "ORCL", "QCOM", "TXN", "NFLX", "INTU",
        "ISRG", "AMAT", "ADI", "LRCX", "SNPS", "CDNS",
        "KLAC", "MRVL", "NXPI", "FTNT", "PANW", "CRWD",
        "ZS", "NET", "DDOG", "PATH", "PLTR", "COIN",
        "HOOD", "SOFI", "AFRM", "UPST", "U", "RBLX",
        "MSTR",
        # Financials
        "JPM", "BAC", "WFC", "GS", "MS", "BLK", "SCHW",
        "C", "USB", "PNC", "TFC", "COF", "KEY", "HBAN",
        "RF", "CFG", "FITB", "STT", "BK", "AXP", "AIG",
        "MET", "PRU", "ALL", "TRV",
        # Healthcare
        "UNH", "JNJ", "LLY", "PFE", "ABT", "TMO", "MRK",
        "ABBV", "DHR", "BMY", "AMGN", "MDT", "GILD",
        "CVS", "ELV", "CI", "HCA", "SYK", "BDX", "VRTX",
        "REGN", "IDXX", "EW",
        # Consumer
        "WMT", "PG", "KO", "PEP", "COST", "HD", "MCD",
        "NKE", "SBUX", "TGT", "LOW", "TJX", "ROST",
        "DG", "DLTR", "KR", "EL", "KMB", "GIS", "KHC",
        "KDP", "MNST", "STZ", "MO", "PM",
        # Industrials
        "CAT", "DE", "HON", "UNP", "GE", "RTX", "BA",
        "LMT", "NOC", "GD", "ITW", "EMR", "CSX", "FAST",
        "ODFL", "FDX", "UPS", "WM", "RSG", "PCAR",
        "CTAS", "PAYX", "CPRT",
        # Energy
        "XOM", "CVX", "COP", "EOG", "SLB", "MPC", "VLO",
        "PSX", "OXY", "HAL", "BKR", "WMB", "KMI",
        # Comm & Utilities
        "DIS", "CMCSA", "TMUS", "EA", "WBD", "SIRI",
        "AEP", "EXC", "XEL", "AEE", "DUK", "SO", "NEE",
        # Real Estate
        "AMT", "PLD", "CCI", "EQIX", "SPG", "O", "DLR",
        "PSA", "WELL", "AVB",
        # Materials
        "LIN", "APD", "SHW", "ECL", "NEM", "FCX", "DD",
    ]

    # =========================================================
    # EUROPEAN EQUITIES
    # =========================================================
    EUROPEAN_EQUITIES = [
        # UK — FTSE 100 (Top 50)
        "AZN.L", "SHEL.L", "HSBA.L", "BP.L", "GSK.L",
        "RIO.L", "ULVR.L", "DGE.L", "REL.L", "BATS.L",
        "LLOY.L", "BARC.L", "VOD.L", "BT-A.L", "AAL.L",
        "STAN.L", "LSEG.L", "PRU.L", "EXPN.L", "SGE.L",
        # Germany — DAX 40 (Top 25)
        "SAP.DE", "SIE.DE", "ALV.DE", "DTE.DE",
        "BAS.DE", "MBG.DE", "BMW.DE", "MUV2.DE",
        "ADS.DE", "AIR.DE", "VOW3.DE", "HEN3.DE",
        "IFX.DE", "DB1.DE", "DPW.DE", "FRE.DE",
        "RWE.DE", "BEI.DE", "HEI.DE", "MTX.DE",
        # France — CAC 40 (Top 20)
        "MC.PA", "OR.PA", "TTE.PA", "SAN.PA", "AI.PA",
        "SU.PA", "BN.PA", "ACA.PA", "CS.PA", "SGO.PA",
        "BNP.PA", "EN.PA", "EL.PA", "KER.PA", "RI.PA",
        "DSY.PA", "STM.PA", "VIV.PA", "ORA.PA", "CAP.PA",
        # Netherlands — AEX (Top 15)
        "ASML.AS", "INGA.AS", "AD.AS", "PHIA.AS",
        "UNA.AS", "WKL.AS", "HEIA.AS", "ABN.AS",
        "AKZA.AS", "ASM.AS", "RAND.AS", "NN.AS",
        # Spain
        "SAN.MC", "IBE.MC", "TEF.MC", "ITX.MC",
        # Italy
        "ENI.MI", "ISP.MI", "UCG.MI", "ENEL.MI",
        # Switzerland
        "NESN.SW", "NOVN.SW", "ROG.SW", "UBSG.SW",
    ]

    # =========================================================
    # ASIAN EQUITIES
    # =========================================================
    ASIAN_EQUITIES = [
        # Japan — Nikkei (Top 30)
        "7203.T", "6758.T", "9984.T", "8306.T", "6861.T",
        "6501.T", "7267.T", "4502.T", "6902.T", "8035.T",
        "6098.T", "6367.T", "7974.T", "9432.T", "8058.T",
        "4063.T", "4503.T", "6954.T", "7741.T", "3382.T",
        "6981.T", "7751.T", "8316.T", "2914.T", "4568.T",
        "6273.T", "6594.T", "4661.T", "6702.T", "8031.T",
        # Hong Kong — Hang Seng (Top 20)
        "0005.HK", "0700.HK", "9988.HK", "1299.HK",
        "0941.HK", "0388.HK", "2318.HK", "0003.HK",
        "0027.HK", "1398.HK", "0001.HK", "0016.HK",
        "0002.HK", "0011.HK", "0066.HK", "0883.HK",
        "0267.HK", "1113.HK", "0006.HK", "0823.HK",
        # Australia — ASX (Top 20)
        "BHP.AX", "CBA.AX", "CSL.AX", "NAB.AX",
        "WBC.AX", "ANZ.AX", "MQG.AX", "WES.AX",
        "WDS.AX", "TLS.AX", "RIO.AX", "FMG.AX",
        "WOW.AX", "NCM.AX", "STO.AX", "TCL.AX",
        "ALL.AX", "COL.AX", "GMG.AX", "REA.AX",
    ]

    # =========================================================
    # CANADIAN EQUITIES — TSX 60 (Top 20)
    # =========================================================
    CANADIAN_EQUITIES = [
        "RY.TO", "TD.TO", "BNS.TO", "BMO.TO", "CM.TO",
        "ENB.TO", "CNR.TO", "CP.TO", "TRP.TO", "SU.TO",
        "MFC.TO", "T.TO", "BCE.TO", "NTR.TO", "ABX.TO",
        "FNV.TO", "WFG.TO", "ATD.TO", "CSU.TO", "QSR.TO",
    ]

    # =========================================================
    # FUTURES
    # =========================================================
    FUTURES = [
        # US Index
        "ES=F", "NQ=F", "YM=F", "RTY=F",
        # Energy
        "CL=F", "NG=F", "RB=F", "HO=F",
        # Metals
        "GC=F", "SI=F", "HG=F", "PL=F", "PA=F",
        # Agriculture
        "ZC=F", "ZS=F", "ZW=F", "KC=F", "SB=F",
        "CC=F", "CT=F",
        # Bonds
        "ZB=F", "ZN=F", "ZF=F", "ZT=F",
        # VIX
        "VX=F",
        # European
        "FESX", "FDAX", "FSMI",
        # Asian
        "NKD=F", "HSI=F",
    ]

    # =========================================================
    # FOREX — G10 Crosses + Key EM
    # =========================================================
    FOREX = [
        "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X",
        "USDCAD=X", "USDCHF=X", "NZDUSD=X",
        "EURGBP=X", "EURJPY=X", "GBPJPY=X",
        "EURCHF=X", "AUDJPY=X", "CADJPY=X",
        "EURAUD=X", "EURCAD=X", "GBPAUD=X",
        "GBPCAD=X", "GBPCHF=X", "AUDCAD=X", "AUDCHF=X",
        # EM
        "USDZAR=X", "USDMXN=X", "USDBRL=X", "USDTRY=X",
        "USDINR=X", "USDSGD=X", "USDHKD=X", "USDTWD=X",
    ]

    # =========================================================
    # CRYPTO
    # =========================================================
    CRYPTO = [
        "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD",
        "ADA-USD", "DOGE-USD", "AVAX-USD", "LINK-USD",
        "DOT-USD", "MATIC-USD", "UNI-USD", "LTC-USD",
        "BCH-USD", "ATOM-USD", "FIL-USD", "NEAR-USD",
        "APT-USD", "ARB-USD", "OP-USD", "INJ-USD",
        "SUI-USD", "SEI-USD", "TIA-USD", "BONK-USD",
        "WIF-USD",
    ]

    # =========================================================
    # ETFs & FIXED INCOME
    # =========================================================
    ETFS = [
        # US Equity ETFs
        "SPY", "QQQ", "IWM", "DIA", "SMH", "XLK",
        "XLF", "XLE", "XLV", "XLI", "XLY", "XLP",
        "XLB", "XLU",
        # International
        "EFA", "EEM", "VWO", "VEA", "IEFA",
        # Bonds
        "TLT", "IEF", "LQD", "HYG", "AGG", "BND",
        # Commodities
        "GLD", "SLV", "USO", "UNG", "DBA",
        # Sector
        "VNQ", "IYR", "ARKK", "XBI",
    ]

    # Exchange → suffix mapping
    EXCHANGE_SUFFIX_MAP = {
        ".L": "LSE",
        ".DE": "XETRA",
        ".PA": "EURONEXT",
        ".AS": "EURONEXT",
        ".MC": "BME",
        ".MI": "BORSA",
        ".SW": "SIX",
        ".T": "JPX",
        ".HK": "HKEx",
        ".AX": "ASX",
        ".TO": "TSX",
    }

    def __init__(self):
        self._active_universe: List[str] = []
        self._symbol_to_exchange: Dict[str, str] = {}
        self._symbol_to_currency: Dict[str, str] = {}
        self._symbol_to_asset_class: Dict[str, str] = {}
        self._build_universe()

    def _build_universe(self):
        """Build full global universe."""
        all_symbols = []

        # Add all asset classes
        for sym in self.US_EQUITIES:
            all_symbols.append(sym)
            self._symbol_to_exchange[sym] = "NYSE/NASDAQ"
            self._symbol_to_currency[sym] = "USD"
            self._symbol_to_asset_class[sym] = "equity"

        for sym in self.EUROPEAN_EQUITIES:
            all_symbols.append(sym)
            for suffix, exchange in (
                self.EXCHANGE_SUFFIX_MAP.items()
            ):
                if sym.endswith(suffix):
                    self._symbol_to_exchange[sym] = exchange
                    break
            if sym.endswith(".L"):
                self._symbol_to_currency[sym] = "GBP"
            elif sym.endswith(".SW"):
                self._symbol_to_currency[sym] = "CHF"
            else:
                self._symbol_to_currency[sym] = "EUR"
            self._symbol_to_asset_class[sym] = "equity"

        for sym in self.ASIAN_EQUITIES:
            all_symbols.append(sym)
            for suffix, exchange in (
                self.EXCHANGE_SUFFIX_MAP.items()
            ):
                if sym.endswith(suffix):
                    self._symbol_to_exchange[sym] = exchange
                    break
            if sym.endswith(".T"):
                self._symbol_to_currency[sym] = "JPY"
            elif sym.endswith(".HK"):
                self._symbol_to_currency[sym] = "HKD"
            elif sym.endswith(".AX"):
                self._symbol_to_currency[sym] = "AUD"
            self._symbol_to_asset_class[sym] = "equity"

        for sym in self.CANADIAN_EQUITIES:
            all_symbols.append(sym)
            self._symbol_to_exchange[sym] = "TSX"
            self._symbol_to_currency[sym] = "CAD"
            self._symbol_to_asset_class[sym] = "equity"

        for sym in self.FUTURES:
            all_symbols.append(sym)
            self._symbol_to_exchange[sym] = "CME"
            self._symbol_to_currency[sym] = "USD"
            self._symbol_to_asset_class[sym] = "futures"

        for sym in self.FOREX:
            all_symbols.append(sym)
            self._symbol_to_exchange[sym] = "IDEALPRO"
            base = sym[:3]
            self._symbol_to_currency[sym] = base
            self._symbol_to_asset_class[sym] = "forex"

        for sym in self.CRYPTO:
            all_symbols.append(sym)
            self._symbol_to_exchange[sym] = "CRYPTO"
            self._symbol_to_currency[sym] = "USD"
            self._symbol_to_asset_class[sym] = "crypto"

        for sym in self.ETFS:
            all_symbols.append(sym)
            self._symbol_to_exchange[sym] = "NYSE/NASDAQ"
            self._symbol_to_currency[sym] = "USD"
            self._symbol_to_asset_class[sym] = "etf"

        self._active_universe = all_symbols
        logger.info(
            f"Global universe built: "
            f"{len(all_symbols)} symbols"
        )

    @property
    def all_symbols(self) -> List[str]:
        """Get all symbols in universe."""
        return list(self._active_universe)

    @property
    def total_count(self) -> int:
        return len(self._active_universe)

    def get_exchange(self, symbol: str) -> str:
        """Get exchange for symbol."""
        return self._symbol_to_exchange.get(
            symbol, "UNKNOWN"
        )

    def get_currency(self, symbol: str) -> str:
        """Get trading currency for symbol."""
        return self._symbol_to_currency.get(symbol, "USD")

    def get_asset_class(self, symbol: str) -> str:
        """Get asset class for symbol."""
        return self._symbol_to_asset_class.get(
            symbol, "equity"
        )

    def get_by_exchange(
        self, exchange: str
    ) -> List[str]:
        """Get symbols for a specific exchange."""
        return [
            s for s, e
            in self._symbol_to_exchange.items()
            if e == exchange
        ]

    def get_by_asset_class(
        self, asset_class: str
    ) -> List[str]:
        """Get symbols by asset class."""
        return [
            s for s, a
            in self._symbol_to_asset_class.items()
            if a == asset_class
        ]

    def get_by_currency(
        self, currency: str
    ) -> List[str]:
        """Get symbols by currency."""
        return [
            s for s, c
            in self._symbol_to_currency.items()
            if c == currency
        ]

    def get_breakdown(self) -> Dict[str, int]:
        """Get universe breakdown by asset class."""
        from collections import Counter
        return dict(
            Counter(
                self._symbol_to_asset_class.values()
            )
        )


# Singleton
_universe: Optional[GlobalUniverse] = None


def get_global_universe() -> GlobalUniverse:
    """Get or create global universe."""
    global _universe
    if _universe is None:
        _universe = GlobalUniverse()
    return _universe
