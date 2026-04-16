"""
PACK 5A: GLOBAL MARKET HOURS ENFORCEMENT
Supports trading in 7+ time zones with intelligent market hours management
"""

import logging
from datetime import datetime, time, timedelta
from typing import Dict, List, Tuple

import pytz

logger = logging.getLogger(__name__)


class GlobalMarketHours:
    """Manage market hours across 7 global markets"""

    # Market definitions: timezone, open_time, close_time
    MARKET_DEFINITIONS = {
        "NYSE": {
            "timezone": "US/Eastern",
            "open": "09:30",
            "close": "16:00",
            "pre_market": "04:00",
            "post_market": "20:00",
        },
        "NASDAQ": {
            "timezone": "US/Eastern",
            "open": "09:30",
            "close": "16:00",
            "pre_market": "04:00",
            "post_market": "20:00",
        },
        "LSE": {
            "timezone": "Europe/London",
            "open": "08:00",
            "close": "16:30",
            "pre_market": "07:00",
            "post_market": "17:00",
        },
        "EURONEXT": {
            "timezone": "Europe/Paris",
            "open": "09:00",
            "close": "17:30",
            "pre_market": "08:00",
            "post_market": "17:45",
        },
        "JPX": {
            "timezone": "Asia/Tokyo",
            "open": "09:00",
            "close": "15:00",
            "pre_market": "08:30",
            "post_market": "15:15",
        },
        "HKEx": {
            "timezone": "Asia/Hong_Kong",
            "open": "09:30",
            "close": "16:00",
            "pre_market": "09:00",
            "post_market": "16:30",
        },
        "ASX": {
            "timezone": "Australia/Sydney",
            "open": "10:00",
            "close": "16:00",
            "pre_market": "09:00",
            "post_market": "16:30",
        },
        "TSX": {
            "timezone": "America/Toronto",
            "open": "09:30",
            "close": "16:00",
            "pre_market": "04:00",
            "post_market": "20:00",
        },
    }

    # Symbol to market mapping
    SYMBOL_MARKET_MAP = {
        # US
        "SPY": "NYSE",
        "AAPL": "NASDAQ",
        "TSLA": "NASDAQ",
        # UK
        "ASML.AS": "EURONEXT",
        "SHELL": "LSE",
        # Japan
        "9984.T": "JPX",  # Softbank
        # Hong Kong
        "0005.HK": "HKEx",  # HSBC
        # Australia
        "CBA": "ASX",  # Commonwealth Bank
        # Canada
        "TD": "TSX",  # TD Bank
    }

    def __init__(self):
        self.market_definitions = self.MARKET_DEFINITIONS
        self.symbol_market_map = self.SYMBOL_MARKET_MAP

    def get_market_for_symbol(self, symbol: str) -> str:
        """Get primary market for a symbol"""
        # Try exact match first
        if symbol in self.symbol_market_map:
            return self.symbol_market_map[symbol]

        # Infer from suffix
        if ".T" in symbol:
            return "JPX"
        elif ".HK" in symbol:
            return "HKEx"
        elif ".AS" in symbol:
            return "EURONEXT"
        elif symbol.endswith(".AX"):
            return "ASX"
        else:
            # Default to NYSE/NASDAQ for unrecognized US symbols
            return "NYSE"

    def is_market_open(
        self, symbol: str, extended_hours: bool = False
    ) -> Tuple[bool, str]:
        """Check if market is open for symbol"""
        market = self.get_market_for_symbol(symbol)
        market_def = self.market_definitions[market]

        tz = pytz.timezone(market_def["timezone"])
        now = datetime.now(tz)

        open_time = datetime.strptime(market_def["open"], "%H:%M").time()
        close_time = datetime.strptime(market_def["close"], "%H:%M").time()

        current_time = now.time()

        # Check weekend
        if now.weekday() >= 5:
            return False, f"[{market}] Weekend - market closed"

        if extended_hours:
            pre_time = datetime.strptime(market_def["pre_market"], "%H:%M").time()
            post_time = datetime.strptime(market_def["post_market"], "%H:%M").time()
            is_open = pre_time <= current_time <= post_time
            session = "pre/regular/post"
        else:
            is_open = open_time <= current_time <= close_time
            session = "regular"

        if not is_open:
            return False, f"[{market}] Closed - outside {session} hours"

        return True, f"[{market}] Open ({session} hours)"

    def get_active_markets(self) -> List[str]:
        """Get list of currently open markets"""
        open_markets = []
        for market in self.market_definitions.keys():
            market_def = self.market_definitions[market]
            tz = pytz.timezone(market_def["timezone"])
            now = datetime.now(tz)

            if now.weekday() >= 5:
                continue

            open_time = datetime.strptime(market_def["open"], "%H:%M").time()
            close_time = datetime.strptime(market_def["close"], "%H:%M").time()
            current_time = now.time()

            if open_time <= current_time <= close_time:
                open_markets.append(market)

        return open_markets

    def get_next_market_opening(self) -> Tuple[str, str]:
        """Get next market opening time"""
        markets_by_opening = []

        for market, market_def in self.market_definitions.items():
            tz = pytz.timezone(market_def["timezone"])
            now = datetime.now(tz)

            open_time = datetime.strptime(market_def["open"], "%H:%M").time()
            open_dt = datetime.combine(now.date(), open_time)

            # If market already opened today, next opening is tomorrow
            if now.time() > open_time:
                open_dt += timedelta(days=1)

            markets_by_opening.append((market, open_dt))

        # Sort by opening time
        markets_by_opening.sort(key=lambda x: x[1])
        next_market, next_time = markets_by_opening[0]

        return next_market, str(next_time)

    def get_market_status_report(self) -> Dict:
        """Get status of all 7 markets"""
        report = {}

        for market, market_def in self.market_definitions.items():
            tz = pytz.timezone(market_def["timezone"])
            now = datetime.now(tz)

            open_time = datetime.strptime(market_def["open"], "%H:%M").time()
            close_time = datetime.strptime(market_def["close"], "%H:%M").time()
            current_time = now.time()

            is_open = open_time <= current_time <= close_time and now.weekday() < 5

            report[market] = {
                "timezone": market_def["timezone"],
                "local_time": now.strftime("%H:%M"),
                "is_open": is_open,
                "opens_at": market_def["open"],
                "closes_at": market_def["close"],
            }

        return report


# Global instance
global_market_hours = GlobalMarketHours()


def check_trading_allowed(symbol: str) -> bool:
    """Quick check if trading is allowed for symbol"""
    is_open, msg = global_market_hours.is_market_open(symbol)
    logger.debug(msg)
    return is_open


if __name__ == "__main__":
    gmh = GlobalMarketHours()

    print("Global Market Status:")
    import json

    print(json.dumps(gmh.get_market_status_report(), indent=2))

    print(f"\nActive markets: {gmh.get_active_markets()}")
    print(f"Next opening: {gmh.get_next_market_opening()}")
