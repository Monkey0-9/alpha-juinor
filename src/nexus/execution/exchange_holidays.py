"""
Exchange Holiday Calendar
==========================
Comprehensive holiday calendars for 8 global exchanges.
Prevents order submission on market holidays.
"""

import logging
from datetime import date, datetime
from typing import Dict, List, Set

logger = logging.getLogger(__name__)


class ExchangeHolidayCalendar:
    """
    Holiday calendars for 8 global exchanges.
    Updated annually — covers 2025-2027.
    """

    # NYSE / NASDAQ holidays
    US_HOLIDAYS: Dict[int, List[date]] = {
        2025: [
            date(2025, 1, 1),    # New Year's Day
            date(2025, 1, 20),   # MLK Day
            date(2025, 2, 17),   # Presidents' Day
            date(2025, 4, 18),   # Good Friday
            date(2025, 5, 26),   # Memorial Day
            date(2025, 6, 19),   # Juneteenth
            date(2025, 7, 4),    # Independence Day
            date(2025, 9, 1),    # Labor Day
            date(2025, 11, 27),  # Thanksgiving
            date(2025, 12, 25),  # Christmas
        ],
        2026: [
            date(2026, 1, 1),
            date(2026, 1, 19),
            date(2026, 2, 16),
            date(2026, 4, 3),
            date(2026, 5, 25),
            date(2026, 6, 19),
            date(2026, 7, 3),    # Observed
            date(2026, 9, 7),
            date(2026, 11, 26),
            date(2026, 12, 25),
        ],
        2027: [
            date(2027, 1, 1),
            date(2027, 1, 18),
            date(2027, 2, 15),
            date(2027, 3, 26),
            date(2027, 5, 31),
            date(2027, 6, 18),   # Observed
            date(2027, 7, 5),    # Observed
            date(2027, 9, 6),
            date(2027, 11, 25),
            date(2027, 12, 24),  # Observed
        ],
    }

    # LSE holidays
    LSE_HOLIDAYS: Dict[int, List[date]] = {
        2025: [
            date(2025, 1, 1),
            date(2025, 4, 18),   # Good Friday
            date(2025, 4, 21),   # Easter Monday
            date(2025, 5, 5),    # May Bank Holiday
            date(2025, 5, 26),   # Spring Bank Holiday
            date(2025, 8, 25),   # Summer Bank Holiday
            date(2025, 12, 25),
            date(2025, 12, 26),  # Boxing Day
        ],
        2026: [
            date(2026, 1, 1),
            date(2026, 4, 3),
            date(2026, 4, 6),
            date(2026, 5, 4),
            date(2026, 5, 25),
            date(2026, 8, 31),
            date(2026, 12, 25),
            date(2026, 12, 28),
        ],
    }

    # EURONEXT (Paris) holidays
    EURONEXT_HOLIDAYS: Dict[int, List[date]] = {
        2025: [
            date(2025, 1, 1),
            date(2025, 4, 18),
            date(2025, 4, 21),
            date(2025, 5, 1),    # Labour Day
            date(2025, 12, 25),
            date(2025, 12, 26),
        ],
        2026: [
            date(2026, 1, 1),
            date(2026, 4, 3),
            date(2026, 4, 6),
            date(2026, 5, 1),
            date(2026, 12, 25),
            date(2026, 12, 28),
        ],
    }

    # JPX (Tokyo) holidays
    JPX_HOLIDAYS: Dict[int, List[date]] = {
        2025: [
            date(2025, 1, 1),
            date(2025, 1, 2),
            date(2025, 1, 3),
            date(2025, 1, 13),   # Coming of Age
            date(2025, 2, 11),   # National Foundation
            date(2025, 2, 24),   # Emperor's Birthday
            date(2025, 3, 20),   # Vernal Equinox
            date(2025, 4, 29),   # Showa Day
            date(2025, 5, 3),
            date(2025, 5, 5),
            date(2025, 5, 6),
            date(2025, 7, 21),   # Marine Day
            date(2025, 8, 11),   # Mountain Day
            date(2025, 9, 15),   # Respect for Aged
            date(2025, 9, 23),   # Autumnal Equinox
            date(2025, 10, 13),  # Sports Day
            date(2025, 11, 3),   # Culture Day
            date(2025, 11, 24),  # Labour Thanksgiving
            date(2025, 12, 31),
        ],
        2026: [
            date(2026, 1, 1),
            date(2026, 1, 2),
            date(2026, 1, 12),
            date(2026, 2, 11),
            date(2026, 2, 23),
            date(2026, 3, 20),
            date(2026, 4, 29),
            date(2026, 5, 4),
            date(2026, 5, 5),
            date(2026, 5, 6),
            date(2026, 7, 20),
            date(2026, 8, 11),
            date(2026, 9, 21),
            date(2026, 9, 23),
            date(2026, 10, 12),
            date(2026, 11, 3),
            date(2026, 11, 23),
            date(2026, 12, 31),
        ],
    }

    # HKEx holidays (simplified)
    HKEX_HOLIDAYS: Dict[int, List[date]] = {
        2025: [
            date(2025, 1, 1),
            date(2025, 1, 29),   # CNY
            date(2025, 1, 30),
            date(2025, 1, 31),
            date(2025, 4, 4),    # Ching Ming
            date(2025, 4, 18),
            date(2025, 4, 21),
            date(2025, 5, 1),
            date(2025, 5, 5),    # Buddha's Birthday
            date(2025, 5, 31),   # Dragon Boat
            date(2025, 7, 1),    # HKSAR
            date(2025, 10, 1),   # National Day
            date(2025, 10, 7),   # Chung Yeung
            date(2025, 12, 25),
            date(2025, 12, 26),
        ],
        2026: [
            date(2026, 1, 1),
            date(2026, 2, 17),
            date(2026, 2, 18),
            date(2026, 2, 19),
            date(2026, 4, 3),
            date(2026, 4, 6),
            date(2026, 4, 7),
            date(2026, 5, 1),
            date(2026, 5, 25),
            date(2026, 6, 19),
            date(2026, 7, 1),
            date(2026, 10, 1),
            date(2026, 10, 26),
            date(2026, 12, 25),
        ],
    }

    # ASX holidays
    ASX_HOLIDAYS: Dict[int, List[date]] = {
        2025: [
            date(2025, 1, 1),
            date(2025, 1, 27),   # Australia Day
            date(2025, 4, 18),
            date(2025, 4, 21),
            date(2025, 4, 25),   # ANZAC Day
            date(2025, 6, 9),    # Queen's Birthday
            date(2025, 12, 25),
            date(2025, 12, 26),
        ],
        2026: [
            date(2026, 1, 1),
            date(2026, 1, 26),
            date(2026, 4, 3),
            date(2026, 4, 6),
            date(2026, 4, 27),
            date(2026, 6, 8),
            date(2026, 12, 25),
            date(2026, 12, 28),
        ],
    }

    # TSX holidays
    TSX_HOLIDAYS: Dict[int, List[date]] = {
        2025: [
            date(2025, 1, 1),
            date(2025, 2, 17),   # Family Day
            date(2025, 4, 18),
            date(2025, 5, 19),   # Victoria Day
            date(2025, 7, 1),    # Canada Day
            date(2025, 8, 4),    # Civic Holiday
            date(2025, 9, 1),    # Labour Day
            date(2025, 10, 13),  # Thanksgiving
            date(2025, 12, 25),
            date(2025, 12, 26),
        ],
        2026: [
            date(2026, 1, 1),
            date(2026, 2, 16),
            date(2026, 4, 3),
            date(2026, 5, 18),
            date(2026, 7, 1),
            date(2026, 8, 3),
            date(2026, 9, 7),
            date(2026, 10, 12),
            date(2026, 12, 25),
            date(2026, 12, 28),
        ],
    }

    # Exchange → calendar mapping
    EXCHANGE_CALENDAR_MAP = {
        "NYSE": "US",
        "NASDAQ": "US",
        "LSE": "LSE",
        "EURONEXT": "EURONEXT",
        "XETRA": "EURONEXT",
        "JPX": "JPX",
        "HKEx": "HKEX",
        "ASX": "ASX",
        "TSX": "TSX",
        "CME": "US",
        "NYMEX": "US",
        "COMEX": "US",
        "CBOT": "US",
        "ICE": "US",
        "EUREX": "EURONEXT",
    }

    def __init__(self):
        self._calendars: Dict[str, Dict[int, Set[date]]] = {
            "US": {
                y: set(d)
                for y, d in self.US_HOLIDAYS.items()
            },
            "LSE": {
                y: set(d)
                for y, d in self.LSE_HOLIDAYS.items()
            },
            "EURONEXT": {
                y: set(d)
                for y, d in self.EURONEXT_HOLIDAYS.items()
            },
            "JPX": {
                y: set(d)
                for y, d in self.JPX_HOLIDAYS.items()
            },
            "HKEX": {
                y: set(d)
                for y, d in self.HKEX_HOLIDAYS.items()
            },
            "ASX": {
                y: set(d)
                for y, d in self.ASX_HOLIDAYS.items()
            },
            "TSX": {
                y: set(d)
                for y, d in self.TSX_HOLIDAYS.items()
            },
        }

    def is_holiday(
        self,
        exchange: str,
        check_date: date = None,
    ) -> bool:
        """Check if date is a holiday for exchange."""
        if check_date is None:
            check_date = date.today()

        cal_key = self.EXCHANGE_CALENDAR_MAP.get(
            exchange, "US"
        )
        cal = self._calendars.get(cal_key, {})
        year_holidays = cal.get(check_date.year, set())
        return check_date in year_holidays

    def is_trading_day(
        self,
        exchange: str,
        check_date: date = None,
    ) -> bool:
        """Check if date is a trading day."""
        if check_date is None:
            check_date = date.today()

        # Weekend check
        if check_date.weekday() >= 5:
            return False

        # Holiday check
        return not self.is_holiday(exchange, check_date)

    def get_holidays(
        self, exchange: str, year: int
    ) -> List[date]:
        """Get all holidays for exchange in year."""
        cal_key = self.EXCHANGE_CALENDAR_MAP.get(
            exchange, "US"
        )
        cal = self._calendars.get(cal_key, {})
        return sorted(cal.get(year, set()))

    def next_trading_day(
        self, exchange: str, from_date: date = None
    ) -> date:
        """Get next trading day after from_date."""
        from datetime import timedelta
        if from_date is None:
            from_date = date.today()

        check = from_date + timedelta(days=1)
        max_look = 10
        for _ in range(max_look):
            if self.is_trading_day(exchange, check):
                return check
            check += timedelta(days=1)
        return check


# Singleton
_calendar = None


def get_holiday_calendar() -> ExchangeHolidayCalendar:
    """Get or create holiday calendar."""
    global _calendar
    if _calendar is None:
        _calendar = ExchangeHolidayCalendar()
    return _calendar
