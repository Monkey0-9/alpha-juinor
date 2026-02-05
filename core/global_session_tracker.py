import pytz
from datetime import datetime, time, timedelta
import logging

logger = logging.getLogger("GlobalSession")

# Market Schedules (Local Times)
# Opening time, Closing time, Timezone
MARKET_SCHEDULES = {
    "NYSE": {"open": time(9, 30), "close": time(16, 0), "tz": "America/New_York"},
    "LSE": {"open": time(8, 0), "close": time(16, 30), "tz": "Europe/London"},  # London Stock Exchange
    "JPX": {"open": time(9, 0), "close": time(15, 0), "tz": "Asia/Tokyo"},      # Tokyo Stock Exchange
    "HKEX": {"open": time(9, 30), "close": time(16, 0), "tz": "Asia/Hong_Kong"}, # Hong Kong
    "ASX": {"open": time(10, 0), "close": time(16, 0), "tz": "Australia/Sydney"},# Australian Securities Exchange
    "TSX": {"open": time(9, 30), "close": time(16, 0), "tz": "America/Toronto"}, # Toronto
    "CRYPTO": {"open": time(0, 0), "close": time(23, 59, 59), "tz": "UTC"}     # Crypto (24/7)
}

class GlobalSessionTracker:
    """
    Tracks global market sessions, liquidity states, and time-to-open/close.
    """
    def __init__(self):
        self.schedules = MARKET_SCHEDULES
        logger.info("GlobalSessionTracker initialized with calendars: %s", list(self.schedules.keys()))

    def get_current_time(self, tz_name: str) -> datetime:
        """Get current time in specific timezone."""
        return datetime.now(pytz.timezone(tz_name))

    def is_market_open(self, exchange: str) -> bool:
        """Check if a specific exchange is currently open."""
        if exchange not in self.schedules:
            logger.warning(f"Exchange {exchange} not found. Defaulting to False.")
            return False

        if exchange == "CRYPTO":
            return True

        sched = self.schedules[exchange]
        tz = pytz.timezone(sched["tz"])
        now_local = datetime.now(tz)

        # Simple check: Weekday (0-4) and Time within range
        # Note: Does not handle holidays yet.
        if now_local.weekday() >= 5: # Sat=5, Sun=6
            return False

        current_time = now_local.time()
        return sched["open"] <= current_time <= sched["close"]

    def get_active_sessions(self) -> list:
        """Return list of currently open exchanges."""
        return [ex for ex in self.schedules if self.is_market_open(ex)]

    def get_liquidity_state(self) -> str:
        """
        Determine global liquidity state based on open markets.
        """
        active = self.get_active_sessions()
        if "NYSE" in active or "LSE" in active:
            return "HIGH" # US or Europe open
        elif "JPX" in active or "HKEX" in active:
            return "MEDIUM" # Asia open
        elif "ASX" in active:
            return "LOW" # Pacific only
        elif "CRYPTO" in active and len(active) == 1:
            return "THIN" # Weekend/Off-hours (Crypto only)
        else:
            return "OFF" # Should rarely happen given Crypto

    def time_to_next_open(self, exchange: str) -> timedelta:
        """Calculate time until next market open."""
        if exchange not in self.schedules or exchange == "CRYPTO":
            return timedelta(0) # Always open or unknown

        sched = self.schedules[exchange]
        tz = pytz.timezone(sched["tz"])
        now_local = datetime.now(tz)

        open_time = sched["open"]
        target_date = now_local.date()

        # If today is weekend or past close, move to tomorrow (or Monday)
        while True:
            # Check if open time is in future today
            candidate_open = datetime.combine(target_date, open_time)
            candidate_open = tz.localize(candidate_open)

            if candidate_open > now_local:
                # Still need to check if target_date is weekend
                if target_date.weekday() < 5:
                    return candidate_open - now_local

            # Move to next day
            target_date += timedelta(days=1)

            # Safety break (should find a weekday within 3-4 days)
            if (target_date - now_local.date()).days > 7:
                break

        return timedelta(hours=24) # Fallback

    def time_to_close(self, exchange: str) -> timedelta:
        """Calculate time remaining in current session."""
        if not self.is_market_open(exchange):
            return timedelta(0)

        if exchange == "CRYPTO":
            return timedelta(hours=24) # Infinite

        sched = self.schedules[exchange]
        tz = pytz.timezone(sched["tz"])
        now_local = datetime.now(tz)

        close_dt = datetime.combine(now_local.date(), sched["close"])
        close_dt = tz.localize(close_dt)

        return close_dt - now_local

# Singleton instance
_tracker = GlobalSessionTracker()

def get_global_session_tracker():
    return _tracker
