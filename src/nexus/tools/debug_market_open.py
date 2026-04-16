
import os
import sys
from datetime import datetime, time, timedelta, timezone

# Add project root to path
sys.path.append(os.getcwd())

def test_market_open():
    print(f"Current Local Time (script): {datetime.now()}")

    try:
        import pytz
        print("pytz is available")
        now_et = datetime.now(pytz.timezone('US/Eastern'))
    except ImportError:
        print("pytz is NOT available, using manual UTC-5")
        now_et = datetime.now(timezone(timedelta(hours=-5)))

    print(f"ET Time: {now_et}")
    print(f"Weekday (0=Mon, 6=Sun): {now_et.weekday()}")

    is_weekend = now_et.weekday() >= 5
    print(f"Is Weekend: {is_weekend}")

    market_open_t = time(9, 30)
    market_close_t = time(16, 0)
    in_hours = market_open_t <= now_et.time() <= market_close_t
    print(f"In Market Hours: {in_hours}")

    is_open = not is_weekend and in_hours
    print(f"Final is_market_open result: {is_open}")

if __name__ == "__main__":
    test_market_open()
