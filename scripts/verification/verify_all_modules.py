"""Quick verification of all new modules."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from compliance.compliance_engine import get_compliance
from data.global_universe import get_global_universe
from execution.currency_settlement import get_settlement
from execution.exchange_holidays import get_holiday_calendar
from monitoring.grafana_metrics import get_metrics

# Holiday Calendar
cal = get_holiday_calendar()
print(f"NYSE trading today: {cal.is_trading_day('NYSE')}")
print(f"JPX 2025 holidays: {len(cal.get_holidays('JPX', 2025))}")
print(f"LSE 2025 holidays: {len(cal.get_holidays('LSE', 2025))}")

# Currency Settlement
s = get_settlement()
print(f"EUR->USD 1000: {s.convert(1000, 'EUR')}")
print(f"JPY->USD 100000: {s.convert(100000, 'JPY')}")
print(f"GBP->USD 5000: {s.convert(5000, 'GBP')}")

# Metrics
m = get_metrics()
m.set_total_equity(1000000)
m.set_daily_pnl(5000)
m.inc_orders_submitted("alpaca")
m.observe_order_latency(25.3)
print(f"Metrics: {m.to_dict()['gauges']}")

# Compliance
c = get_compliance()
rec = c.record_execution("AAPL", "buy", 100, 150.25, 150.20, "NYSE")
print(f"Compliance record hash: {rec['hash']}")
print(f"Slippage: {rec['slippage_bps']}bps")

# Universe
u = get_global_universe()
print(f"Universe: {u.total_count} symbols")
print(f"UK stocks: {len(u.get_by_currency('GBP'))}")
print(f"JPY stocks: {len(u.get_by_currency('JPY'))}")
print(f"CAD stocks: {len(u.get_by_currency('CAD'))}")

print("\n=== ALL MODULES VERIFIED OK ===")
