import sys
import os
sys.path.append(os.getcwd())
from database.schema import IngestionAuditRecord, DailyPriceRecord
import inspect

print("IngestionAuditRecord fields:", IngestionAuditRecord.__annotations__)
print("DailyPriceRecord fields:", DailyPriceRecord.__annotations__)

try:
    rec = IngestionAuditRecord(run_id="1", symbol="S", asset_class="S", provider="P", status="OK")
    print("IngestionAuditRecord init success with status/asset_class")
except Exception as e:
    print(f"IngestionAuditRecord init failed: {e}")

try:
    rec = DailyPriceRecord(
        symbol="S", date="D", open=1.0, high=1.0, low=1.0, close=1.0,
        adjusted_close=1.0, volume=100, provider="P", raw_hash="H",
        spike_flag=1, volume_spike_flag=1
    )
    print("DailyPriceRecord init success with spike_flag")
except Exception as e:
    print(f"DailyPriceRecord init failed: {e}")
