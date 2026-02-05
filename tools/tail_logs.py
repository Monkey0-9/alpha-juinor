import json
import os

log_path = r"c:\mini-quant-fund\runtime\logs\live.jsonl"
try:
    with open(log_path, 'r') as f:
        lines = f.readlines()
        last_lines = lines[-20:]
        print(f"--- Reading last {len(last_lines)} lines from {log_path} ---")
        for line in last_lines:
            try:
                entry = json.loads(line)
                print(f"[{entry.get('level')}] {entry.get('message')}")
            except json.JSONDecodeError:
                print(f"[RAW] {line.strip()}")
except Exception as e:
    print(f"Error reading log: {e}")
