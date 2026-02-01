# safety/kill_switch.py
"""
Global kill switch service (file-backed simple implementation).
Operations: engage, release, status.
Integrate: orchestrator checks kill_switch.is_engaged() before scheduling any cycle.
"""
import os, json, time

KILL_PATH = os.environ.get("KILL_SWITCH_PATH", "runs/kill_switch.json")

class KillSwitch:
    def __init__(self, path=KILL_PATH):
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        if not os.path.exists(self.path):
            self._write({"engaged": False, "ts": None, "reason": None})

    def _write(self, obj):
        with open(self.path, "w") as f:
            json.dump(obj, f, indent=2)

    def engage(self, reason="manual"):
        self._write({"engaged": True, "ts": time.time(), "reason": reason})

    def release(self):
        self._write({"engaged": False, "ts": time.time(), "reason": None})

    def is_engaged(self) -> bool:
        with open(self.path, "r") as f:
            obj = json.load(f)
        return bool(obj.get("engaged", False))

    def status(self):
        with open(self.path, "r") as f:
            return json.load(f)
