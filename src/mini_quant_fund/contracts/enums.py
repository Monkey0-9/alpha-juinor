"""
contracts/enums.py

Enumerations for the contracts package.
"""
from enum import Enum

class decision_enum(Enum):
    EXECUTE = "EXECUTE"
    HOLD = "HOLD"
    REJECT = "REJECT"
    ERROR = "ERROR"
