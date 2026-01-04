# backtest/execution.py
"""
Institutional-grade execution module for backtests.

Features:
- Order lifecycle state machine (NEW -> PARTIALLY_FILLED -> FILLED / CANCELLED)
- Partial fills using bar-level liquidity + ADV participation caps
- Real LIMIT order crossing behavior (Low/High checks)
- ADV-based market impact model (with safe fallbacks)
- Commission & slippage accounting
- Per-order notional cap to prevent unrealistically large fills
- Thread-safe, append-only TradeBlotter with atomic CSV export
- Structured logging and defensive input validation
- Deterministic outputs (no silent failures)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import uuid
import numpy as np
import pandas as pd
import logging
import threading
import os
import tempfile
import shutil

# -------------------------
# Logging
# -------------------------
logger = logging.getLogger("execution")
if not logger.handlers:
    # default handler if not configured by app
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(ch)
logger.setLevel(logging.INFO)

# -------------------------
# Exceptions
# -------------------------
class ExecutionError(Exception):
    pass


# -------------------------
# Enums & Models
# -------------------------

@dataclass
class BarData:
    open: float
    high: float
    low: float
    close: float
    volume: float
    timestamp: datetime
    ticker: Optional[str] = None

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class OrderStatus(Enum):
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"

class AlgoType(Enum):
    NONE = "NONE"      # Immediate fill (limited by max_participation_rate)
    VWAP = "VWAP"      # Volume Weighted Average Price (Participation)
    TWAP = "TWAP"      # Time Weighted Average Price (Pacing)

@dataclass
class Order:
    """
    Order object representing an order submitted by strategy/engine.
    - quantity: positive for buy quantity, negative for sell quantity
    - limit_price: required for LIMIT orders
    """
    ticker: str
    quantity: float
    order_type: OrderType
    timestamp: datetime
    limit_price: Optional[float] = None
    strategy_id: str = "default"
    meta: Dict[str, Any] = field(default_factory=dict)
    
    # Institutional Algos
    algo: AlgoType = AlgoType.NONE
    target_participation_rate: Optional[float] = None # Defaults to handler's rate if None

    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    remaining_qty: float = field(init=False)
    status: OrderStatus = field(init=False)

    def __post_init__(self):
        # Normalize numeric types and initialize state
        self.quantity = float(self.quantity)
        self.remaining_qty = float(self.quantity)
        self.status = OrderStatus.NEW
        if self.order_type == OrderType.LIMIT and self.limit_price is None:
            raise ExecutionError("LIMIT order created without limit_price")
活跃
