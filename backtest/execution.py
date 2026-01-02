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
class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class OrderStatus(Enum):
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"


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


@dataclass
class Trade:
    trade_id: str
    order_id: str
    ticker: str
    quantity: float
    fill_price: float
    expected_price: float
    market_impact: float
    slippage: float
    commission: float
    cost: float
    timestamp: datetime
    meta: Dict[str, Any] = field(default_factory=dict)


# -------------------------
# Trade Blotter (thread-safe)
# -------------------------
class TradeBlotter:
    """
    Append-only in-memory blotter. Thread-safe.
    Use save_trades_csv() to atomically write to disk.
    """

    def __init__(self):
        self._orders: Dict[str, Order] = {}
        self._trades: List[Trade] = []
        self._lock = threading.Lock()

    def record_order(self, order: Order) -> None:
        with self._lock:
            if order.id in self._orders:
                logger.warning("Order %s already recorded; overwriting", order.id)
            self._orders[order.id] = order

    def record_trade(self, trade: Trade) -> None:
        with self._lock:
            self._trades.append(trade)

    def get_order(self, order_id: str) -> Optional[Order]:
        with self._lock:
            return self._orders.get(order_id)

    def orders_df(self) -> pd.DataFrame:
        with self._lock:
            rows = []
            for o in self._orders.values():
                rows.append({
                    "order_id": o.id,
                    "timestamp": o.timestamp,
                    "ticker": o.ticker,
                    "quantity": o.quantity,
                    "remaining_qty": o.remaining_qty,
                    "order_type": o.order_type.value,
                    "status": o.status.value,
                    "limit_price": o.limit_price,
                    "strategy_id": o.strategy_id,
                    "meta": o.meta,
                })
            return pd.DataFrame(rows)

    def trades_df(self) -> pd.DataFrame:
        with self._lock:
            rows = []
            for t in self._trades:
                rows.append({
                    "trade_id": t.trade_id,
                    "order_id": t.order_id,
                    "timestamp": t.timestamp,
                    "ticker": t.ticker,
                    "quantity": t.quantity,
                    "fill_price": t.fill_price,
                    "expected_price": t.expected_price,
                    "market_impact": t.market_impact,
                    "slippage": t.slippage,
                    "commission": t.commission,
                    "cost": t.cost,
                    "meta": t.meta,
                })
            return pd.DataFrame(rows)

    def save_trades_csv(self, path: str) -> None:
        """
        Atomically write trades.csv. Creates directories as needed.
        """
        df = self.trades_df()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        # Write to temp file then move
        dirpath = os.path.dirname(path)
        fd, tmp = tempfile.mkstemp(dir=dirpath, prefix=".tmp_trades_", suffix=".csv")
        os.close(fd)
        try:
            df.to_csv(tmp, index=False)
            shutil.move(tmp, path)
            logger.info("Wrote trades CSV to %s (rows=%d)", path, len(df))
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except Exception:
                    pass


# -------------------------
# Execution Handler
# -------------------------
class RealisticExecutionHandler:
    """
    Institutional execution handler with conservative defaults.

    Parameters:
    - commission_pct: commission applied to notional (e.g., 0.0005 = 0.05%)
    - max_participation_rate: fraction of bar volume per bar (e.g., 0.1 = 10%)
    - impact_coeff: coefficient for market impact model (higher => larger impact)
    - adv_lookback: days to compute ADV (used for participation vs ADV)
    - vol_lookback: lookback window for realized vol (in bars)
    - min_vol_fallback: fallback annualized vol if estimation fails
    - max_notional_per_order: maximal notional allowed per order (if >0)
    - min_fill_size: ignore fills smaller than this absolute size (avoid dust)
    """

    def __init__(
        self,
        commission_pct: float = 0.0005,
        max_participation_rate: float = 0.10,
        impact_coeff: float = 0.15,
        adv_lookback: int = 20,
        vol_lookback: int = 21,
        min_vol_fallback: float = 0.02,
        max_notional_per_order: float = 0.0,
        min_fill_size: float = 1.0,
    ):
        # Param validation
        if commission_pct < 0 or commission_pct > 0.1:
            raise ValueError("commission_pct out of range")
        if not (0 < max_participation_rate <= 1.0):
            raise ValueError("max_participation_rate must be in (0,1]")
        if adv_lookback < 1 or vol_lookback < 1:
            raise ValueError("lookbacks must be >= 1")

        self.commission_pct = float(commission_pct)
        self.max_participation_rate = float(max_participation_rate)
        self.impact_coeff = float(impact_coeff)
        self.adv_lookback = int(adv_lookback)
        self.vol_lookback = int(vol_lookback)
        self.min_vol_fallback = float(min_vol_fallback)
        self.max_notional_per_order = float(max_notional_per_order)
        self.min_fill_size = float(min_fill_size)

    # ---- Helpers ----
    def _validate_bar(self, bar: Dict[str, float]) -> None:
        required = ["Open", "High", "Low", "Close", "Volume"]
        for k in required:
            if k not in bar:
                raise ExecutionError(f"Bar missing required field: {k}")
        if bar["Volume"] is None or bar["Volume"] <= 0:
            raise ExecutionError("Bar volume must be > 0 for reliable execution")

    def _estimate_vol_and_adv(
        self,
        price_history: pd.Series,
        volume_history: pd.Series,
    ) -> Tuple[float, float]:
        """
        Estimate annualized volatility and ADV robustly.
        Will raise ExecutionError only if both estimates are impossible.
        """
        vol = None
        adv = None

        # Estimate vol: rolling std of pct_change
        try:
            if price_history is None or len(price_history) < 5:
                raise RuntimeError("insufficient price history")
            ret = price_history.astype(float).pct_change().dropna()
            if len(ret) == 0:
                raise RuntimeError("no returns")
            roll_win = min(self.vol_lookback, max(2, len(ret)))
            vol_raw = ret.rolling(roll_win).std().iloc[-1]
            if np.isnan(vol_raw) or vol_raw <= 0:
                raise RuntimeError("bad vol estimate")
            vol = float(vol_raw) * np.sqrt(252)
        except Exception as e:
            logger.warning("Vol estimation failed (%s). Using fallback vol=%s", e, self.min_vol_fallback)
            vol = float(self.min_vol_fallback)

        # Estimate ADV
        try:
            if volume_history is None or len(volume_history) < 1:
                raise RuntimeError("insufficient volume history")
            adv_win = min(self.adv_lookback, len(volume_history))
            adv = float(volume_history.tail(adv_win).mean())
            if np.isnan(adv) or adv <= 0:
                raise RuntimeError("bad adv")
        except Exception as e:
            logger.warning("ADV estimation failed (%s). Using bar-level volume as fallback", e)
            # adv fallback will be set by caller if necessary; return adv=None to indicate fallback
            adv = None

        return vol, adv

    # ---- Main fill logic ----
    def fill_order(
        self,
        order: Order,
        bar: Dict[str, float],
        bar_timestamp: datetime,
        price_history: pd.Series,
        volume_history: pd.Series,
    ) -> Optional[Trade]:
        """
        Attempt to fill (part of) the order against a single bar.

        Returns a Trade if any quantity was filled, otherwise None.

        Notes:
        - Order.remaining_qty is updated in-place.
        - Order.status is updated to PARTIALLY_FILLED or FILLED.
        - The engine that calls fill_order should record the returned trade via blotter.record_trade(trade).
        """

        # Basic state checks
        if order is None:
            raise ExecutionError("order is None")
        if order.status in (OrderStatus.FILLED, OrderStatus.CANCELLED):
            logger.debug("Order %s already %s; skipping", order.id, order.status.value)
            return None

        # Validate bar
        try:
            self._validate_bar(bar)
        except ExecutionError as e:
            logger.error("Invalid bar provided: %s", e)
            raise

        close_price = float(bar["Close"])
        high = float(bar["High"])
        low = float(bar["Low"])
        bar_volume = float(bar["Volume"])

        # LIMIT logic: require price crossing for execution
        if order.order_type == OrderType.LIMIT:
            limit = float(order.limit_price)
            side = np.sign(order.remaining_qty)
            # Buy: limit is maximum price buyer is willing to pay -> executable if low <= limit
            if side > 0 and not (low <= limit):
                logger.debug("Limit buy %s not crossed (low=%s > limit=%s) on bar", order.id, low, limit)
                return None
            # Sell: limit is minimum price seller wants -> executable if high >= limit
            if side < 0 and not (high >= limit):
                logger.debug("Limit sell %s not crossed (high=%s < limit=%s) on bar", order.id, high, limit)
                return None

        # Estimate vol and adv
        vol, adv = self._estimate_vol_and_adv(price_history, volume_history)

        # If adv fallback (None), use bar_volume conservatively but log
        if adv is None:
            logger.warning("ADV unavailable, using bar volume %s as conservative adv", bar_volume)
            adv = max(bar_volume, 1.0)

        # Max quantity by bar-level participation (bar-level liquidity cap)
        max_by_bar = bar_volume * self.max_participation_rate

        # Determine requested fill (clamp to remaining_qty)
        desired_qty = order.remaining_qty
        # For safety, enforce min_fill_size to avoid dust
        if abs(desired_qty) < self.min_fill_size:
            logger.debug("Order %s remaining_qty %s below min_fill_size %s; marking filled",
                         order.id, order.remaining_qty, self.min_fill_size)
            # consider it filled with zero trade
            order.remaining_qty = 0.0
            order.status = OrderStatus.FILLED
            return None

        # Max fill also limited by max_notional_per_order if configured
        # Compute notional potential using close_price. Use abs values.
        max_fill_by_bar_qty = max_by_bar
        if self.max_notional_per_order > 0:
            max_qty_by_notional = self.max_notional_per_order / max(close_price, 1e-12)
            max_fill_by_bar_qty = min(max_fill_by_bar_qty, max_qty_by_notional)

        # Final fill quantity this bar
        fill_qty = np.sign(desired_qty) * min(abs(desired_qty), max_fill_by_bar_qty)

        if abs(fill_qty) < 1e-8:
            logger.debug("Order %s cannot be filled this bar due to participation/notional caps", order.id)
            return None

        # Compute participation relative to ADV (conservative measure)
        participation = min(1.0, abs(fill_qty) / max(adv, 1.0))

        # Market impact model (signed)
        market_impact = float(self.impact_coeff) * float(vol) * np.sqrt(participation)

        # Determine expected_price and fill_price robustly:
        # - expected_price = close_price (engine's midpoint expectation for bar)
        # - fill_price adjusts by sign*impact, then clipped to bar high/low to avoid unrealistic prices
        expected_price = float(close_price)
        side = np.sign(fill_qty)
        raw_fill_price = expected_price * (1.0 + side * market_impact)

        # Clip fill price into [low, high] to be conservative
        fill_price = float(np.clip(raw_fill_price, low, high))

        # For LIMIT orders, ensure fill price is not worse than limit:
        if order.order_type == OrderType.LIMIT:
            limit_price = float(order.limit_price)
            if side > 0:
                # Buyer: cannot pay more than limit; if clipped fill_price > limit, set to limit (if within bar range)
                if fill_price > limit_price and low <= limit_price <= high:
                    fill_price = limit_price
                # otherwise, if limit isn't inside bar range, it's unexpected (but we already checked crossing)
            else:
                # Seller: cannot sell lower than limit; ensure fill_price >= limit
                if fill_price < limit_price and low <= limit_price <= high:
                    fill_price = limit_price

        # Costs
        notional = abs(fill_qty) * fill_price
        commission = notional * float(self.commission_pct)
        slippage = abs(fill_price - expected_price) * abs(fill_qty)
        total_cost = commission + slippage

        # Build trade object (timestamp is bar_timestamp — correct attribution)
        trade = Trade(
            trade_id=uuid.uuid4().hex,
            order_id=order.id,
            ticker=order.ticker,
            quantity=float(fill_qty),
            fill_price=float(fill_price),
            expected_price=float(expected_price),
            market_impact=float(market_impact),
            slippage=float(slippage),
            commission=float(commission),
            cost=float(total_cost),
            timestamp=bar_timestamp,
            meta={
                "vol_used": float(vol),
                "adv_used": float(adv),
                "participation": float(participation),
                "bar_volume": float(bar_volume),
                "bar_low": float(low),
                "bar_high": float(high),
            },
        )

        # Update order state (in-place)
        order.remaining_qty = float(order.remaining_qty - fill_qty)
        # Guard against numerical drift
        if abs(order.remaining_qty) < 1e-8:
            order.remaining_qty = 0.0
            order.status = OrderStatus.FILLED
        else:
            order.status = OrderStatus.PARTIALLY_FILLED

        logger.debug(
            "Filled order %s qty=%s fill_price=%s expected=%s impact=%s commission=%s remaining=%s",
            order.id, trade.quantity, trade.fill_price, trade.expected_price,
            trade.market_impact, trade.commission, order.remaining_qty
        )

        return trade
