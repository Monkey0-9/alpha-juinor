# backtest/execution.py

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, List
import uuid
import numpy as np
import pandas as pd
from datetime import datetime


# =========================
# Order & Trade definitions
# =========================

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


@dataclass
class Order:
    ticker: str
    quantity: float
    order_type: OrderType
    timestamp: datetime
    strategy_id: str = "default"
    meta: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex)


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


# =========================
# Trade Blotter
# =========================

class TradeBlotter:
    """
    Central registry for orders and trades.
    """

    def __init__(self):
        self.orders: List[Order] = []
        self.trades: List[Trade] = []

    def record_order(self, order: Order):
        self.orders.append(order)

    def record_trade(self, trade: Trade):
        self.trades.append(trade)

    def orders_df(self) -> pd.DataFrame:
        return pd.DataFrame([o.__dict__ for o in self.orders])

    def trades_df(self) -> pd.DataFrame:
        return pd.DataFrame([t.__dict__ for t in self.trades])


# =========================
# Realistic Execution Model
# =========================

class RealisticExecutionHandler:
    """
    Institutional execution model:
    - Volume constrained
    - Volatility-based market impact
    - Commission-aware
    """

    def __init__(
        self,
        commission_pct: float = 0.001,
        max_participation_rate: float = 0.10,
        impact_coeff: float = 0.1,
        vol_lookback: int = 20,
    ):
        self.commission_pct = commission_pct
        self.max_participation_rate = max_participation_rate
        self.impact_coeff = impact_coeff
        self.vol_lookback = vol_lookback

    def fill_order(
        self,
        order: Order,
        bar: Dict[str, float],
        price_history: pd.Series,
    ) -> Optional[Trade]:
        """
        bar must contain: Open, High, Low, Close, Volume
        """

        price = bar.get("Close")
        volume = bar.get("Volume")

        if price is None or price <= 0:
            return None

        if volume is None or volume <= 0:
            # Liquidity missing → assume illiquid market
            volume = abs(order.quantity) * 10

        # =========================
        # Volume constraint
        # =========================
        max_qty = volume * self.max_participation_rate
        fill_qty = np.sign(order.quantity) * min(abs(order.quantity), max_qty)

        if fill_qty == 0:
            return None

        # =========================
        # Volatility estimation
        # =========================
        returns = price_history.pct_change().dropna()
        vol = returns.rolling(self.vol_lookback).std().iloc[-1] if len(returns) > self.vol_lookback else 0.01

        # =========================
        # Market impact (square-root)
        # =========================
        participation = abs(fill_qty) / volume
        market_impact = self.impact_coeff * vol * np.sqrt(participation)

        # =========================
        # Execution price
        # =========================
        side = np.sign(fill_qty)
        expected_price = price
        fill_price = price * (1 + side * market_impact)

        # =========================
        # Costs
        # =========================
        notional = abs(fill_qty) * fill_price
        commission = notional * self.commission_pct
        slippage = abs(fill_price - expected_price) * abs(fill_qty)
        cost = commission + slippage

        return Trade(
            trade_id=uuid.uuid4().hex,
            order_id=order.id,
            ticker=order.ticker,
            quantity=fill_qty,
            fill_price=fill_price,
            expected_price=expected_price,
            market_impact=market_impact,
            slippage=slippage,
            commission=commission,
            cost=cost,
            timestamp=order.timestamp,
        )
