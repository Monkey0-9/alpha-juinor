"""
=============================================================================
UNIFIED 13-TYPE TRADING ENGINE
=============================================================================
ONE intelligent engine managing ALL 13 trading types with dynamic routing.

TRADING TYPES:
1. Day Trading (intraday)
2. Swing Trading (days/weeks)
3. Scalping (seconds)
4. Position Trading (months/years)
5. Momentum Trading (trend following)
6. Algorithmic Trading (automated rules)
7. Social Trading (crowd signals)
8. Copy Trading (mirror external)
9. News Trading (events)
10. Technical Trading (chart patterns)
11. Fundamental Trading (valuation)
12. Delivery Trading (settlement)
13. Event-Driven Trading (corporate actions)

This engine:
- Routes each symbol to optimal trading type(s)
- Generates type-specific signals
- Aggregates into final trade decision
- Manages positions per type with appropriate parameters
- Tracks performance attribution by type
- Coordinates without conflicts using priority resolution
"""

import logging
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TradingType(Enum):
    """All 13 trading types"""

    DAY_TRADING = 1
    SWING_TRADING = 2
    SCALPING = 3
    POSITION_TRADING = 4
    MOMENTUM_TRADING = 5
    ALGORITHMIC = 6
    SOCIAL_TRADING = 7
    COPY_TRADING = 8
    NEWS_TRADING = 9
    TECHNICAL_TRADING = 10
    FUNDAMENTAL_TRADING = 11
    DELIVERY_TRADING = 12
    EVENT_DRIVEN = 13


class UnifiedTradingEngine:
    """
    Unified engine managing all 13 trading types simultaneously.

    For each symbol each cycle:
    1. Select active trading types (1-3 per symbol)
    2. Generate signals from each active type
    3. Aggregate signals with conflict resolution
    4. Execute with type-specific parameters
    5. Manage positions per type
    """

    def __init__(self, nav: float, config: Dict):
        self.nav = nav
        self.config = config
        self.positions = {}  # symbol -> position metadata
        self.completed_trades = []  # For performance attribution

        # Type-specific parameters
        self.type_params = self._init_type_parameters()

    def _init_type_parameters(self) -> Dict:
        """Initialize parameters for each 13 trading types"""
        return {
            TradingType.DAY_TRADING: {
                "hold_minutes_min": 1,
                "hold_minutes_max": 390,  # 6.5 hours
                "position_size_pct": 0.03,
                "stop_loss_pct": 0.02,
                "take_profit_pct": [0.05, 0.10],
                "auto_close_time": "15:50",  # 3:50 PM ET
                "max_daily": 10,
            },
            TradingType.SWING_TRADING: {
                "hold_minutes_min": 120,  # 2 hours
                "hold_minutes_max": 28800,  # 20 days
                "position_size_pct": 0.03,
                "stop_loss_pct": 0.05,
                "take_profit_pct": [0.05, 0.10, 0.15],
                "max_concurrent": 15,
            },
            TradingType.SCALPING: {
                "hold_minutes_min": 0.15,  # 10 seconds
                "hold_minutes_max": 5,
                "position_size_pct": 0.01,
                "stop_loss_pct": 0.003,
                "take_profit_pct": [0.005, 0.01],
                "max_daily": 200,
                "order_book_monitoring": True,
            },
            TradingType.POSITION_TRADING: {
                "hold_minutes_min": 43200,  # 30 days
                "hold_minutes_max": 525600,  # 1 year
                "position_size_pct": 0.05,
                "stop_loss_pct": 0.10,
                "take_profit_pct": [0.20, 0.50],
                "trailing_stop": True,
            },
            TradingType.MOMENTUM_TRADING: {
                "hold_minutes_min": 60,  # 1 hour
                "hold_minutes_max": 1440,  # 1 day
                "position_size_pct": 0.04,
                "stop_loss_pct": 0.04,
                "take_profit_pct": [0.08, 0.15],
            },
            TradingType.ALGORITHMIC: {
                "hold_minutes_min": 1,
                "hold_minutes_max": 10080,  # 1 week
                "position_size_pct": 0.04,
                "stop_loss_pct": self.config.get("risk_max_loss_pct", 0.02),
                "take_profit_pct": self.config.get("target_profit_pct", [0.05, 0.10]),
            },
            TradingType.SOCIAL_TRADING: {
                "hold_minutes_min": 60,
                "hold_minutes_max": 1440,
                "position_size_pct": 0.02,
                "stop_loss_pct": 0.03,
                "take_profit_pct": [0.05, 0.10],
            },
            TradingType.COPY_TRADING: {
                "hold_minutes_min": 30,
                "hold_minutes_max": 2880,  # 2 days
                "position_size_pct": 0.025,
                "stop_loss_pct": 0.04,
                "take_profit_pct": [0.05, 0.12],
                "scale_to_nav": True,
            },
            TradingType.NEWS_TRADING: {
                "hold_minutes_min": 5,
                "hold_minutes_max": 120,  # 2 hours
                "position_size_pct": 0.02,
                "stop_loss_pct": 0.04,
                "take_profit_pct": [0.03, 0.08],
            },
            TradingType.TECHNICAL_TRADING: {
                "hold_minutes_min": 60,
                "hold_minutes_max": 2880,
                "position_size_pct": 0.03,
                "stop_loss_pct": 0.03,
                "take_profit_pct": [0.05, 0.12],
            },
            TradingType.FUNDAMENTAL_TRADING: {
                "hold_minutes_min": 1440,  # 1 day
                "hold_minutes_max": 262080,  # 6 months
                "position_size_pct": 0.04,
                "stop_loss_pct": 0.08,
                "take_profit_pct": [0.15, 0.30],
            },
            TradingType.DELIVERY_TRADING: {
                "hold_minutes_min": 525600,  # 1 year
                "hold_minutes_max": 5256000,  # 10 years
                "position_size_pct": 0.05,
                "dividend_reinvestment": True,
            },
            TradingType.EVENT_DRIVEN: {
                "hold_minutes_min": 60,
                "hold_minutes_max": 1440,
                "position_size_pct": 0.03,
                "stop_loss_pct": 0.05,
                "take_profit_pct": [0.05, 0.20],
            },
        }

    def select_trading_types(
        self,
        symbol: str,
        volatility: float,
        adv: float,
        spread_bps: float,
        has_earnings: bool = False,
        has_news: bool = False,
    ) -> List[TradingType]:
        """
        Select active trading types for this symbol based on market conditions.

        Returns list of 1-3 optimal types per symbol.
        """
        active_types = []

        # Liquidity-driven types
        if adv > 10_000_000 and spread_bps < 5:
            active_types.extend([TradingType.SCALPING, TradingType.DAY_TRADING])
        elif adv > 5_000_000 and spread_bps < 10:
            active_types.extend([TradingType.DAY_TRADING, TradingType.SWING_TRADING])

        # Volatility-driven types
        if volatility < 0.15:
            active_types.append(TradingType.POSITION_TRADING)
            active_types.append(TradingType.SWING_TRADING)
        elif volatility >= 0.35:
            active_types.extend([TradingType.NEWS_TRADING, TradingType.EVENT_DRIVEN])
        else:
            active_types.extend(
                [TradingType.MOMENTUM_TRADING, TradingType.TECHNICAL_TRADING]
            )

        # Event-driven types
        if has_earnings:
            active_types.append(TradingType.EVENT_DRIVEN)

        if has_news:
            active_types.append(TradingType.NEWS_TRADING)

        # Always-on types
        active_types.append(TradingType.ALGORITHMIC)

        # Return unique list
        return list(set(active_types))

    def generate_signals(
        self,
        symbol: str,
        signal_data: Dict,
        active_types: List[TradingType],
    ) -> Dict[TradingType, Optional[Dict]]:
        """
        Generate signals from each active trading type for this symbol.

        Returns: {
            TradingType.DAY_TRADING: {...signal...},
            TradingType.SWING_TRADING: {...signal...},
            ...
        }
        """
        signals = {}

        for trading_type in active_types:
            signal = self._generate_signal_for_type(symbol, trading_type, signal_data)
            signals[trading_type] = signal

        return signals

    def _generate_signal_for_type(
        self,
        symbol: str,
        trading_type: TradingType,
        signal_data: Dict,
    ) -> Optional[Dict]:
        """Generate signal for specific trading type"""

        # Stub implementations - expand based on actual analysis
        if trading_type == TradingType.DAY_TRADING:
            # ORB or momentum spike
            if signal_data.get("is_orb"):
                return {"action": "BUY", "confidence": 0.85, "type": "orb"}

        elif trading_type == TradingType.SWING_TRADING:
            # Support bounce or trend continuation
            if signal_data.get("at_support"):
                return {"action": "BUY", "confidence": 0.80, "type": "support"}

        elif trading_type == TradingType.SCALPING:
            # Order book imbalance
            if signal_data.get("bid_ask_imbalance") > 1.7:
                return {"action": "BUY", "confidence": 0.80, "type": "book"}

        elif trading_type == TradingType.MOMENTUM_TRADING:
            # Breakout detected
            if signal_data.get("is_breakout"):
                return {"action": "BUY", "confidence": 0.75, "type": "breakout"}

        elif trading_type == TradingType.NEWS_TRADING:
            # News event detected
            if signal_data.get("has_news_event"):
                return {
                    "action": "BUY",
                    "confidence": signal_data.get("sentiment", 0.5),
                }

        elif trading_type == TradingType.EVENT_DRIVEN:
            # Corporate event detected
            if signal_data.get("has_corporate_event"):
                return {"action": "BUY", "confidence": 0.75, "type": "event"}

        # ... Other types return None if no signal
        return None

    def aggregate_signals(
        self,
        symbol: str,
        signals: Dict[TradingType, Optional[Dict]],
    ) -> Optional[Tuple[TradingType, Dict]]:
        """
        Aggregate signals from multiple types into final decision.

        Returns: (selected_trading_type, final_signal) or None

        Priority: Position > Swing > Technical > Day > Scalp
        """

        # Filter valid signals
        valid_signals = {k: v for k, v in signals.items() if v is not None}

        if not valid_signals:
            return None

        # Priority resolution
        priority_order = [
            TradingType.POSITION_TRADING,
            TradingType.SWING_TRADING,
            TradingType.TECHNICAL_TRADING,
            TradingType.FUNDAMENTAL_TRADING,
            TradingType.DAY_TRADING,
            TradingType.MOMENTUM_TRADING,
            TradingType.EVENT_DRIVEN,
            TradingType.NEWS_TRADING,
            TradingType.ALGORITHMIC,
            TradingType.SOCIAL_TRADING,
            TradingType.COPY_TRADING,
            TradingType.SCALPING,
            TradingType.DELIVERY_TRADING,
        ]

        # Check for conflicts (buy vs sell)
        buy_signals = [s for s in valid_signals.values() if s.get("action") == "BUY"]
        sell_signals = [s for s in valid_signals.values() if s.get("action") == "SELL"]

        if buy_signals and sell_signals:
            # Conflict: use highest priority type to break tie
            for trading_type in priority_order:
                if trading_type in valid_signals:
                    return (trading_type, valid_signals[trading_type])

        # Consensus: use highest priority type
        for trading_type in priority_order:
            if trading_type in valid_signals:
                return (trading_type, valid_signals[trading_type])

        return None

    def execute_trade(
        self,
        symbol: str,
        trading_type: TradingType,
        signal: Dict,
        nav: float,
        current_price: float,
    ) -> Optional[Dict]:
        """
        Execute trade with type-specific parameters.
        """

        params = self.type_params[trading_type]
        position_size_pct = params["position_size_pct"]
        stop_loss_pct = params["stop_loss_pct"]
        take_profit_list = params.get("take_profit_pct", [0.05])

        # Calculate position
        position_value = nav * position_size_pct
        quantity = int(position_value / current_price)

        if quantity <= 0:
            return None

        # Calculate stops/targets
        stop_loss = current_price * (1 - stop_loss_pct)
        take_profits = [current_price * (1 + tp) for tp in take_profit_list]

        # Create position record
        position = {
            "symbol": symbol,
            "trading_type": trading_type,
            "quantity": quantity,
            "entry_price": current_price,
            "entry_time": datetime.now(),
            "stop_loss": stop_loss,
            "take_profits": take_profits,
            "side": signal.get("action", "BUY"),
        }

        self.positions[symbol] = position
        return position

    def manage_positions(self):
        """
        Manage open positions based on their trading type:
        - Day trades: Close at 3:50 PM
        - Scalps: Close after 5 min
        - Position trades: Hold until thesis broken
        - Etc.
        """

        for symbol, position in list(self.positions.items()):
            trading_type = position.get("trading_type")

            # Check auto-close time for day trades
            if trading_type == TradingType.DAY_TRADING:
                if self._is_market_close_time():
                    self._close_position(symbol, reason="day_auto_close")
                    continue

            # Check hold time limits
            hold_time = (datetime.now() - position["entry_time"]).total_seconds() / 60
            params = self.type_params[trading_type]
            hold_min = params.get("hold_minutes_min", 0)
            hold_max = params.get("hold_minutes_max", float("inf"))

            if hold_time > hold_max:
                self._close_position(symbol, reason="max_hold_exceeded")
                continue

            # Check profit targets
            current_price = 0  # Would be fetched from market data
            if current_price > 0:
                unrealized = (current_price - position["entry_price"]) / position[
                    "entry_price"
                ]

                # Stop loss hit
                if current_price <= position["stop_loss"]:
                    self._close_position(symbol, reason="stop_loss_hit")
                    continue

                # Take profit hit
                for i, tp in enumerate(position["take_profits"]):
                    if current_price >= tp:
                        self._close_position(symbol, reason=f"tp{i+1}_hit")
                        break

    def _close_position(self, symbol: str, reason: str):
        """Close a position and record in completed trades"""
        if symbol in self.positions:
            position = self.positions.pop(symbol)
            position["close_reason"] = reason
            position["close_time"] = datetime.now()
            self.completed_trades.append(position)
            logger.info(f"Closed {symbol} ({position['trading_type'].name}): {reason}")

    def _is_market_close_time(self) -> bool:
        """Check if it's near market close (3:50 PM ET)"""
        import pytz

        now = datetime.now(pytz.timezone("US/Eastern"))
        return now.hour == 15 and now.minute >= 50

    def get_performance_attribution(self) -> Dict:
        """Get performance metrics broken down by trading type"""
        attribution = {}

        for trading_type in TradingType:
            trades = [
                t
                for t in self.completed_trades
                if t.get("trading_type") == trading_type
            ]

            if trades:
                pnls = []
                hold_times = []
                for trade in trades:
                    # Calculate PnL
                    entry = trade["entry_price"]
                    # Close price would be tracked - using entry as stub
                    close = entry * 1.01  # Stub
                    pnl = (close - entry) * trade["quantity"]
                    pnls.append(pnl)

                    # Hold time
                    hold_time = (
                        trade.get("close_time", datetime.now()) - trade["entry_time"]
                    ).total_seconds() / 3600
                    hold_times.append(hold_time)

                win_trades = len([p for p in pnls if p > 0])

                attribution[trading_type.name] = {
                    "trades": len(trades),
                    "total_pnl": sum(pnls),
                    "avg_pnl": np.mean(pnls) if pnls else 0,
                    "win_rate": (win_trades / len(trades) * 100) if trades else 0,
                    "avg_hold_hours": np.mean(hold_times) if hold_times else 0,
                }

        return attribution
