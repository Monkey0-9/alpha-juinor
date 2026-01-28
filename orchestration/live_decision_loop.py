#!/usr/bin/env python3
"""
Live Decision Loop Engine

Per-second decision loop that:
1. Aggregates real-time signals from MetaBrain
2. Updates position and P&L tracking
3. Displays live trading status in terminal
4. Manages order generation and execution

Timing:
- Per-second decision loop (configurable)
- Data refresh every 30-60 minutes (separate thread)
"""

import os
import sys
import time
import json
import logging
import threading
import signal
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict
from copy import deepcopy

import numpy as np
import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.manager import DatabaseManager, get_db
from agents.meta_brain import MetaBrain, SymbolDecision, DECISION_BUY, DECISION_SELL, DECISION_HOLD, DECISION_REJECT
from data.collectors.data_router import DataRouter
from brokers.alpaca_broker import AlpacaExecutionHandler
from brokers.mock_broker import MockBroker
from governance.lifecycle_manager import LifecycleManager
from monitoring.prometheus_exporter import metrics
from utils.tracing import trace_span

logger = logging.getLogger(__name__)


@dataclass
class LivePosition:
    """Real-time position tracking"""
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    entry_time: datetime
    last_update: datetime

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price

    @property
    def cost_basis(self) -> float:
        return abs(self.quantity) * self.entry_price

    @property
    def unrealized_pnl(self) -> float:
        if self.quantity > 0:
            return (self.current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - self.current_price) * abs(self.quantity)

    @property
    def pnl_pct(self) -> float:
        if self.cost_basis > 0:
            return (self.unrealized_pnl / self.cost_basis) * 100
        return 0.0


@dataclass
class LiveSignal:
    """
    Real-time signal for a symbol.

    Aligned with GovernanceDecision from governance/institutional_specification.py
    for consistent decision structure across the system.
    """
    # Core decision fields (aligned with GovernanceDecision)
    symbol: str
    signal: str  # BUY, SELL, HOLD
    cycle_id: str = ""
    timestamp: str = ""  # ISO format string for alignment

    # Model/output metrics
    mu_hat: float = 0.0
    sigma_hat: float = 0.0
    conviction: float = 0.0
    cvar: float = 0.0
    model_confidence: float = 0.0

    # Quality metrics
    data_quality: float = 1.0
    expected_return: float = 0.0
    expected_risk: float = 0.0
    position_size: float = 0.0

    # Reason tracking
    reason_codes: List[str] = field(default_factory=list)
    vetoed: bool = False
    veto_reason: str = ""

    # Risk check flags (aligned with GovernanceDecision.veto_triggers)
    cvar_limit_check: bool = True
    leverage_limit_check: bool = True
    drawdown_limit_check: bool = True
    correlation_limit_check: bool = True
    sector_limit_check: bool = True

    # Additional tracking
    agent_breakdown: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    current_price: float = 0.0
    last_decision: Optional[SymbolDecision] = None
    strategy_id: str = ""
    strategy_stage: str = ""


@dataclass
class LiveTradingState:
    """Complete live trading state"""
    cycle_count: int = 0
    start_time: Optional[datetime] = None
    last_decision_time: Optional[datetime] = None
    last_data_refresh: Optional[datetime] = None
    positions: Dict[str, LivePosition] = field(default_factory=dict)
    signals: Dict[str, LiveSignal] = field(default_factory=dict)
    orders_pending: List[Dict] = field(default_factory=list)
    orders_executed: int = 0
    risk_metrics: Dict[str, float] = field(default_factory=dict)
    system_status: str = "INITIALIZING"
    error_count: int = 0
    consecutive_errors: int = 0


class LiveDecisionLoop:
    """
    Per-second decision loop engine for live trading.

    Responsibilities:
    - Run decision loop at configurable interval (default: 1 second)
    - Aggregate signals from MetaBrain
    - Track positions and P&L in real-time
    - Generate and manage orders
    - Display live status in terminal
    """

    def __init__(
        self,
        tick_interval: float = 1.0,
        data_refresh_interval_min: int = 30,
        paper_mode: bool = True,
        market_hours_only: bool = True,
        symbols: Optional[List[str]] = None,
        on_decision_callback: Optional[Callable[[str, SymbolDecision], None]] = None
    ):
        """
        Initialize live decision loop.

        Args:
            tick_interval: Seconds between decision ticks (default: 1.0)
            data_refresh_interval_min: Minutes between data refreshes (default: 30)
            paper_mode: Run in paper mode (no real trades)
            market_hours_only: Only make decisions during market hours
            symbols: List of symbols to trade (loads from universe if None)
            on_decision_callback: Optional callback when decision is made
        """
        self.tick_interval = tick_interval
        self.data_refresh_interval = data_refresh_interval_min * 60  # Convert to seconds
        self.paper_mode = paper_mode
        self.market_hours_only = market_hours_only
        self.symbols = symbols or []
        self.on_decision_callback = on_decision_callback

        # State
        self.running = False
        self.paused = False
        self.state = LiveTradingState()
        self.state.start_time = datetime.utcnow()

        # Components
        self.db = DatabaseManager()
        self.router = DataRouter()
        self.lifecycle_manager = LifecycleManager(self.db, self.router)
        self.meta_brain = MetaBrain()

        # Broker setup
        self._setup_broker()

        # Cached data (updated every 30-60 min)
        self._cached_prices: Dict[str, float] = {}
        self._cached_features: Dict[str, Dict[str, Any]] = {}
        self._last_price_update: Optional[datetime] = None

        # Signal smoothing (for reducing noise)
        self._signal_history: Dict[str, List[Dict]] = defaultdict(list)
        self._signal_smoothing_window = 10  # Use last 10 signals

        # Performance tracking
        self._tick_times: List[float] = []
        self._decision_latency: List[float] = []

        # Thread locks
        self._state_lock = threading.Lock()
        self._data_lock = threading.Lock()

        # Load universe
        self._load_universe()

        # Initialize positions from database
        self._load_positions()

        logger.info(f"LiveDecisionLoop initialized | Tick: {tick_interval}s | Data Refresh: {data_refresh_interval_min}min | Symbols: {len(self.symbols)}")

    def _setup_broker(self):
        """Setup execution broker"""
        if self.paper_mode:
            self.broker = MockBroker()
            logger.info("Using MockBroker (paper mode)")
        else:
            try:
                api_key = os.getenv('ALPACA_API_KEY')
                secret_key = os.getenv('ALPACA_SECRET_KEY')
                if api_key and secret_key:
                    self.broker = AlpacaExecutionHandler(api_key, secret_key)
                    logger.info("Using AlpacaExecutionHandler (live mode)")
                else:
                    self.broker = MockBroker()
                    logger.warning("No Alpaca credentials found, using MockBroker")
            except Exception as e:
                logger.error(f"Failed to connect to broker: {e}")
                self.broker = MockBroker()

    def _enforce_startup_governance(self):
        """
        Phase 3: Startup Governance Gate.
        Block trading if history requirements are not met.
        """
        logger.info("[GOVERNANCE] Running startup history checks...")

        # Check coverage for last 5 years
        end_date = datetime.utcnow().strftime("%Y-%m-%d")
        start_date = (datetime.utcnow() - timedelta(days=5*365 + 20)).strftime("%Y-%m-%d")

        # Determine strictness based on mode
        is_strict = not self.paper_mode

        try:
            coverage = self.db.get_symbol_coverage(start_date, end_date)
            details = {r['symbol']: r['row_count'] for r in coverage.get('details', [])}

            failures = []
            for symbol in self.symbols:
                count = details.get(symbol, 0)
                if count < 1260:
                    failures.append(f"{symbol} ({count} rows < 1260)")

            if failures:
                msg = f"[GOVERNANCE_FAILURE] Insufficient history for {len(failures)} symbols: {', '.join(failures[:5])}..."
                logger.error(msg)

                if is_strict:
                    logger.critical("RELENTLESS GOVERNANCE: HALTING LIVE SYSTEM DUE TO HISTORY VIOLATION.")
                    logger.critical("Rule: Survival > Profit. Cannot trade blind.")
                    raise SystemExit("GOVERNANCE_VIOLATION: Insufficient History")
                else:
                    logger.warning("PAPER MODE: Proceeding despite governance failures (would allow for testing).")
            else:
                 logger.info("[GOVERNANCE] All symbols meet institutional history requirements (>1260 days).")

            # [GOVERNANCE] Feature Freshness Startup Check
            logger.info("[GOVERNANCE] Verifying feature freshness...")

            # 2. Run Lifecycle Checks (Task 4)
            logger.info("[GOVERNANCE] Running symbol lifecycle checks...")
            for symbol in self.symbols:
                try:
                    self.lifecycle_manager.run_lifecycle_check(symbol)
                except Exception as e:
                    logger.error(f"Lifecycle check failed for {symbol}: {e}")

            from mini_quant_fund.intelligence.feature_store import FeatureStore
            store = FeatureStore()
            features_map = store.get_latest(self.symbols)

            stale_symbols = []
            now = datetime.utcnow()
            for sym in self.symbols:
                f_data = features_map.get(sym, {})
                f_date = f_data.get('date')
                if not f_date:
                    stale_symbols.append(f"{sym} (MISSING)")
                    continue

                try:
                    ts = pd.to_datetime(f_date)
                    if ts.tzinfo is None: ts = ts.tz_localize('UTC')

                    # Ensure now is TZ aware
                    if now.tzinfo is None: now_aware = now.replace(tzinfo=ts.tzinfo)
                    else: now_aware = now.astimezone(ts.tzinfo)

                    age = (now_aware - ts).total_seconds() / 3600.0
                    if age > 24:
                         stale_symbols.append(f"{sym} ({age:.1f}h old)")
                except:
                     stale_symbols.append(f"{sym} (ERROR)")

            if stale_symbols:
                msg = f"[GOVERNANCE_FAILURE] Stale/Missing features for {len(stale_symbols)} symbols: {', '.join(stale_symbols[:5])}..."
                logger.error(msg)

                if is_strict:
                    logger.critical("RELENTLESS GOVERNANCE: HALTING LIVE SYSTEM DUE TO STALE FEATURES.")
                    logger.critical("Run 'python scripts/feature_refresh.py' to update.")
                    raise SystemExit("GOVERNANCE_VIOLATION: Stale Features")
                else:
                    logger.warning("PAPER MODE: Proceeding despite stale features.")
            else:
                logger.info("[GOVERNANCE] All features are fresh (<24h).")

        except Exception as e:
            logger.error(f"Governance check failed: {e}")
            if is_strict:
                raise SystemExit("GOVERNANCE_CHECK_FAILED")

    def _load_universe(self):
        """Load trading universe"""
        if not self.symbols:
            try:
                with open("configs/universe.json", "r") as f:
                    universe = json.load(f)
                self.symbols = universe.get("active_tickers", [])
            except Exception as e:
                logger.error(f"Failed to load universe: {e}")
                self.symbols = []

        logger.info(f"Loaded universe: {len(self.symbols)} symbols")

    def _load_positions(self):
        """Load current positions from broker/database"""
        try:
            positions = self.broker.get_positions() if hasattr(self.broker, 'get_positions') else {}
            for symbol, qty in positions.items():
                if qty != 0:
                    self.state.positions[symbol] = LivePosition(
                        symbol=symbol,
                        quantity=qty,
                        entry_price=0.0,  # Would need to fetch from broker
                        current_price=self._cached_prices.get(symbol, 0.0),
                        entry_time=datetime.utcnow(),
                        last_update=datetime.utcnow()
                    )
            logger.info(f"Loaded {len(self.state.positions)} positions from broker")
        except Exception as e:
            logger.warning(f"Failed to load positions: {e}")

    def _is_market_hours(self) -> bool:
        """Check if current time is within market hours (NYSE 9:30 AM - 4:00 PM ET)"""
        if not self.market_hours_only:
            return True

        from datetime import datetime
        import pytz

        et_tz = pytz.timezone('US/Eastern')
        now_et = datetime.now(et_tz)

        # Weekend check
        if now_et.weekday() >= 5:
            return False

        # Market hours check
        market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)

        return market_open <= now_et <= market_close

    def _refresh_market_data(self):
        """Refresh cached market data (called every 30-60 minutes)"""
        logger.info(f"[DATA_REFRESH] Starting data refresh for {len(self.symbols)} symbols...")
        refresh_start = time.time()

        with self._data_lock:
            try:
                # Fetch latest prices in parallel
                latest_prices = self.router.get_latest_prices_parallel(self.symbols)
                self._cached_prices = latest_prices
                self._last_price_update = datetime.utcnow()

                # Update position current prices
                with self._state_lock:
                    for symbol, pos in self.state.positions.items():
                        if symbol in self._cached_prices:
                            pos.current_price = self._cached_prices[symbol]
                            pos.last_update = datetime.utcnow()

                # Update signal current prices
                for symbol in self.state.signals:
                    if symbol in self._cached_prices:
                        self.state.signals[symbol].current_price = self._cached_prices[symbol]

                refresh_duration = time.time() - refresh_start
                logger.info(f"[DATA_REFRESH] Completed in {refresh_duration:.2f}s | Updated {len(latest_prices)} prices")
                self.state.last_data_refresh = datetime.utcnow()

            except Exception as e:
                logger.error(f"[DATA_REFRESH] Failed: {e}")
                self.state.error_count += 1

    def _should_refresh_data(self) -> bool:
        """Check if it's time to refresh market data"""
        if self._last_price_update is None:
            return True
        elapsed = (datetime.utcnow() - self._last_price_update).total_seconds()
        return elapsed >= self.data_refresh_interval

    def _get_latest_prices(self) -> Dict[str, float]:
        """Get latest prices (from cache or fetch if needed)"""
        if self._should_refresh_data():
            self._refresh_market_data()

        with self._data_lock:
            return deepcopy(self._cached_prices)

    def _fetch_fresh_price(self, symbol: str) -> Optional[float]:
        """Fetch a fresh price for a single symbol"""
        try:
            price = self.router.get_latest_price(symbol)
            if price:
                with self._data_lock:
                    self._cached_prices[symbol] = price
                return price
        except Exception as e:
            logger.debug(f"Failed to fetch fresh price for {symbol}: {e}")

        # Fallback to cached
        with self._data_lock:
            return self._cached_prices.get(symbol)

    def _compute_signals(self) -> Dict[str, LiveSignal]:
        """
        Compute live signals for all symbols.

        Uses cached historical data + latest prices to make decisions.
        """
        signals = {}

        # Get latest prices
        latest_prices = self._get_latest_prices()

        for symbol in self.symbols:
            try:
                # Get fresh price for this symbol
                current_price = self._fetch_fresh_price(symbol)
                if current_price is None:
                    logger.warning(f"No price available for {symbol}")
                    continue

                # Build agent outputs (simulated for real-time)
                # In production, this would call actual agent models
                agent_outputs = self._generate_agent_outputs(symbol, current_price, latest_prices)

                # Get positions
                current_position = self.state.positions.get(symbol, LivePosition(
                    symbol=symbol,
                    quantity=0.0,
                    entry_price=0.0,
                    current_price=current_price,
                    entry_time=datetime.utcnow(),
                    last_update=datetime.utcnow()
                ))

                # Make decision via MetaBrain
                decision = self._make_decision(
                    symbol=symbol,
                    agent_outputs=agent_outputs,
                    current_price=current_price,
                    current_position_qty=current_position.quantity
                )

                # Smooth signals
                smoothed_signal = self._smooth_signal(symbol, decision)

                # Create LiveSignal with all fields aligned to GovernanceDecision
                signals[symbol] = LiveSignal(
                    symbol=symbol,
                    signal=decision.final_decision,
                    cycle_id=decision.cycle_id,
                    timestamp=decision.timestamp,
                    mu_hat=decision.mu_hat,
                    sigma_hat=decision.sigma_hat,
                    conviction=decision.conviction,
                    cvar=0.0,  # Would be calculated from risk_state if available
                    model_confidence=decision.provider_confidence,
                    data_quality=decision.data_quality_score,
                    expected_return=decision.mu_hat,
                    expected_risk=decision.sigma_hat,
                    position_size=decision.position_size,
                    reason_codes=decision.reason_codes,
                    vetoed=False,
                    veto_reason="",
                    cvar_limit_check=True,
                    leverage_limit_check=True,
                    drawdown_limit_check=True,
                    correlation_limit_check=True,
                    sector_limit_check=True,
                    agent_breakdown={
                        name: {
                            'mu': contrib.mu,
                            'sigma': contrib.sigma,
                            'confidence': contrib.confidence,
                            'weight': contrib.weight
                        }
                        for name, contrib in (decision.agent_results or {}).items()
                    },
                    current_price=current_price,
                    last_decision=decision,
                    strategy_id="",
                    strategy_stage=""
                )

                # Update signal history
                self._signal_history[symbol].append({
                    'signal': decision.final_decision,
                    'mu_hat': decision.mu_hat,
                    'conviction': decision.conviction,
                    'timestamp': datetime.utcnow()
                })

                # Trim history
                if len(self._signal_history[symbol]) > self._signal_smoothing_window:
                    self._signal_history[symbol].pop(0)

                # Callback
                if self.on_decision_callback:
                    self.on_decision_callback(symbol, decision)

            except Exception as e:
                logger.error(f"Signal computation failed for {symbol}: {e}")
                self.state.consecutive_errors += 1

        # [GOVERNANCE] KILL SWITCH: STALE FEATURE ENFORCEMENT
        # Count rejections due to stale features
        stale_count = 0
        active_count = 0

        for sig in signals.values():
            active_count += 1
            # Check if REJECT and reason is STALE_FEATURES
            # LiveSignal object construction above maps 'signal' to decision.final_decision
            # We need to access reason codes. LiveSignal doesn't store reason codes directly
            # but we can infer or pass it.
            # Or better, check the last_decision object stored in LiveSignal
            if sig.last_decision and sig.last_decision.final_decision == "REJECT":
                # Check reason codes
                if sig.last_decision.reason_codes and "STALE_FEATURES" in sig.last_decision.reason_codes:
                    stale_count += 1

        if active_count > 0:
            stale_pct = (stale_count / active_count) * 100.0
            if stale_pct > 10.0: # 10% tolerance
                msg = f"[KILL_SWITCH] HALTING LIVE TRADING: {stale_count}/{active_count} ({stale_pct:.1f}%) symbols have STALE FEATURES (>24h)."
                logger.critical(msg)
                logger.critical("Immediate Action Required: Run 'python scripts/feature_refresh.py' to update data.")

                # Halt the loop
                self.running = False
                self.state.system_status = "HALTED_GOVERNANCE"
                raise SystemExit(msg)

        return signals

    def _generate_agent_outputs(self, symbol: str, current_price: float, latest_prices: Dict[str, float]) -> List[Dict]:
        """Generate agent outputs for real-time decision making"""
        outputs = []

        # Get historical data for this symbol
        try:
            from database.manager import get_db
            db = get_db()

            # Fetch recent price data from database
            prices = db.get_price_history(symbol, limit=60)

            if prices is not None and len(prices) > 20:
                df = prices.set_index('date') if 'date' in prices.columns else prices
                close = df['Close'] if 'Close' in df.columns else df.iloc[:, 0]
                returns = np.log(close / close.shift(1)).dropna()

                # Technical Momentum Agent
                mom_1m = returns.tail(21).sum() if len(returns) >= 21 else 0.0
                mom_3m = returns.tail(63).sum() if len(returns) >= 63 else 0.0
                outputs.append({
                    'agent_name': 'MomentumAgent',
                    'mu': 0.3 * mom_1m + 0.3 * mom_3m,
                    'sigma': 0.15,
                    'confidence': 0.7,
                    'metadata': {'type': 'technical'}
                })

                # Mean Reversion Agent
                short_term = close.pct_change(5).iloc[-1] if len(close) >= 5 else 0.0
                long_term = close.pct_change(60).iloc[-1] if len(close) >= 60 else 0.0
                outputs.append({
                    'agent_name': 'MeanReversionAgent',
                    'mu': long_term - short_term,
                    'sigma': 0.12,
                    'confidence': 0.6,
                    'metadata': {'type': 'statistical'}
                })

                # Volatility Signal
                current_vol = returns.tail(20).std() * np.sqrt(252) if len(returns) >= 20 else 0.15
                hist_vol = returns.tail(252).std() * np.sqrt(252) if len(returns) >= 252 else 0.15
                vol_ratio = current_vol / (hist_vol + 1e-10)
                outputs.append({
                    'agent_name': 'VolatilityAgent',
                    'mu': -0.5 * (vol_ratio - 1.0),
                    'sigma': 0.10,
                    'confidence': 0.65,
                    'metadata': {'type': 'garch'}
                })

                # Pattern Signal
                sma_20 = close.rolling(20).mean().iloc[-1] if len(close) >= 20 else close.iloc[-1]
                sma_50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else close.iloc[-1]
                trend = 0.1 if close.iloc[-1] > sma_20 > sma_50 else (-0.1 if close.iloc[-1] < sma_20 < sma_50 else 0.0)

                # RSI
                delta = close.diff()
                gain = delta.where(delta > 0, 0).rolling(14).mean().iloc[-1] if len(close) >= 14 else 0.0
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean().iloc[-1] if len(close) >= 14 else 0.0
                rs = gain / (loss + 1e-10)
                rsi = 100 - (100 / (1 + rs))
                rsi_signal = 0.1 if rsi < 30 else (-0.1 if rsi > 70 else 0.0)

                outputs.append({
                    'agent_name': 'PatternAgent',
                    'mu': trend + rsi_signal,
                    'sigma': 0.08,
                    'confidence': 0.55,
                    'metadata': {'type': 'pattern'}
                })

            else:
                # Fallback: Use cached data or simple signals
                outputs = self._generate_fallback_outputs(symbol, current_price)

        except Exception as e:
            logger.debug(f"Agent output generation failed for {symbol}: {e}")
            outputs = self._generate_fallback_outputs(symbol, current_price)

        return outputs

    def _generate_fallback_outputs(self, symbol: str, current_price: float) -> List[Dict]:
        """Generate simple fallback outputs when historical data unavailable"""
        return [
            {
                'agent_name': 'MomentumAgent',
                'mu': 0.02,
                'sigma': 0.15,
                'confidence': 0.5,
                'metadata': {'type': 'fallback'}
            },
            {
                'agent_name': 'MeanReversionAgent',
                'mu': -0.01,
                'sigma': 0.12,
                'confidence': 0.5,
                'metadata': {'type': 'fallback'}
            },
            {
                'agent_name': 'VolatilityAgent',
                'mu': 0.0,
                'sigma': 0.10,
                'confidence': 0.5,
                'metadata': {'type': 'fallback'}
            }
        ]

    def _make_decision(
        self,
        symbol: str,
        agent_outputs: List[Dict],
        current_price: float,
        current_position_qty: float
    ) -> SymbolDecision:
        """Make trading decision via MetaBrain"""
        cycle_id = f"live_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        # Organize outputs by symbol
        symbol_outputs = {symbol: agent_outputs}

        # Get portfolio state
        portfolio_state = {
            'nav': 100000.0,  # Would load from broker
            'cash': 10000.0,
            'positions': {symbol: current_position_qty for symbol in self.symbols},
            'leverage_used': sum(abs(pos.quantity) for pos in self.state.positions.values()) / 100000.0
        }

        # Get risk state
        risk_state = {
            'is_risk_on': True,
            'regime': 'BULL_QUIET',
            'regime_scalar': 1.0,
            'risk_override': False,
            'cvar_breach': False,
            'portfolio_leverage': portfolio_state['leverage_used'],
            'violations': []
        }

        # Get features (cached or empty)
        features = self._cached_features.get(symbol, {})

        # Make decision
        start_time = time.time()

        # [GOVERNANCE] Check if signal was rejected due to freshness in strategy layer
        # (This logic is usually handled by the strategy returning a REJECT decision,
        # but we need to count them for the kill switch).
        # We don't have direct access to internal strategy rejections here unless we inspect
        # the decision object returned by the strategy (MetaBrain delegating to Strategy).
        # MetaBrain uses InstitutionalStrategy.

        decisions = self.meta_brain.make_decisions(
            cycle_id=cycle_id,
            symbol_agent_outputs=symbol_outputs,
            symbol_features={symbol: features},
            symbol_positions={symbol: current_position_qty},
            portfolio_state=portfolio_state,
            risk_state=risk_state
        )

        # [GOVERNANCE] Kill Switch Logic
        # We need to track how many symbols are being rejected due to STALE_FEATURES
        # Since this method is called per-symbol in the loop (inefficient but matches current arch),
        # we can't easily aggregate across the universe here.
        # However, _run_decision_tick calls _compute_signals which loops all symbols.
        # Check _compute_signals implementation.

        latency = time.time() - start_time
        self._decision_latency.append(latency)

        decision = decisions.get(symbol)
        if decision:
            return decision

        # Fallback
        return SymbolDecision(
            symbol=symbol,
            cycle_id=cycle_id,
            final_decision=DECISION_HOLD,
            reason_codes=['fallback'],
            mu_hat=0.0,
            sigma_hat=0.1,
            conviction=0.0
        )

    def _smooth_signal(self, symbol: str, decision: SymbolDecision) -> SymbolDecision:
        """
        Smooth signals to reduce noise.
        Uses moving average of recent mu_hat values.
        """
        history = self._signal_history.get(symbol, [])

        if len(history) < 3:
            return decision

        # Average recent mu_hat values
        recent_mu = sum(h['mu_hat'] for h in history[-5:]) / min(5, len(history))
        recent_conviction = sum(h['conviction'] for h in history[-5:]) / min(5, len(history))

        # Apply smoothing (50% weight to smoothed value)
        smoothed_mu = 0.5 * recent_mu + 0.5 * decision.mu_hat
        smoothed_conviction = 0.5 * recent_conviction + 0.5 * decision.conviction

        # Update decision
        decision.mu_hat = smoothed_mu
        decision.conviction = smoothed_conviction

        return decision

    def _update_risk_metrics(self):
        """Update portfolio-level risk metrics"""
        try:
            total_exposure = sum(pos.market_value for pos in self.state.positions.values())
            total_pnl = sum(pos.unrealized_pnl for pos in self.state.positions.values())

            # Simple risk metrics
            leverage = total_exposure / 100000.0  # Assuming 100k NAV

            self.state.risk_metrics = {
                'total_exposure': total_exposure,
                'total_pnl': total_pnl,
                'leverage': leverage,
                'positions_count': len(self.state.positions),
                'symbols_tracked': len(self.state.signals)
            }

        except Exception as e:
            logger.debug(f"Risk metric update failed: {e}")




    @trace_span("run_decision_tick")
    def _run_decision_tick(self):
        """Execute a single decision tick"""
        tick_start = time.time()

        with metrics.metrics['cycle_latency'].time():
            try:
                # Check market hours
                if not self._is_market_hours():
                    logger.debug("[MARKET_HOURS] Outside trading hours, skipping tick")
                    return

                # Compute signals
                signals = self._compute_signals()

                with self._state_lock:
                    self.state.signals = signals
                    self.state.cycle_count += 1
                    self.state.last_decision_time = datetime.utcnow()
                    self.state.consecutive_errors = 0
                    self.state.system_status = "RUNNING"

                # Update risk metrics
                self._update_risk_metrics()

                # Update Observability
                metrics.set_nav(self.state.risk_metrics.get('total_exposure', 0), str(self.state.cycle_count))

                # Log heartbeat
                logger.debug(f"[HEARTBEAT] Cycle #{self.state.cycle_count} | Signals: {len(signals)}")

            except Exception as e:
                logger.error(f"[TICK_ERROR] {e}")
                with self._state_lock:
                    self.state.error_count += 1
                    self.state.consecutive_errors += 1
                    self.state.system_status = "ERROR"

        # Track timing
        tick_duration = time.time() - tick_start
        self._tick_times.append(tick_duration)


        # Log timing stats periodically
        if self.state.cycle_count % 100 == 0:
            avg_tick = np.mean(self._tick_times[-100:])
            avg_latency = np.mean(self._decision_latency[-100:])
            logger.info(f"[PERF] Avg tick: {avg_tick*1000:.1f}ms | Avg decision latency: {avg_latency*1000:.1f}ms")

    def _get_uptime(self) -> str:
        """Get formatted uptime string"""
        elapsed = datetime.utcnow() - self.state.start_time
        hours, remainder = divmod(int(elapsed.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours}h {minutes}m {seconds}s"

    def get_status(self) -> Dict[str, Any]:
        """Get current system status"""
        with self._state_lock:
            return {
                'cycle_count': self.state.cycle_count,
                'uptime': self._get_uptime(),
                'system_status': self.state.system_status,
                'positions': {
                    sym: {
                        'quantity': pos.quantity,
                        'current_price': pos.current_price,
                        'unrealized_pnl': pos.unrealized_pnl,
                        'pnl_pct': pos.pnl_pct
                    }
                    for sym, pos in self.state.positions.items()
                },
                'signals': {
                    sym: {
                        'signal': sig.signal,
                        'mu_hat': sig.mu_hat,
                        'conviction': sig.conviction,
                        'data_quality': sig.data_quality
                    }
                    for sym, sig in self.state.signals.items()
                },
                'last_decision_time': self.state.last_decision_time.isoformat() if self.state.last_decision_time else None,
                'last_data_refresh': self.state.last_data_refresh.isoformat() if self.state.last_data_refresh else None,
                'error_count': self.state.error_count,
                'consecutive_errors': self.state.consecutive_errors,
                'risk_metrics': self.state.risk_metrics
            }

    def start(self):
        """Start the live decision loop"""
        self.running = True
        logger.info("=" * 80)
        logger.info("LIVE DECISION LOOP STARTED")
        logger.info(f"Tick Interval: {self.tick_interval}s")
        logger.info(f"Data Refresh: Every {self.data_refresh_interval}s ({self.data_refresh_interval/60:.0f} min)")
        logger.info(f"Market Hours Only: {self.market_hours_only}")
        logger.info(f"Symbols: {len(self.symbols)}")
        logger.info("=" * 80)

        # Initial data refresh
        self._refresh_market_data()

        # STARTUP GOVERNANCE GATE (Phase 3)
        self._enforce_startup_governance()

        # Main loop
        while self.running:
            try:
                self._run_decision_tick()

                # Calculate sleep time to maintain interval
                elapsed = time.time() - (time.time() - self.tick_interval)  # Approximate
                sleep_time = max(0, self.tick_interval - elapsed)
                time.sleep(sleep_time)

            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received")
                break
            except Exception as e:
                logger.error(f"Unexpected error in decision loop: {e}")
                time.sleep(self.tick_interval)

    def stop(self):
        """Stop the live decision loop"""
        logger.info("Stopping live decision loop...")
        self.running = False


if __name__ == "__main__":
    # Test the live decision loop
    loop = LiveDecisionLoop(
        tick_interval=1.0,
        data_refresh_interval_min=30,
        paper_mode=True,
        market_hours_only=True
    )

    try:
        loop.start()
    except KeyboardInterrupt:
        loop.stop()

