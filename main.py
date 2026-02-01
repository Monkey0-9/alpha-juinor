#!/usr/bin/env python3
"""
Institutional Live Trading Agent

Responsibilities:
- 24/7 opportunity scanner with per-second decision loop.
- Mandated Phase 0 Governance Gate (1260 days of history).
- Uses cached historical features (no multi-year fetches in loop).
- Model persistence (load once).
"""

import argparse
import json
import os
import signal
import sys
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
import warnings

import numpy as np
import pandas as pd
try:
    from dotenv import load_dotenv
except ImportError:
    # Fail gracefully if python-dotenv is not installed
    warnings.warn(
        "python-dotenv not found. Falling back to OS environment variables."
    )

    def load_dotenv(**kwargs):
        pass

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from brokers.alpaca_broker import AlpacaExecutionHandler  # noqa: E402
from brokers.mock_broker import MockBroker  # noqa: E402
from configs.config_manager import ConfigManager  # noqa: E402
from database.manager import DatabaseManager  # noqa: E402
from portfolio.allocator import InstitutionalAllocator  # noqa: E402
from risk.engine import RiskManager  # noqa: E402
from strategies.factory import StrategyFactory  # noqa: E402
from utils.logging_config import setup_logging  # noqa: E402
from utils.metrics import metrics  # noqa: E402
from risk.kill_switch import GlobalKillSwitch  # noqa: E402
from governance.do_not_trade import allow_trading  # noqa: E402
from execution.gates import ExecutionGatekeeper  # noqa: E402

# Institutional Warning Throttling
warnings.filterwarnings("once", category=UserWarning)
warnings.filterwarnings("once", category=DeprecationWarning)
warnings.filterwarnings(
    "ignore", message="X does not have valid feature names"
)

# Configure Institutional Logging
logger = setup_logging("LIVE_AGENT", log_dir="runtime/logs")


class GovernanceError(Exception):
    """Custom exception for institutional governance violations."""

    def __init__(self, code, message):
        self.code = code
        self.message = message
        super().__init__(f"[{code}] {message}")


# =============================================================================
# INSTITUTIONAL GOVERNANCE CONSTANTS
# =============================================================================
REQUIRED_HISTORY_ROWS = 1260  # 5 years of daily data
KILL_SWITCH_PATH = "runtime/KILL_SWITCH"


def check_kill_switch() -> bool:
    """
    Check for binary Kill Switch file.
    If 'runtime/KILL_SWITCH' exists, the system must HALT immediately.

    Returns:
        True if kill switch is active, False otherwise
    """
    if os.path.exists(KILL_SWITCH_PATH):
        msg = f"[KILL_SWITCH] Activated: '{KILL_SWITCH_PATH}' found. " \
              f"System HALTING."
        logger.critical(msg)
        return True
    return False


def governance_halt(
    affected_symbols: List[str], reason: str = "Missing historical data"
) -> None:
    """
    Emit institutional governance log and halt the system.

    This function MUST be called when:
    - Historical data is missing for active symbols
    - Data quality threshold is not met
    - Any other governance violation that prevents trading

    Args:
        affected_symbols: List of symbols affected by the governance issue
        reason: Reason for the halt
    """
    logger.critical("[DATA_GOVERNANCE] Missing historical data detected")
    logger.critical(f"Symbols affected: {len(affected_symbols)}")
    logger.critical(f"Required rows per symbol: {REQUIRED_HISTORY_ROWS}")
    logger.critical("Action required: Run ingest_history.py")
    logger.critical("System halted intentionally")

    # Write to governance log file
    os.makedirs("runtime", exist_ok=True)
    with open("runtime/governance_halt.log", "a") as f:
        f.write("[DATA_GOVERNANCE]\n")
        f.write(f"Timestamp: {datetime.utcnow().isoformat()}\n")
        f.write(f"Reason: {reason}\n")
        f.write(f"Symbols affected: {len(affected_symbols)}\n")
        f.write(f"Required rows per symbol: {REQUIRED_HISTORY_ROWS}\n")
        f.write("Action required: Run ingest_history.py\n")
        f.write("System halted intentionally\n\n")

    sys.exit(1)


def heartbeat_worker(logger, interval=5):
    """Daemon thread for institutional heartbeat logging."""
    while True:
        try:
            # Calculate system state
            state = "OK"
            if metrics.model_errors > 100:
                state = "HALTED"
            elif metrics.model_errors > 0 or metrics.arima_fallbacks > 10:
                state = "DEGRADED"

            # Calculate ML state
            # Note: We can't directly access MLAlpha instance here,
            # so we infer from metrics
            from configs.config_manager import ConfigManager
            try:
                cfg = ConfigManager().config
                ml_enabled_config = cfg.get("features", {}).get(
                    "ml_enabled", False
                )
            except Exception:
                ml_enabled_config = False

            if not ml_enabled_config:
                ml_state = "DISABLED_CONFIG"
            elif metrics.model_errors > 50:
                ml_state = "DISABLED_GOVERNANCE"
            elif metrics.model_errors > 10:
                ml_state = "ENABLED|DEGRADED"
            else:
                ml_state = "ENABLED|OK"

            msg = (
                f"[HEARTBEAT] uptime={metrics.uptime_sec}s | "
                f"symbols={metrics.symbols_count} | cycles={metrics.cycles} | "
                f"state={state} | ml_state={ml_state} | "
                f"model_errors={metrics.model_errors} | "
                f"arima_fb={metrics.arima_fallbacks}"
            )
            logger.info(msg)
        except Exception as e:
            logger.error(f"Heartbeat worker error: {e}")
        time.sleep(interval)


def check_history_completeness(
    db: DatabaseManager, symbols: List[str]
) -> Dict[str, Any]:
    """
    Check if all symbols have the required 1260 rows of history.

    Args:
        db: DatabaseManager instance
        symbols: List of symbols to check

    Returns:
        Dict with 'compliant' (bool), 'missing' (list), 'counts' (dict)
    """
    if not symbols:
        return {"compliant": True, "missing": [], "counts": {}}

    placeholders = ",".join(["?"] * len(symbols))

    query = f"""
        SELECT symbol, COUNT(*) as row_count
        FROM price_history
        WHERE symbol IN ({placeholders})
        GROUP BY symbol
    """

    with db.get_connection() as conn:
        cursor = conn.execute(query, symbols)
        results = cursor.fetchall()

    def _get_val(row, key, index):
        try:
            return row[key]
        except Exception:
            return row[index]

    count_map = {
        _get_val(row, "symbol", 0): _get_val(row, "row_count", 1)
        for row in results
    }
    missing = []

    for symbol in symbols:
        count = count_map.get(symbol, 0)
        if count < REQUIRED_HISTORY_ROWS:
            missing.append({
                "symbol": symbol,
                "actual": count,
                "required": REQUIRED_HISTORY_ROWS
            })

    return {
        "compliant": len(missing) == 0,
        "missing": missing,
        "counts": count_map
    }


def check_1260_rows_requirement(
    db: DatabaseManager, symbols: List[str]
) -> bool:
    """
    ABSOLUTE PRECHECK: Verify all symbols have >= 1260 rows.

    This is the MANDATORY governance gate before trading can start.
    If any symbol has fewer than 1260 rows, the system MUST HALT.

    Args:
        db: DatabaseManager instance
        symbols: List of symbols to verify

    Returns:
        True if all symbols have required history, False otherwise
    """
    if not symbols:
        logger.warning("[DATA_GOVERNANCE] No symbols to check")
        return True

    msg = f"[DATA_GOVERNANCE] Verifying {len(symbols)} symbols have " \
          f">= {REQUIRED_HISTORY_ROWS} rows..."
    logger.info(msg)

    result = check_history_completeness(db, symbols)

    if result["compliant"]:
        msg = f"[DATA_GOVERNANCE] [PASS] All {len(symbols)} symbols have " \
              f">= {REQUIRED_HISTORY_ROWS} rows"
        logger.info(msg)
        return True
    else:
        missing_symbols = [m["symbol"] for m in result["missing"]]
        msg = f"[DATA_GOVERNANCE] [FAIL] {len(missing_symbols)} symbols " \
              f"missing required history"
        logger.critical(msg)

        # Log details of failed symbols
        for m in result["missing"][:10]:  # Log first 10
            msg = f"  - {m['symbol']}: {m['actual']} rows " \
                  f"(need {m['required']})"
            logger.critical(msg)
        if len(result["missing"]) > 10:
            msg = f"  ... and {len(result['missing']) - 10} more"
            logger.critical(msg)

        # Emit governance halt
        governance_halt(missing_symbols, "Insufficient historical data")
        return False


class InstitutionalLiveAgent:
    """
    Institutional Live Trading Agent.
    Strictly follows Master Prompt (Section 4).
    """

    def __init__(self, tickers: Optional[List[str]] = None):
        load_dotenv(override=True)
        self.cm = ConfigManager()
        self.cfg = self.cm.config
        self.db = DatabaseManager()

        # Universe loading
        if tickers:
            self.tickers = tickers
        else:
            try:
                with open("configs/universe.json", "r") as f:
                    universe = json.load(f)
                self.tickers = universe.get("active_tickers", [])
            except Exception:
                self.tickers = []

        metrics.symbols_count = len(self.tickers)

        # Components
        self.running = False
        self.safe_mode = False

        # DEFENSIVE: Initialize error tracking (Emergency Fix)
        self.loop_errors = 0
        self.unstable_count = 0

        # DEFENSIVE: Initialize components as None
        # (will be set in initialize_system)
        self.strategy = None
        self.allocator = None
        self.risk_mgr = None
        self.handler = None
        self.market_data = None

        self.execution_gate = ExecutionGatekeeper()

        # High-Priority Integrations
        from risk.regime_controller import RegimeController
        from ui.terminal_dashboard import TerminalDashboard
        self.regime_controller = RegimeController()
        self.dashboard = TerminalDashboard()
        self.kill_switch = GlobalKillSwitch()
        logger.info(
            "[AGENT] RegimeController, Dashboard, and KillSwitch initialized"
        )

        # State
        self.last_heartbeat = datetime.utcnow()
        self.last_feature_refresh = None
        self.portfolio_state = {}  # symbol: qty

        # Signal handlers
        signal.signal(signal.SIGINT, self._handle_exit)
        signal.signal(signal.SIGTERM, self._handle_exit)

    def _handle_exit(self, signum, frame):
        logger.info("Shutdown signal received. Stopping live agent...")
        self.running = False

    def check_kill_switch(self) -> bool:
        """
        Check for binary Kill Switch file.
        If 'kill_switch.txt' exists, the system must HALT immediately.
        """
        if os.path.exists("kill_switch.txt"):
            msg = "KILL SWITCH ACTIVATED: 'kill_switch.txt' found. " \
                  "System HALTING."
            logger.critical(msg)
            return True
        return False

    def check_governance_gate(self) -> bool:
        """
        ABSOLUTE PRECHECK: Load ONLY symbols with state = ACTIVE.
        Do NOT error if zero active symbols; start in Safe Mode.
        """
        msg = "[DATA_GOVERNANCE] Phase 0: Verifying Institutional " \
              "Symbol Governance..."
        logger.info(msg)

        try:
            active_symbols = self.db.get_active_symbols()
        except Exception as e:
            logger.error(
                f"[DATA_GOVERNANCE] Failed to query active symbols: {e}"
            )
            active_symbols = []

        if self.tickers:
            original_tickers = set(self.tickers)
            self.tickers = [s for s in self.tickers if s in active_symbols]
            excluded = original_tickers - set(self.tickers)
            if excluded:
                msg = f"[DATA_GOVERNANCE] Excluded {len(excluded)} " \
                      f"non-ACTIVE symbols."
                logger.warning(msg)
        else:
            self.tickers = active_symbols

        if not self.tickers:
            msg = "[DATA_GOVERNANCE] Zero ACTIVE symbols detected. " \
                  "System running in safe mode."
            logger.info(msg)
            return True

        msg = f"[DATA_GOVERNANCE] [READY] System starting with " \
              f"{len(self.tickers)} ACTIVE symbols."
        logger.info(msg)
        return True

    def load_252d_market_data(self) -> pd.DataFrame:
        """
        MANDATORY: Load last 252 daily bars for ACTIVE symbols ONLY.
        Returns a multi-index DataFrame with (symbol, OHLCV) columns.
        """
        from utils.market_loader import load_market_data

        if not self.tickers:
            return pd.DataFrame()

        msg = f"[DATA_GOVERNANCE] Loading 252-day window for " \
              f"{len(self.tickers)} symbols..."
        logger.info(msg)

        # Use simple list of symbols
        market_data_dict = load_market_data(self.tickers, lookback=252)

        if not market_data_dict:
            logger.critical("[DATA_GOVERNANCE] Missing historical data")
            logger.critical(f"Symbols affected: {len(self.tickers)}")
            logger.critical("Required rows per symbol: 1260")
            logger.critical("Action required: Run ingest_history.py")
            logger.critical("System halted intentionally")
            import sys

            sys.exit(1)

        logger.info(f"Loaded market_data for {len(market_data_dict)} symbols")

        # Convert dictionary {symbol: df} to MultiIndex DataFrame
        # matching expected format.
        # Expected: MultiIndex columns (Symbol, Field) or similar?
        # Existing code: combined = pd.concat(all_data, axis=1) ->
        # resulting in (Field, Symbol)? Or (Symbol, Field)?
        # pd.concat({symbol: df}, axis=1) produces (Symbol, Field)
        # as columns if keys are passed.

        # combined = pd.concat(all_data, axis=1)
        # This results in MultiIndex columns: Level 0 = Symbol,
        # Level 1 = Field (Open, High, etc.)

        # But wait, load_market_data returns dict of DF.
        # We need to ensure columns are capitalized (Open, High...)
        # The utils.market_loader selects lowercase: date, open, high...
        # We need to rename them.

        processed_data = {}
        for sym, df in market_data_dict.items():
            df = df.rename(
                columns={
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "close": "Close",
                    "volume": "Volume",
                }
            )
            processed_data[sym] = df[
                ["Open", "High", "Low", "Close", "Volume"]
            ]

        if not processed_data:
            return pd.DataFrame()

        combined = pd.concat(processed_data, axis=1)
        return combined

    def initialize_system(self):
        """Load models and components ONCE and keep warm."""
        logger.info("Initializing trading components...")

        self.risk_mgr = RiskManager(
            max_leverage=self.cfg["risk"]["max_gross_leverage"],
            target_vol_limit=self.cfg["risk"]["target_volatility_annualized"],
            initial_capital=self.cfg["execution"]["initial_capital"],
        )

        self.allocator = InstitutionalAllocator(
            risk_manager=self.risk_mgr,
            max_leverage=self.cfg["risk"]["max_gross_leverage"],
        )

        # 252-day Data Loading Rule
        self.market_data = self.load_252d_market_data()

        # Strategy setup with ACTIVE tickers that have data
        if isinstance(self.market_data.columns, pd.MultiIndex):
            symbols_with_data = (
                self.market_data.columns.get_level_values(0).unique().tolist()
            )
        else:
            if not self.market_data.empty:
                symbols_with_data = self.tickers
            else:
                symbols_with_data = []

        self.strategy = StrategyFactory.create_strategy(
            {
                "type": "institutional",
                "tickers": symbols_with_data,
                "use_ml": self.cfg["alpha"]["ml_weight"] > 0,
            }
        )

        # Broker setup
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")
        if (
            self.cfg["execution"]["mode"] in ["paper", "live"]
            and api_key
            and secret_key
        ):
            base_url = os.getenv(
                "ALPACA_API_URL", "https://paper-api.alpaca.markets"
            )
            self.handler = AlpacaExecutionHandler(
                api_key=str(api_key),
                secret_key=str(secret_key),
                base_url=base_url
            )
            logger.info(f"Using AlpacaExecutionHandler with {base_url}")
        else:
            self.handler = MockBroker()
            logger.info("Using MockBroker")

        logger.info("System initialized and warm.")

    def run_per_second_loop(self):
        """
        Main decision loop running every second.
        Uses in-memory features/cached data.
        """
        self.running = True
        logger.info("=" * 80)
        logger.info("INSTITUTIONAL LIVE ENGINE STARTED (PER-SECOND LOOP)")
        if self.market_data is None or self.market_data.empty:
            msg = "[DATA_GOVERNANCE] No active market data. " \
                  "System will remain in observation mode."
            logger.warning(msg)
        logger.info("=" * 80)

        # Start Heartbeat Thread (Institutional Monitoring)
        args = (logger, 5)
        h_thread = threading.Thread(
            target=heartbeat_worker, args=args, daemon=True
        )
        h_thread.start()

        while self.running:
            loop_start = time.time()

            try:
                # 0. Safety & Governance Gate (Consolidated)

                # Gather metrics for governance
                current_nav = 1_000_000.0  # Default
                try:
                    if self.handler:
                        acct = self.handler.get_account()
                        if acct:
                            current_nav = float(getattr(acct, 'equity', 1e6))
                except Exception:
                    pass

                sys_metrics = {
                    "nav_usd": current_nav,
                    # Placeholder until regime engine provides distinct score
                    "regime_confidence": 1.0,
                    "hit_rate": getattr(metrics, 'hit_rate', 0.5),
                    "is_replay": False  # Live mode
                }

                gov_decision = allow_trading(sys_metrics)
                if not gov_decision["allow"]:
                    msg = f"[GOVERNANCE_HALT] Trading skipped: " \
                          f"{gov_decision['reason']}"
                    logger.warning(msg)
                    time.sleep(1)
                    continue

                # Legacy Kill Switch checks for redundancy
                if os.path.exists("runtime/KILL_SWITCH"):
                    logger.critical(
                        "[HALT] Manual KILL_SWITCH detected! Shutting down."
                    )
                    self.running = False
                    break

                self.last_heartbeat = datetime.utcnow()

                # 1. Regime Detection (High-Priority Integration)
                # Compute inputs for regime detection

                # 1a. VIX Fetch (from market data if available)
                vix_value = None
                try:
                    cols = self.market_data.columns
                    if (
                        isinstance(cols, pd.MultiIndex) and
                        '^VIX' in cols.get_level_values(0)
                    ):
                        vix_value = float(
                            self.market_data['^VIX']['Close'].iloc[-1]
                        )
                except Exception:
                    pass  # VIX not in universe, use None

                # 1b. Drawdown Computation
                drawdown_value = None
                try:
                    if hasattr(self.handler, 'get_account'):
                        acct = self.handler.get_account()
                        equity = float(getattr(acct, 'equity', 1e6))
                        le = float(getattr(acct, 'last_equity', equity))
                        # Simple DD = (current - peak) / peak
                        peak = max(equity, le)
                        dd_val = (equity - peak) / peak if peak > 0 else 0.0
                        drawdown_value = dd_val
                except Exception:
                    pass

                # 1c. Correlation Computation (Average pairwise)
                avg_corr = None
                try:
                    if (
                        not self.market_data.empty and
                        isinstance(self.market_data.columns, pd.MultiIndex)
                    ):
                        cols = self.market_data.columns
                        symbols = cols.get_level_values(0).unique().tolist()
                        if len(symbols) > 1:
                            # Filter to symbols actually in columns
                            valid_syms = [
                                s for s in symbols
                                if s in cols.get_level_values(0)
                            ]
                            closes = pd.DataFrame({
                                s: self.market_data[s]['Close']
                                for s in valid_syms
                            })
                            rets = closes.pct_change(fill_method=None).dropna()
                            if len(rets) > 20:  # Need enough data
                                corr_mat = rets.corr()
                                # Upper triangle mean (exclude diagonal)
                                mask = np.triu(
                                    np.ones(corr_mat.shape), k=1
                                ).astype(bool)
                                upper = corr_mat.where(mask)
                                avg_corr = upper.stack().mean()
                except Exception:
                    pass

                current_regime = self.regime_controller.detect_regime(
                    vix=vix_value,
                    drawdown=drawdown_value,
                    avg_correlation=avg_corr
                )

                # Apply regime overrides to config
                if current_regime.value != "NORMAL":
                    self.cfg = self.regime_controller.apply_overrides(
                        current_regime, self.cfg
                    )
                    msg = f"[REGIME] {current_regime.value} mode active"
                    logger.warning(msg)

                # 2. Error Rate Monitoring & Safe Mode (Priority 3 Fix)
                if getattr(self, 'loop_errors', 0) > 5:
                    msg = "[SAFE_MODE] High error rate detected (>5 " \
                          "consecutive). Reducing exposure."
                    logger.critical(msg)
                    self.safe_mode = True

                # Dynamic Safe Mode based on signal stability
                if self.market_data is None or self.market_data.empty:
                    # Safe Mode: Neutral Signals
                    time.sleep(1)
                    continue

                # Defensive: Ensure strategy initialized
                if self.strategy is not None:
                    signals = self.strategy.generate_signals(self.market_data)
                else:
                    logger.warning(
                        "Strategy not initialized, skipping signal generation"
                    )
                    signals = None

                # 4. Allocator Decision (with hedging overlay)
                if signals is not None and not signals.empty:
                    # Apply Safe Mode Scaling (Priority 3 Fix)
                    if getattr(self, "safe_mode", False):
                        logger.warning(
                            "[SAFE_MODE] Scaling signals by 0.1 to minimize."
                        )
                        signals = signals * 0.1

                    # Defensive: Ensure allocator initialized
                    if self.allocator is not None:
                        target_weights = self.allocator.allocate(
                            signals, data=self.market_data
                        )
                    else:
                        logger.warning(
                            "Allocator not initialized, using empty targets"
                        )
                        target_weights = {}

                    self.loop_errors = 0  # Reset on success
                    metrics.cycles += 1

                    # Generate Cycle ID for audit
                    ts = int(time.time())
                    current_cycle_id = f"live_{ts}_{metrics.cycles}"

                    # Log signal summary
                    if len(target_weights) > 0:
                        msg = f"Generated {len(target_weights)} " \
                              f"position targets"
                        logger.info(msg)

                        # --- Execution Decision Layer ---

                        from governance.execution_decision import \
                            decide_execution

                        executed_count = 0
                        skipped_count = 0
                        skip_reasons = {}  # reason -> count

                        # Get NAV
                        nav = 1000000.0  # Default fallback
                        if hasattr(self.handler, 'get_account'):
                            try:
                                acct = self.handler.get_account()
                                if hasattr(acct, 'equity'):  # Alpaca
                                    nav = float(acct.equity)
                                elif (
                                    isinstance(acct, dict) and 'equity' in acct
                                ):
                                    nav = float(acct['equity'])
                            except Exception:
                                pass

                        market_open_flag = self.execution_gate.is_market_open()
                        # Override check disabled for debug
                        if not market_open_flag:
                            # Check if broker can override (e.g. crypto)
                            if hasattr(self.handler, 'is_market_open'):
                                status = self.handler.is_market_open()
                                market_open_flag = status

                        # Skipping history (track in memory for live agent)
                        if not hasattr(self, "skipping_history"):
                            self.skipping_history = {}

                        e_cfg = self.cfg.get('execution', {})
                        exec_enabled = e_cfg.get('enabled', True)

                        # --- BATCH OPTIMIZATION: Fetch all positions once ---
                        current_positions_map = {}
                        if (
                            self.handler and
                            hasattr(self.handler, "get_positions")
                        ):
                            try:
                                raw_positions = self.handler.get_positions()
                                # Handle both Alpaca (list of dicts) and
                                # internal format
                                if isinstance(raw_positions, list):
                                    for p in raw_positions:
                                        # Alpaca returns dict with 'symbol' and
                                        # 'qty'
                                        sym = p.get('symbol')
                                        qty = float(p.get('qty', 0.0))
                                        current_positions_map[sym] = qty
                                elif isinstance(raw_positions, dict):
                                    current_positions_map = raw_positions
                            except Exception as e:
                                logger.error(
                                    f"Failed to fetch batch positions: {e}"
                                )

                        msg = f"[EXECUTION] Processing " \
                              f"{len(target_weights)} target signals..."
                        logger.info(msg)

                        for i, (symbol, target_w) in enumerate(
                            target_weights.items()
                        ):
                            # Periodic heartbeat for large loops
                            if i > 0 and i % 50 == 0:
                                msg = f"[EXECUTION] Processed {i}/" \
                                      f"{len(target_weights)} symbols..."
                                logger.info(msg)

                            # Optimized Lookup
                            qty_p = self.portfolio_state.get(symbol, 0.0)
                            current_qty = current_positions_map.get(
                                symbol, qty_p
                            )

                            price = 0.0
                            cols = self.market_data.columns
                            if (
                                not self.market_data.empty and
                                symbol in cols.get_level_values(0)
                            ):
                                price = float(
                                    self.market_data[symbol]['Close'].iloc[-1]
                                )

                            if price <= 0:
                                continue

                            current_val = current_qty * price
                            current_w = current_val / nav

                            # Conviction? Recover from signals.
                            conviction = 0.5  # Default
                            if (
                                signals is not None and
                                symbol in signals.columns
                            ):
                                val = signals[symbol].iloc[-1]
                                # Map signal to conviction.
                                if val > 0.8 or val < 0.2:
                                    conviction = 0.8

                            # Call Decision Layer
                            # Risk handled by allocator
                            risk_scaled_w = target_w

                            exec_res = {}
                            final_decision = 'SKIP_INTERNAL_ERROR'
                            reason_codes = ['UNKNOWN']

                            if exec_enabled:
                                exec_res = decide_execution(
                                    cycle_id=current_cycle_id,
                                    symbol=symbol,
                                    target_weight=target_w,
                                    current_weight=current_w,
                                    nav_usd=nav,
                                    price=price,
                                    conviction=conviction,
                                    data_quality=1.0,  # Passed gov check
                                    risk_scaled_weight=risk_scaled_w,
                                    skipping_history=self.skipping_history,
                                    market_open=market_open_flag,
                                    config=self.cfg
                                )

                                final_decision = exec_res.get(
                                    'decision', 'ERROR'
                                )
                                reason_codes = exec_res.get('reason_codes', [])

                                audit_payload = {
                                    "cycle_id": current_cycle_id,
                                    "symbol": symbol,
                                    "timestamp": datetime.utcnow().isoformat(),
                                    "decision_type": "execution",
                                    "final_decision": final_decision,
                                    "reason_codes": reason_codes,
                                    "target_weight": target_w,
                                    "current_weight": current_w,
                                    "target_qty": exec_res.get(
                                        "target_qty", 0
                                    ),
                                    "rounded_qty": exec_res.get(
                                        "target_qty", 0
                                    ),
                                    "notional_usd": exec_res.get(
                                        "notional_usd", 0
                                    ),
                                    "conviction": conviction,
                                    "market_open": market_open_flag,
                                    "broker_error": None
                                }

                                # Order Data
                                audit_payload["order"] = {
                                    "notional": exec_res.get(
                                        "notional_usd", 0
                                    ),
                                    "target_qty": exec_res.get(
                                        'target_qty', 0
                                    ),
                                    "broker_error": None
                                }

                                if final_decision != 'EXECUTE':
                                    # SKIP Logic
                                    skipped_count += 1
                                    self.skipping_history[symbol] = (
                                        self.skipping_history.get(symbol, 0)
                                        + 1
                                    )
                                    first_reason = (
                                        reason_codes[0] if reason_codes
                                        else "UNKNOWN"
                                    )
                                    skip_reasons[first_reason] = (
                                        skip_reasons.get(first_reason, 0) + 1
                                    )

                                    # Write Audit
                                    try:
                                        from audit.decision_log import \
                                            write_audit
                                        write_audit(audit_payload)
                                    except Exception as e:
                                        logger.error(
                                            f"Audit Write Failed: {e}"
                                        )

                                    continue
                                else:
                                    # EXECUTE Logic
                                    self.skipping_history[symbol] = 0  # Reset

                            # Execute Order
                            try:
                                # Calculate DELTA for Order
                                target_val = target_w * nav
                                target_shares = target_val / price
                                cur_shares = current_w * nav / price
                                order_shares = target_shares - cur_shares

                                # Rounding
                                if isinstance(
                                    self.handler,
                                    AlpacaExecutionHandler
                                ):
                                    # Use Alpaca wrapper
                                    pass

                                if abs(order_shares) > 0:
                                    side = 'buy' \
                                        if order_shares > 0 else 'sell'
                                    qty_to_trade = abs(order_shares)

                                    # --- FINAL EXECUTION GATE ---
                                    v_res = self.execution_gate \
                                        .validate_execution(
                                            symbol=symbol,
                                            qty=qty_to_trade,
                                            side=side,
                                            price=price,
                                            adv_30d=1e6,
                                            volatility=0.02
                                        )
                                    is_ok, reason, scaled_qty = v_res

                                    if not is_ok:
                                        msg = f"Final Gate REJECT: " \
                                              f"{symbol} | {reason}"
                                        logger.warning(msg)
                                        final_decision = \
                                            f"SKIP_{reason}"
                                        audit_payload[
                                            'final_decision'
                                        ] = final_decision
                                        rcs = audit_payload[
                                            'reason_codes'
                                        ]
                                        if reason not in rcs:
                                            rcs.append(reason)

                                        # Write Audit
                                        try:
                                            from audit.decision_log \
                                                import write_audit
                                            write_audit(audit_payload)
                                        except Exception:
                                            pass
                                        continue

                                    qty_to_trade = scaled_qty

                                    # Submit to handler
                                    if (
                                        self.handler and
                                        hasattr(self.handler, "submit_order")
                                    ):
                                        result = self.handler.submit_order(
                                            symbol=symbol,
                                            qty=qty_to_trade,
                                            side=side,
                                            type="market",
                                            time_in_force="day",
                                            price=price
                                        )

                                        if result["success"]:
                                            executed_count += 1
                                            # Optimistic update
                                            self.portfolio_state[symbol] = \
                                                target_shares
                                        else:
                                            # LOG BROKER FAILURE
                                            msg = f"Broker Submission " \
                                                  f"Failed: {result['error']}"
                                            logger.error(msg)
                                            audit_payload["decision"] = \
                                                "BROKER_FAILURE"
                                            audit_payload["broker_error"] = \
                                                result["error"]
                                            # Rewrite audit with error? Or
                                            # append?
                                            # Valid trade was attempted but
                                            # failed.
                                            # We should log this specific
                                            # outcome.

                                    else:
                                        # Mock fallback
                                        executed_count += 1

                                # Write Execution Audit (Success or Broker
                                # Failure)
                                try:
                                    from audit.decision_log import write_audit
                                    write_audit(audit_payload)
                                except Exception as e:
                                    logger.error(f"Audit Write Failed: {e}")

                            except Exception as e:
                                logger.error(
                                    f"Order Execution Logic Failed for "
                                    f"{symbol}: {e}"
                                )
                                # Try to audit crash
                                try:
                                    audit_payload['decision'] = 'CRASH'
                                    audit_payload['broker_error'] = str(e)
                                    from audit.decision_log import write_audit
                                    write_audit(audit_payload)
                                except Exception:
                                    pass

                        # Log Summary
                        sp = [f"{k}={v}" for k, v in skip_reasons.items()]
                        skip_summary = ", ".join(sp)
                        logger.info(
                            f"Execution summary: executed={executed_count} "
                            f"skipped={skipped_count} ({skip_summary})"
                        )

                else:
                    logger.debug("No signals generated, skipping allocation")

                # Update Dashboard
                try:
                    # ML Health calculation
                    ml_health = 1.0
                    if (
                        metrics.model_errors > 0 or
                        metrics.arima_fallbacks > 20
                    ):
                        ml_health = 0.7 if metrics.model_errors < 50 else 0.4

                    self.dashboard.update("ml", {
                        "health": ml_health,
                        "active_models": len(self.tickers),
                        "arima_fb": metrics.arima_fallbacks
                    })

                    self.dashboard.update("execution", {
                        "executed_count": executed_count,
                        "blocked_count": skipped_count
                    })

                    # Render every 5 seconds or if run_once
                    if (
                        metrics.cycles % 5 == 0 or
                        getattr(self, 'run_once', False)
                    ):
                        self.dashboard.render()
                except Exception as e:
                    logger.debug(f"Dashboard update failed: {e}")

                elapsed = time.time() - loop_start

                # Check run_once
                if getattr(self, 'run_once', False):
                    logger.info("Run-once complete. Exiting.")
                    self.running = False
                    break

                sleep_time = max(0, 60.0 - elapsed)
                time.sleep(sleep_time)

            except KeyboardInterrupt:
                logger.info(
                    "Shutdown signal received (Ctrl+C). Stopping live agent..."
                )
                self.running = False
                break
            except GovernanceError as ge:
                logger.critical(f"[GOVERNANCE_HALT] {ge.code}: {ge.message}")
                self.running = False
                break
            except Exception as e:
                import traceback

                logger.error(f"[ERROR] Unexpected loop crash: {e}")
                logger.error(traceback.format_exc())
                time.sleep(2)  # Prevent rapid fire logs

    def start(self):
        """
        Start the institutional live trading agent.

        MANDATORY SEQUENCE:
        1. Check Kill Switch
        2. Run Symbol Classification
        3. Get Active Symbols
        4. Verify 1260-Row Requirement (HALT if failed)
        5. Initialize System
        6. Run Trading Loop
        """
        # 0. Check Kill Switch
        if check_kill_switch():
            sys.exit(1)

        # 0. Check Runtime Kill Switch (legacy support)
        if self.check_kill_switch():
            sys.exit(1)

        # CRITICAL: Audit DB health check
        # Week 1 Blocker: Audit DB Mandatory at Startup
        try:
            if not self.db.check_table_exists("decisions"):
                logger.critical("[FATAL] Audit DB missing 'decisions' table")
                sys.exit(1)
        except Exception as e:
            logger.error(f"DB Check failed: {e}")
            # proceed or exit? exit.
            sys.exit(1)

        # 1. Institutional Governance Sweep (Classify symbols)
        from data.governance.governance_agent import SymbolGovernor

        governor = SymbolGovernor()
        logger.info(
            "[GOVERNANCE] Starting pre-flight symbol classification..."
        )
        governor.classify_all()

        # 2. Get Active Symbols (those with state=ACTIVE in symbol_governance)
        logger.info(
            "[DATA_GOVERNANCE] Fetching ACTIVE symbols from governance "
            "table..."
        )
        active_symbols = self.db.get_active_symbols()
        logger.info(
            f"[DATA_GOVERNANCE] Found {len(active_symbols)} ACTIVE symbols"
        )

        if not active_symbols:
            logger.critical("[DATA_GOVERNANCE] No ACTIVE symbols in database")
            governance_halt(
                [], "No ACTIVE symbols - run ingest_history.py first"
            )

        # 3. ABSOLUTE GOVERNANCE GATE: Verify 1260-Row Requirement
        # This MUST pass before any trading can occur
        logger.info("[DATA_GOVERNANCE] Running 1260-row requirement check...")
        check_1260_rows_requirement(self.db, active_symbols)

        # 4. Filter tickers to only ACTIVE symbols
        if self.tickers:
            original_tickers = set(self.tickers)
            self.tickers = [s for s in self.tickers if s in active_symbols]
            excluded = original_tickers - set(self.tickers)
            if excluded:
                logger.warning(
                    f"[DATA_GOVERNANCE] Excluded {len(excluded)} "
                    "non-ACTIVE symbols from command line"
                )
        else:
            self.tickers = active_symbols

        logger.info(
            f"[DATA_GOVERNANCE] [READY] System starting with "
            f"{len(self.tickers)} ACTIVE symbols"
        )

        # DEFENSIVE ASSERTIONS (Emergency Fix)
        assert self.db is not None, "Database manager not initialized"
        assert self.cfg is not None, "Configuration not loaded"
        assert len(self.tickers) > 0, "No active tickers for trading"
        logger.info("[ORCHESTRATION] Pre-flight assertions passed")

        # 5. Initialize System (Loads data for ACTIVE symbols)
        self.initialize_system()

        # DEFENSIVE ASSERTIONS (Post-Init)
        assert self.strategy is not None, "Strategy not initialized"
        assert self.allocator is not None, "Allocator not initialized"
        assert self.handler is not None, "Execution handler not initialized"
        logger.info("[ORCHESTRATION] System initialization verified")

        # 6. Run Trading Loop
        self.run_per_second_loop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Institutional Live Trading Agent"
    )
    parser.add_argument("--tickers", type=str, help="Filter tickers")
    parser.add_argument(
        "--run-once", action="store_true", help="Run a single cycle"
    )
    parser.add_argument(
        "--mode", type=str, choices=["paper", "live", "backtest"],
        help="Override trading mode"
    )
    args, unknown = parser.parse_known_args()

    tickers = args.tickers.split(",") if args.tickers else None

    agent = InstitutionalLiveAgent(tickers=tickers)

    # Apply CLI overrides
    if args.mode:
        agent.cfg["execution"]["mode"] = args.mode

    # Store run_once flag
    agent.run_once = args.run_once

    agent.start()
