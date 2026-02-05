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
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from statsmodels.tools.sm_exceptions import ConvergenceWarning, ValueWarning

from alpha.elite_factor_library import get_factor_library
from alpha.network_alpha import get_network_alpha
from analytics.advanced_technical import get_technical_analyzer
from analytics.pattern_recognition import get_pattern_engine
from analytics.precision_analyzer import get_precision_analyzer
from analytics.quant_engine import get_quant_engine
from analytics.regime_analyzer import get_regime_analyzer
from brokers.alpaca_broker import AlpacaExecutionHandler
from brokers.mock_broker import MockBroker
from compliance.compliance_engine import get_compliance_engine
from configs.config_manager import ConfigManager
from control.rl_meta_controller import get_rl_controller

# Phase 18: Global Financial AI (Temporal Intelligence & Proprietary Edge)
from core.global_session_tracker import get_global_session_tracker
from data.alternative.insider_tracker import get_insider_tracker
from data.alternative.options_flow import get_options_flow_analyzer
from data.alternative.order_flow import get_order_flow_analyzer
from database.manager import DatabaseManager
from execution.advanced_execution import get_execution_engine
from execution.gates import ExecutionGatekeeper
from execution.smart_order_router import get_smart_order_router
from execution.tca_engine import get_tca_engine
from execution.trade_manager import get_trade_manager
from execution.ultimate_executor import get_ultimate_executor
from governance.do_not_trade import allow_trading
from intelligence.autonomous_brain import get_autonomous_brain

# Phase 11: Autonomous Intelligence Layer
from intelligence.autonomous_reasoner import get_reasoner
from intelligence.crisis_alpha_scanner import get_crisis_scanner
from intelligence.decision_validator import get_validator
from intelligence.deep_alpha_brain import get_deep_brain
from intelligence.genius_picker import get_genius_picker

# Phase 17b: Learn-Trade-Learn Continuous Loop
from intelligence.learning_feedback import get_learning_feedback

# Phase 12: Ultimate 300 IQ Brain
from intelligence.master_orchestrator import get_master_brain
from intelligence.meta_learning_allocator import get_meta_allocator
from intelligence.perfect_timing import get_timing_engine

# Phase 16: Top 1% Institutional Intelligence
from intelligence.smart_money_detector import get_smart_money_detector
from intelligence.stock_scanner import get_scanner
from intelligence.strategy_selector import get_strategy_selector
from intelligence.supreme_engine import get_supreme_engine
from intelligence.ultimate_orchestrator import get_orchestrator

# Phase 13: Zero-Loss System
from intelligence.zero_loss_guardian import get_guardian

# Phase 1-10 Elite Intelligence Modules
from orchestration.smart_orchestrator import get_smart_orchestrator
from portfolio.allocator import InstitutionalAllocator
from portfolio.elite_optimizer import get_elite_optimizer
from research.network_analyzer import get_network_analyzer
from research_platform.cross_asset_flow import get_cross_asset_mapper
from research_platform.ensemble_predictor import get_ensemble_predictor
from research_platform.optimizer import get_optimizer

# Phase 17: Alpha Discovery Pipeline
from research_platform.research_engine import get_alpha_pipeline
from research_platform.strategy_validator import get_kill_switch
from risk.advanced_risk_manager import get_risk_engine
from risk.engine import RiskManager
from risk.kill_switch import GlobalKillSwitch
from risk.portfolio_guardian import get_portfolio_guardian
from risk.realtime_monitor import get_risk_monitor
from risk.stress_tester import get_stress_tester
from risk.stress_testing import get_stress_framework

# Phase 15: High-Risk High-Reward Strategies
from strategies.aggressive_recovery import get_hrhr_engine
from strategies.dip_buyer import get_dip_buyer
from strategies.factory import StrategyFactory

# Phase 14: Multi-Strategy Intelligence
from strategies.strategy_universe import get_strategy_universe
from utils.logging_config import setup_logging
from utils.metrics import metrics

# We use simplefilter to be aggressive about these specific categories
# because statsmodels emits warnings that get caught in race conditions.
warnings.simplefilter("ignore", category=UserWarning)
warnings.filterwarnings("once", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*numpy.core.numeric.*")
warnings.filterwarnings("ignore", message=".*declarative_base.*")
warnings.filterwarnings("ignore", message="X does not have valid feature name.*")
_pandas4_warning = getattr(getattr(pd, "errors", None), "Pandas4Warning", None)
if _pandas4_warning is not None:
    warnings.filterwarnings("ignore", category=_pandas4_warning)
warnings.filterwarnings("ignore", message=".*feature names.*")

# Statsmodels aggressive suppression
warnings.simplefilter("ignore", category=ValueWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)
warnings.simplefilter("ignore", category=RuntimeWarning)

warnings.filterwarnings("ignore", message=".*No supported index is available.*")
warnings.filterwarnings(
    "ignore", message=".*Maximum Likelihood optimization failed to converge.*"
)
warnings.filterwarnings("ignore", message=".*A date index has been provided.*")
warnings.filterwarnings("ignore", message=".*date index.*not monotonic.*")

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
        msg = (
            f"[KILL_SWITCH] Activated: '{KILL_SWITCH_PATH}' found. " f"System HALTING."
        )
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
            elif metrics.model_errors > 0 or metrics.arima_fallbacks > 100:
                state = "DEGRADED"

            # Calculate ML state
            # Note: We can't directly access MLAlpha instance here,
            # so we infer from metrics
            from configs.config_manager import ConfigManager

            try:
                cfg = ConfigManager().config
                ml_enabled_config = cfg.get("features", {}).get("ml_enabled", False)
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
        _get_val(row, "symbol", 0): _get_val(row, "row_count", 1) for row in results
    }
    missing = []

    for symbol in symbols:
        count = count_map.get(symbol, 0)
        if count < REQUIRED_HISTORY_ROWS:
            missing.append(
                {"symbol": symbol, "actual": count, "required": REQUIRED_HISTORY_ROWS}
            )

    return {"compliant": len(missing) == 0, "missing": missing, "counts": count_map}


def check_1260_rows_requirement(db: DatabaseManager, symbols: List[str]) -> bool:
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

    msg = (
        f"[DATA_GOVERNANCE] Verifying {len(symbols)} symbols have "
        f">= {REQUIRED_HISTORY_ROWS} rows..."
    )
    logger.info(msg)

    result = check_history_completeness(db, symbols)

    if result["compliant"]:
        msg = (
            f"[DATA_GOVERNANCE] [PASS] All {len(symbols)} symbols have "
            f">= {REQUIRED_HISTORY_ROWS} rows"
        )
        logger.info(msg)
        return True
    else:
        missing_symbols = [m["symbol"] for m in result["missing"]]
        msg = (
            f"[DATA_GOVERNANCE] [FAIL] {len(missing_symbols)} symbols "
            f"missing required history"
        )
        logger.critical(msg)

        # Log details of failed symbols
        for m in result["missing"][:10]:  # Log first 10
            msg = f"  - {m['symbol']}: {m['actual']} rows " f"(need {m['required']})"
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

        # Universe loading - FULL MARKET MODE
        # Load ALL active symbols from database for full market trading
        if tickers:
            self.tickers = tickers
        else:
            # Full market: Get ALL active symbols from database
            try:
                all_active = self.db.get_active_symbols()
                if all_active:
                    self.tickers = all_active
                    logger.info(
                        f"[FULL_MARKET] Loaded {len(self.tickers)} symbols "
                        "from database"
                    )
                else:
                    # Fallback to universe.json if database empty
                    with open("configs/universe.json", "r") as f:
                        universe = json.load(f)
                    self.tickers = universe.get("active_tickers", [])
                    logger.warning("[FULL_MARKET] Database empty, using universe.json")
            except Exception as e:
                logger.warning(f"[FULL_MARKET] DB query failed: {e}")
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
        logger.info("[AGENT] RegimeController, Dashboard, and KillSwitch initialized")

        # Phase 11: Autonomous Intelligence Layer
        try:
            self.autonomous_reasoner = get_reasoner()
            self.quant_engine = get_quant_engine()
            # Legacy feedback loop - we use Phase 17 one now
            # self.feedback_loop = get_feedback_loop()
            logger.info("[AGENT] Autonomous Intelligence Layer initialized")
        except Exception as e:
            logger.warning(f"[AGENT] Intelligence layer partial: {e}")

        # Phase 12: Ultimate 300 IQ Brain System
        try:
            self.master_brain = get_master_brain()
            self.deep_brain = get_deep_brain()
            self.decision_validator = get_validator()
            self.stock_scanner = get_scanner()
            logger.info("[AGENT] 300 IQ Master Brain System initialized")
        except Exception as e:
            logger.warning(f"[AGENT] Master Brain partial: {e}")

        # Phase 1-10 & 13-16: Elite Intelligence Integration
        try:
            # Phase 1-10
            self.smart_orchestrator = get_smart_orchestrator()
            self.factor_library = get_factor_library()
            self.tca_engine = get_tca_engine()
            self.stress_framework = get_stress_framework()
            self.risk_monitor = get_risk_monitor()
            self.compliance_engine = get_compliance_engine()

            # Phase 13: Zero-Loss
            self.guardian = get_guardian()
            self.timing_engine = get_timing_engine()
            self.genius_picker = get_genius_picker()
            self.ultimate_executor = get_ultimate_executor()

            # Phase 14: Multi-Strategy
            self.strategy_universe = get_strategy_universe()
            self.strategy_selector = get_strategy_selector()
            self.precision_analyzer = get_precision_analyzer()
            self.supreme_engine = get_supreme_engine()

            # Phase 15: HRHR
            self.hrhr_engine = get_hrhr_engine()
            self.dip_buyer = get_dip_buyer()

            # Phase 16: Top 1%
            self.smart_money = get_smart_money_detector()
            self.regime_analyzer = get_regime_analyzer()
            self.autonomous_brain = get_autonomous_brain()
            self.advanced_risk = get_risk_engine()
            self.elite_optimizer = get_elite_optimizer()
            self.technical_analyzer = get_technical_analyzer()
            self.pattern_engine = get_pattern_engine()
            self.advanced_execution = get_execution_engine()
            self.trade_manager = get_trade_manager()
            self.ultimate_orchestrator = get_orchestrator()

            logger.info("[AGENT] Elite Intelligence (Phases 1-16) initialized")
        except Exception as e:
            logger.warning(f"[AGENT] Elite partial init: {e}")

        # Phase 17: Alpha Discovery & Continuous Learning Loop
        try:
            # Research & Validation
            self.alpha_pipeline = get_alpha_pipeline()
            self.kill_switch = get_kill_switch()  # The BRUTAL one
            self.optimizer = get_optimizer()

            # Alternative Data
            self.options_flow = get_options_flow_analyzer()
            self.insider_tracker = get_insider_tracker()
            self.order_flow = get_order_flow_analyzer()

            # Risk & Execution
            self.stress_tester = get_stress_tester()
            self.smart_order_router = get_smart_order_router()

            # Continuous Learning Loop
            self.learning_feedback = get_learning_feedback()
            self.meta_allocator = get_meta_allocator()
            self.crisis_scanner = get_crisis_scanner()
            self.ensemble_predictor = get_ensemble_predictor()
            self.cross_asset = get_cross_asset_mapper()

            logger.info("[AGENT] Phase 17 Learn-Trade-Learn Loop ACTIVE")
        except Exception as e:
            logger.error(f"[AGENT] Phase 17 Initialization Failed: {e}")
            # Non-critical for now, but logged as error

        # Phase 18: Global Financial AI (New)
        try:
            self.global_session = get_global_session_tracker()
            self.rl_controller = get_rl_controller()
            self.network_alpha = get_network_alpha()
            self.network_analyzer = get_network_analyzer()
            logger.info("[AGENT] Phase 18 Global Financial AI initialized")
        except Exception as e:
            logger.error(f"[AGENT] Phase 18 Initialization Failed: {e}")

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
            msg = "KILL SWITCH ACTIVATED: 'kill_switch.txt' found. " "System HALTING."
            logger.critical(msg)
            return True
        return False

    def check_governance_gate(self) -> bool:
        """
        ABSOLUTE PRECHECK: Load ONLY symbols with state = ACTIVE.
        Do NOT error if zero active symbols; start in Safe Mode.
        """
        msg = (
            "[DATA_GOVERNANCE] Phase 0: Verifying Institutional " "Symbol Governance..."
        )
        logger.info(msg)

        try:
            active_symbols = self.db.get_active_symbols()
        except Exception as e:
            logger.error(f"[DATA_GOVERNANCE] Failed to query active symbols: {e}")
            active_symbols = []

        if self.tickers:
            original_tickers = set(self.tickers)
            self.tickers = [s for s in self.tickers if s in active_symbols]
            excluded = original_tickers - set(self.tickers)
            if excluded:
                msg = (
                    f"[DATA_GOVERNANCE] Excluded {len(excluded)} "
                    f"non-ACTIVE symbols."
                )
                logger.warning(msg)
        else:
            self.tickers = active_symbols

        if not self.tickers:
            msg = (
                "[DATA_GOVERNANCE] Zero ACTIVE symbols detected. "
                "System running in safe mode."
            )
            logger.info(msg)
            return True

        msg = (
            f"[DATA_GOVERNANCE] [READY] System starting with "
            f"{len(self.tickers)} ACTIVE symbols."
        )
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

        msg = (
            f"[DATA_GOVERNANCE] Loading 252-day window for "
            f"{len(self.tickers)} symbols..."
        )
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
            processed_data[sym] = df[["Open", "High", "Low", "Close", "Volume"]]

        if not processed_data:
            return pd.DataFrame()

        combined = pd.concat(processed_data, axis=1, sort=False)
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
            base_url = os.getenv("ALPACA_API_URL", "https://paper-api.alpaca.markets")
            self.handler = AlpacaExecutionHandler(
                api_key=str(api_key), secret_key=str(secret_key), base_url=base_url
            )
            logger.info(f"Using AlpacaExecutionHandler with {base_url}")
        else:
            self.handler = MockBroker()
            logger.info("Using MockBroker")

        logger.info("System initialized and warm.")

    def _process_exits(self):
        """Review and execute exit signals from TradeManager (Auto-Sell)."""
        from decimal import Decimal

        # 1. Update prices for ALL open trades
        latest_prices = {}
        if self.market_data is not None and not self.market_data.empty:
            open_trades = self.trade_manager.get_open_trades()
            if not open_trades:
                return  # No open trades, no exits needed

            cols = self.market_data.columns
            level0 = cols.get_level_values(0)

            for trade in open_trades:
                sym = trade.symbol
                if sym in level0:
                    try:
                        # Get latest Close
                        val = self.market_data[sym]["Close"].iloc[-1]
                        latest_prices[sym] = Decimal(str(val))
                    except:
                        pass

            if latest_prices:
                self.trade_manager.update_prices(latest_prices)

        # 1b. ELITE "THINKING": Re-evaluate Open Positions based on Future Prediction
        # "Don't hold unprofitable; Boost winners if prediction is high"
        if self.network_alpha:
            for trade in self.trade_manager.get_open_trades():
                sym = trade.symbol

                # GET FUTURE PREDICTION SCORE
                score = self.network_alpha.get_score(sym)  # 0.0 to 1.0

                # A. PREDICTION IS LOSS (Score < 0.3) -> QUICK CUT
                if score < 0.3:
                    # If PnL is also negative/flat, Kill it.
                    if trade.unrealized_pnl <= 0:
                        logger.warning(
                            f"[SMART_CUT] Prediction turned BAD for {sym} (Score: {score:.2f}). Exiting."
                        )
                        from execution.trade_manager import ExitReason

                        self.trade_manager.close_trade(
                            trade.trade_id,
                            trade.current_price,
                            reason=ExitReason.SMART_CUT,
                        )
                        continue

            # C. PRECISION EXIT TIMING (Phase 12)
            if self.timing_engine:
                for trade in self.trade_manager.get_open_trades():
                    try:
                        timing_sig = self.timing_engine.find_exit_timing(
                            symbol=trade.symbol,
                            action=trade.side.upper(),
                            entry_price=float(trade.entry_price),
                            current_price=float(trade.current_price),
                            stop_loss=float(trade.stop_loss),
                            target=float(trade.take_profit_1),
                            market_data=self.market_data,
                        )

                        if timing_sig and timing_sig.should_exit:
                            logger.warning(
                                f"[PRECISION_EXIT] {trade.symbol} triggered by Timing Engine. "
                                f"Reason: {timing_sig.exit_reason.value if hasattr(timing_sig.exit_reason, 'value') else timing_sig.exit_reason}"
                            )
                            from execution.trade_manager import ExitReason

                            self.trade_manager.close_trade(
                                trade.trade_id,
                                trade.current_price,
                                reason=ExitReason.SMART_CUT,  # Using Smart Cut as reason
                            )
                    except Exception as e:
                        logger.debug(
                            f"[TIMING] Exit check failed for {trade.symbol}: {e}"
                        )

        # 2. Check Exits
        exits = self.trade_manager.check_exits()
        if not exits:
            return

        # 3. Execute
        from execution.ultimate_executor import get_ultimate_executor

        executor = get_ultimate_executor(self.handler)

        for exit_order in exits:
            logger.info(
                f"[EXIT] Processing {exit_order.side} for {exit_order.symbol} ({exit_order.reason})"
            )

            # Current price for limit calculation
            current_price_dec = latest_prices.get(exit_order.symbol, Decimal("100.0"))
            current_price = float(current_price_dec)

            try:
                plan = executor.create_execution_plan(
                    symbol=exit_order.symbol,
                    action=exit_order.side,
                    quantity=float(exit_order.quantity),
                    current_price=current_price,
                    urgency="HIGH",
                )

                result = executor.execute(plan)

                if result.fully_filled or result.partially_filled:
                    self.trade_manager.process_exit(
                        trade_id=exit_order.trade_id,
                        quantity=int(result.filled_qty),
                        exit_price=Decimal(str(result.avg_fill_price)),
                        reason=exit_order.reason,
                    )
                    logger.info(
                        f"[EXIT] CONFIRMED: {exit_order.symbol} closed @ {result.avg_fill_price}"
                    )
                else:
                    logger.error(f"[EXIT] FAILED: {result.error_message}")
            except Exception as e:
                logger.error(f"[EXIT] EXCEPTION: {e}")

    def run_per_second_loop(self):
        """
        Main decision loop running every second.
        Uses in-memory features/cached data.
        """
        self.running = True
        logger.info("=" * 80)
        logger.info("INSTITUTIONAL LIVE ENGINE STARTED (PER-SECOND LOOP)")

        # Add basic logging structure for debugging if needed (though already imported)
        # logger is already imported at module level

        if self.market_data is None or self.market_data.empty:
            msg = (
                "[DATA_GOVERNANCE] No active market data. "
                "System will remain in observation mode."
            )
            logger.warning(msg)
        logger.info("=" * 80)
        if self.market_data is None or self.market_data.empty:
            msg = (
                "[DATA_GOVERNANCE] No active market data. "
                "System will remain in observation mode."
            )
            logger.warning(msg)
        logger.info("=" * 80)

        # Start Heartbeat Thread (Institutional Monitoring)
        args = (logger, 5)
        h_thread = threading.Thread(target=heartbeat_worker, args=args, daemon=True)
        h_thread.start()

        while self.running:
            # Check Duration Limit
            if (
                hasattr(self, "end_time")
                and self.end_time
                and datetime.now() > self.end_time
            ):
                logger.info("[STOP] Duration limit reached. Shutting down agent.")
                self.running = False
                break

            loop_start = time.time()
            self._grade_map = {}
            self._sm_aligned_map = {}

            # 0. REFRESH MARKET DATA (CRITICAL FIX)
            try:
                # Reload only if needed, but for MVP let's ensure freshness
                new_data = self.load_252d_market_data()
                if not new_data.empty:
                    self.market_data = new_data
                    logger.debug("[DATA] Market data refreshed.")
            except Exception as e:
                logger.error(f"[DATA] Refresh failed: {e}")

            # 0. Auto-Exit Check (Stop Loss / Take Profit)
            self._process_exits()

            # 0a. Global Session Check
            current_sessions = self.global_session.get_active_sessions()
            if not current_sessions:
                # If no major markets open, maybe sleep longer or check crypto
                if "CRYPTO" not in self.global_session.schedules:
                    time.sleep(10)  # Low power mode

            # 0b. RL Allocation Update (every ~1000 ticks or daily)
            # For now, just logging the state
            # state = self.rl_controller.get_state(None)
            # action = self.rl_controller.predict_action(state)

            try:
                # 0. Safety & Governance Gate (Consolidated)

                # Gather metrics for governance
                current_nav = 1_000_000.0  # Default
                try:
                    if self.handler:
                        acct = self.handler.get_account()
                        if acct:
                            current_nav = float(getattr(acct, "equity", 1e6))
                except Exception:
                    pass

                sys_metrics = {
                    "nav_usd": current_nav,
                    # Placeholder until regime engine provides distinct score
                    "regime_confidence": 1.0,
                    "hit_rate": getattr(metrics, "hit_rate", 0.5),
                    "is_replay": False,  # Live mode
                }

                gov_decision = allow_trading(sys_metrics)
                if not gov_decision["allow"]:
                    msg = (
                        f"[GOVERNANCE_HALT] Trading skipped: "
                        f"{gov_decision['reason']}"
                    )
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
                    if isinstance(
                        cols, pd.MultiIndex
                    ) and "^VIX" in cols.get_level_values(0):
                        vix_value = float(self.market_data["^VIX"]["Close"].iloc[-1])
                except Exception:
                    pass  # VIX not in universe, use None

                # 1b. Drawdown Computation
                drawdown_value = None
                try:
                    if hasattr(self.handler, "get_account"):
                        acct = self.handler.get_account()
                        equity = float(getattr(acct, "equity", 1e6))
                        le = float(getattr(acct, "last_equity", equity))
                        # Simple DD = (current - peak) / peak
                        peak = max(equity, le)
                        dd_val = (equity - peak) / peak if peak > 0 else 0.0
                        drawdown_value = dd_val
                except Exception:
                    pass

                # --- INSTITUTIONAL MONITORING UPDATE (Phase 6) ---
                from monitoring.dashboard_backend import get_monitor

                monitor = get_monitor()
                monitor.update_pnl(drawdown_value if drawdown_value else 0.0)
                monitor.record_latency("loop", (time.time() - loop_start) * 1000)

                # --- ALTERNATIVE DATA INGEST (Phase 5.3) ---
                # In a real loop, might run this less frequently
                # from data.alternative_data import get_alt_data_engine
                # alt_engine = get_alt_data_engine()
                # for ticker in self.tickers[:5]: # Sample 5 for perf
                #     sig = alt_engine.get_aggregated_signal(ticker)
                #     # Store in FeatureStore...

                # 1c. Correlation Computation (Average pairwise)
                avg_corr = None
                try:
                    if not self.market_data.empty and isinstance(
                        self.market_data.columns, pd.MultiIndex
                    ):
                        cols = self.market_data.columns
                        symbols = cols.get_level_values(0).unique().tolist()
                        if len(symbols) > 1:
                            # Filter to symbols actually in columns
                            valid_syms = [
                                s for s in symbols if s in cols.get_level_values(0)
                            ]
                            closes = pd.DataFrame(
                                {s: self.market_data[s]["Close"] for s in valid_syms}
                            )
                            rets = closes.pct_change(fill_method=None).dropna()
                            if len(rets) > 20:  # Need enough data
                                corr_mat = rets.corr()
                                # Upper triangle mean (exclude diagonal)
                                mask = np.triu(np.ones(corr_mat.shape), k=1).astype(
                                    bool
                                )
                                upper = corr_mat.where(mask)
                                avg_corr = upper.stack().mean()
                except Exception:
                    pass

                current_regime = self.regime_controller.detect_regime(
                    vix=vix_value, drawdown=drawdown_value, avg_correlation=avg_corr
                )

                # Apply regime overrides to config
                if current_regime.value != "NORMAL":
                    self.cfg = self.regime_controller.apply_overrides(
                        current_regime, self.cfg
                    )
                    msg = f"[REGIME] {current_regime.value} mode active"
                    logger.warning(msg)

                # 2. Error Rate Monitoring & Safe Mode (Priority 3 Fix)
                if getattr(self, "loop_errors", 0) > 5:
                    msg = (
                        "[SAFE_MODE] High error rate detected (>5 "
                        "consecutive). Reducing exposure."
                    )
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
                        logger.warning("Allocator not initialized, using empty targets")
                        target_weights = {}

                    self.loop_errors = 0  # Reset on success
                    metrics.cycles += 1

                    # Generate Cycle ID for audit
                    ts = int(time.time())
                    current_cycle_id = f"live_{ts}_{metrics.cycles}"

                    # Log signal summary
                    if len(target_weights) > 0:
                        msg = f"Generated {len(target_weights)} " f"position targets"
                        logger.info(msg)

                        # --- Execution Decision Layer ---

                        from governance.execution_decision import decide_execution

                        executed_count = 0
                        skipped_count = 0
                        skip_reasons = {}  # reason -> count

                        # Get NAV
                        nav = 1000000.0  # Default fallback
                        if hasattr(self.handler, "get_account"):
                            try:
                                acct = self.handler.get_account()
                                if hasattr(acct, "equity"):  # Alpaca
                                    nav = float(acct.equity)
                                elif isinstance(acct, dict) and "equity" in acct:
                                    nav = float(acct["equity"])
                            except Exception:
                                pass

                        market_open_flag = self.execution_gate.is_market_open()
                        # Override check disabled for debug
                        if not market_open_flag:
                            # Check if broker can override (e.g. crypto)
                            if hasattr(self.handler, "is_market_open"):
                                status = self.handler.is_market_open()
                                market_open_flag = status

                        # Skipping history (track in memory for live agent)
                        if not hasattr(self, "skipping_history"):
                            self.skipping_history = {}

                        e_cfg = self.cfg.get("execution", {})
                        exec_enabled = e_cfg.get("enabled", True)

                        # --- BATCH OPTIMIZATION: Fetch all positions once ---
                        current_positions_map = {}
                        if self.handler and hasattr(self.handler, "get_positions"):
                            try:
                                raw_positions = self.handler.get_positions()
                                # Handle both Alpaca (list of dicts) and
                                # internal format
                                if isinstance(raw_positions, list):
                                    for p in raw_positions:
                                        # Alpaca returns dict with 'symbol' and
                                        # 'qty'
                                        sym = p.get("symbol")
                                        qty = float(p.get("qty", 0.0))
                                        current_positions_map[sym] = qty
                                elif isinstance(raw_positions, dict):
                                    current_positions_map = raw_positions
                            except Exception as e:
                                logger.error(f"Failed to fetch batch positions: {e}")

                        # --- STAGE 2: AUTOMATIC EXPOSURE REDUCTION ---
                        # Check portfolio health and inject SELL signals for trim
                        try:
                            guardian = get_portfolio_guardian()
                            if guardian and raw_positions and nav > 0:
                                # Convert to format expected by guardian
                                pos_list = []
                                for p in raw_positions:
                                    if isinstance(p, dict):
                                        pos_list.append(
                                            {
                                                "symbol": p.get("symbol"),
                                                "market_value": float(
                                                    p.get("market_value", 0)
                                                ),
                                            }
                                        )
                                # Stage 2: 60% threshold
                                trim_syms = guardian.check_portfolio_health(
                                    pos_list, nav, max_exposure_pct=0.60
                                )
                                if trim_syms:
                                    logger.warning(
                                        f"[STAGE2] Auto-trim triggered for: "
                                        f"{trim_syms}"
                                    )
                                    for sym in trim_syms:
                                        if sym not in target_weights:
                                            # Inject SELL signal
                                            target_weights[sym] = -0.05
                                            logger.info(
                                                f"[STAGE2] Injected SELL for {sym}"
                                            )
                        except Exception as e:
                            logger.error(f"[STAGE2] Guardian check failed: {e}")

                        msg = (
                            f"[EXECUTION] Processing "
                            f"{len(target_weights)} target signals..."
                        )
                        logger.info(msg)

                        # --- SMART INTELLIGENCE LAYER (Phase 1-10) ---
                        # 1. Regime-Based Sizing Adjustment
                        current_regime_val = current_regime.value
                        regime_multiplier = 1.0
                        if current_regime_val == "VOLATILE":
                            regime_multiplier = 0.5
                            logger.info("[SMART] Volatile Regime: Cutting sizes by 50%")
                        elif current_regime_val == "CRASH":
                            regime_multiplier = 0.0
                            logger.warn("[SMART] Crash Regime: HALTING BUYING")

                        # 2. Alternative Data Overlay
                        # Boost conviction if Alt Data supports the trade
                        for sym in target_weights:
                            alt_signal = self.alt_engine.get_aggregated_signal(sym)
                            if alt_signal and alt_signal.signal_value > 0.6:
                                if target_weights[sym] > 0:
                                    target_weights[sym] *= 1.2
                                    logger.info(f"[SMART] Alt Data Boost for {sym}")

                        # 3. LLM Veto for Largest Position
                        if target_weights:
                            top_sym = max(
                                target_weights, key=lambda k: abs(target_weights[k])
                            )
                            if (
                                abs(target_weights[top_sym]) > 0.05
                            ):  # Only check significant trades
                                try:
                                    llm_analysis = self.llm_analyzer.analyze_trade(
                                        symbol=top_sym,
                                        price=100.0,  # Mock price
                                        features={},
                                        ensemble_score=0.8,
                                        models={},
                                        risk_data={},
                                        position_state={},
                                    )
                                    if (
                                        llm_analysis
                                        and llm_analysis.recommendation == "AVOID"
                                    ):
                                        logger.warning(
                                            f"[SMART] LLM VETOED trade for {top_sym}"
                                        )
                                        target_weights[top_sym] = 0.0
                                except Exception as e:
                                    logger.error(f"LLM Check Failed: {e}")

                        # 4. Final Regime Scaling
                        for sym in target_weights:
                            target_weights[sym] *= regime_multiplier

                        # --- TOP 1% INSTITUTIONAL BRAIN SCRUTINY (Phase 16) ---
                        if self.autonomous_brain and target_weights:
                            logger.info(
                                "[BRAIN] Scrutinizing all candidates for Elite-only trades..."
                            )
                            brain_signals = []
                            for sym, w in target_weights.items():
                                if abs(w) > 0.001:  # Significant trades only
                                    # Basic action mapping for brain
                                    action = "BUY" if w > 0 else "SELL"
                                    # Determine if we are opening or closing
                                    is_closing = False
                                    if sym in current_positions_map:
                                        cur_qty = current_positions_map[sym]
                                        if (cur_qty > 0 and w <= 0) or (
                                            cur_qty < 0 and w >= 0
                                        ):
                                            is_closing = True

                                    if not is_closing:
                                        brain_signals.append(
                                            {
                                                "symbol": sym,
                                                "action": action,
                                                "confidence": 0.5,  # Base
                                                "strategy": "Ensemble_Institutional",
                                            }
                                        )

                            if brain_signals:
                                try:
                                    ranked_picks = (
                                        self.autonomous_brain._rank_opportunities(
                                            brain_signals, self.market_data
                                        )
                                    )
                                    # Create a map of symbol -> grade and reason
                                    grade_map = {
                                        p["symbol"]: p.get("grade", "F")
                                        for p in ranked_picks
                                    }
                                    reason_map = {
                                        p["symbol"]: p.get("reasoning", "Unknown")
                                        for p in ranked_picks
                                    }

                                    # VETO Logic: Only A+, A, B allowed for NEW positions
                                    for sym in list(target_weights.keys()):
                                        if abs(target_weights[sym]) > 0.001:
                                            # Skip veto for closing positions
                                            is_closing = False
                                            if sym in current_positions_map:
                                                cur_qty = current_positions_map[sym]
                                                if (
                                                    cur_qty > 0
                                                    and target_weights[sym] <= 0
                                                ) or (
                                                    cur_qty < 0
                                                    and target_weights[sym] >= 0
                                                ):
                                                    is_closing = True

                                            if not is_closing:
                                                grade = grade_map.get(sym, "F")
                                                reason = reason_map.get(sym, "N/A")

                                                if grade not in ["A+", "A", "B"]:
                                                    logger.warning(
                                                        f"[BRAIN] VETO: {sym} rejected with grade {grade}. Reason: {reason}"
                                                    )
                                                    target_weights[sym] = 0.0
                                                else:
                                                    logger.info(
                                                        f"[BRAIN] APPROVED: {sym} Grade {grade}. Reason: {reason}"
                                                    )

                                            # EMERGENCY BUY STOP (Phase 32)
                                            if target_weights[sym] > 0:
                                                logger.warning(
                                                    f"[EMERGENCY] BUY STOPPED for {sym}. Accumulation suspended."
                                                )
                                                target_weights[sym] = 0.0
                                                continue

                                            # SMART MONEY ALIGNMENT (Phase 16)
                                            if self.smart_money:
                                                sm_flow = self.smart_money.analyze(
                                                    sym, self.market_data
                                                )
                                                if sm_flow:
                                                    # If we want to BUY, but institutions are SELLING (Distribution)
                                                    if (
                                                        target_weights[sym] > 0
                                                        and sm_flow.regime
                                                        == "DISTRIBUTION"
                                                    ):
                                                        logger.warning(
                                                            f"[SMART_MONEY] VETO BUY for {sym}: Institutional Distribution detected."
                                                        )
                                                        target_weights[sym] = 0.0
                                                    # If we want to SELL, but institutions are BUYING (Accumulation)
                                                    elif (
                                                        target_weights[sym] < 0
                                                        and sm_flow.regime
                                                        == "ACCUMULATION"
                                                    ):
                                                        # RELAXATION: Allow exit if closing
                                                        if is_closing:
                                                            logger.info(
                                                                f"[SMART_MONEY] ALLOWING EXIT for {sym} despite Accumulation (Profit Taking)."
                                                            )
                                                        else:
                                                            logger.warning(
                                                                f"[SMART_MONEY] VETO SHORT for {sym}: Institutional Accumulation detected."
                                                            )
                                                            target_weights[sym] = 0.0
                                    # Exception handling removed to fix syntax error
                                    # except Exception as e:
                                    #     logger.error(f"[BRAIN] Scrutiny error: {e}")

                                    # SYMMETRY MONITORING
                                    if abs(target_weights[sym]) > 0.001:
                                        if target_weights[sym] > 0:
                                            self.symmetry_buys += 1
                                        elif target_weights[sym] < 0:
                                            self.symmetry_sells += 1

                                    # Hourly Symmetry Log
                                    if time.time() - self.last_symmetry_log > 3600:
                                        logger.info(
                                            f"[SYMMETRY] Last Hour: Buys={self.symmetry_buys}, Sells={self.symmetry_sells}"
                                        )
                                        self.symmetry_buys = 0
                                        self.symmetry_sells = 0
                                        self.last_symmetry_log = time.time()

                                except Exception as e:
                                    logger.error(
                                        f"[BRAIN] Intelligence Layer Error: {e}"
                                    )

                        # ---------------------------------------------

                        for i, (symbol, target_w) in enumerate(target_weights.items()):
                            # Periodic heartbeat for large loops
                            if i > 0 and i % 50 == 0:
                                msg = (
                                    f"[EXECUTION] Processed {i}/"
                                    f"{len(target_weights)} symbols..."
                                )
                                logger.info(msg)

                            # Optimized Lookup
                            qty_p = self.portfolio_state.get(symbol, 0.0)
                            current_qty = current_positions_map.get(symbol, qty_p)

                            # --- SAFETY GUARD CHECK (Phase 10) ---
                            from production.safety_guards import get_safety_guard

                            safety = get_safety_guard()

                            # Calculate Smart Metrics on the Fly
                            adv_val = 1000000  # Default
                            spread_val = 5  # Default
                            move_val = 0.0  # Default

                            if (
                                self.market_data is not None
                                and not self.market_data.empty
                                and symbol
                                in self.market_data.columns.get_level_values(0)
                            ):
                                try:
                                    # 1. Volume (Approx ADV)
                                    vol = (
                                        self.market_data[symbol]["Volume"]
                                        .iloc[-20:]
                                        .mean()
                                    )
                                    close = self.market_data[symbol]["Close"].iloc[-1]
                                    adv_val = vol * close

                                    # 2. Daily Move
                                    open_p = self.market_data[symbol]["Open"].iloc[-1]
                                    move_val = (
                                        abs((close - open_p) / open_p)
                                        if open_p > 0
                                        else 0
                                    )

                                    # 3. Spread (Simulated if no L2 data)
                                    # If highvol, assume wider spread
                                    spread_val = 15 if move_val > 0.03 else 5
                                except:
                                    pass

                            # Construct order preview
                            order_preview = {
                                "symbol": symbol,
                                "quantity": abs(target_w * 100),  # Mock size
                                "price": 100.0,  # Mock price if unavailable
                                "adv_30d": adv_val,
                                "spread_bps": spread_val,
                                "daily_move_pct": move_val,
                            }
                            if not safety.check_pre_trade(order_preview):
                                logger.warning(f"[SAFETY] Blocked trade for {symbol}")
                                continue

                            price = 0.0
                            cols = self.market_data.columns
                            if (
                                not self.market_data.empty
                                and symbol in cols.get_level_values(0)
                            ):
                                price = float(
                                    self.market_data[symbol]["Close"].iloc[-1]
                                )

                            if price <= 0:
                                continue

                            current_val = current_qty * price
                            current_w = current_val / nav

                            # Conviction? Recover from signals.
                            conviction = 0.5  # Default
                            if signals is not None and symbol in signals.columns:
                                val = signals[symbol].iloc[-1]
                                # Map signal to conviction.
                                if val > 0.8 or val < 0.2:
                                    conviction = 0.8

                            # Call Decision Layer
                            # Risk handled by allocator
                            risk_scaled_w = target_w

                            exec_res = {}
                            final_decision = "SKIP_INTERNAL_ERROR"
                            reason_codes = ["UNKNOWN"]

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
                                    config=self.cfg,
                                    intelligence_grade=self._grade_map.get(
                                        symbol, "N/A"
                                    ),
                                    smart_money_aligned=self._sm_aligned_map.get(
                                        symbol, True
                                    ),
                                )

                                final_decision = exec_res.get("decision", "ERROR")
                                reason_codes = exec_res.get("reason_codes", [])

                                audit_payload = {
                                    "cycle_id": current_cycle_id,
                                    "symbol": symbol,
                                    "timestamp": datetime.utcnow().isoformat(),
                                    "decision_type": "execution",
                                    "final_decision": final_decision,
                                    "reason_codes": reason_codes,
                                    "target_weight": target_w,
                                    "current_weight": current_w,
                                    "target_qty": exec_res.get("target_qty", 0),
                                    "rounded_qty": exec_res.get("target_qty", 0),
                                    "notional_usd": exec_res.get("notional_usd", 0),
                                    "conviction": conviction,
                                    "market_open": market_open_flag,
                                    "broker_error": None,
                                }

                                # Order Data
                                audit_payload["order"] = {
                                    "notional": exec_res.get("notional_usd", 0),
                                    "target_qty": exec_res.get("target_qty", 0),
                                    "broker_error": None,
                                }

                                if final_decision != "EXECUTE":
                                    # SKIP Logic
                                    skipped_count += 1
                                    self.skipping_history[symbol] = (
                                        self.skipping_history.get(symbol, 0) + 1
                                    )
                                    first_reason = (
                                        reason_codes[0] if reason_codes else "UNKNOWN"
                                    )
                                    skip_reasons[first_reason] = (
                                        skip_reasons.get(first_reason, 0) + 1
                                    )

                                    # Write Audit
                                    try:
                                        from audit.decision_log import write_audit

                                        write_audit(audit_payload)
                                    except Exception as e:
                                        logger.error(f"Audit Write Failed: {e}")

                                    continue
                                else:
                                    # EXECUTE Logic
                                    self.skipping_history[symbol] = 0  # Reset

                            # --- ELITE PORTFOLIO GUARDIAN (Phase 6) ---
                            # "Still more smart" - Correlation & Volatility Check
                            from risk.portfolio_guardian import get_portfolio_guardian

                            guardian = get_portfolio_guardian()

                            current_holdings = list(current_positions_map.keys())

                            # 0. Underwater Check (STOP BUYING LOSERS)
                            if (
                                symbol in current_positions_map
                                and current_positions_map[symbol] != 0
                            ):
                                # Get Unrealized PnL from TradeManager
                                position_info = self.trade_manager.get_position(symbol)
                                if (
                                    position_info
                                    and position_info.get("unrealized_pnl", 0) < 0
                                ):
                                    if target_w > current_w:  # Trying to BUY more
                                        logger.warning(
                                            f"[GUARDIAN] REJECT {symbol}: Position is underwater. No averaging down."
                                        )
                                        continue

                            # 1. Correlation Check
                            if not guardian.check_new_trade(
                                symbol,
                                self.market_data,
                                current_holdings,
                                current_weight=current_w,
                            ):
                                logger.warning(
                                    f"[GUARDIAN] Blocked {symbol} due to Portfolio Correlation Risk."
                                )
                                continue

                            # 2. Volatility Scalar
                            vol_scalar = guardian.get_volatility_scalar(
                                symbol, self.market_data
                            )
                            if vol_scalar != 1.0:
                                old_w = target_w
                                target_w *= vol_scalar
                                logger.info(
                                    f"[GUARDIAN] Scaled {symbol} by {vol_scalar:.2f}x (Vol Targeting) W: {old_w:.4f}->{target_w:.4f}"
                                )

                            # Execute Order
                            try:
                                # Calculate DELTA for Order
                                target_val = target_w * nav
                                target_shares = target_val / price
                                cur_shares = current_w * nav / price
                                order_shares = target_shares - cur_shares

                                # Rounding
                                if isinstance(self.handler, AlpacaExecutionHandler):
                                    # Use Alpaca wrapper
                                    pass

                                if abs(order_shares) > 0:
                                    side = "buy" if order_shares > 0 else "sell"
                                    qty_to_trade = abs(order_shares)

                                    # --- FINAL EXECUTION GATE ---
                                    v_res = self.execution_gate.validate_execution(
                                        symbol=symbol,
                                        qty=qty_to_trade,
                                        side=side,
                                        price=price,
                                        adv_30d=1e6,
                                        volatility=0.02,
                                    )
                                    is_ok, reason, scaled_qty = v_res

                                    if not is_ok:
                                        msg = (
                                            f"Final Gate REJECT: "
                                            f"{symbol} | {reason}"
                                        )
                                        logger.warning(msg)
                                        final_decision = f"SKIP_{reason}"
                                        audit_payload["final_decision"] = final_decision
                                        rcs = audit_payload["reason_codes"]
                                        if reason not in rcs:
                                            rcs.append(reason)

                                        # Write Audit
                                        try:
                                            from audit.decision_log import write_audit

                                            write_audit(audit_payload)
                                        except Exception:
                                            pass
                                        continue

                                    qty_to_trade = scaled_qty

                                    # --- PRECISION ENTRY TIMING (Phase 12) ---
                                    entry_type = "market"
                                    limit_price = price
                                    if self.timing_engine:
                                        try:
                                            timing_res = (
                                                self.timing_engine.find_perfect_entry(
                                                    symbol=symbol,
                                                    action=side.upper(),
                                                    current_price=price,
                                                    market_data=self.market_data,
                                                )
                                            )
                                            if (
                                                timing_res
                                                and timing_res.decision == "WAIT"
                                            ):
                                                logger.info(
                                                    f"[TIMING] {symbol}: Entry not optimal (Patience score: {timing_res.patience_score:.2f}). Waiting..."
                                                )
                                                continue
                                            elif (
                                                timing_res
                                                and timing_res.decision == "LIMIT"
                                            ):
                                                entry_type = "limit"
                                                limit_price = float(
                                                    timing_res.limit_price
                                                )
                                                logger.info(
                                                    f"[TIMING] {symbol}: Using LIMIT entry at {limit_price}"
                                                )
                                        except Exception as e:
                                            logger.debug(
                                                f"[TIMING] Entry check failed for {symbol}: {e}"
                                            )

                                    # Submit to handler
                                    if self.handler and hasattr(
                                        self.handler, "submit_order"
                                    ):
                                        result = self.handler.submit_order(
                                            symbol=symbol,
                                            qty=qty_to_trade,
                                            side=side,
                                            type=entry_type,
                                            time_in_force="day",
                                            price=limit_price,
                                        )

                                        if result["success"]:
                                            executed_count += 1
                                            # Optimistic update
                                            self.portfolio_state[symbol] = target_shares

                                            # REGISTER WITH TRADE MANAGER (AUTO-SELL LOGIC)
                                            if side == "buy":
                                                from decimal import Decimal

                                                try:
                                                    self.trade_manager.open_trade(
                                                        symbol=symbol,
                                                        side="LONG",
                                                        entry_price=Decimal(str(price)),
                                                        quantity=int(qty_to_trade),
                                                        stop_loss=Decimal(
                                                            str(price * 0.95)
                                                        ),  # 5% SL
                                                        take_profit_1=Decimal(
                                                            str(price * 1.05)
                                                        ),  # 5% TP
                                                        take_profit_2=Decimal(
                                                            str(price * 1.15)
                                                        ),  # 15% TP
                                                        trailing_stop_pct=0.03,  # 3% Trailing
                                                        strategy="RL_Hybrid",
                                                    )
                                                    logger.info(
                                                        f"[TRADE_MANAGER] Registered {symbol} for Auto-Sell"
                                                    )
                                                except Exception as e:
                                                    logger.error(
                                                        f"[TRADE_MANAGER] Registration Failed: {e}"
                                                    )
                                        else:
                                            # LOG BROKER FAILURE
                                            msg = (
                                                f"Broker Submission "
                                                f"Failed: {result['error']}"
                                            )
                                            logger.error(msg)
                                            audit_payload["decision"] = "BROKER_FAILURE"
                                            audit_payload["broker_error"] = result[
                                                "error"
                                            ]
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
                                    audit_payload["decision"] = "CRASH"
                                    audit_payload["broker_error"] = str(e)
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
                    if metrics.model_errors > 0 or metrics.arima_fallbacks > 20:
                        ml_health = 0.7 if metrics.model_errors < 50 else 0.4

                    self.last_dashboard_update = 0
                    self.last_symmetry_log = time.time()
                    self.symmetry_buys = 0
                    self.symmetry_sells = 0

                    # Create Tables
                    self.dashboard.update(
                        "ml",
                        {
                            "health": ml_health,
                            "active_models": len(self.tickers),
                            "arima_fb": metrics.arima_fallbacks,
                        },
                    )

                    self.dashboard.update(
                        "execution",
                        {
                            "executed_count": executed_count,
                            "blocked_count": skipped_count,
                        },
                    )

                    # Render every 5 seconds or if run_once
                    if metrics.cycles % 5 == 0 or getattr(self, "run_once", False):
                        self.dashboard.render()
                except Exception as e:
                    logger.debug(f"Dashboard update failed: {e}")

                elapsed = time.time() - loop_start

                # Check run_once
                if getattr(self, "run_once", False):
                    logger.info("Run-once complete. Exiting.")
                    self.running = False
                    break

                sleep_time = max(0, 60.0 - elapsed)
                time.sleep(sleep_time)

            except KeyboardInterrupt:
                logger.info("Shutdown signal received (Ctrl+C). Stopping live agent...")
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
        logger.info("[GOVERNANCE] Starting pre-flight symbol classification...")
        governor.classify_all()

        # 2. Get Active Symbols (those with state=ACTIVE in symbol_governance)
        logger.info(
            "[DATA_GOVERNANCE] Fetching ACTIVE symbols from governance " "table..."
        )
        active_symbols = self.db.get_active_symbols()
        logger.info(f"[DATA_GOVERNANCE] Found {len(active_symbols)} ACTIVE symbols")

        if not active_symbols:
            logger.critical("[DATA_GOVERNANCE] No ACTIVE symbols in database")
            governance_halt([], "No ACTIVE symbols - run ingest_history.py first")

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
    parser = argparse.ArgumentParser(description="Institutional Live Trading Agent")
    import logging

    parser.add_argument("--tickers", type=str, help="Filter tickers")
    parser.add_argument("--run-once", action="store_true", help="Run a single cycle")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["paper", "live", "backtest"],
        help="Override trading mode",
    )
    parser.add_argument(
        "--duration", type=int, help="Run duration in days (default: infinite)"
    )
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    args, unknown = parser.parse_known_args()

    # Configure logging level
    numeric_level = getattr(logging, args.log_level.upper(), None)
    if isinstance(numeric_level, int):
        logging.getLogger().setLevel(numeric_level)

    tickers = args.tickers.split(",") if args.tickers else None

    agent = InstitutionalLiveAgent(tickers=tickers)

    # Apply CLI overrides
    if args.mode:
        agent.cfg["execution"]["mode"] = args.mode

    # Store run_once flag
    agent.run_once = args.run_once

    # Store duration limit (if any)
    if args.duration:
        agent.end_time = datetime.now() + pd.Timedelta(days=args.duration)
        logger.info(f"System will run for {args.duration} days until {agent.end_time}")
    else:
        agent.end_time = None

    agent.start()
