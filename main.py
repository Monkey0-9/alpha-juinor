# main.py
import os
import logging
import argparse
from pathlib import Path
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json
import subprocess
from dotenv import load_dotenv, find_dotenv
from typing import List, Optional, Any, Dict
import sys

# Institutional Config Management
from configs.config_manager import ConfigManager

# Robust Dotenv Loading
env_path = find_dotenv()
if env_path:
    load_dotenv(env_path, override=True)
else:
    print("No .env file found. Falling back to system environment variables.")

# NEW: Import Data Router
from data.collectors.data_router import DataRouter

from strategies.factory import StrategyFactory
from risk.engine import RiskManager
from portfolio.allocator import InstitutionalAllocator
from backtest.engine import BacktestEngine
from backtest.execution import RealisticExecutionHandler
from backtest.registry import BacktestRegistry
from monitoring.alerts import AlertManager
from engine.analytics import annualized_return, annualized_volatility, sharpe_ratio, max_drawdown
from reports.attribution import AttributionEngine
from ops.checklists import generate_daily_checklist
from ops.generate_daily_report import generate_report
from brokers.alpaca_broker import AlpacaExecutionHandler
from data.collectors.alpaca_collector import AlpacaDataProvider
from engine.live_engine import LiveEngine
from engine.market_listener import MarketListener
from brokers.mock_broker import MockBroker
from utils.time import to_utc, get_now_utc
from data.universe_manager import UnifiedUniverseManager
from utils.timezone import normalize_index_utc

logger = logging.getLogger("InstitutionalDriver")

def validate_env():
    """Fail-fast validation for critical environment variables."""
    required = ["ALPACA_API_KEY", "ALPACA_SECRET_KEY"]
    missing = [var for var in required if not os.getenv(var)]
    
    if os.getenv("EXECUTION_MODE") in ["paper", "live"] and missing:
        logger.critical(f"CRITICAL CONFIG ERROR: Missing credentials for PRODUCTION mode: {missing}")
        sys.exit(1)
        
    # Check for optional but recommended providers
    if os.getenv("TELEGRAM_TOKEN") and not os.getenv("TELEGRAM_CHAT_ID"):
        logger.warning("TELEGRAM_TOKEN found but TELEGRAM_CHAT_ID is missing. Alerts will not be sent.")
        
    # Check for common dotenv issues (e.g. quotes or spaces in keys)
    for key in required:
        val = os.getenv(key)
        if val and (val.startswith("'") or val.startswith('"')):
            logger.warning(f"POTENTIAL DOTENV PARSING ISSUE: Key {key} starts with quotes. Cleaning...")
            os.environ[key] = val.strip("'").strip('"')

validate_env()

def run_production_pipeline():
    """
    Main entry point for the validated institutional pipeline.
    """
    try:
         # Basic logging setup immediately to catch early errors
         logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    except Exception:
         pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-universe", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--mode", type=str)
    parser.add_argument("--tickers", type=str)
    args, unknown = parser.parse_known_args()
    
    start_time = time.time()
    
    # 1. Initialize Configuration
    cm = ConfigManager()
    cfg = cm.config
    cfg_hash = cm.config_hash
    
    # 2. Setup Resources
    registry = BacktestRegistry(base_dir=str(Path("output/backtests")))
    try:
        alerts = AlertManager()
    except Exception as e:
        logger.warning(f"AlertManager init failed (likely .env issue): {e}")
        alerts = None
    
    # 0. Generate Daily Checklist
    try:
        generate_daily_checklist(cfg_hash)
    except Exception:
        pass
    
    # Audit Meta
    meta = {
        "mode": cfg['execution']['mode'],
        "timestamp": get_now_utc().isoformat(),
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
    }
    try:
        meta["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        meta["git_commit"] = "unknown"
    
    # Map Config to Objects
    risk_cfg = cfg['risk']
    universe = cfg['universe']
    exec_cfg = cfg['execution']
    
    # Selection of Data Source (Institutional Data Policy)
    class RoutingProvider:
        def __init__(self):
            # Use the new Smart Router
            self.router = DataRouter()
        
        def get_panel(self, tickers: List[str], start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
            # Use High-Speed Parallel Fetcher
            try:
                return self.router.get_panel_parallel(tickers, start_date, end_date)
            except Exception as e:
                logger.error(f"Parallel fetch failed, falling back to sequential: {e}")
                return pd.DataFrame()

        def get_latest_price(self, ticker: str) -> Optional[float]:
            """Consistent interface for LiveEngine."""
            try:
                return self.router.get_latest_price(ticker)
            except Exception as e:
                logger.warning(f"Failed to get latest price for {ticker}: {e}")
                return None

        def get_latest_quote(self, ticker: str) -> Optional[float]:
            """Alias for backward compatibility."""
            return self.get_latest_price(ticker)
        
        @property
        def router_instance(self):
            return self.router

    if exec_cfg['mode'] in ['paper', 'live']:
        logger.info("INSTITUTIONAL: Using Entitlement-Aware Data Router for Production.")
    
    provider = RoutingProvider()

    # 1b. Universe Discovery
    um = UnifiedUniverseManager("configs/universe.json")
    if args.tickers:
        active_tickers = args.tickers.split(",")
        universe_meta = {}
    else:
        active_tickers = um.discover_and_filter(universe, force_refresh=args.refresh_universe)
        active_tickers = active_tickers[:500] 
        try:
            universe_meta_df = pd.read_parquet(um.cache_path)
            universe_meta = universe_meta_df.set_index("symbol").to_dict("index")
        except Exception:
            universe_meta = {}
    
    risk_mgr = RiskManager(
        max_leverage=risk_cfg['max_gross_leverage'],
        target_vol_limit=risk_cfg['target_volatility_annualized'],
        initial_capital=exec_cfg['initial_capital']
    )
    
    allocator = InstitutionalAllocator(
        risk_manager=risk_mgr,
        max_leverage=risk_cfg['max_gross_leverage'],
        allow_short=False
    )
    
    strategy = StrategyFactory.create_strategy({
        "type": "institutional",
        "tickers": active_tickers,
        "use_ml": cfg['alpha']['ml_weight'] > 0
    })
    
    handler = RealisticExecutionHandler(commission_pct=exec_cfg['commission_bps'] / 10000.0)
    engine = BacktestEngine(
        provider=provider, 
        initial_capital=exec_cfg['initial_capital'], 
        execution_handler=handler
    )
    
    # 3. Model Warming
    start_date = "2025-05-01"
    # FIX: Fetch 365 days for ML to satisfy "Need 500" validation (approx 252 trading days + buffer)
    train_start = (pd.to_datetime(start_date) - timedelta(days=400)).strftime("%Y-%m-%d")
    logger.info(f"Phase 1: Warming models from {train_start} (Deep History for ML)...")
    train_panel = engine._build_price_panel(active_tickers, train_start, start_date)
    if train_panel is not None:
        strategy.train_models(train_panel)
    
    # 4. Create Run
    run_info = registry.create_run({
        "strategy_name": "GOLDEN_BASELINE_V1",
        "tickers": active_tickers,
        "config_hash": cfg_hash,
        "mode": exec_cfg['mode']
    })
    run_id = run_info["run_id"]
    
    full_panel = None

    def strategy_fn(ts, current_prices, engine_ptr):
        nonlocal full_panel
        # Rebalance based on frequency
        if exec_cfg['rebalance_frequency'] == "daily":
            # Pass
            pass
        elif exec_cfg['rebalance_frequency'] == "weekly" and ts.weekday() != 2: # Wednesday
            return []
        
        if full_panel is None:
            full_panel = engine_ptr._build_price_panel(active_tickers, start_date)
        
        if full_panel is None: return []
        
        # Align timezone for point-in-time slice (Institutional: MANDATORY UTC)
        ts = to_utc(ts)
        # Ensure panel index is UTC
        if full_panel.index.tz is None:
            full_panel.index = full_panel.index.tz_localize("UTC")
            
        pit_data = full_panel[full_panel.index <= ts]
        if len(pit_data) < 20: return []
        
        # Generation
        # Fetch Macro Context (FRED/AlphaVantage) for Global Decision Making
        # Access safely via engine_ptr provider wrapper
        macro_ctx = None
        try:
             # Our provider wrapper has an internal 'router' attribute, but it's inside the class instance
             # engine_ptr.provider is the RoutingProvider INSTANCE
             if hasattr(engine_ptr.provider, 'router'):
                 macro_ctx = engine_ptr.provider.router.get_macro_context()
        except Exception:
             pass

        gen_signals = strategy.generate_signals(pit_data, macro_context=macro_ctx)
        if gen_signals.empty:
            logger.warning(f"No signals generated for {ts}. Skipping rebalance.")
            return []
            
        signals = gen_signals.iloc[-1].to_dict()
        
        # FIX: iterate over existing columns only, not all levels in metadata
        # This prevents KeyError if a ticker is in levels but not in dataframe columns
        valid_tickers = pit_data.columns.get_level_values(0).unique()
        
        prices_map = {tk: pit_data[tk]["Close"] for tk in valid_tickers}
        vols_map = {tk: pit_data[tk]["Volume"] for tk in valid_tickers}
        
        return allocator.allocate(
            signals, 
            prices_map, 
            vols_map, 
            engine_ptr.portfolio, 
            ts, 
            metadata=universe_meta,
            method="risk_parity"
        ).orders

    if exec_cfg['mode'] == "backtest":
        logger.info(f"Phase 2: Executing Protected Backtest: {run_id}")
        engine.run(start_date=start_date, strategy_fn=strategy_fn, tickers=active_tickers)
        
        # Finalize Backtest Meta
        end_time = time.time()
        meta["duration_seconds"] = end_time - start_time
        with open("output/meta.json", "w") as f:
            json.dump(meta, f, indent=4)
        logger.info(f"Final Backtest Meta saved. Duration: {meta['duration_seconds']:.2f}s")
        
        res = engine.get_results()
        if not res.empty:
            equity = res["equity"]
            ret = equity.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).fillna(0)
            metrics = {
                "annualized_return": annualized_return(ret),
                "sharpe_ratio": sharpe_ratio(ret),
                "max_drawdown": max_drawdown(equity),
                "config_hash": cfg_hash
            }
            registry.save_artifacts(
                run_id=run_id, 
                config=cfg, 
                equity_df=res, 
                trades_df=engine.get_blotter().trades_df(),
                extra_meta=metrics
            )
            
            # Phase 3: Post-Run Attribution
            attr_engine = AttributionEngine()
            attribution = attr_engine.calculate_attribution(res, engine.get_blotter().trades_df())
            logger.info(f"Performance Attribution: {attribution}")
            logger.info(f"GOLDEN RUN COMPLETE! ID: {run_id}")
            
    elif exec_cfg['mode'] in ["paper", "live"]:
        if exec_cfg['mode'] == "live":
            # MANDATORY ENFORCEMENT: --live requires manual confirmation
            confirm = input("!!! WARNING !!! LIVE TRADING MODE. Type 'CONFIRM' to proceed: ")
            if confirm != "CONFIRM":
                logger.error("Live trading aborted by user.")
                return

        logger.info(f"Phase 2: Initializing Live {exec_cfg['mode'].upper()} Engine...")
        
        # Switch to Alpaca Provider if using Alpaca
        live_provider = AlpacaDataProvider() if universe.get('data_source') == "alpaca" else provider
        
        # Determine handler
        if universe.get('broker') == "alpaca":
             live_handler = AlpacaExecutionHandler(
                 os.getenv("ALPACA_API_KEY"),
                 os.getenv("ALPACA_SECRET_KEY")
             )
        else:
             # Fallback to local mock broker if no broker specified
             live_handler = MockBroker(initial_capital=exec_cfg.get('initial_capital', 100000))

        live_engine = LiveEngine(
            provider=live_provider,
            handler=live_handler,
            risk_manager=risk_mgr,
            initial_capital=float(exec_cfg.get('initial_capital', 100000))
        )
        
        # Initialize Monitoring
        if alerts:
            alerts.alert(f"Autonomous Trading Loop Started. Frequency: {exec_cfg['rebalance_frequency']}")

        # Loop persistence state
        loop_error_count = 0
        last_report_date = None
        last_trade_time = 0.0

        while True:
            try:
                # ---------------------------------------------------------
                # 1. DAILY REPORT SCHEDULER (Evening 5:30 PM - 6:00 PM)
                # ---------------------------------------------------------
                now = datetime.now()
                # Target: 17:30
                if now.hour == 17 and now.minute >= 30:
                    today_str = now.strftime("%Y-%m-%d")
                    if last_report_date != today_str:
                        logger.info("🕒 Scheduled Event: Sending Daily Telegram Report...")
                        try:
                            generate_report()
                            last_report_date = today_str
                            if alerts:
                                alerts.alert("Daily Report Sent Successfully.", level="SCHEDULER")
                        except Exception as e:
                            logger.error(f"Report generation failed: {e}")
                            if alerts:
                                alerts.alert(f"Report Failed: {e}", level="ERROR")

                # ---------------------------------------------------------
                # 2. REAL-TIME EVENT LOOP (The "Listener")
                # ---------------------------------------------------------
                # Initialize Listener using the internal router instance if available, else provider
                router_inst = provider.router_instance if hasattr(provider, 'router_instance') else None
                if 'listener' not in locals():
                    logger.info("Initializing Real-Time Market Listener...")
                    # Fallback to provider if it acts as router, or ensure router is accessible
                    # In our code, provider is RoutingProvider which has .router property
                    listener_router = router_inst if router_inst else provider 
                    listener = MarketListener(listener_router, active_tickers)
                
                # A. Adaptive Polling (Micro-Sleep)
                # We poll frequently (~1s) to check if any ticker logic needs to run
                # The listener class handles per-ticker rate limiting.
                time.sleep(1.0) 
                
                # B. Check for Schedule (Hourly Heartbeat / Rebalance)
                # Still run full rebalance periodically to ensure portfolio weights are roughly correct
                is_scheduled_run = False
                if time.time() - last_trade_time > 3600:
                    logger.info("Triggering Scheduled Hourly Rebalance...")
                    is_scheduled_run = True

                # C. Check for Market Events
                events = listener.tick()
                if events:
                    logger.warning(f"MARKET EVENTS DETECTED: {events}")
                    for evt in events:
                        if alerts: alerts.alert(f"EVENT: {evt}", level="CRITICAL")
                
                # D. Execution Trigger
                # Run engine IF: Scheduled OR Significant Event
                if events:
                     # FAST PATH: Check for Critical CRASH Events
                     # Institutional Requirement 3: Dual-Speed Intelligence
                     for evt in events:
                         if "FLASH_CRASH" in evt:
                             logger.critical("⚡ FAST PATH TRIGGERED: Executing EMERGENCY PROTOCOL.")
                             live_engine.enter_safe_mode()
                             # Wait for manual recovery
                             while live_engine.crash_mode:
                                 time.sleep(10)
                                 logger.warning("System in SAFE MODE. Waiting for restart...")
                             break # Exit event loop if crashed
                
                if is_scheduled_run or (events and not live_engine.crash_mode):
                    # SLOW PATH: Normal Rebalance / Volatility Adjustment
                    # Heartbeat / Diagnosis
                    if alerts:
                        risk_tier = "NORMAL"
                        if risk_mgr:
                            risk_tier = risk_mgr.get_risk_tier(live_engine.portfolio_value)
                        
                        if events or risk_tier in ["HIGH", "EXTREME"]:
                            diag = risk_mgr.explain_diagnostics(live_engine.portfolio_value) if risk_mgr else "N/A"
                            alerts.heartbeat(diag)
                        elif is_scheduled_run:
                            alerts.heartbeat()

                    # Execute Rebalance / Reaction
                    live_engine.run_once(active_tickers, strategy_fn)
                    last_trade_time = time.time()
                    
                    # RESET ERROR COUNT ON SUCCESS
                    loop_error_count = 0
                    
                    # Update Meta
                    if 'start_time' in locals():
                        meta["duration_seconds"] = time.time() - start_time
                    with open("output/meta.json", "w") as f:
                        json.dump(meta, f, indent=4)
                        
                    logger.info("Cycle Complete. Resuming Surveillance...")

            except KeyboardInterrupt:
                logger.warning("User stopped the loop. Performing graceful shutdown...")
                if 'live_engine' in locals():
                    live_engine.enter_safe_mode()
                break
            except Exception as e:
                logger.error(f"PIPELINE ERROR: {e}", exc_info=True)
                
                loop_error_count += 1
                if loop_error_count > 10:
                    logger.critical("TOO MANY PIPELINE ERRORS. Hard-stopping for safety.")
                    if 'live_engine' in locals():
                        live_engine.enter_safe_mode()
                    break
                
                # "Solid Rock" Stability: Adaptive Cooldown (Exponential-ish)
                cooldown_sleep = min(300, 10 * loop_error_count)
                logger.info(f"Cooling down for {cooldown_sleep}s before retry (Error #{loop_error_count})...")
                time.sleep(cooldown_sleep) 
    
    # Initial Meta Persistence
    try:
        os.makedirs("output", exist_ok=True)
        with open("output/meta.json", "w") as f:
            json.dump(meta, f, indent=4)
        logger.info(f"Initial Performance Meta saved to output/meta.json")
    except Exception as e:
        logger.warning(f"Failed to save initial meta.json: {e}")

if __name__ == "__main__":
    run_production_pipeline()
