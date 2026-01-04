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
from dotenv import load_dotenv

# Institutional Config Management
from configs.config_manager import ConfigManager

load_dotenv()

from data.collectors.yahoo_collector import YahooDataProvider
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
from brokers.alpaca_broker import AlpacaExecutionHandler
from data.collectors.alpaca_collector import AlpacaDataProvider
from engine.live_engine import LiveEngine

logger = logging.getLogger("InstitutionalDriver")

def run_production_pipeline():
    """
    Main entry point for the validated institutional pipeline.
    Uses Golden Config for all parameters.
    """
    start_time = time.time()
    
    # 1. Initialize Configuration
    cm = ConfigManager()
    cfg = cm.config
    cfg_hash = cm.config_hash
    
    # 2. Setup Resources
    registry = BacktestRegistry(base_dir=str(Path("output/backtests")))
    alerts = AlertManager()
    
    # 0. Generate Daily Checklist
    generate_daily_checklist(cfg_hash)
    
    # Audit Meta
    meta = {
        "mode": cfg['execution']['mode'],
        "timestamp": datetime.now().isoformat(),
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
    
    provider = YahooDataProvider() # Future: use universe['data_source']
    
    risk_mgr = RiskManager(
        max_leverage=risk_cfg['max_gross_leverage'],
        target_vol_limit=risk_cfg['target_volatility_annualized']
    )
    
    allocator = InstitutionalAllocator(
        risk_manager=risk_mgr,
        max_leverage=risk_cfg['max_gross_leverage']
    )
    
    strategy = StrategyFactory.create_strategy({
        "type": "institutional",
        "tickers": universe['tickers'],
        "use_ml": cfg['alpha']['ml_weight'] > 0
    })
    
    handler = RealisticExecutionHandler(commission_pct=exec_cfg['commission_bps'] / 10000.0)
    engine = BacktestEngine(
        provider=provider, 
        initial_capital=exec_cfg['initial_capital'], 
        execution_handler=handler
    )
    
    # 3. Model Warming
    start_date = "2023-01-01" # Baseline start
    train_start = (pd.to_datetime(start_date) - timedelta(days=90)).strftime("%Y-%m-%d")
    logger.info(f"Phase 1: Warming models from {train_start}...")
    train_panel = engine._build_price_panel(universe['tickers'], train_start, start_date)
    if train_panel is not None:
        strategy.train_models(train_panel)
    
    # 4. Create Run
    run_info = registry.create_run({
        "strategy_name": "GOLDEN_BASELINE_V1",
        "tickers": universe['tickers'],
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
            full_panel = engine_ptr._build_price_panel(universe['tickers'], start_date)
        
        if full_panel is None: return []
        
        pit_data = full_panel[full_panel.index <= ts]
        if len(pit_data) < 20: return []
        
        # Generation
        signals = strategy.generate_signals(pit_data).iloc[-1].to_dict()
        prices_map = {tk: pit_data[tk]["Close"] for tk in pit_data.columns.levels[0]}
        vols_map = {tk: pit_data[tk]["Volume"] for tk in pit_data.columns.levels[0]}
        
        return allocator.allocate(signals, prices_map, vols_map, engine_ptr.portfolio, ts, method="signal").orders

    if exec_cfg['mode'] == "backtest":
        logger.info(f"Phase 2: Executing Protected Backtest: {run_id}")
        engine.run(start_date=start_date, strategy_fn=strategy_fn, tickers=universe['tickers'])
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
             from brokers.mock_broker import MockBroker
             live_handler = MockBroker(initial_capital=exec_cfg.get('initial_capital', 100000))

        live_engine = LiveEngine(
            provider=live_provider,
            handler=live_handler,
            risk_manager=risk_mgr,
            initial_capital=float(exec_cfg.get('initial_capital', 100000))
        )
        
        # Initialize Monitoring
        from monitoring.alerts import AlertManager
        alert_mgr = AlertManager()
        alert_mgr.alert(f"Autonomous Trading Loop Started. Frequency: {exec_cfg['rebalance_frequency']}")

        while True:
            try:
                # Heartbeat
                alert_mgr.heartbeat()

                # Execute today's rebalance
                live_engine.run_once(universe['tickers'], strategy_fn)
                
                # Update meta
                meta["duration_seconds"] = time.time() - start_time
                with open("output/meta.json", "w") as f:
                    json.dump(meta, f, indent=4)
                
                # Sleep based on frequency (Daily -> check every hour, etc)
                # For daily, we'll sleep 1 hour and re-check if it's a new day/market hours
                logger.info("Sleeping for 1 hour...")
                time.sleep(3600) 
            except Exception as e:
                logger.error(f"Error in 24/7 loop: {e}")
                time.sleep(60) # Short sleep on error before retry
    
    # Initial Meta Persistence
    try:
        os.makedirs("output", exist_ok=True)
        with open("output/meta.json", "w") as f:
            json.dump(meta, f, indent=4)
        logger.info(f"Initial Performance Meta saved to output/meta.json")
    except Exception as e:
        logger.warning(f"Failed to save initial meta.json: {e}")

    if exec_cfg['mode'] == "backtest":
        logger.info(f"Phase 2: Executing Protected Backtest: {run_id}")
        engine.run(start_date=start_date, strategy_fn=strategy_fn, tickers=universe['tickers'])
        
        # Finalize Backtest Meta
        end_time = time.time()
        meta["duration_seconds"] = end_time - start_time
        with open("output/meta.json", "w") as f:
            json.dump(meta, f, indent=4)
        logger.info(f"Final Backtest Meta saved. Duration: {meta['duration_seconds']:.2f}s")

    elif exec_cfg['mode'] in ["paper", "live"]:
        # ... setup ...
        res = engine.get_results()
        if not res.empty:
            equity = res["equity"]
            ret = equity.pct_change().fillna(0)
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
    else:
        logger.info(f"LIVE {exec_cfg['mode'].upper()} CYCLE COMPLETE.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    run_production_pipeline()
