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
from typing import List, Optional, Any, Dict

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
from brokers.mock_broker import MockBroker
from utils.time import to_utc, get_now_utc
from data.universe_manager import UnifiedUniverseManager

logger = logging.getLogger("InstitutionalDriver")

def run_production_pipeline():
    """
    Main entry point for the validated institutional pipeline.
    """
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
    alerts = AlertManager()
    
    # 0. Generate Daily Checklist
    generate_daily_checklist(cfg_hash)
    
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
    # Crypto -> Alpaca (Unlimited free tier), Equities -> Yahoo (Silent fallback)
    def router_provider(ticker_list: List[str]):
        """Specialized routing provider that respects entitlements."""
        class RoutingProvider:
            def __init__(self):
                self.alpaca = AlpacaDataProvider()
                self.yahoo = YahooDataProvider()
            
            def get_panel(self, tickers: List[str], start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
                data = {}
                for tk in tickers:
                    # ROUTE: -USD is crypto -> Alpaca, else -> Yahoo (to avoid 403)
                    if "-USD" in tk or "BTC" in tk or "ETH" in tk:
                        df = self.alpaca.fetch_ohlcv(tk, start_date, end_date)
                    else:
                        df = self.yahoo.fetch_ohlcv(tk, start_date, end_date)
                    
                    if not df.empty:
                        for col in df.columns:
                            data[(tk, col)] = df[col]
                
                if not data: return pd.DataFrame()
                panel = pd.DataFrame(data)
                panel.columns = pd.MultiIndex.from_tuples(panel.columns)
                return panel

            def get_latest_quote(self, ticker: str) -> Optional[float]:
                if "-USD" in ticker or "BTC" in ticker or "ETH" in ticker:
                    return self.alpaca.get_latest_quote(ticker)
                return self.yahoo.get_latest_quote(ticker)
        
        return RoutingProvider()

    if exec_cfg['mode'] in ['paper', 'live']:
        logger.info("INSTITUTIONAL: Using Entitlement-Aware Data Router for Production.")
        base_tickers = universe['tickers']
    else:
        base_tickers = universe['tickers']

    # 1b. Universe Discovery (Phase 15)
    provider_raw = AlpacaDataProvider() # Access to discovery
    um = UnifiedUniverseManager(provider_raw)
    
    # If using full market config or tickers is not explicit
    if args.tickers:
        active_tickers = args.tickers.split(",")
        universe_meta = {}
    else:
        # Full Market Discovery
        active_tickers = um.discover_and_filter(universe, force_refresh=args.refresh_universe)
        # Limit to Top 500 for compute safety (Scaling heuristic)
        active_tickers = active_tickers[:500] 
        universe_meta_df = pd.read_parquet(um.cache_path)
        universe_meta = universe_meta_df.set_index("symbol").to_dict("index")

    provider = router_provider(active_tickers)
    
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
    start_date = "2023-01-01" # Baseline start
    train_start = (pd.to_datetime(start_date) - timedelta(days=90)).strftime("%Y-%m-%d")
    logger.info(f"Phase 1: Warming models from {train_start}...")
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
        signals = strategy.generate_signals(pit_data).iloc[-1].to_dict()
        prices_map = {tk: pit_data[tk]["Close"] for tk in pit_data.columns.levels[0]}
        vols_map = {tk: pit_data[tk]["Volume"] for tk in pit_data.columns.levels[0]}
        
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
        alert_mgr = AlertManager()
        alert_mgr.alert(f"Autonomous Trading Loop Started. Frequency: {exec_cfg['rebalance_frequency']}")

        while True:
            try:
                # Heartbeat (Only send diagnosis if HIGH or EXTREME risk)
                if 'live_engine' in locals() and risk_mgr:
                    tier = risk_mgr.get_risk_tier(live_engine.portfolio_value)
                    if tier in ["HIGH", "EXTREME"]:
                        diag = risk_mgr.explain_diagnostics(live_engine.portfolio_value)
                        alert_mgr.heartbeat(diagnosis=diag)
                    else:
                        alert_mgr.heartbeat()  # Normal heartbeat without diagnosis
                else:
                    alert_mgr.heartbeat()

                # Execute today's rebalance
                live_engine.run_once(active_tickers, strategy_fn)
                
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
    else:
        logger.info(f"LIVE {exec_cfg['mode'].upper()} CYCLE COMPLETE.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    run_production_pipeline()
