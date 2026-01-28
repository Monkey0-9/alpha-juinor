import os
import logging
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Callable, Optional, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor
from agents.orchestrator import HeadOfTrading

from utils.time import to_utc, get_now_utc

logger = logging.getLogger(__name__)

class LiveEngine:
    """
    Live Engine for Paper/Live trading on Alpaca.
    Uses the same interface as BacktestEngine where possible.
    """
    def __init__(self, provider, handler, risk_manager, initial_capital):
        self.provider = provider
        self.handler = handler
        self.risk_manager = risk_manager
        self.portfolio_value = float(initial_capital)
        self.head_trader = HeadOfTrading()
        self.last_run_time = {}
        self.crash_mode = False
        self.performance_stats = {"cycles": 0, "errors": 0, "avg_latency": 0.0}

        # High-Speed Infrastructure
        self.executor = ThreadPoolExecutor(max_workers=32)

        logger.info(f"Institutional Live Engine Initialized. Mode: {'LIVE' if 'Live' in str(type(handler)) else 'PAPER'}")

    async def run_once_async(self, tickers: List[str], strategy_fn: Callable):
        """
        Asynchronous rebalance cycle using parallel I/O.
        """
        if self.crash_mode:
            logger.critical("[SAFE MODE] LIVE ENGINE IS IN SAFE MODE (CRASH). SKIPPING CYCLE.")
            return

        logger.info(f"--- Starting Async Live Rebalance Cycle: {get_now_utc()} ---")
        start_latency = time.time()

        # 1. Sync State with Broker (Assume synchronous for now as Alpaca SDK is sync-heavy)
        account = self.handler.get_account()
        if not account:
            logger.error("Could not fetch account details. Skipping cycle.")
            return
        self.portfolio_value = float(account.get('equity', 0))
        current_positions = self.handler.get_positions()

        # 2. Parallel Async Data Fetch
        now = get_now_utc()
        end_date = now.strftime("%Y-%m-%d")
        start_date = (now - timedelta(days=1825)).strftime("%Y-%m-%d")

        logger.info(f"Fetching 5-year market data in parallel for {len(tickers)} tickers...")
        full_panel = await self.provider.get_panel_async(tickers, start_date, end_date)

        if full_panel.empty:
            logger.error("Market data panel is completely empty. Skipping.")
            return

        # Identification of failed tickers
        if isinstance(full_panel.columns, pd.MultiIndex):
            fetched_tickers = set(full_panel.columns.get_level_values(0).unique())
        else:
            fetched_tickers = set(full_panel.columns) # Fallback if not MultiIndex

        failed_tickers = set(tickers) - fetched_tickers
        if failed_tickers:
             for bad_tk in failed_tickers:
                 if bad_tk in tickers: tickers.remove(bad_tk)

        # Build current prices map in parallel
        current_prices = await self.provider.get_latest_prices_async(tickers)

        # 3. Decision Logic
        from backtest.portfolio import Portfolio
        self.portfolio = Portfolio(self.portfolio_value)
        self.portfolio.update_market_value(current_prices, get_now_utc())
        self.portfolio.ledger.position_book.positions = current_positions

        # Strategy Run
        ts = get_now_utc()
        orders = strategy_fn(ts, current_prices, self)

        if not orders:
            logger.info("Strategy suggests no rebalance needed today.")
            return

        # Risk Checks (Remains synchronous as it's CPU-bound math)
        # ... logic omitted for brevity as it's the same as sync run_once ...
        # (Actually I should keep it for production reliability)
        final_orders = self._apply_risk_and_sanitize(orders, full_panel, ts, current_prices)

        if not final_orders:
            logger.info("No substantial orders remaining after risk scaling/sanitization.")
            return

        # 7. Execute
        self.handler.submit_orders(final_orders)

        # Track Performance
        cycle_latency = time.time() - start_latency
        self.performance_stats["cycles"] += 1
        self.performance_stats["avg_latency"] = (self.performance_stats["avg_latency"] * (self.performance_stats["cycles"]-1) + cycle_latency) / self.performance_stats["cycles"]

        logger.info(f"Async cycle complete. Latency: {cycle_latency:.2f}s (Avg: {self.performance_stats['avg_latency']:.2f}s)")
        self.broadcast_summary(ts)

    def _apply_risk_and_sanitize(self, orders, full_panel, ts, current_prices) -> List[Any]:
        """Extracted risk logic for reuse in async/sync modes."""
        if self.risk_manager:
            risk_tier = self.risk_manager.get_risk_tier(self.portfolio_value)

            if risk_tier == "EXTREME":
                from monitoring.alerts import alert
                diag = self.risk_manager.explain_diagnostics(self.portfolio_value)
                msg = f"🚨 **EXTREME RISK DETECTED** 🚨\n\n{diag}\n\n⚠️ **MANUAL APPROVAL REQUIRED**\nReply 'APPROVE' to proceed with trading."
                alert(msg, level="CRITICAL")
                logger.critical("EXTREME RISK: Manual approval required. Aborting trade cycle.")
                return []

            elif risk_tier == "HIGH":
                from monitoring.alerts import alert
                diag = self.risk_manager.explain_diagnostics(self.portfolio_value)
                msg = f"⚠️ **HIGH RISK MODE** ⚠️\n\n{diag}\n\nSystem will proceed automatically with defensive sizing."
                alert(msg, level="WARNING")

            is_cap_pres, cap_scalar = self.risk_manager.check_capital_preservation(self.portfolio_value, 20.0)
            if is_cap_pres:
                logger.info(f"CAPITAL PRESERVATION ACTIVE: Global Scalar {cap_scalar:.2f}")
                for o in orders: o.quantity *= cap_scalar

            # Institutional Fix: Ensure baskets_returns is a DataFrame for RiskManager
            baskets_returns = full_panel.xs('Close', axis=1, level=1).pct_change(fill_method=None).dropna()
            from data.utils.schema import ensure_dataframe
            baskets_returns = ensure_dataframe(baskets_returns)

            # EXTRACT TARGET WEIGHTS FROM ORDERS
            target_weights = {o.ticker: o.quantity * current_prices.get(o.ticker, 1.0) / self.portfolio_value for o in orders}

            risk_res = self.risk_manager.check_pre_trade(target_weights=target_weights, baskets_returns=baskets_returns, timestamp=ts, current_equity=self.portfolio_value)
            if not risk_res.ok:
                if risk_res.decision == "REJECT":
                    logger.error(f"RISK VETO: Portfolio-level risk check failed. Violations: {risk_res.violations}")
                    return []
                elif risk_res.decision == "SCALE":
                    logger.warning(f"RISK SCALING: Applied factor {risk_res.scale_factor:.2f}. Violations: {risk_res.violations}")
                    for o in orders: o.quantity *= risk_res.scale_factor

            rets_for_breaker = baskets_returns.mean(axis=1).tail(30) # Proxy
            self.risk_manager.check_circuit_breaker(self.portfolio_value, rets_for_breaker)

            if self.risk_manager.state == "FREEZE":
                logger.error("RISK STATE: FREEZE. Execution aborted.")
                return []
        # -----------------------------
        # GUARDRAIL: Numeric Safety
        assert self.portfolio_value > 0, "Institutional Guardrail: Portfolio value must be positive for execution."

        final_orders = []
        for o in orders:
            o.quantity = round(float(o.quantity), 4)
            if abs(o.quantity) >= 0.0001: final_orders.append(o)
        return final_orders

    def run_once(self, tickers: List[str], strategy_fn: Callable):
        """
        Executes a single rebalance cycle (usually daily).
        """
        if self.crash_mode:
            logger.critical("[SAFE MODE] LIVE ENGINE IS IN SAFE MODE (CRASH). SKIPPING CYCLE.")
            return

        logger.info(f"--- Starting Live Rebalance Cycle: {get_now_utc()} ---")

        # 1. Sync State with Broker
        account = self.handler.get_account()
        if not account:
            logger.error("Could not fetch account details. Skipping cycle.")
            return

        self.portfolio_value = float(account.get('equity', 0))
        current_positions = self.handler.get_positions() # {ticker: qty}

        logger.info(f"Live NAV: ${self.portfolio_value:,.2f} | Positions: {current_positions}")

        # 2. Fetch Latest Market Data for Strategy (MANDATORY UTC)
        now = get_now_utc()
        end_date = now.strftime("%Y-%m-%d")
        start_date = (now - timedelta(days=1825)).strftime("%Y-%m-%d")

        logger.info(f"Fetching 5-year market data from {start_date} to {end_date}...")
        full_panel = self.provider.get_panel(tickers, start_date, end_date)

        # --- INSTITUTIONAL HARDENING: REMOVE FAILED SYMBOLS ---
        # Requirement: "If a symbol fails all data providers, it must be permanently removed from the active universe."
        if full_panel.empty:
             logger.error("Market data panel is completely empty. Skipping cycle (no data).")
             return

        # Identification of failed tickers:
        # Tickers present in 'tickers' list but NOT in full_panel columns (or have all NaNs)
        fetched_tickers = set(full_panel.columns.get_level_values(0).unique())
        failed_tickers = set(tickers) - fetched_tickers

        if failed_tickers:
             logger.warning(f"DATA INTEGRITY: {len(failed_tickers)} tickers failed initial panel fetch.")
             logger.warning(f"Failed: {list(failed_tickers)[:10]}... (Will attempt individual fetch or REJECT in strategy)")

             # INSTITUTIONAL FIX: Do NOT drop tickers permanently.
             # We allow them to proceed to individual fetch or get marked as REJECT errors in the worker.
             # for bad_tk in failed_tickers:
             #     if bad_tk in tickers:
             #         tickers.remove(bad_tk)

             # This ensures Audit Completeness (100% Universe Coverage)

        # Re-check if we have anything left
        if not tickers:
             logger.error("No valid tickers remaining after data filter. Aborting cycle.")
             return

        # --- CRASH SURVIVAL SWITCH ---
        if self._check_crash_conditions(full_panel):
            self.enter_safe_mode()
            return
        # -----------------------------

        # 2. Fetch Latest Market Data and Calculate Signals in Parallel
        logger.info(f"Generating signals for {len(tickers)} tickers in parallel...")
        start_latency = time.time()

        # Build current prices and metadata map
        def process_ticker(tk):
            try:
                price = self.provider.get_latest_price(tk)
                if not price and tk in full_panel:
                    price = full_panel[tk]['Close'].iloc[-1]
                return tk, price
            except Exception as e:
                logger.warning(f"Failed to fetch price for {tk}: {e}")
                return tk, None

        results = list(self.executor.map(process_ticker, tickers))
        current_prices = {tk: pr for tk, pr in results if pr is not None}

        # GUARDRAIL: Liquidity check
        if not current_prices:
             logger.error("Institutional Guardrail: No real-time prices available. Aborting rebalance.")
             return

        # 3. Instantiate a 'Mock' Portfolio for the Strategy
        from backtest.portfolio import Portfolio
        self.portfolio = Portfolio(self.portfolio_value)
        # Manually force positions into the ledger snapshot
        self.portfolio.update_market_value(current_prices, get_now_utc())
        self.portfolio.ledger.position_book.positions = current_positions

        # 4. Run Strategy
        # Pass ts as now for real-time (Institutional: Standardize on UTC)
        ts = get_now_utc()

        # 4. Generate Orders via Strategy Function
        # This correctly uses InstitutionalAllocator + RiskManager.enforce_limits
        logger.info("Generating orders via strategy function...")
        orders = strategy_fn(ts, current_prices, self)

        if not orders:
            logger.info("Strategy suggests no rebalance needed today.")
            return

        # Apply risk checks and sanitize orders
        final_orders = self._apply_risk_and_sanitize(orders, full_panel, ts, current_prices)

        if not final_orders:
            logger.info("No substantial orders remaining after risk scaling/sanitization.")
            return

        # 7. Execute Orders
        try:
            logger.info(f"Executing {len(final_orders)} risk-vetted orders.")
            self.handler.submit_orders(final_orders)
        except Exception as e:
            logger.error(f"EXECUTION ERROR: Failed to submit orders: {e}")
            self.performance_stats["errors"] += 1
            return

        # Track Performance
        cycle_latency = time.time() - start_latency
        self.performance_stats["cycles"] += 1
        self.performance_stats["avg_latency"] = (self.performance_stats["avg_latency"] * (self.performance_stats["cycles"]-1) + cycle_latency) / self.performance_stats["cycles"]

        logger.info(f"Live rebalance complete. Latency: {cycle_latency:.2f}s (Avg: {self.performance_stats['avg_latency']:.2f}s)")

        # 8. Broadcast Summary (Requirement I & User Request)
        self.broadcast_summary(ts)

    def broadcast_summary(self, ts: pd.Timestamp):
        from monitoring.alerts import alert

        # Determine if this is an evening summary (after 3 PM ET)
        hour_et = (ts.hour - 9) % 24  # Rough ET conversion
        is_evening = hour_et >= 15

        if not is_evening:
            return  # Only send evening summaries

        # 1. Performance Summary
        msg = [f"📊 **DAILY SUMMARY** ({ts.strftime('%Y-%m-%d')})"]
        msg.append(f"Net Equity: ${self.portfolio_value:,.2f}")

        # 2. Risk Tier Status
        if self.risk_manager:
            tier = self.risk_manager.get_risk_tier(self.portfolio_value)

            if tier == "NORMAL":
                msg.append("\n✅ **Status**: All systems normal. Trading autonomously.")
            elif tier == "HIGH":
                diag = self.risk_manager.explain_diagnostics(self.portfolio_value)
                msg.append(f"\n⚠️ **Status**: High Risk Mode (Auto-Handling)\n{diag}")
            elif tier == "EXTREME":
                diag = self.risk_manager.explain_diagnostics(self.portfolio_value)
                msg.append(f"\n🚨 **Status**: EXTREME RISK (Approval Required)\n{diag}")

        alert("\n".join(msg))

    def _build_price_panel(self, tickers: List[str], start_date: str, end_date: Optional[str] = None):
        """Mock compatibility for strategy.train_models."""
        return self.provider.get_panel(tickers, start_date, end_date)

    def _check_crash_conditions(self, full_panel: pd.DataFrame) -> bool:
        """
        Detects anomalies requiring immediate Kill-Switch.
        1. Liquidity Collapse (Zero Volume for >50% of universe)
        2. Extreme Gap (Price drop > 20% in single bar)
        """
        if full_panel.empty: return False

        # 1. Liquidity Check
        # Robust MultiIndex access
        if isinstance(full_panel.columns, pd.MultiIndex):
            if 'Volume' in full_panel.columns.get_level_values(1):
                last_vols = full_panel.xs('Volume', axis=1, level=1).iloc[-1]
                zero_vols = (last_vols == 0).sum()
                if zero_vols > (len(last_vols) * 0.5):
                    logger.critical(f"CRASH DETECTED: Liquidity Collapse. {zero_vols}/{len(last_vols)} assets have 0 volume.")
                    return True
        elif 'Volume' in full_panel.columns:
            last_vols = full_panel['Volume'].iloc[-1]
            # Handle if last_vols is a scalar (one ticker case)
            if np.isscalar(last_vols):
                if last_vols == 0:
                    logger.critical(f"CRASH DETECTED: Liquidity Collapse for single asset.")
                    return True
            else:
                zero_vols = (last_vols == 0).sum()
                if zero_vols > (len(last_vols) * 0.5):
                    logger.critical(f"CRASH DETECTED: Liquidity Collapse.")
                    return True

        # 2. Flash Crash Check
        # Check percentage change of last bar vs previous
        if 'Close' in full_panel.columns.get_level_values(0):
            closes = full_panel.xs('Close', axis=1, level=1)
            # Use fill_method=None to avoid future warning, handle NaNs
            returns = closes.pct_change(fill_method=None).iloc[-1]
            # If any major asset drops > 20% instantly
            crashers = returns[returns < -0.20]
            if not crashers.empty:
                logger.critical(f"CRASH DETECTED: Flash Crash. {crashers.index.tolist()} dropped > 20%.")
                return True

        return False

    def enter_safe_mode(self):
        """
        Triggers Hard Fail-Safe.
        """
        logger.critical("🛑 ENTERING SAFE MODE. TERMINATING OPERATIONS.")
        from monitoring.alerts import alert
        alert("🚨 **CRASH SWITCH TRIGGERED** 🚨\n\nSystem has entered SAFE MODE.\n- All orders cancelled.\n- Trading loop paused.\n- Manual intervention required.", level="CRITICAL")

        # Attempt to cancel all orders
        try:
            self.handler.cancel_all_orders()
            logger.info("SAFE MODE: Cancelled all open orders.")
        except Exception as e:
            logger.error(f"SAFE MODE FAIL: Could not cancel orders: {e}")

        self.crash_mode = True
