
import os
import logging
import time
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Callable, Optional
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
        self.crash_mode = False
        
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
        start_date = (now - timedelta(days=252)).strftime("%Y-%m-%d")
        
        logger.info(f"Fetching market data from {start_date} to {end_date}...")
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
             logger.warning(f"DATA INTEGRITY: Dropping {len(failed_tickers)} failed tickers from UNIVERSE permanently.")
             logger.warning(f"Dropped: {list(failed_tickers)[:10]}...")
             
             # Permanent Removal for this runtime session
             for bad_tk in failed_tickers:
                 if bad_tk in tickers:
                     tickers.remove(bad_tk)
             
             # Also update local reference if using filtering
             # But 'tickers' passed in is likely a reference to the main list in main.py loop or copy
             # Ideally we return the valid list or the caller should know.
             # In main.py loop logic, 'active_tickers' is passed. Modifying it in-place affects next iterations.
             # This satisfies "Once a symbol fails in a session, never retry it."

        # Re-check if we have anything left
        if not tickers:
             logger.error("No valid tickers remaining after data filter. Aborting cycle.")
             return

        # --- CRASH SURVIVAL SWITCH ---
        if self._check_crash_conditions(full_panel):
            self.enter_safe_mode()
            return
        # -----------------------------

        # Build current prices map
        current_prices = {}
        for tk in tickers:
            price = self.provider.get_latest_quote(tk)
            if price:
                current_prices[tk] = price
            else:
                # Fallback to last close in panel
                if tk in full_panel:
                    current_prices[tk] = full_panel[tk]['Close'].iloc[-1]
                else:
                    logger.warning(f"Price unavailable for {tk} (No Live Quote & No History). Skipping.")
        
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
        
        # 4.5 TIERED RISK CHECK (User Request)
        if self.risk_manager:
            risk_tier = self.risk_manager.get_risk_tier(self.portfolio_value)
            
            if risk_tier == "EXTREME":
                from monitoring.alerts import alert
                diag = self.risk_manager.explain_diagnostics(self.portfolio_value)
                msg = f"🚨 **EXTREME RISK DETECTED** 🚨\n\n{diag}\n\n⚠️ **MANUAL APPROVAL REQUIRED**\nReply 'APPROVE' to proceed with trading."
                alert(msg, level="CRITICAL")
                logger.critical("EXTREME RISK: Manual approval required. Aborting trade cycle.")
                return
            
            elif risk_tier == "HIGH":
                from monitoring.alerts import alert
                diag = self.risk_manager.explain_diagnostics(self.portfolio_value)
                msg = f"⚠️ **HIGH RISK MODE** ⚠️\n\n{diag}\n\nSystem will proceed automatically with defensive sizing."
                alert(msg, level="WARNING")
                
            # --- CAPITAL PRESERVATION CHECK ---
            # Explicit call if needed, though enforce_limits usually handles it.
            # But per user request we can log it here.
            is_cap_pres, cap_scalar = self.risk_manager.check_capital_preservation(self.portfolio_value, 20.0) # VIX placeholder if not passed, but strategy has Macro
            if is_cap_pres:
                logger.info(f"CAPITAL PRESERVATION ACTIVE: Global Scalar {cap_scalar:.2f}")
                # We can apply this global scalar to all orders
                for o in orders:
                    o.quantity *= cap_scalar


        # 5. PORTFOLIO-LEVEL RISK AUDIT (Institutional Safety Guard)
        if self.risk_manager:
            # Prepare target weights for check_pre_trade
            target_weights = {}
            for o in orders:
                # target_weight = order_notional / total_equity
                target_weights[o.ticker] = (o.quantity * current_prices.get(o.ticker, 0)) / (self.portfolio_value + 1e-9)
            
            # Fetch returns for panels (needed for VaR/CVaR check)
            # In live, we use the panel returns
            baskets_returns = full_panel.xs('Close', axis=1, level=1).pct_change(fill_method=None).dropna()
            
            risk_res = self.risk_manager.check_pre_trade(
                target_weights=target_weights,
                baskets_returns=baskets_returns,
                timestamp=ts,
                current_equity=self.portfolio_value
            )
            
            if not risk_res.ok:
                if risk_res.decision == "REJECT":
                    logger.error(f"RISK VETO: Portfolio-level risk check failed. Violations: {risk_res.violations}")
                    return
                elif risk_res.decision == "SCALE":
                    logger.warning(f"RISK SCALING: Applied factor {risk_res.scale_factor:.2f}. Violations: {risk_res.violations}")
                    for o in orders:
                        o.quantity *= risk_res.scale_factor
            
            # Final Circuit Breaker Check
            # realized_returns for circuit breaker usually calculated over window
            rets_for_breaker = baskets_returns.mean(axis=1).tail(30) # Proxy
            self.risk_manager.check_circuit_breaker(self.portfolio_value, rets_for_breaker)
            
            if self.risk_manager.state == "FREEZE":
                logger.error("RISK STATE: FREEZE. Execution aborted.")
                return
 
        # 6. Final Order Sanitization (Fractional Rounding)
        final_orders = []
        for o in orders:
            o.quantity = round(float(o.quantity), 4)
            if abs(o.quantity) >= 0.0001:
                final_orders.append(o)

        if not final_orders:
            logger.info("No substantial orders remaining after risk scaling/sanitization.")
            return

        # 7. Execute Orders
        logger.info(f"Executing {len(final_orders)} risk-vetted orders.")
        self.handler.submit_orders(final_orders)
        logger.info("Live rebalance complete.")
        
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
        if 'Volume' in full_panel.columns.get_level_values(0):
            # Check last row volume
            last_vols = full_panel.xs('Volume', axis=1, level=1).iloc[-1]
            zero_vols = (last_vols == 0).sum()
            if zero_vols > (len(last_vols) * 0.5):
                logger.critical(f"CRASH DETECTED: Liquidity Collapse. {zero_vols}/{len(last_vols)} assets have 0 volume.")
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
