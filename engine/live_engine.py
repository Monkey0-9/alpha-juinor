
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
        
    def run_once(self, tickers: List[str], strategy_fn: Callable):
        """
        Executes a single rebalance cycle (usually daily).
        """
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
        
        if full_panel.empty:
            logger.error("Market data panel is empty. Cannot generate signals.")
            return

        # Build current prices map
        current_prices = {}
        for tk in tickers:
            price = self.provider.get_latest_quote(tk)
            if price:
                current_prices[tk] = price
            else:
                # Fallback to last close in panel
                current_prices[tk] = full_panel[tk]['Close'].iloc[-1]
        
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
