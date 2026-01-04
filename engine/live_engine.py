
import os
import logging
import time
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Callable, Optional
from agents.orchestrator import HeadOfTrading

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
        logger.info(f"--- Starting Live Rebalance Cycle: {datetime.now()} ---")
        
        # 1. Sync State with Broker
        account = self.handler.get_account()
        if not account:
            logger.error("Could not fetch account details. Skipping cycle.")
            return
            
        self.portfolio_value = float(account.get('equity', 0))
        current_positions = self.handler.get_positions() # {ticker: qty}
        
        logger.info(f"Live NAV: ${self.portfolio_value:,.2f} | Positions: {current_positions}")
        
        # 2. Fetch Latest Market Data for Strategy
        # We need historical panel + current prices
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=252)).strftime("%Y-%m-%d")
        
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
        self.portfolio.update_market_value(current_prices, datetime.now())
        self.portfolio.ledger.position_book.positions = current_positions
        
        # 4. Run Strategy
        # Pass ts as now for real-time
        ts = pd.Timestamp(datetime.now())
        
        # 4. Agentic Consensus & Execution
        logger.info(f"--- Consulting AI Investment Committee for {len(tickers)} assets ---")
        
        from backtest.execution import Order
        agent_orders = []
        
        for tk in tickers:
            # Consult the AI committee
            agent_data = {
                "prices": full_panel[tk] if tk in full_panel.columns.levels[0] else None,
                "current_price": current_prices.get(tk)
            }
            decision = self.head_trader.get_consensus_signal(tk, agent_data)
            
            if abs(decision['signal']) > 0.01: # Minimum 1% weight change to act
                # The agent signal is already a target weight (e.g. 0.10 for 10%)
                target_weight = decision['signal']
                target_value = self.portfolio_value * target_weight
                
                # Calculate current position value
                current_qty = current_positions.get(tk, 0)
                current_value = current_qty * current_prices[tk]
                
                # Order is the difference
                order_value = target_value - current_value
                qty = order_value / current_prices[tk]
                from backtest.execution import Order, OrderType
                
                logger.info(f"[DECISION] {tk}: Target Weight {target_weight:.1%} | Rebalance Qty: {qty:.4f}")
                agent_orders.append(Order(
                    ticker=tk, 
                    quantity=qty, 
                    order_type=OrderType.MARKET, 
                    timestamp=ts
                ))

        if not agent_orders:
            logger.info("Investment Committee suggests no portfolio changes today.")
            return
            
        logger.info(f"Executing {len(agent_orders)} agent-driven orders.")
        
        # 5. Execute Orders
        self.handler.submit_orders(agent_orders)
        logger.info("Live rebalance complete.")

    def _build_price_panel(self, tickers: List[str], start_date: str, end_date: Optional[str] = None):
        """Mock compatibility for strategy.train_models."""
        return self.provider.get_panel(tickers, start_date, end_date)
