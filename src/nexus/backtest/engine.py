from typing import List, Dict, Optional, Any
from datetime import datetime
from ..models.market import MarketBar
from ..models.trade import Order, Trade, OrderSide, OrderType, OrderStatus
from ..data.engine import DataEngine
from ..research.base import BaseAlpha
from .portfolio import PortfolioTracker
from .models import InstitutionalCostModel
from .stats import PerformanceStats
from ..core.context import engine_context

class BacktestEngine:
    """
    Institutional backtest engine that iterates through time, 
    evaluates strategies, and simulates realistic execution.
    """
    def __init__(self, data_engine: DataEngine, initial_cash: float = 100000.0):
        self.data_engine = data_engine
        self.portfolio = PortfolioTracker(initial_cash=initial_cash)
        self.cost_model = InstitutionalCostModel()
        self.logger = engine_context.get_logger("backtest_engine")

    async def run(
        self, 
        symbols: List[str], 
        strategy: BaseAlpha, 
        start: datetime, 
        end: datetime,
        interval: str = "1d"
    ) -> Dict[str, Any]:
        """
        Runs the backtest across a list of symbols and a strategy.
        """
        self.logger.info(f"Starting backtest for {strategy.get_name()} on {symbols}")
        
        # 1. Fetch data for all symbols
        all_data: Dict[str, List[MarketBar]] = {}
        for symbol in symbols:
            all_data[symbol] = await self.data_engine.get_data(symbol, start, end, interval=interval)

        # 2. Synchronize data into a chronological bar sequence
        # (Simplified: assumes all symbols have the same timestamps for daily bars)
        # In a real HFT system, this would be a priority queue of events.
        timestamps = sorted(list(set(bar.timestamp for symbol_bars in all_data.values() for bar in symbol_bars)))
        
        for ts in timestamps:
            # Current prices for M2M
            current_prices = {}
            
            for symbol in symbols:
                symbol_bars = all_data[symbol]
                # Find the bar for this timestamp
                current_bar = next((b for b in symbol_bars if b.timestamp == ts), None)
                if not current_bar:
                    continue
                
                current_prices[symbol] = current_bar.close
                
                # A. Evaluate Strategy
                # Strategy only sees data up to ts (inclusive of previous bars)
                history = [b for b in symbol_bars if b.timestamp <= ts]
                signal = strategy.generate_signal(history)
                
                if signal and signal.side:
                    # B. Simulate Order Execution
                    # In a real system, the signal would go to a Risk Engine first.
                    order = Order(
                        symbol=symbol,
                        side=signal.side,
                        order_type=OrderType.MARKET,
                        quantity=100, # Simplified: fixed size
                        timestamp=ts
                    )
                    
                    # C. Apply realistic execution gaps and costs
                    exec_price = self.cost_model.get_execution_price(order, current_bar.close, current_bar.volume)
                    total_cost = self.cost_model.calculate_cost(order, current_bar.close, current_bar.volume)
                    
                    trade = Trade(
                        order_id=order.order_id,
                        symbol=symbol,
                        side=order.side,
                        quantity=order.quantity,
                        price=exec_price,
                        commission=total_cost,
                        timestamp=ts
                    )
                    
                    # D. Update Portfolio
                    self.portfolio.update_with_trade(trade)
            
            # E. Mark-to-Market at end of bar
            self.portfolio.mark_to_market(ts, current_prices)

        # 3. Finalize Metrics
        metrics = PerformanceStats.calculate_metrics(self.portfolio.equity_history)
        self.logger.info(f"Backtest complete. Final Sharpe: {metrics.get('sharpe_ratio', 0):.2f}")
        
        return {
            "strategy": strategy.get_name(),
            "metrics": metrics,
            "equity_history": self.portfolio.equity_history,
            "trades": [t.model_dump() for t in self.portfolio.trades]
        }
