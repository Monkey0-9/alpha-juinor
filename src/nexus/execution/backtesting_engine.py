"""
Production Backtesting Engine using Backtrader
"""

import os
import sys
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

try:
    import backtrader as bt
    BACKTRADER_AVAILABLE = True
except ImportError:
    BACKTRADER_AVAILABLE = False
    logging.warning("Backtrader not available. Install with: pip install backtrader")

from .market_impact import MarketImpactModel, ImpactEstimate

logger = logging.getLogger(__name__)

class MiniQuantStrategy(bt.Strategy):
    """Base strategy class for MiniQuantFund algorithms"""
    
    params = (
        ('printlog', True),
        ('risk_aversion', 1e-5),
        ('max_position_size', 1000000),
        ('commission', 0.001),
    )
    
    def __init__(self):
        self.dataclose = self.datas[0].close
        self.order = None
        self.buyprice = None
        self.buycomm = None
        self.impact_model = MarketImpactModel()
        self.trades = []
        self.portfolio_value = []
        
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
            
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.2f}, '
                        f'Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.2f}, '
                        f'Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
                
            # Record trade for impact analysis
            self.trades.append({
                'datetime': bt.num2date(order.executed.dt),
                'type': 'BUY' if order.isbuy() else 'SELL',
                'price': order.executed.price,
                'size': order.executed.size,
                'commission': order.executed.comm,
                'value': order.executed.value
            })
            
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f'Order Canceled/Margin/Rejected: {order.status}')
            
        self.order = None
        
    def notify_trade(self, trade):
        if not trade.isclosed:
            return
            
        self.log(f'TRADE CLOSED - PnL: {trade.pnl:.2f}, PnL Net: {trade.pnlcomm:.2f}')
        
    def log(self, txt, dt=None):
        if self.params.printlog:
            dt = dt or self.datas[0].datetime.date(0)
            logging.info(f'{dt.isoformat()} - {txt}')
            
    def next(self):
        # Record portfolio value
        self.portfolio_value.append({
            'datetime': self.datas[0].datetime.date(0),
            'cash': self.broker.getcash(),
            'value': self.broker.getvalue(),
            'position': self.getposition().size
        })

class VWAPBacktestStrategy(MiniQuantStrategy):
    """VWAP strategy with market impact modeling"""
    
    params = (
        ('vwap_period', 20),
        ('entry_threshold', 0.002),
        ('exit_threshold', 0.001),
    )
    
    def __init__(self):
        super().__init__()
        
        # VWAP calculation
        self.vwap = bt.indicators.WeightedMovingAverage(
            self.data.close, period=self.params.vwap_period
        )
        
        # Market impact estimation
        self.impact_threshold = 0.001  # 10 bps
        
    def next(self):
        super().next()
        
        # Ensure we have enough data
        if len(self.data) < self.params.vwap_period:
            return
            
        current_price = self.dataclose[0]
        vwap_price = self.vwap[0]
        
        # Calculate deviation from VWAP
        deviation = (current_price - vwap_price) / vwap_price
        
        # Position sizing with market impact
        position_size = self.getposition().size
        available_cash = self.broker.getcash()
        
        # Market impact estimation
        if deviation > self.params.entry_threshold and position_size <= 0:
            # Price above VWAP - consider selling
            size = min(1000, int(available_cash * 0.1 / current_price))
            
            if size > 0:
                impact = self.impact_model.estimate_impact(
                    symbol=self.data._name,
                    side='sell',
                    quantity=size,
                    price=current_price
                )
                
                if impact.impact_bps < self.impact_threshold:
                    self.log(f'SELL SIGNAL - Deviation: {deviation:.4f}, Impact: {impact.impact_bps:.2f} bps')
                    self.order = self.sell(size=size)
                    
        elif deviation < -self.params.entry_threshold and position_size >= 0:
            # Price below VWAP - consider buying
            size = min(1000, int(available_cash * 0.1 / current_price))
            
            if size > 0:
                impact = self.impact_model.estimate_impact(
                    symbol=self.data._name,
                    side='buy',
                    quantity=size,
                    price=current_price
                )
                
                if impact.impact_bps < self.impact_threshold:
                    self.log(f'BUY SIGNAL - Deviation: {deviation:.4f}, Impact: {impact.impact_bps:.2f} bps')
                    self.order = self.buy(size=size)
                    
        # Exit conditions
        elif position_size > 0 and deviation > self.params.exit_threshold:
            self.log(f'EXIT LONG - Deviation: {deviation:.4f}')
            self.order = self.close()
            
        elif position_size < 0 and deviation < -self.params.exit_threshold:
            self.log(f'EXIT SHORT - Deviation: {deviation:.4f}')
            self.order = self.close()

class TWAPBacktestStrategy(MiniQuantStrategy):
    """TWAP strategy with time-based execution"""
    
    params = (
        ('execution_period', 60),  # minutes
        ('target_quantity', 10000),
        ('num_slices', 10),
    )
    
    def __init__(self):
        super().__init__()
        self.execution_start = None
        self.executed_quantity = 0
        self.current_slice = 0
        
    def next(self):
        super().next()
        
        current_time = self.datas[0].datetime.time(0)
        current_price = self.dataclose[0]
        
        # Start execution if not started
        if self.execution_start is None:
            self.execution_start = current_time
            self.log(f'Starting TWAP execution - Target: {self.params.target_quantity} shares')
            
        # Calculate elapsed minutes
        elapsed_minutes = (current_time.hour * 60 + current_time.minute) - \
                         (self.execution_start.hour * 60 + self.execution_start.minute)
        
        # Execute slices
        if elapsed_minutes > 0 and self.current_slice < self.params.num_slices:
            slice_size = self.params.target_quantity / self.params.num_slices
            remaining = self.params.target_quantity - self.executed_quantity
            
            if remaining > 0 and slice_size > 0:
                # Market impact estimation
                impact = self.impact_model.estimate_impact(
                    symbol=self.data._name,
                    side='buy',
                    quantity=slice_size,
                    price=current_price
                )
                
                self.log(f'EXECUTING SLICE {self.current_slice + 1}/{self.params.num_slices} - '
                        f'Size: {slice_size:.0f}, Impact: {impact.impact_bps:.2f} bps')
                
                self.order = self.buy(size=slice_size)
                self.executed_quantity += slice_size
                self.current_slice += 1

class BacktestingEngine:
    """Production backtesting engine"""
    
    def __init__(self, initial_cash: float = 1000000.0):
        self.initial_cash = initial_cash
        self.cerebro = bt.Cerebro()
        
    def add_data(self, data: pd.DataFrame, name: str = 'DATA'):
        """Add price data to backtesting engine"""
        
        if not BACKTRADER_AVAILABLE:
            raise ImportError("Backtrader not available. Install with: pip install backtrader")
            
        # Convert to Backtrader format
        data_feed = bt.feeds.PandasData(
            dataname=data,
            datetime=None,  # Use index as datetime
            open='open',
            high='high',
            low='low',
            close='close',
            volume='volume',
            openinterest=-1
        )
        
        self.cerebro.adddata(data_feed, name=name)
        
    def add_strategy(self, strategy_class: bt.Strategy, **kwargs):
        """Add strategy to backtesting engine"""
        self.cerebro.addstrategy(strategy_class, **kwargs)
        
    def add_analyzer(self, analyzer_class: bt.Analyzer, **kwargs):
        """Add analyzer to backtesting engine"""
        self.cerebro.addanalyzer(analyzer_class, **kwargs)
        
    def run_backtest(self, plot_results: bool = False) -> Dict[str, Any]:
        """Run backtest and return results"""
        
        # Set initial cash
        self.cerebro.broker.setcash(self.initial_cash)
        
        # Add commission
        self.cerebro.broker.setcommission(commission=0.001)
        
        # Add analyzers
        self.cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        self.cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        self.cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        self.cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        
        # Run backtest
        logging.info("Starting backtest...")
        results = self.cerebro.run()
        strategy = results[0]
        
        # Get analysis results
        sharpe_analysis = strategy.analyzers.sharpe.get_analysis()
        drawdown_analysis = strategy.analyzers.drawdown.get_analysis()
        returns_analysis = strategy.analyzers.returns.get_analysis()
        trades_analysis = strategy.analyzers.trades.get_analysis()
        
        # Calculate portfolio metrics
        final_value = self.cerebro.broker.getvalue()
        total_return = (final_value - self.initial_cash) / self.initial_cash
        
        # Compile results
        backtest_results = {
            'initial_cash': self.initial_cash,
            'final_value': final_value,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'sharpe_ratio': sharpe_analysis.get('sharperatio', 0),
            'max_drawdown': drawdown_analysis.get('max', {}).get('drawdown', 0),
            'max_drawdown_pct': drawdown_analysis.get('max', {}).get('drawdown', 0) * 100,
            'total_trades': trades_analysis.get('total', {}).get('closed', 0),
            'winning_trades': trades_analysis.get('won', {}).get('total', 0),
            'losing_trades': trades_analysis.get('lost', {}).get('total', 0),
            'win_rate': trades_analysis.get('won', {}).get('total', 0) / max(trades_analysis.get('total', {}).get('closed', 1), 1),
            'portfolio_history': strategy.portfolio_value,
            'trade_history': strategy.trades,
            'returns_analysis': returns_analysis
        }
        
        # Log results
        logging.info(f"Backtest completed - Return: {total_return:.2%}, "
                    f"Sharpe: {backtest_results['sharpe_ratio']:.3f}, "
                    f"Max DD: {backtest_results['max_drawdown_pct']:.2f}%")
        
        # Plot results if requested
        if plot_results:
            self.cerebro.plot(style='candlestick', barup='green', bardown='red')
            
        return backtest_results
        
    def optimize_strategy(self, strategy_class: bt.Strategy, 
                         parameter_ranges: Dict[str, List], 
                         objective: str = 'sharpe') -> Dict[str, Any]:
        """Optimize strategy parameters"""
        
        if not BACKTRADER_AVAILABLE:
            raise ImportError("Backtrader not available. Install with: pip install backtrader")
            
        # Add strategy with parameter ranges
        self.cerebro.optstrategy(
            strategy_class,
            **{k: v for k, v in parameter_ranges.items()}
        )
        
        # Run optimization
        logging.info("Starting strategy optimization...")
        results = self.cerebro.run()
        
        # Find best parameters
        best_result = None
        best_score = -float('inf')
        
        for result in results:
            if hasattr(result[0], 'analyzers') and result[0].analyzers:
                sharpe = result[0].analyzers.sharpe.get_analysis().get('sharperatio', 0)
                
                if sharpe > best_score:
                    best_score = sharpe
                    best_result = result
                    
        if best_result:
            best_params = {k: getattr(best_result[0].params, k) for k in parameter_ranges.keys()}
            
            logging.info(f"Optimization completed - Best Sharpe: {best_score:.3f}, "
                        f"Best params: {best_params}")
            
            return {
                'best_score': best_score,
                'best_parameters': best_params,
                'all_results': results
            }
        else:
            return {'error': 'No valid results found'}

def create_sample_data(symbol: str = 'AAPL', days: int = 252) -> pd.DataFrame:
    """Create sample market data for backtesting"""
    
    # Generate realistic price data
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, days)
    
    # Create price series
    prices = [100]
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
    
    # Create DataFrame
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    data = pd.DataFrame({
        'open': [p * (1 + np.random.normal(0, 0.001)) for p in prices[:-1]],
        'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices[:-1]],
        'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices[:-1]],
        'close': prices[:-1],
        'volume': np.random.randint(1000000, 5000000, days)
    }, index=dates)
    
    return data

def run_vwap_backtest_demo():
    """Run VWAP backtesting demonstration"""
    
    if not BACKTRADER_AVAILABLE:
        logging.error("Backtrader not available. Install with: pip install backtrader")
        return None
        
    # Create backtesting engine
    engine = BacktestingEngine(initial_cash=1000000)
    
    # Add sample data
    data = create_sample_data('AAPL', 252)
    engine.add_data(data, 'AAPL')
    
    # Add VWAP strategy
    engine.add_strategy(VWAPBacktestStrategy, 
                       vwap_period=20,
                       entry_threshold=0.002,
                       exit_threshold=0.001)
    
    # Run backtest
    results = engine.run_backtest(plot_results=False)
    
    return results

def run_twap_backtest_demo():
    """Run TWAP backtesting demonstration"""
    
    if not BACKTRADER_AVAILABLE:
        logging.error("Backtrader not available. Install with: pip install backtrader")
        return None
        
    # Create backtesting engine
    engine = BacktestingEngine(initial_cash=1000000)
    
    # Add sample data
    data = create_sample_data('AAPL', 252)
    engine.add_data(data, 'AAPL')
    
    # Add TWAP strategy
    engine.add_strategy(TWAPBacktestStrategy,
                       execution_period=60,
                       target_quantity=10000,
                       num_slices=10)
    
    # Run backtest
    results = engine.run_backtest(plot_results=False)
    
    return results

if __name__ == "__main__":
    # Run demonstration backtests
    logging.basicConfig(level=logging.INFO)
    
    print("Running VWAP Backtest Demo...")
    vwap_results = run_vwap_backtest_demo()
    
    print("\nRunning TWAP Backtest Demo...")
    twap_results = run_twap_backtest_demo()
    
    print("\nBacktesting demonstrations completed!")
