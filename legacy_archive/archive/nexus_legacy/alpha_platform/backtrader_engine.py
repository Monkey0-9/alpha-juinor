
import backtrader as bt
import pandas as pd
from typing import Dict, Any, List
import logging
from mini_quant_fund.data.collectors.data_router import DataRouter

logger = logging.getLogger(__name__)

class AlphaStrategy(bt.Strategy):
    """Generic strategy that uses provided alpha weights."""
    params = (
        ('weights', None),
    )

    def __init__(self):
        self.inds = {data: data.close for data in self.datas}

    def next(self):
        dt = self.datas[0].datetime.date(0)
        if self.p.weights is not None and dt in self.p.weights.index:
            target_weights = self.p.weights.loc[dt]
            for i, data in enumerate(self.datas):
                symbol = data._name
                if symbol in target_weights:
                    self.order_target_percent(data, target_weights[symbol])

class BacktraderEngine:
    """Backtrader-based backtesting engine for high-fidelity simulation."""
    
    def __init__(self, initial_cash: float = 1000000.0, commission: float = 0.0001):
        self.initial_cash = initial_cash
        self.commission = commission
        
    def run_backtest(self, 
                    symbols: List[str], 
                    start_date: str, 
                    end_date: str, 
                    alpha_weights: pd.DataFrame) -> Dict[str, Any]:
        """
        Run backtest using Backtrader.
        alpha_weights: DataFrame with date index and symbol columns.
        """
        cerebro = bt.Cerebro()
        cerebro.broker.setcash(self.initial_cash)
        cerebro.broker.setcommission(commission=self.commission)
        
        router = DataRouter()
        
        for symbol in symbols:
            df = router.get_price_history(symbol, start_date, end_date, allow_long_history=True)
            if df is not None and not df.empty:
                # Map columns to Backtrader format
                df = df.rename(columns={
                    'Open': 'open', 'High': 'high', 'Low': 'low', 
                    'Close': 'close', 'Volume': 'volume'
                })
                data = bt.feeds.PandasData(dataname=df, name=symbol)
                cerebro.adddata(data)
            else:
                logger.warning(f"No data found for {symbol}")
                
        cerebro.addstrategy(AlphaStrategy, weights=alpha_weights)
        
        # Add analyzers
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        
        logger.info("Starting Backtrader execution...")
        results = cerebro.run()
        strat = results[0]
        
        # Extract metrics
        sharpe = strat.analyzers.sharpe.get_analysis()
        drawdown = strat.analyzers.drawdown.get_analysis()
        returns = strat.analyzers.returns.get_analysis()
        trades = strat.analyzers.trades.get_analysis()
        
        final_value = cerebro.broker.getvalue()
        total_return = (final_value / self.initial_cash) - 1
        
        return {
            "final_value": final_value,
            "total_return": total_return,
            "sharpe_ratio": sharpe.get('sharperatio', 0),
            "max_drawdown": drawdown.get('max', {}).get('drawdown', 0),
            "total_trades": trades.get('total', {}).get('total', 0),
            "win_rate": trades.get('won', {}).get('total', 0) / trades.get('total', {}).get('total', 1)
        }
