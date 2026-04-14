import pandas as pd
import structlog
from typing import List, Dict
from mini_quant_fund.portfolio.allocator import Order

logger = structlog.get_logger()

class Simulator:
    """
    Walk-forward backtesting simulator with slippage and market impact.
    """
    def __init__(self, initial_nav: float = 1_000_000.0, slippage_bps: float = 5.0):
        self.nav = initial_nav
        self.slippage = slippage_bps / 10000.0
        self.positions = {} # symbol: quantity
        self.equity_curve = []

    def run_simulation(self, historical_data: Dict[str, pd.DataFrame], orders: List[Order]):
        """
        Processes orders against historical data.
        """
        for order in orders:
            symbol = order.symbol
            if symbol not in historical_data:
                continue

            # Simple simulation: assume execution at next day's open
            price = historical_data[symbol]["Open"].iloc[-1]
            execution_price = price * (1 + self.slippage if order.side == "BUY" else 1 - self.slippage)

            qty = order.size / execution_price
            if order.side == "BUY":
                self.positions[symbol] = self.positions.get(symbol, 0) + qty
                self.nav -= order.size
            elif order.side == "SELL":
                sell_qty = min(qty, self.positions.get(symbol, 0))
                self.positions[symbol] -= sell_qty
                self.nav += sell_qty * execution_price

        logger.info("Simulation step complete", nav=self.nav)
        self.equity_curve.append(self.nav)
