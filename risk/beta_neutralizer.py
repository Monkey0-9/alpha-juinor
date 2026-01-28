import pandas as pd
import numpy as np
from typing import Dict, Any
import logging

# risk/beta_neutralizer.py

logger = logging.getLogger("BETA_NEUTRALIZER")

class BetaNeutralizer:
    """
    Maintains portfolio market-neutrality (Beta = 0.0).
    Uses SPY as the market proxy.
    """

    def __init__(self, market_proxy: str = "SPY", lookback: int = 63):
        self.market_proxy = market_proxy
        self.lookback = lookback

    def calculate_beta_neutralization(self, weights: Dict[str, float], prices: pd.DataFrame) -> Dict[str, float]:
        """
        Input: Strategy weights + Price history (including market proxy).
        Output: SPY weight required to bring portfolio beta to zero.
        """
        if prices.empty or self.market_proxy not in prices.columns:
            logger.warning("BetaNeutralizer: Insufficient price data or SPY missing.")
            return {}

        # 1. Calculate Returns
        returns = prices.pct_change().dropna()
        if len(returns) < self.lookback:
            return {}

        market_rets = returns[self.market_proxy]
        market_var = market_rets.var()
        if market_var == 0: return {}

        # 2. Calculate Beta for each symbol in the portfolio
        portfolio_beta = 0.0
        active_symbols = [s for s in weights.keys() if s in returns.columns]

        for symbol in active_symbols:
            # Beta = Cov(r_s, r_m) / Var(r_m)
            cov = returns[symbol].cov(market_rets)
            beta = cov / market_var

            portfolio_beta += weights[symbol] * beta

        # 3. Neutralization logic
        # Portfolio_Beta_Total = Portfolio_Beta + W_spy * Beta_spy
        # Since Beta_spy = 1.0, W_spy = -Portfolio_Beta

        hedge_weight = -portfolio_beta

        if abs(hedge_weight) > 0.001:
            logger.info(f"[HEDGE] Portfolio_Beta={portfolio_beta:.3f} Action=NEUTRALIZING ETF={self.market_proxy} Weight={hedge_weight:.2%}")
            return {self.market_proxy: hedge_weight}

        return {}
