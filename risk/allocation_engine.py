"""
Allocation Engine
=================

This module implements institutional-grade capital allocation logic.
It focuses on Volatility Targeting and Strict Leverage Constraints.

Core Logic:
1. Estimate Asset Volatility.
2. Calculate Position Scalar: Target_Vol / Asset_Vol.
3. Apply Factor Weights.
4. Constrain Gross Leverage.
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class VolatilityTargetingAllocator:
    """
    Allocates capital to target a specific annualized volatility profile.
    """

    def __init__(self, target_vol: float = 0.20, max_leverage: float = 4.0):
        """
        Args:
            target_vol: Annualized volatility target (default 20%).
            max_leverage: Hard cap on gross leverage (sum of absolute weights).
        """
        self.target_vol = target_vol
        self.max_leverage = max_leverage

    def allocate(self,
                 factor_scores: pd.Series,
                 market_data: pd.DataFrame,
                 capital: float) -> pd.DataFrame:
        """
        Generate target positions (units) based on factor scores and risk.

        Args:
            factor_scores: Series of alpha scores (Ticker -> Score).
                          High positive score = Long. High negative = Short.
            market_data: DataFrame of recent price history (for vol estimation).
            capital: Current equity capital.

        Returns:
            DataFrame with columns ['weight', 'units', 'value', 'side']
        """
        if factor_scores.empty:
            logger.warning("Allocator received empty factor scores.")
            return pd.DataFrame()

        # 1. Estimate Asset Volatility (Annualized)
        closes = self._extract_closes(market_data)
        daily_rets = closes.pct_change(1).iloc[-30:] # Use last 30 days for responsive vol

        # Annualized Volatility
        asset_vol = daily_rets.std() * np.sqrt(252)
        asset_vol = asset_vol.replace(0, np.nan).fillna(1.0) # Defend against div/0

        # 2. Calculate Risk Scalars
        # How much leverage can we take on this asset to hit target vol?
        # e.g., if Asset Vol = 10% and Target = 20%, Scalar = 2.0x
        risk_scalars = self.target_vol / asset_vol

        # Align with factor scores
        common_tickers = factor_scores.index.intersection(risk_scalars.index)
        if len(common_tickers) == 0:
            logger.warning("No overlap between factor scores and market data assets.")
            return pd.DataFrame()

        active_scores = factor_scores.loc[common_tickers]
        active_scalers = risk_scalars.loc[common_tickers]

        # 3. Calculate Raw Target Weights
        # We assume factor scores are somewhat normalized (e.g. Z-scores ~ -2 to 2)
        # We treat the Score as a "Conviction" multiplier on the Risk Scalar.
        # Weight = Score * (Target / Vol)
        # Note: We scale down the Z-score logic slightly so a score of 1.0 isn't auto 100% allocation
        # Implementation Heuristic: Base weight 10% * Score

        conviction_multiplier = 0.10
        raw_weights = active_scores * active_scalers * conviction_multiplier

        # 4. Global Leverage Constraint
        # Calculate Gross Leverage
        gross_leverage = raw_weights.abs().sum()

        if gross_leverage > self.max_leverage:
            logger.info(f"Gross Leverage {gross_leverage:.2f}x exceeds limit {self.max_leverage}x. De-levering.")
            scale_down = self.max_leverage / gross_leverage
            final_weights = raw_weights * scale_down
        else:
            final_weights = raw_weights

        # 5. Convert to Units
        current_prices = closes.iloc[-1].loc[common_tickers]

        allocations = pd.DataFrame(index=common_tickers)
        allocations['weight'] = final_weights
        allocations['value'] = final_weights * capital

        # Units = Value / Price
        allocations['units'] = (allocations['value'] / current_prices).fillna(0).astype(int)
        allocations['side'] = np.where(allocations['units'] > 0, 'LONG',
                              np.where(allocations['units'] < 0, 'SHORT', 'FLAT'))

        return allocations

    def _extract_closes(self, market_data):
        """Helper to extract closes from various input formats."""
        if isinstance(market_data.columns, pd.MultiIndex):
            closes = pd.DataFrame()
            tickers = market_data.columns.get_level_values(0).unique()
            for ticker in tickers:
                if 'Close' in market_data[ticker].columns:
                    closes[ticker] = market_data[ticker]['Close']
            return closes
        elif 'Close' in market_data.columns:
             return market_data[['Close']]
        else:
             return market_data
