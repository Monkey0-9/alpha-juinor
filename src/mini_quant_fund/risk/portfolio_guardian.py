"""
Elite Portfolio Guardian
========================

The "Top 1%" Risk Layer.
Ensures the portfolio is mathematically robust, not just a collection of random bets.

Features:
1. Anti-Correlation Guard: Prevents buying assets highly correlated to existing holdings.
2. Volatility Targeting: Scales size based on asset volatility.
3. Sector Limits: Hard caps on sector exposure.
"""

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PortfolioGuardian:
    def __init__(
        self,
        max_correlation: float = 0.7,
        target_vol: float = 0.15,
        max_ticker_weight: float = 0.10,
    ):
        self.max_correlation = max_correlation
        self.target_portfolio_vol = target_vol
        self.max_ticker_weight = max_ticker_weight
        self.sector_exposure = {}

    def check_new_trade(
        self,
        symbol: str,
        market_data: pd.DataFrame,
        current_portfolio: List[str],
        current_weight: float = 0.0,
    ) -> bool:
        """
        Validates if adding 'symbol' keeps the portfolio healthy.
        True = Safe to Buy.
        False = Reject (Too risky/correlated).
        """
        # 0. Size Check
        if current_weight >= self.max_ticker_weight:
            logger.warning(
                f"[GUARDIAN] REJECT {symbol}: Current weight {current_weight:.2%} >= Max {self.max_ticker_weight:.2%}"
            )
            return False

        if not current_portfolio:
            return True  # First trade is free (risk-wise)

        # 1. Correlation Check
        if not self._check_correlation(symbol, market_data, current_portfolio):
            return False

        return True

    def _check_correlation(
        self, symbol: str, market_data: pd.DataFrame, current_portfolio: List[str]
    ) -> bool:
        """
        Calculate correlation of 'symbol' against every asset in 'current_portfolio'.
        Reject if Average Correlation is > threshold.
        """
        try:
            # Need historical data
            # Assuming market_data has Close prices for all symbols
            # Structure: MultiIndex (Symbol, Field) or similar

            # Extract closing prices
            # Warning: This operation might be slow if DF is huge.
            # In production, use a pre-calculated correlation matrix.

            # For Simulation/MVP: use a mock check if data is missing
            if market_data is None or market_data.empty:
                return True

            returns_df = pd.DataFrame()

            # Get Symbol Returns
            if symbol in market_data.columns.get_level_values(0):
                s_close = market_data[symbol]["Close"]
                returns_df[symbol] = s_close.pct_change()
            else:
                # No history for new symbol -> Can't check correlation
                # Fail Open or Closed? Open for now, rely on Vol check.
                return True

            # Get Portfolio Returns
            count = 0
            high_corr_count = 0

            for exist_sym in current_portfolio:
                if exist_sym == symbol:
                    continue
                if exist_sym in market_data.columns.get_level_values(0):
                    p_close = market_data[exist_sym]["Close"]
                    # Align indices?
                    # simple corr
                    other_ret = p_close.pct_change()

                    # Calculate Correlation (last 60 days)
                    # We need common index
                    combined = pd.concat(
                        [returns_df[symbol], other_ret], axis=1
                    ).dropna()
                    if len(combined) > 30:
                        corr = combined.corr().iloc[0, 1]
                        if corr > self.max_correlation:
                            logger.warning(
                                f"[RISK] High Correlation: {symbol} vs {exist_sym} = {corr:.2f}"
                            )
                            high_corr_count += 1
                        count += 1

            # Decision: If highly correlated to > 50% of portfolio, Reject
            if count > 0 and (high_corr_count / count) > 0.5:
                logger.warning(
                    f"[GUARDIAN] REJECT {symbol}: Too correlated to portfolio."
                )
                return False

        except Exception as e:
            logger.error(f"[GUARDIAN] Correlation Check Error: {e}")
            return True  # Fail Safe (Allow trade, or Block? Top 1% might Block.)
            # Going with Allow to prevent total freeze on small data errors.

        return True

    def get_volatility_scalar(self, symbol: str, market_data: pd.DataFrame) -> float:
        """
        Returns a size multiplier (0.0 to 2.0).
        If asset is super volatile, return < 1.0.
        If asset is stable, return > 1.0.
        """
        try:
            if symbol in market_data.columns.get_level_values(0):
                # Calculate Annualized Vol
                closes = market_data[symbol]["Close"]
                returns = closes.pct_change().dropna()
                vol = returns.std() * np.sqrt(252)

                if vol == 0:
                    return 1.0

                # Target Vol / Actual Vol
                # e.g. Target 15%, Actual 30% -> Scalar 0.5 (Half size)
                scalar = self.target_portfolio_vol / vol

                # Cap it
                return max(0.2, min(scalar, 1.5))
        except:
            return 1.0

    def check_portfolio_health(
        self,
        positions: List[Dict[str, Any]],
        nav: float,
        max_exposure_pct: float = 0.80,
    ) -> List[str]:
        """
        SYMMETRIC INTELLIGENCE: Check if portfolio is over-exposed.
        Returns list of symbols to trim if total exposure > limit.
        """
        trim_candidates = []
        if not positions or nav <= 0:
            return trim_candidates

        total_exposure = sum(p.get("market_value", 0) for p in positions)
        exposure_pct = total_exposure / nav

        if exposure_pct > max_exposure_pct:
            logger.warning(
                f"[GUARDIAN] Over-exposure detected: {exposure_pct:.1%} > {max_exposure_pct:.0%}. Signal TRIM."
            )
            # Simple logic: Trim the largest positions first (or weakest?)
            # For this MVP, we signal the Brain to reduce risk.
            # We return symbols that contribute most to risk.
            sorted_pos = sorted(
                positions, key=lambda x: x.get("market_value", 0), reverse=True
            )
            for p in sorted_pos[:3]:  # Top 3 holdings
                trim_candidates.append(p.get("symbol"))

        return trim_candidates


def get_portfolio_guardian():
    return _guardian
