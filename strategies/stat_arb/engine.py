"""
Statistical Arbitrage Engine
============================

Manages the lifecycle of Pairs Trading:
1. Discovery: Finding cointegrated pairs.
2. Modeling: Fitting OU processes to spreads.
3. Signaling: Generating signals based on mean reversion.
"""

import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from strategies.stat_arb.math_utils import check_cointegration, fit_ou_process, calculate_ou_score

logger = logging.getLogger(__name__)

class StatArbEngine:
    def __init__(self, update_frequency_days: int = 30):
        self.pairs: List[Dict] = []
        self.last_scan_ts = None
        self.update_frequency_days = update_frequency_days

        # Hardcoded candidates for MVP (Sector Peers)
        # In full version, we'd scan the whole universe.
        self.candidates = [
            ('AAPL', 'MSFT'),
            ('GOOGL', 'META'),
            ('AMD', 'NVDA'),
            ('MA', 'V'),
            ('PEP', 'KO'),
            ('CVX', 'XOM'),
            ('JPM', 'BAC')
        ]

    def update_pairs(self, market_data: pd.DataFrame):
        """
        Scan candidates for cointegration and update valid pairs list.
        Should be run periodically (e.g., monthly).
        """
        valid_pairs = []

        # Ensure we have data
        if market_data.empty:
            return

        # Extract closes
        closes = self._extract_closes(market_data)
        if closes.empty:
            return

        logger.info(f"StatArb: Scanning {len(self.candidates)} candidate pairs...")

        for s1, s2 in self.candidates:
            if s1 not in closes.columns or s2 not in closes.columns:
                continue

            series_a = closes[s1].dropna()
            series_b = closes[s2].dropna()

            # Align indices
            common_idx = series_a.index.intersection(series_b.index)
            if len(common_idx) < 100: # Need enough history
                continue

            series_a = series_a.loc[common_idx]
            series_b = series_b.loc[common_idx]

            is_coint, p_val, hedge_ratio = check_cointegration(series_a, series_b)

            if is_coint:
                # Fit OU params immediately to verify mean reversion speed
                spread = np.log(series_a) - hedge_ratio * np.log(series_b)
                ou_params = fit_ou_process(spread)

                # Filter: Half-life must be reasonable (e.g., < 20 days) for "Quick" trading
                if 1.0 < ou_params['half_life'] < 60.0:
                    valid_pairs.append({
                        'leg1': s1,
                        'leg2': s2,
                        'hedge_ratio': hedge_ratio,
                        'ou_params': ou_params,
                        'last_z': 0.0
                    })
                    logger.info(f"StatArb: Found Pair {s1}-{s2} | HR={hedge_ratio:.2f} | HL={ou_params['half_life']:.1f}d")

        self.pairs = valid_pairs
        self.last_scan_ts = pd.Timestamp.utcnow()
        logger.info(f"StatArb: Updated pair universe. Count={len(self.pairs)}")

    def generate_signals(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals for active pairs.
        Returns DataFrame with 'signal', 'leg1', 'leg2', 'hedge_ratio'.
        """
        if not self.pairs:
            # Try to init if data available
            self.update_pairs(market_data)

        if not self.pairs:
            return pd.DataFrame()

        closes = self._extract_closes(market_data)
        signals = []

        for pair in self.pairs:
            s1, s2 = pair['leg1'], pair['leg2']
            if s1 not in closes.columns or s2 not in closes.columns:
                continue

            # Get latest prices
            p1 = closes[s1].iloc[-1]
            p2 = closes[s2].iloc[-1]
            hr = pair['hedge_ratio']

            if pd.isna(p1) or pd.isna(p2):
                continue

            # Calculate current spread
            try:
                spread_val = np.log(p1) - hr * np.log(p2)

                # Check recalibration (e.g. constant in regression might be missing in logic above,
                # but OU Z-score handles mean centering via 'mu')

                z_score = calculate_ou_score(spread_val, pair['ou_params'])
                pair['last_z'] = z_score

                # Signal Logic
                # Z > 2.0 -> Spread Rich -> Short Spread (Short Leg1, Long Leg2)
                # Z < -2.0 -> Spread Cheap -> Long Spread (Long Leg1, Short Leg2)
                # Exit criteria could be handled here or by risk manager.
                # For alpha signal, we output magnitude.

                signal_val = 0.0
                if z_score > 2.0:
                    signal_val = -1.0 # Short Spread
                elif z_score < -2.0:
                    signal_val = 1.0 # Long Spread
                elif abs(z_score) < 0.5:
                    signal_val = 0.0 # Exit / Flat

                # Damping for existing positions vs new?
                # For now, raw alpha signal.

                if signal_val != 0.0:
                    signals.append({
                        'type': 'pair',
                        'leg1': s1,
                        'leg2': s2,
                        'signal': signal_val,
                        'hedge_ratio': hr,
                        'z_score': z_score
                    })
            except Exception as e:
                logger.error(f"Error calc signal for {s1}-{s2}: {e}")

        return pd.DataFrame(signals)

    def _extract_closes(self, market_data):
        if isinstance(market_data.columns, pd.MultiIndex):
            closes_dict = {}
            tickers = market_data.columns.get_level_values(0).unique()
            for ticker in tickers:
                if 'Close' in market_data[ticker].columns:
                    closes_dict[ticker] = market_data[ticker]['Close']
            return pd.DataFrame(closes_dict)
        elif 'Close' in market_data.columns:
             return market_data[['Close']]
        else:
             return market_data
