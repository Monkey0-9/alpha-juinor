"""
Institutional Filters for Signal Validation.

Includes liquidity, volatility, correlation, exposure caps, and veto logic.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
import logging

logger = logging.getLogger(__name__)


class InstitutionalFilters:
    """
    Institutional-grade filters for signal validation and veto logic.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.liquidity_threshold = config.get('liquidity_threshold', 0.01)  # 1% of ADV
        self.volatility_cap = config.get('volatility_cap', 0.05)  # 5% daily vol
        self.correlation_limit = config.get('correlation_limit', 0.8)
        self.exposure_cap = config.get('exposure_cap', 0.2)  # 20% per asset
        self.max_leverage = config.get('max_leverage', 2.0)

        # Veto triggers
        self.veto_triggers = {
            'drawdown_threshold': config.get('drawdown_threshold', 0.05),  # 5%
            'vol_spike_threshold': config.get('vol_spike_threshold', 0.03),  # 3%
            'correlation_breakdown': config.get('correlation_breakdown', 0.9),
        }

        # State tracking
        self.portfolio_exposure = {}
        self.current_drawdown = 0.0
        self.baseline_volatility = 0.02  # Rolling baseline

    def apply_filters(self, signals: Dict[str, float], market_data: pd.DataFrame,
                     regime_context: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """
        Apply all institutional filters to signals.

        Args:
            signals: Raw signals {ticker: signal}
            market_data: Market data
            regime_context: Current regime info

        Returns:
            Tuple of (filtered_signals, filter_metadata)
        """
        filtered_signals = signals.copy()
        filter_metadata = {
            'liquidity_filter': {},
            'volatility_filter': {},
            'correlation_filter': {},
            'exposure_filter': {},
            'veto_status': 'pass'
        }

        # 1. Liquidity Filter
        filtered_signals, liquidity_meta = self._liquidity_filter(filtered_signals, market_data)
        filter_metadata['liquidity_filter'] = liquidity_meta

        # 2. Volatility Filter
        filtered_signals, vol_meta = self._volatility_filter(filtered_signals, market_data)
        filter_metadata['volatility_filter'] = vol_meta

        # 3. Correlation Filter
        filtered_signals, corr_meta = self._correlation_filter(filtered_signals, market_data)
        filter_metadata['correlation_filter'] = corr_meta

        # 4. Exposure Caps
        filtered_signals, exposure_meta = self._exposure_filter(filtered_signals)
        filter_metadata['exposure_filter'] = exposure_meta

        # 5. Veto Logic
        veto_result = self._check_veto_conditions(filtered_signals, market_data, regime_context)
        if veto_result['veto_triggered']:
            filtered_signals = {k: 0.5 for k in filtered_signals.keys()}  # Neutral signals
            filter_metadata['veto_status'] = veto_result['reason']

        return filtered_signals, filter_metadata

    def _liquidity_filter(self, signals: Dict[str, float], market_data: pd.DataFrame) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """Filter based on liquidity constraints."""
        filtered = signals.copy()
        metadata = {}

        for ticker, signal in signals.items():
            if abs(signal - 0.5) < 0.01:  # Neutral signal
                continue

            try:
                ticker_data = market_data[ticker] if isinstance(market_data.columns, pd.MultiIndex) else market_data
                if ticker not in ticker_data.columns:
                    continue

                recent_volume = ticker_data[ticker]['Volume'].iloc[-5:].mean()
                current_price = ticker_data[ticker]['Close'].iloc[-1]

                # Estimate trade size impact
                est_trade_value = 100000  # Assume $100k trade
                est_shares = est_trade_value / current_price
                participation_rate = est_shares / recent_volume if recent_volume > 0 else 1.0

                if participation_rate > self.liquidity_threshold:
                    # Reduce signal strength
                    reduction_factor = self.liquidity_threshold / participation_rate
                    filtered[ticker] = 0.5 + (signal - 0.5) * reduction_factor
                    metadata[ticker] = f"liquidity_reduced_{reduction_factor:.2f}"

            except Exception as e:
                logger.warning(f"Liquidity filter failed for {ticker}: {e}")
                metadata[ticker] = "liquidity_check_failed"

        return filtered, metadata

    def _volatility_filter(self, signals: Dict[str, float], market_data: pd.DataFrame) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """Filter based on volatility constraints."""
        filtered = signals.copy()
        metadata = {}

        for ticker, signal in signals.items():
            if abs(signal - 0.5) < 0.01:
                continue

            try:
                ticker_data = market_data[ticker] if isinstance(market_data.columns, pd.MultiIndex) else market_data
                if ticker not in ticker_data.columns:
                    continue

                returns = ticker_data[ticker]['Close'].pct_change().dropna()
                current_vol = returns.iloc[-20:].std()  # 20-day vol

                if current_vol > self.volatility_cap:
                    # Reduce signal in high vol
                    vol_ratio = self.volatility_cap / current_vol
                    filtered[ticker] = 0.5 + (signal - 0.5) * vol_ratio
                    metadata[ticker] = f"volatility_reduced_{vol_ratio:.2f}"

            except Exception as e:
                logger.warning(f"Volatility filter failed for {ticker}: {e}")
                metadata[ticker] = "volatility_check_failed"

        return filtered, metadata

    def _correlation_filter(self, signals: Dict[str, float], market_data: pd.DataFrame) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """Filter based on correlation constraints."""
        filtered = signals.copy()
        metadata = {}

        active_signals = {k: v for k, v in signals.items() if abs(v - 0.5) >= 0.01}
        if len(active_signals) < 2:
            return filtered, metadata

        try:
            # Calculate correlations
            returns_data = {}
            for ticker in active_signals.keys():
                ticker_data = market_data[ticker] if isinstance(market_data.columns, pd.MultiIndex) else market_data
                if ticker in ticker_data.columns:
                    returns = ticker_data[ticker]['Close'].pct_change().dropna()
                    returns_data[ticker] = returns

            if len(returns_data) >= 2:
                returns_df = pd.DataFrame(returns_data)
                corr_matrix = returns_df.corr()

                # Check for high correlations
                for i, ticker1 in enumerate(active_signals.keys()):
                    for j, ticker2 in enumerate(active_signals.keys()):
                        if i < j and ticker1 in corr_matrix.index and ticker2 in corr_matrix.columns:
                            corr = abs(corr_matrix.loc[ticker1, ticker2])
                            if corr > self.correlation_limit:
                                # Reduce correlated signals
                                reduction = (self.correlation_limit / corr) ** 2
                                filtered[ticker1] = 0.5 + (filtered[ticker1] - 0.5) * reduction
                                filtered[ticker2] = 0.5 + (filtered[ticker2] - 0.5) * reduction
                                metadata[f"{ticker1}_{ticker2}"] = f"correlation_reduced_{reduction:.2f}"

        except Exception as e:
            logger.warning(f"Correlation filter failed: {e}")
            metadata['error'] = "correlation_check_failed"

        return filtered, metadata

    def _exposure_filter(self, signals: Dict[str, float]) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """Apply exposure caps."""
        filtered = signals.copy()
        metadata = {}

        total_exposure = sum(abs(s - 0.5) for s in signals.values())

        if total_exposure > self.max_leverage:
            # Scale down all signals
            scale_factor = self.max_leverage / total_exposure
            filtered = {k: 0.5 + (v - 0.5) * scale_factor for k, v in signals.items()}
            metadata['global'] = f"exposure_scaled_{scale_factor:.2f}"

        # Individual exposure caps
        for ticker, signal in signals.items():
            exposure = abs(signal - 0.5)
            if exposure > self.exposure_cap:
                filtered[ticker] = 0.5 + np.sign(signal - 0.5) * self.exposure_cap
                metadata[ticker] = f"exposure_capped_{self.exposure_cap}"

        return filtered, metadata

    def _check_veto_conditions(self, signals: Dict[str, float], market_data: pd.DataFrame,
                             regime_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check veto conditions that would halt all trading.

        Returns:
            Dict with veto status and reason
        """
        result = {'veto_triggered': False, 'reason': None}

        try:
            # Drawdown check
            if hasattr(self, 'current_drawdown') and self.current_drawdown > self.veto_triggers['drawdown_threshold']:
                result['veto_triggered'] = True
                result['reason'] = f"drawdown_{self.current_drawdown:.3f}"

            # Volatility spike check
            if regime_context and regime_context.get('vol_regime') == 'high_vol':
                current_vol = self._calculate_market_volatility(market_data)
                if current_vol > self.baseline_volatility * (1 + self.veto_triggers['vol_spike_threshold']):
                    result['veto_triggered'] = True
                    result['reason'] = f"vol_spike_{current_vol:.3f}"

            # Correlation breakdown (market correlation too high)
            if len(signals) > 1:
                market_corr = self._calculate_market_correlation(market_data)
                if market_corr > self.veto_triggers['correlation_breakdown']:
                    result['veto_triggered'] = True
                    result['reason'] = f"correlation_breakdown_{market_corr:.3f}"

        except Exception as e:
            logger.error(f"Veto check failed: {e}")
            # Don't veto on error, but log it

        return result

    def _calculate_market_volatility(self, market_data: pd.DataFrame) -> float:
        """Calculate overall market volatility."""
        try:
            all_returns = []
            for col in market_data.columns.levels[0] if isinstance(market_data.columns, pd.MultiIndex) else market_data.columns:
                if 'Close' in market_data[col].columns:
                    returns = market_data[col]['Close'].pct_change().dropna()
                    all_returns.extend(returns.values[-20:])  # Last 20 days

            return np.std(all_returns) if all_returns else 0.02
        except:
            return 0.02

    def _calculate_market_correlation(self, market_data: pd.DataFrame) -> float:
        """Calculate average market correlation."""
        try:
            returns_data = {}
            for col in market_data.columns.levels[0] if isinstance(market_data.columns, pd.MultiIndex) else market_data.columns:
                if 'Close' in market_data[col].columns:
                    returns = market_data[col]['Close'].pct_change().dropna()
                    returns_data[col] = returns

            if len(returns_data) > 1:
                returns_df = pd.DataFrame(returns_data)
                corr_matrix = returns_df.corr()
                # Average absolute correlation
                return np.mean(np.abs(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]))

            return 0.0
        except:
            return 0.0

    def update_portfolio_state(self, current_pnl: float, peak_pnl: float):
        """Update portfolio state for veto checks."""
        self.current_drawdown = (peak_pnl - current_pnl) / peak_pnl if peak_pnl > 0 else 0.0
