"""
Portfolio Optimizer - Elite Quant Fund System
Full Black-Litterman implementation, Risk Parity, Min Variance, Max Diversification
Post-optimization volatility targeting
Built to Renaissance Technologies / Jane Street standards
"""

import logging
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Callable, Any, Literal
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
from numpy.linalg import inv, LinAlgError
from scipy.optimize import minimize, minimize_scalar

from elite_quant_fund.core.types import (
    Portfolio, TargetAllocation, AlphaSignal, RiskLimits,
    MarketBar, calc_cvar, Result
)
from elite_quant_fund.risk.engine import LedoitWolfCovariance

logger = logging.getLogger(__name__)


# ============================================================================
# BLACK-LITTERMAN MODEL
# ============================================================================

class BlackLittermanModel:
    """
    Black-Litterman portfolio optimization
    Combines market equilibrium (CAPM) with investor views
    
    Key equation:
    E[R] = [(tau*Sigma)^-1 + P' * Omega^-1 * P]^-1 * 
           [(tau*Sigma)^-1 * Pi + P' * Omega^-1 * Q]
    
    Where:
    - Pi: equilibrium excess returns (CAPM)
    - tau: uncertainty scalar
    - Sigma: covariance matrix
    - P: view matrix (which assets in each view)
    - Q: view vector (expected returns for each view)
    - Omega: uncertainty matrix of views
    """
    
    def __init__(
        self,
        tau: float = 0.025,  # Uncertainty parameter (typically 0.025-0.05)
        risk_aversion: float = 2.5  # Risk aversion coefficient
    ):
        self.tau = tau
        self.risk_aversion = risk_aversion
        
        # Market equilibrium returns (CAPM)
        self.market_weights: Optional[np.ndarray] = None
        self.equilibrium_returns: Optional[np.ndarray] = None
        
        # Views
        self.views: List[Dict[str, Any]] = []
        self.view_matrix: Optional[np.ndarray] = None
        self.view_returns: Optional[np.ndarray] = None
        self.view_uncertainty: Optional[np.ndarray] = None
    
    def set_market_equilibrium(
        self,
        symbols: List[str],
        market_caps: Dict[str, float],
        cov_matrix: np.ndarray
    ) -> None:
        """
        Set market equilibrium (CAPM) implied returns
        Pi = risk_aversion * Sigma * w_mkt
        """
        
        self.symbols = symbols
        n = len(symbols)
        
        # Market capitalization weights
        total_cap = sum(market_caps.get(s, 0) for s in symbols)
        self.market_weights = np.array([
            market_caps.get(s, 1.0) / total_cap if total_cap > 0 else 1.0 / n
            for s in symbols
        ])
        
        # Equilibrium excess returns (CAPM)
        self.equilibrium_returns = self.risk_aversion * cov_matrix @ self.market_weights
        
        self.covariance = cov_matrix
    
    def add_view(
        self,
        assets: List[str],
        expected_return: float,
        confidence: float = 0.5,
        view_type: Literal['absolute', 'relative'] = 'absolute'
    ) -> None:
        """
        Add investor view to the model
        
        Args:
            assets: List of assets in view (2 for relative, 1 for absolute)
            expected_return: Expected return (spread for relative)
            confidence: View confidence (0-1, 1 = certain)
            view_type: 'absolute' or 'relative'
        """
        
        self.views.append({
            'assets': assets,
            'expected_return': expected_return,
            'confidence': confidence,
            'type': view_type
        })
    
    def add_alpha_signals(
        self,
        signals: Dict[str, AlphaSignal],
        confidence_base: float = 0.3
    ) -> None:
        """
        Convert alpha signals to BL views
        """
        
        for symbol, signal in signals.items():
            # Signal strength maps to confidence
            confidence = confidence_base * signal.confidence
            
            # Expected return proportional to signal strength
            # Assume 10% annual return for full strength signal
            expected_return = signal.strength * 0.10
            
            self.add_view(
                assets=[symbol],
                expected_return=expected_return,
                confidence=confidence,
                view_type='absolute'
            )
    
    def _build_view_matrices(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build P (view matrix), Q (returns), Omega (uncertainty)"""
        
        n = len(self.symbols)
        k = len(self.views)
        
        P = np.zeros((k, n))
        Q = np.zeros(k)
        Omega = np.zeros((k, k))
        
        for i, view in enumerate(self.views):
            if view['type'] == 'absolute':
                # Absolute view
                asset = view['assets'][0]
                if asset in self.symbols:
                    idx = self.symbols.index(asset)
                    P[i, idx] = 1.0
                    Q[i] = view['expected_return']
            
            elif view['type'] == 'relative':
                # Relative view (spread between two assets)
                if len(view['assets']) >= 2:
                    asset1, asset2 = view['assets'][0], view['assets'][1]
                    if asset1 in self.symbols and asset2 in self.symbols:
                        idx1 = self.symbols.index(asset1)
                        idx2 = self.symbols.index(asset2)
                        P[i, idx1] = 1.0
                        P[i, idx2] = -1.0
                        Q[i] = view['expected_return']
            
            # View uncertainty (lower confidence = higher uncertainty)
            # Omega_ii = P_i @ (tau * Sigma) @ P_i' * (1 - confidence) / confidence
            view_variance = P[i] @ (self.tau * self.covariance) @ P[i].T
            uncertainty = view_variance * (1 - view['confidence']) / (view['confidence'] + 1e-10)
            Omega[i, i] = uncertainty
        
        return P, Q, Omega
    
    def optimize(self) -> Optional[TargetAllocation]:
        """
        Compute Black-Litterman expected returns and optimal weights
        """
        
        if self.equilibrium_returns is None or self.covariance is None:
            logger.error("Market equilibrium not set")
            return None
        
        # Build view matrices
        if len(self.views) > 0:
            P, Q, Omega = self._build_view_matrices()
            
            # BL expected returns
            # E[R] = [(tau*Sigma)^-1 + P' * Omega^-1 * P]^-1 * 
            #        [(tau*Sigma)^-1 * Pi + P' * Omega^-1 * Q]
            
            tau_sigma_inv = inv(self.tau * self.covariance)
            
            try:
                omega_inv = inv(Omega)
            except LinAlgError:
                # Omega is diagonal, use element-wise inverse
                omega_inv = np.diag(1.0 / np.diag(Omega))
            
            M_left = tau_sigma_inv + P.T @ omega_inv @ P
            M = inv(M_left)
            
            M_right = tau_sigma_inv @ self.equilibrium_returns + P.T @ omega_inv @ Q
            
            posterior_returns = M @ M_right
            posterior_cov = self.covariance + M
            
        else:
            # No views, use equilibrium
            posterior_returns = self.equilibrium_returns
            posterior_cov = self.covariance
        
        # Mean-variance optimization with BL returns
        optimal_weights = self._mean_variance_opt(
            posterior_returns,
            posterior_cov,
            self.risk_aversion
        )
        
        # Calculate metrics
        expected_return = optimal_weights @ posterior_returns
        expected_vol = np.sqrt(optimal_weights @ posterior_cov @ optimal_weights)
        sharpe = expected_return / expected_vol if expected_vol > 0 else 0
        
        weights_dict = {
            sym: float(w) for sym, w in zip(self.symbols, optimal_weights)
        }
        
        return TargetAllocation(
            timestamp=datetime.now(),
            weights=weights_dict,
            expected_return=float(expected_return),
            expected_volatility=float(expected_vol),
            sharpe_ratio=float(sharpe),
            method='black_litterman'
        )
    
    def _mean_variance_opt(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        risk_aversion: float
    ) -> np.ndarray:
        """Closed-form mean-variance optimization"""
        
        n = len(expected_returns)
        
        # Unconstrained optimal: w = (1/lambda) * Sigma^-1 * mu
        try:
            cov_inv = inv(cov_matrix)
        except LinAlgError:
            # Add small regularization
            cov_inv = inv(cov_matrix + np.eye(n) * 1e-6)
        
        w_unconstrained = (1.0 / risk_aversion) * cov_inv @ expected_returns
        
        # Apply constraints (long-only, sum to 1)
        def neg_utility(w):
            ret = w @ expected_returns
            vol = np.sqrt(w @ cov_matrix @ w)
            return -(ret - 0.5 * risk_aversion * vol**2)
        
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        ]
        
        bounds = [(0, 1) for _ in range(n)]
        
        result = minimize(
            neg_utility,
            w_unconstrained,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        if result.success:
            return result.x
        else:
            logger.warning(f"Optimization failed: {result.message}")
            # Fall back to market weights
            return self.market_weights if self.market_weights is not None else np.ones(n) / n


# ============================================================================
# RISK PARITY
# ============================================================================

class RiskParityOptimizer:
    """
    Risk Parity optimization - equal risk contribution from all assets
    More stable than mean-variance, diversifies across risk sources
    """
    
    def __init__(self, target_volatility: float = 0.10):
        self.target_volatility = target_volatility
    
    def optimize(
        self,
        symbols: List[str],
        cov_matrix: np.ndarray
    ) -> Optional[TargetAllocation]:
        """
        Compute risk parity weights
        """
        
        n = len(symbols)
        
        # Initial guess: inverse volatility weights
        vols = np.sqrt(np.diag(cov_matrix))
        inv_vols = 1.0 / (vols + 1e-10)
        w0 = inv_vols / np.sum(inv_vols)
        
        # Risk budgeting objective
        def risk_budget_objective(w):
            # Portfolio volatility
            port_vol = np.sqrt(w @ cov_matrix @ w)
            
            # Marginal risk contributions
            mrc = (cov_matrix @ w) / port_vol
            
            # Risk contributions (should be equal for risk parity)
            rc = w * mrc
            
            # Deviation from equal risk contribution
            target_rc = port_vol / n
            deviation = np.sum((rc - target_rc) ** 2)
            
            return deviation
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        ]
        
        bounds = [(0.001, 1) for _ in range(n)]  # Min weight 0.1%
        
        result = minimize(
            risk_budget_objective,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        if not result.success:
            logger.warning(f"Risk parity optimization failed: {result.message}")
            return None
        
        weights = result.x
        
        # Calculate metrics
        port_vol = np.sqrt(weights @ cov_matrix @ weights)
        
        # Scale to target volatility
        if port_vol > 0:
            scale = self.target_volatility / port_vol
            weights = weights * scale
        
        expected_return = 0.03  # Placeholder, risk parity doesn't optimize return
        sharpe = expected_return / port_vol if port_vol > 0 else 0
        
        weights_dict = {sym: float(w) for sym, w in zip(symbols, weights)}
        
        return TargetAllocation(
            timestamp=datetime.now(),
            weights=weights_dict,
            expected_return=float(expected_return),
            expected_volatility=float(port_vol),
            sharpe_ratio=float(sharpe),
            method='risk_parity'
        )


# ============================================================================
# MINIMUM VARIANCE
# ============================================================================

class MinimumVarianceOptimizer:
    """
    Minimum variance portfolio - lowest possible volatility
    Conservative approach, ignores expected returns
    """
    
    def optimize(
        self,
        symbols: List[str],
        cov_matrix: np.ndarray
    ) -> Optional[TargetAllocation]:
        """
        Compute minimum variance weights
        """
        
        n = len(symbols)
        
        # Objective: minimize portfolio variance
        def portfolio_variance(w):
            return w @ cov_matrix @ w
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        ]
        
        bounds = [(0, 1) for _ in range(n)]
        
        # Initial guess
        w0 = np.ones(n) / n
        
        result = minimize(
            portfolio_variance,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        if not result.success:
            logger.warning(f"Min variance optimization failed: {result.message}")
            return None
        
        weights = result.x
        
        # Calculate metrics
        port_vol = np.sqrt(weights @ cov_matrix @ weights)
        expected_return = 0.02  # Conservative estimate
        sharpe = expected_return / port_vol if port_vol > 0 else 0
        
        weights_dict = {sym: float(w) for sym, w in zip(symbols, weights)}
        
        return TargetAllocation(
            timestamp=datetime.now(),
            weights=weights_dict,
            expected_return=float(expected_return),
            expected_volatility=float(port_vol),
            sharpe_ratio=float(sharpe),
            method='min_variance'
        )


# ============================================================================
# MAXIMUM DIVERSIFICATION
# ============================================================================

class MaximumDiversificationOptimizer:
    """
    Maximum diversification ratio portfolio
    Maximizes weighted average asset volatility / portfolio volatility
    """
    
    def optimize(
        self,
        symbols: List[str],
        cov_matrix: np.ndarray
    ) -> Optional[TargetAllocation]:
        """
        Compute maximum diversification weights
        """
        
        n = len(symbols)
        vols = np.sqrt(np.diag(cov_matrix))
        
        # Diversification ratio: (w' * vols) / sqrt(w' * Sigma * w)
        def neg_div_ratio(w):
            port_vol = np.sqrt(w @ cov_matrix @ w)
            weighted_vol = w @ vols
            
            if port_vol == 0:
                return 0
            
            return -weighted_vol / port_vol
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        ]
        
        bounds = [(0, 1) for _ in range(n)]
        
        # Initial guess: inverse volatility
        w0 = (1.0 / (vols + 1e-10))
        w0 = w0 / np.sum(w0)
        
        result = minimize(
            neg_div_ratio,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        if not result.success:
            logger.warning(f"Max diversification optimization failed: {result.message}")
            return None
        
        weights = result.x
        
        # Calculate metrics
        port_vol = np.sqrt(weights @ cov_matrix @ weights)
        expected_return = 0.025
        sharpe = expected_return / port_vol if port_vol > 0 else 0
        
        weights_dict = {sym: float(w) for sym, w in zip(symbols, weights)}
        
        return TargetAllocation(
            timestamp=datetime.now(),
            weights=weights_dict,
            expected_return=float(expected_return),
            expected_volatility=float(port_vol),
            sharpe_ratio=float(sharpe),
            method='max_diversification'
        )


# ============================================================================
# VOLATILITY TARGETING
# ============================================================================

class VolatilityTargeter:
    """
    Post-optimization volatility targeting
    Scales allocation to hit target volatility
    """
    
    def __init__(self, target_volatility: float = 0.10):
        self.target_volatility = target_volatility
    
    def target_allocation(
        self,
        allocation: TargetAllocation,
        cov_matrix: np.ndarray,
        symbols: List[str]
    ) -> TargetAllocation:
        """
        Scale allocation to target volatility
        """
        
        weights = np.array([allocation.weights.get(s, 0) for s in symbols])
        
        # Current portfolio volatility
        current_vol = np.sqrt(weights @ cov_matrix @ weights)
        
        if current_vol == 0:
            return allocation
        
        # Scale factor
        scale = self.target_volatility / current_vol
        
        # Scale weights (may need leverage)
        scaled_weights = weights * scale
        
        # Create new allocation
        weights_dict = {sym: float(w) for sym, w in zip(symbols, scaled_weights)}
        
        return TargetAllocation(
            timestamp=datetime.now(),
            weights=weights_dict,
            expected_return=allocation.expected_return * scale,
            expected_volatility=self.target_volatility,
            sharpe_ratio=allocation.sharpe_ratio,
            method=f"{allocation.method}_vol_targeted"
        )


# ============================================================================
# PORTFOLIO OPTIMIZER ORCHESTRATOR
# ============================================================================

class PortfolioOptimizer:
    """
    Main portfolio optimizer with multiple methods
    """
    
    def __init__(
        self,
        symbols: List[str],
        method: Literal['black_litterman', 'risk_parity', 'min_variance', 'max_div'] = 'black_litterman',
        target_volatility: float = 0.10
    ):
        self.symbols = symbols
        self.method = method
        self.target_volatility = target_volatility
        
        # Components
        self.bl_model = BlackLittermanModel()
        self.rp_optimizer = RiskParityOptimizer(target_volatility)
        self.mv_optimizer = MinimumVarianceOptimizer()
        self.md_optimizer = MaximumDiversificationOptimizer()
        self.vol_targeter = VolatilityTargeter(target_volatility)
        
        # Covariance estimator
        self.cov_estimator = LedoitWolfCovariance()
        
        # Market cap data (placeholder - would be real data)
        self.market_caps: Dict[str, float] = {sym: 1.0 for sym in symbols}
        
        # Return history for covariance
        self.return_history: Dict[str, deque] = {
            sym: deque(maxlen=252) for sym in symbols
        }
        
        # Latest allocation
        self.current_allocation: Optional[TargetAllocation] = None
    
    def update_returns(self, symbol: str, ret: float) -> None:
        """Add return to history"""
        if symbol in self.return_history:
            self.return_history[symbol].append(ret)
    
    def _estimate_covariance(self) -> Optional[np.ndarray]:
        """Estimate covariance matrix from returns"""
        
        # Build return matrix
        returns_data = {}
        for sym in self.symbols:
            if len(self.return_history[sym]) >= 20:
                returns_data[sym] = list(self.return_history[sym])[-60:]
        
        if len(returns_data) < len(self.symbols):
            logger.warning("Insufficient return data for covariance estimation")
            # Return diagonal matrix as fallback
            n = len(self.symbols)
            return np.diag([0.02] * n)  # 2% daily variance
        
        # Build DataFrame
        min_len = min(len(v) for v in returns_data.values())
        df_data = {
            sym: returns[-min_len:] for sym, returns in returns_data.items()
        }
        returns_df = pd.DataFrame(df_data)
        
        # Estimate using Ledoit-Wolf
        cov_matrix = self.cov_estimator.estimate(returns_df)
        
        return cov_matrix
    
    def optimize(
        self,
        signals: Optional[Dict[str, AlphaSignal]] = None
    ) -> Optional[TargetAllocation]:
        """
        Run portfolio optimization
        """
        
        # Estimate covariance
        cov_matrix = self._estimate_covariance()
        if cov_matrix is None:
            logger.error("Failed to estimate covariance matrix")
            return None
        
        allocation = None
        
        if self.method == 'black_litterman':
            # Set market equilibrium
            self.bl_model.set_market_equilibrium(
                self.symbols,
                self.market_caps,
                cov_matrix
            )
            
            # Add alpha signals as views
            if signals:
                self.bl_model.add_alpha_signals(signals)
            
            # Optimize
            allocation = self.bl_model.optimize()
        
        elif self.method == 'risk_parity':
            allocation = self.rp_optimizer.optimize(self.symbols, cov_matrix)
        
        elif self.method == 'min_variance':
            allocation = self.mv_optimizer.optimize(self.symbols, cov_matrix)
        
        elif self.method == 'max_div':
            allocation = self.md_optimizer.optimize(self.symbols, cov_matrix)
        
        # Apply volatility targeting
        if allocation and self.target_volatility > 0:
            allocation = self.vol_targeter.target_allocation(
                allocation, cov_matrix, self.symbols
            )
        
        self.current_allocation = allocation
        
        return allocation
    
    def rebalance_orders(
        self,
        current_portfolio: Portfolio,
        target_allocation: TargetAllocation
    ) -> Dict[str, float]:
        """
        Calculate rebalancing orders to reach target allocation
        Returns: {symbol: target_shares}
        """
        
        total_value = current_portfolio.total_value
        
        orders = {}
        
        for symbol, target_weight in target_allocation.weights.items():
            target_value = target_weight * total_value
            
            # Get current position
            current_position = current_portfolio.positions.get(symbol)
            
            if current_position:
                current_value = current_position.market_value
                price = current_position.current_price
            else:
                current_value = 0
                # Get price from history or use default
                price = 100.0
            
            # Calculate target shares
            target_shares = int(target_value / price)
            
            current_shares = current_position.quantity if current_position else 0
            
            # Order quantity
            order_qty = target_shares - current_shares
            
            if order_qty != 0:
                orders[symbol] = order_qty
        
        return orders
    
    def get_stats(self) -> Dict[str, Any]:
        """Get optimizer statistics"""
        return {
            'method': self.method,
            'target_volatility': self.target_volatility,
            'symbols': len(self.symbols),
            'current_allocation': self.current_allocation is not None
        }


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'BlackLittermanModel',
    'RiskParityOptimizer',
    'MinimumVarianceOptimizer',
    'MaximumDiversificationOptimizer',
    'VolatilityTargeter',
    'PortfolioOptimizer',
]
