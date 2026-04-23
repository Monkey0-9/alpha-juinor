"""
Alpha Engine - Elite Quant Fund System
Three stacked models:
  1. OU-calibrated stat arb (Vasicek OLS)
  2. Cross-sectional factor model (PCA residuals)
  3. ML ensemble (LightGBM + Ridge)
IC-weighted blending with Bayesian shrinkage
Built to Renaissance Technologies / Jane Street standards
"""

import logging
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
import warnings

from elite_quant_fund.core.types import (
    MarketBar, AlphaSignal, SignalBundle, SignalType, OUState, Result
)

# Optional LightGBM
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    warnings.warn("LightGBM not available, using fallback models")

logger = logging.getLogger(__name__)


# ============================================================================
# ORNSTEIN-UHLENBECK PROCESS - Statistical Arbitrage
# ============================================================================

class OrnsteinUhlenbeckModel:
    """
    OU process for mean-reversion modeling
    dX(t) = kappa * (theta - X(t)) * dt + sigma * dW(t)
    
    Calibrated via OLS on discrete version:
    X(t+1) = a + b * X(t) + epsilon
    where kappa = -ln(b) / dt, theta = a / (1 - b)
    """
    
    def __init__(self, symbol: str, window: int = 50, dt: float = 1.0):
        self.symbol = symbol
        self.window = window
        self.dt = dt
        
        self.prices: deque = deque(maxlen=window)
        self.log_prices: deque = deque(maxlen=window)
        self.state: Optional[OUState] = None
        
        # Calibration parameters
        self.kappa: float = 0.0  # Mean reversion speed
        self.theta: float = 0.0  # Long-term mean
        self.sigma: float = 0.0  # Volatility
        self.half_life: timedelta = timedelta(days=999)
        
        # Signal generation thresholds
        self.entry_zscore = 2.0
        self.exit_zscore = 0.5
    
    def update(self, bar: MarketBar) -> Optional[AlphaSignal]:
        """Update OU model and generate signal if appropriate"""
        
        # Store price
        self.prices.append(bar.close)
        log_price = np.log(bar.close)
        self.log_prices.append(log_price)
        
        # Need minimum data
        if len(self.prices) < 30:
            return None
        
        # Calibrate OU parameters via OLS
        self._calibrate()
        
        # Check if mean-reverting
        if self.kappa <= 0 or self.half_life.days > 30:
            return None  # Not mean-reverting, no signal
        
        # Calculate z-score
        current_log = log_price
        z_score = (current_log - self.theta) / self.sigma if self.sigma > 0 else 0
        
        # Generate signal
        if abs(z_score) > self.entry_zscore:
            # Strong mean-reversion signal
            strength = np.clip(-z_score / 3.0, -1, 1)  # Mean-reversion: negative z = buy
            
            signal = AlphaSignal(
                symbol=self.symbol,
                timestamp=bar.timestamp,
                signal_type=SignalType.MEAN_REVERSION,
                strength=strength,
                horizon=self.half_life,
                half_life=self.half_life,
                metadata={
                    'model': 'OU_process',
                    'kappa': self.kappa,
                    'theta': self.theta,
                    'z_score': z_score,
                    'half_life_days': self.half_life.days
                }
            )
            
            return signal
        
        return None
    
    def _calibrate(self) -> None:
        """Calibrate OU parameters using OLS on discrete version"""
        
        if len(self.log_prices) < 20:
            return
        
        # Discrete OU: X(t+1) = a + b * X(t) + epsilon
        X = np.array(list(self.log_prices)[:-1])  # X(t)
        Y = np.array(list(self.log_prices)[1:])   # X(t+1)
        
        # OLS regression
        X_with_const = np.column_stack([np.ones(len(X)), X])
        
        try:
            beta = np.linalg.lstsq(X_with_const, Y, rcond=None)[0]
            a, b = beta[0], beta[1]
            
            # Convert to OU parameters
            if b > 0 and b < 1:  # Stationary condition
                self.kappa = -np.log(b) / self.dt
                self.theta = a / (1 - b)
                
                # Calculate sigma from residuals
                residuals = Y - (a + b * X)
                self.sigma = np.std(residuals) * np.sqrt(2 * self.kappa / (1 - b**2))
                
                # Half-life
                if self.kappa > 0:
                    half_life_days = np.log(2) / self.kappa
                    self.half_life = timedelta(days=float(half_life_days))
                
                # Update state
                current_log = list(self.log_prices)[-1]
                z_score = (current_log - self.theta) / self.sigma if self.sigma > 0 else 0
                
                self.state = OUState(
                    symbol=self.symbol,
                    timestamp=datetime.now(),
                    mean=self.theta,
                    speed=self.kappa,
                    volatility=self.sigma,
                    half_life=self.half_life,
                    z_score=z_score
                )
        
        except Exception as e:
            logger.warning(f"OU calibration failed for {self.symbol}: {e}")
    
    def get_state(self) -> Optional[OUState]:
        """Get current OU state"""
        return self.state


# ============================================================================
# CROSS-SECTIONAL FACTOR MODEL
# ============================================================================

class CrossSectionalFactorModel:
    """
    Cross-sectional factor model with PCA residuals
    Factors: momentum, low-volatility, liquidity
    """
    
    def __init__(self, symbols: List[str], lookback: int = 20):
        self.symbols = symbols
        self.lookback = lookback
        
        # Data storage
        self.returns_data: Dict[str, deque] = {
            sym: deque(maxlen=lookback) for sym in symbols
        }
        self.price_data: Dict[str, deque] = {
            sym: deque(maxlen=lookback) for sym in symbols
        }
        self.volume_data: Dict[str, deque] = {
            sym: deque(maxlen=lookback) for sym in symbols
        }
        
        # Factor loadings
        self.factor_exposures: Dict[str, np.ndarray] = {}
        self.factor_returns: Optional[np.ndarray] = None
        
        # PCA model
        self.pca = PCA(n_components=3)
        self.scaler = StandardScaler()
        
        # IC tracking
        self.ic_history: deque = deque(maxlen=50)
        self.mean_ic = 0.0
    
    def update(self, bar: MarketBar) -> Optional[AlphaSignal]:
        """Update factor model and generate signal"""
        
        symbol = bar.symbol
        
        # Store data
        if len(self.price_data[symbol]) > 0:
            prev_price = list(self.price_data[symbol])[-1]
            ret = (bar.close - prev_price) / prev_price
            self.returns_data[symbol].append(ret)
        
        self.price_data[symbol].append(bar.close)
        self.volume_data[symbol].append(bar.volume)
        
        # Check if we have enough data for all symbols
        if not all(len(self.returns_data[sym]) >= 10 for sym in self.symbols):
            return None
        
        # Build factor matrix
        factor_matrix = self._build_factor_matrix()
        if factor_matrix is None:
            return None
        
        # Run PCA on factors
        try:
            factor_scores = self.pca.fit_transform(factor_matrix)
            
            # Store factor exposure for this symbol
            sym_idx = self.symbols.index(symbol)
            self.factor_exposures[symbol] = factor_scores[sym_idx]
            
            # Calculate residual (idiosyncratic alpha)
            # Reconstruct from factors
            reconstructed = self.pca.inverse_transform(factor_scores[sym_idx:sym_idx+1])
            residual = factor_matrix[sym_idx] - reconstructed[0]
            
            # Use residual as alpha signal (mean-reversion on residuals)
            residual_score = np.mean(residual) / (np.std(residual) + 1e-10)
            
            if abs(residual_score) > 1.0:
                strength = np.clip(-residual_score / 2.0, -1, 1)
                
                signal = AlphaSignal(
                    symbol=symbol,
                    timestamp=bar.timestamp,
                    signal_type=SignalType.FACTOR,
                    strength=strength,
                    horizon=timedelta(days=5),
                    metadata={
                        'model': 'cross_sectional_factor',
                        'residual_score': residual_score,
                        'explained_variance': float(np.sum(self.pca.explained_variance_ratio_)),
                        'factors': ['momentum', 'low_vol', 'liquidity']
                    }
                )
                
                return signal
        
        except Exception as e:
            logger.warning(f"Factor model failed for {symbol}: {e}")
        
        return None
    
    def _build_factor_matrix(self) -> Optional[np.ndarray]:
        """Build factor matrix (momentum, low-vol, liquidity)"""
        
        factors = []
        
        for sym in self.symbols:
            if len(self.returns_data[sym]) < 5:
                return None
            
            returns = np.array(list(self.returns_data[sym]))
            prices = np.array(list(self.price_data[sym]))
            volumes = np.array(list(self.volume_data[sym]))
            
            # Momentum factor (12-day cumulative return)
            momentum = np.prod(1 + returns[-12:]) - 1 if len(returns) >= 12 else np.sum(returns)
            
            # Low-volatility factor (inverse of realized vol)
            vol = np.std(returns) * np.sqrt(252)
            low_vol = 1.0 / (vol + 1e-10)
            
            # Liquidity factor (dollar volume)
            avg_volume = np.mean(volumes) if len(volumes) > 0 else 0
            avg_price = np.mean(prices) if len(prices) > 0 else 0
            liquidity = avg_volume * avg_price
            
            factors.append([momentum, low_vol, liquidity])
        
        # Normalize
        factor_matrix = np.array(factors)
        
        # Standardize
        for i in range(factor_matrix.shape[1]):
            col = factor_matrix[:, i]
            if np.std(col) > 0:
                factor_matrix[:, i] = (col - np.mean(col)) / np.std(col)
        
        return factor_matrix
    
    def get_factor_exposure(self, symbol: str) -> Optional[np.ndarray]:
        """Get factor exposure for symbol"""
        return self.factor_exposures.get(symbol)


# ============================================================================
# ML ENSEMBLE - LightGBM + Ridge
# ============================================================================

class MLEnsemble:
    """
    ML ensemble combining LightGBM and Ridge regression
    For non-linear pattern detection
    """
    
    def __init__(self, symbol: str, lookback: int = 50):
        self.symbol = symbol
        self.lookback = lookback
        
        # Data storage
        self.features: deque = deque(maxlen=lookback)
        self.targets: deque = deque(maxlen=lookback)
        self.prices: deque = deque(maxlen=lookback)
        
        # Models
        self.lgb_model: Optional[Any] = None
        self.ridge_model: Optional[Ridge] = None
        self.scaler = StandardScaler()
        
        # Training state
        self.is_trained = False
        self.last_train_time: Optional[datetime] = None
        self.train_interval = timedelta(hours=1)
        
        # Feature importance
        self.feature_importance: Optional[Dict[str, float]] = None
    
    def _extract_features(self, bar: MarketBar) -> Optional[np.ndarray]:
        """Extract technical features from bar"""
        
        if len(self.prices) < 10:
            return None
        
        prices = np.array(list(self.prices))
        
        # Price-based features
        returns = np.diff(prices) / prices[:-1]
        
        # Momentum
        mom_5 = (bar.close - prices[-5]) / prices[-5] if len(prices) >= 5 else 0
        mom_10 = (bar.close - prices[-10]) / prices[-10] if len(prices) >= 10 else 0
        
        # Volatility
        vol_10 = np.std(returns[-10:]) * np.sqrt(252) if len(returns) >= 10 else 0.2
        vol_20 = np.std(returns[-20:]) * np.sqrt(252) if len(returns) >= 20 else 0.2
        
        # Trend
        sma_10 = np.mean(prices[-10:]) if len(prices) >= 10 else bar.close
        sma_20 = np.mean(prices[-20:]) if len(prices) >= 20 else bar.close
        
        # Volume features
        avg_volume = np.mean([p.volume for p in list(self.features)[-10:]]) if len(self.features) >= 10 else bar.volume
        
        # Bar features
        body = abs(bar.close - bar.open) / bar.open
        range_pct = (bar.high - bar.low) / bar.close
        
        features = np.array([
            mom_5,
            mom_10,
            vol_10,
            vol_20,
            (bar.close - sma_10) / sma_10,
            (bar.close - sma_20) / sma_20,
            body,
            range_pct,
            bar.volume / (avg_volume + 1),
            len(self.prices) / self.lookback  # Data completeness
        ])
        
        return features
    
    def update(self, bar: MarketBar) -> Optional[AlphaSignal]:
        """Update ML models and generate prediction"""
        
        # Extract features
        features = self._extract_features(bar)
        if features is None:
            self.prices.append(bar.close)
            return None
        
        # Calculate target (forward return for training)
        if len(self.prices) > 0:
            prev_price = list(self.prices)[-1]
            forward_return = (bar.close - prev_price) / prev_price
            
            # Store features with delay (for next prediction)
            if len(self.features) > 0:
                self.targets.append(forward_return)
        
        self.features.append(features)
        self.prices.append(bar.close)
        
        # Train if needed
        now = datetime.now()
        if (not self.is_trained or 
            self.last_train_time is None or 
            now - self.last_train_time > self.train_interval):
            
            self._train()
        
        # Generate prediction
        if not self.is_trained or features is None:
            return None
        
        try:
            # Scale features
            if len(self.features) >= 20:
                feature_matrix = np.array(list(self.features))
                self.scaler.fit(feature_matrix)
                scaled_features = self.scaler.transform(features.reshape(1, -1))
            else:
                scaled_features = features.reshape(1, -1)
            
            # Ensemble prediction
            predictions = []
            
            if self.ridge_model is not None:
                ridge_pred = self.ridge_model.predict(scaled_features)[0]
                predictions.append(ridge_pred)
            
            if LIGHTGBM_AVAILABLE and self.lgb_model is not None:
                lgb_pred = self.lgb_model.predict(scaled_features)[0]
                predictions.append(lgb_pred)
            
            if len(predictions) == 0:
                return None
            
            # Average predictions
            ensemble_pred = np.mean(predictions)
            
            # Convert to signal strength
            pred_std = np.std(predictions) if len(predictions) > 1 else 0.02
            z_score = ensemble_pred / (pred_std + 1e-10)
            strength = np.clip(z_score / 2.0, -1, 1)
            
            if abs(strength) > 0.3:  # Minimum confidence threshold
                signal = AlphaSignal(
                    symbol=self.symbol,
                    timestamp=bar.timestamp,
                    signal_type=SignalType.MACHINE_LEARNING,
                    strength=strength,
                    horizon=timedelta(hours=4),
                    metadata={
                        'model': 'ml_ensemble',
                        'ridge_pred': predictions[0] if len(predictions) > 0 else 0,
                        'lgb_pred': predictions[1] if len(predictions) > 1 else 0,
                        'ensemble_std': pred_std,
                        'is_trained': self.is_trained
                    }
                )
                
                return signal
        
        except Exception as e:
            logger.warning(f"ML prediction failed for {self.symbol}: {e}")
        
        return None
    
    def _train(self) -> None:
        """Train models on accumulated data"""
        
        if len(self.features) < 30 or len(self.targets) < 30:
            return
        
        try:
            X = np.array(list(self.features)[:-1])  # Features (exclude last)
            y = np.array(list(self.targets))[:len(X)]  # Targets
            
            if len(X) != len(y) or len(X) < 20:
                return
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train Ridge regression
            self.ridge_model = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
            self.ridge_model.fit(X_scaled, y)
            
            # Train LightGBM if available
            if LIGHTGBM_AVAILABLE:
                train_data = lgb.Dataset(X_scaled, label=y)
                params = {
                    'objective': 'regression',
                    'metric': 'rmse',
                    'boosting_type': 'gbdt',
                    'num_leaves': 31,
                    'learning_rate': 0.05,
                    'feature_fraction': 0.9,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5,
                    'verbose': -1
                }
                
                self.lgb_model = lgb.train(
                    params,
                    train_data,
                    num_boost_round=100,
                    valid_sets=[train_data],
                    callbacks=[lgb.early_stopping(stopping_rounds=10), lgb.log_evaluation(0)]
                )
                
                # Store feature importance
                importance = self.lgb_model.feature_importance(importance_type='gain')
                self.feature_importance = {
                    f'feat_{i}': float(imp) for i, imp in enumerate(importance)
                }
            
            self.is_trained = True
            self.last_train_time = datetime.now()
            
            logger.info(f"ML models trained for {self.symbol} on {len(X)} samples")
        
        except Exception as e:
            logger.error(f"ML training failed for {self.symbol}: {e}")


# ============================================================================
# IC-WEIGHTED BLENDING
# ============================================================================

class ICWeightedBlender:
    """
    Information Coefficient weighted blending with Bayesian shrinkage
    Combines multiple alpha models with dynamic weighting
    """
    
    def __init__(self, symbols: List[str], ic_lookback: int = 20):
        self.symbols = symbols
        self.ic_lookback = ic_lookback
        
        # IC tracking per model
        self.model_ics: Dict[str, Dict[str, deque]] = {
            'ou': {sym: deque(maxlen=ic_lookback) for sym in symbols},
            'factor': {sym: deque(maxlen=ic_lookback) for sym in symbols},
            'ml': {sym: deque(maxlen=ic_lookback) for sym in symbols}
        }
        
        # Model weights (updated dynamically)
        self.model_weights: Dict[str, float] = {
            'ou': 0.33,
            'factor': 0.33,
            'ml': 0.34
        }
        
        # Bayesian shrinkage parameter
        self.shrinkage_tau = 0.1  # Shrink toward zero
        self.prior_weight = 0.2
        
        # Signal history for IC calculation
        self.signal_history: Dict[str, Dict[str, deque]] = {
            sym: {
                'ou': deque(maxlen=ic_lookback),
                'factor': deque(maxlen=ic_lookback),
                'ml': deque(maxlen=ic_lookback)
            }
            for sym in symbols
        }
        
        self.return_history: Dict[str, deque] = {
            sym: deque(maxlen=ic_lookback) for sym in symbols
        }
    
    def add_signal(self, signal: AlphaSignal, model_type: str) -> None:
        """Add signal to history for IC calculation"""
        
        symbol = signal.symbol
        
        if model_type in self.signal_history[symbol]:
            self.signal_history[symbol][model_type].append({
                'timestamp': signal.timestamp,
                'strength': signal.strength,
                'horizon': signal.horizon
            })
    
    def update_returns(self, symbol: str, returns: float) -> None:
        """Update realized returns for IC calculation"""
        
        self.return_history[symbol].append(returns)
        
        # Calculate ICs if we have enough data
        if len(self.return_history[symbol]) >= 10:
            self._update_ics(symbol)
            self._update_weights(symbol)
    
    def _update_ics(self, symbol: str) -> None:
        """Calculate information coefficients"""
        
        returns = np.array(list(self.return_history[symbol]))
        
        for model_type, signals in self.signal_history[symbol].items():
            if len(signals) < 10:
                continue
            
            # Get signal strengths aligned with returns
            signal_strengths = [s['strength'] for s in list(signals)[-len(returns):]]
            
            if len(signal_strengths) != len(returns):
                continue
            
            # Calculate IC (Spearman rank correlation)
            try:
                from scipy.stats import spearmanr
                ic, pvalue = spearmanr(signal_strengths, returns)
                
                if not np.isnan(ic):
                    self.model_ics[model_type][symbol].append(ic)
            except:
                pass
    
    def _update_weights(self, symbol: str) -> None:
        """Update model weights using Bayesian shrinkage"""
        
        # Calculate mean ICs
        mean_ics = {}
        for model_type, ics_dict in self.model_ics.items():
            ics = list(ics_dict[symbol])
            if len(ics) > 0:
                mean_ics[model_type] = np.mean(ics)
            else:
                mean_ics[model_type] = 0.0
        
        # Bayesian shrinkage: weight = IC^2 / sum(IC^2)
        # But shrink toward uniform weights
        ic_squared = {k: max(0, v**2) for k, v in mean_ics.items()}
        total_ic = sum(ic_squared.values())
        
        if total_ic > 0:
            # MLE weights
            mle_weights = {k: v / total_ic for k, v in ic_squared.items()}
            
            # Bayesian shrinkage toward uniform
            uniform_weight = 1.0 / len(mle_weights)
            self.model_weights = {
                k: self.prior_weight * uniform_weight + (1 - self.prior_weight) * w
                for k, w in mle_weights.items()
            }
        
        # Normalize
        total_weight = sum(self.model_weights.values())
        if total_weight > 0:
            self.model_weights = {k: v / total_weight for k, v in self.model_weights.items()}
    
    def blend_signals(
        self,
        symbol: str,
        signals: Dict[str, AlphaSignal]
    ) -> Optional[AlphaSignal]:
        """
        Blend multiple signals into consensus using IC-weighted approach
        """
        
        if not signals:
            return None
        
        # Calculate blended strength
        blended_strength = 0.0
        total_weight = 0.0
        
        metadata = {
            'weights': {},
            'contributions': {},
            'models_used': list(signals.keys())
        }
        
        for model_type, signal in signals.items():
            weight = self.model_weights.get(model_type, 0.33)
            
            # Bayesian shrinkage on signal strength
            shrunk_signal = self.shrinkage_tau * 0 + (1 - self.shrinkage_tau) * signal.strength
            
            blended_strength += weight * shrunk_signal
            total_weight += weight
            
            metadata['weights'][model_type] = weight
            metadata['contributions'][model_type] = weight * shrunk_signal
        
        if total_weight > 0:
            blended_strength /= total_weight
        
        # Clip to valid range
        blended_strength = np.clip(blended_strength, -1, 1)
        
        # Get average horizon
        avg_horizon = timedelta(hours=2)
        if signals:
            horizons = [s.horizon for s in signals.values()]
            avg_seconds = np.mean([h.total_seconds() for h in horizons])
            avg_horizon = timedelta(seconds=int(avg_seconds))
        
        return AlphaSignal(
            symbol=symbol,
            timestamp=datetime.now(),
            signal_type=SignalType.MACHINE_LEARNING,
            strength=blended_strength,
            horizon=avg_horizon,
            metadata=metadata
        )


# ============================================================================
# ALPHA ENGINE ORCHESTRATOR
# ============================================================================

class AlphaEngine:
    """
    Main alpha engine orchestrating all models
    """
    
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        
        # Model instances
        self.ou_models: Dict[str, OrnsteinUhlenbeckModel] = {
            sym: OrnsteinUhlenbeckModel(sym) for sym in symbols
        }
        self.factor_model = CrossSectionalFactorModel(symbols)
        self.ml_models: Dict[str, MLEnsemble] = {
            sym: MLEnsemble(sym) for sym in symbols
        }
        
        # Signal blender
        self.blender = ICWeightedBlender(symbols)
        
        # Price tracking for IC calculation
        self.last_prices: Dict[str, float] = {}
        
        # Callbacks
        self.signal_callbacks: List[Callable[[AlphaSignal], None]] = []
        
        # Stats
        self.signals_generated = 0
        self.signals_by_type: Dict[str, int] = {}
    
    def register_signal_callback(self, callback: Callable[[AlphaSignal], None]) -> None:
        """Register callback for new signals"""
        self.signal_callbacks.append(callback)
    
    def process_bar(self, bar: MarketBar) -> SignalBundle:
        """
        Process bar through all alpha models and return signal bundle
        """
        
        symbol = bar.symbol
        signals: Dict[str, AlphaSignal] = {}
        
        # Update ICs with realized returns
        if symbol in self.last_prices:
            realized_return = (bar.close - self.last_prices[symbol]) / self.last_prices[symbol]
            self.blender.update_returns(symbol, realized_return)
        
        self.last_prices[symbol] = bar.close
        
        # OU Mean Reversion Model
        ou_signal = self.ou_models[symbol].update(bar)
        if ou_signal:
            signals['ou'] = ou_signal
            self.blender.add_signal(ou_signal, 'ou')
        
        # Factor Model (only process every symbol once per bar)
        factor_signal = self.factor_model.update(bar)
        if factor_signal:
            signals['factor'] = factor_signal
            self.blender.add_signal(factor_signal, 'factor')
        
        # ML Ensemble
        ml_signal = self.ml_models[symbol].update(bar)
        if ml_signal:
            signals['ml'] = ml_signal
            self.blender.add_signal(ml_signal, 'ml')
        
        # Blend signals
        blended = self.blender.blend_signals(symbol, signals)
        
        # Notify callbacks for individual signals
        for signal in signals.values():
            self.signals_generated += 1
            sig_type = signal.signal_type.name
            self.signals_by_type[sig_type] = self.signals_by_type.get(sig_type, 0) + 1
            
            for callback in self.signal_callbacks:
                try:
                    callback(signal)
                except Exception as e:
                    logger.error(f"Signal callback error: {e}")
        
        # Create bundle
        bundle = SignalBundle(
            timestamp=bar.timestamp,
            signals={symbol: list(signals.values())}
        )
        
        return bundle
    
    def get_consensus_signal(self, symbol: str) -> Optional[AlphaSignal]:
        """Get blended consensus signal for symbol"""
        # This would require storing recent signals
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics"""
        return {
            'signals_generated': self.signals_generated,
            'signals_by_type': self.signals_by_type,
            'model_weights': self.blender.model_weights,
            'symbols': len(self.symbols)
        }


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'OrnsteinUhlenbeckModel',
    'CrossSectionalFactorModel',
    'MLEnsemble',
    'ICWeightedBlender',
    'AlphaEngine',
]
