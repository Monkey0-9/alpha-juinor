# strategies/institutional_strategy.py
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import logging
import time
from .base import BaseStrategy
from .alpha import CompositeAlpha, TrendAlpha, MeanReversionAlpha, RSIAlpha, MACDAlpha, BollingerBandAlpha
from .ml_models.ml_alpha import MLAlpha
from data.processors.features import FeatureEngineer

# New Institutional Modules
from regime.markov import RegimeModel
from market_structure.wyckoff import structure_filter
from market_structure.auction import AuctionConfidence
from market_structure.market_profile import compute_market_profile
from timing.gann_cycles import GannTimeFilter
from micro.vpin import compute_vpin

logger = logging.getLogger(__name__)

class InstitutionalStrategy(BaseStrategy):
    """
    Maximized High-Frequency Alpha Strategy.
    Enriched with Institutional Regimes, Structures, and Filters.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.tickers = config.get('tickers', [])
        
        # Feature Flags
        features = config.get('features', {})
        self.use_regime = features.get('use_regime_detection', True)
        self.use_wyckoff = features.get('use_wyckoff_filter', True)
        self.use_auction = features.get('use_auction_market_confidence', True)
        self.use_market_profile = features.get('use_market_profile_value_area', True)
        self.use_gann = features.get('use_gann_time_filter', True)
        self.use_vpin = features.get('use_vpin_filter', True)

        # Ensemble of 5 technical alphas
        self.technical_alpha = CompositeAlpha(alphas=[
            TrendAlpha(short=5, long=20),
            MeanReversionAlpha(),
            RSIAlpha(),
            MACDAlpha(),
            BollingerBandAlpha()
        ], window=30)
        # Per-Ticker ML Models (Institutional Hardening)
        fe = FeatureEngineer()
        self.ml_models: Dict[str, MLAlpha] = {
            tk: MLAlpha(feature_engineer=fe) for tk in self.tickers
        }
        
        # Initialize Filters
        self.regime_engine = RegimeModel()
        self.auction_engine = AuctionConfidence()
        self.gann_filter = GannTimeFilter()
        
        # Optimization: Pre-allocate Fallback Logging Set & Throttle
        self._logged_fallback = set()
        self._log_throttle = 0

    def generate_signals(self, market_data: pd.DataFrame) -> pd.DataFrame:
        t_start = time.perf_counter()
        signals = {}
        ts = market_data.index[-1]
        
        # 0. Global Timing Filter (Gann) - O(1) Check
        if self.use_gann and not self.gann_filter.can_trade(ts):
            # Fast Exit
            for tk in self.tickers: signals[tk] = 0.5
            return pd.DataFrame([signals], index=[ts])

        # LATENCY OPTIMIZATION:
        # Enforce "Time Budget" of 50ms (soft target)
        
        # Rate Limiting for Logging (Prevent I/O storms during spikes)
        if not hasattr(self, '_log_throttle'): self._log_throttle = 0
        
        # GLOBAL PRE-SLICING (The "Still More Better" Optimization)
        # Instead of slicing deeper in the loop, we slice the whole panel once.
        # This is strictly O(1) for the loop.
        if len(market_data) > 60:
            market_window = market_data.iloc[-60:]
        else:
            market_window = market_data
            
        for tk in self.tickers:
            try:
                # OPTIMIZATION: Check column existence cheaply
                if tk not in market_window.columns.levels[0]: continue
                
                # Access pre-sliced window. 
                # Still need dropna() for specific ticker in case of partial data
                df_slice = market_window[tk].dropna()
                
                if df_slice.empty or len(df_slice) < 20:
                    signals[tk] = 0.5
                    continue
                
                # 1. Alpha (Hybrid)
                # Technical (Vectorized inside needs more data? TrendAlpha long=20. 60 is fine.)
                tech_score = float(self.technical_alpha.compute(df_slice).iloc[-1])
                
                # ML (Inference)
                # CRITICAL: Do not alter ML models as per user instruction.
                ml_model = self.ml_models.get(tk)
                conviction = 0.5
                if ml_model and ml_model.is_trained and ml_model.ml_ready:
                    ml_score = ml_model.predict_conviction(df_slice)
                    conviction = 0.5 * tech_score + 0.5 * ml_score
                else:
                    conviction = 0.5 + (tech_score - 0.5) * 0.7
                    
                # 2. Institutional Filters (Conditional Execution - Fail Fast)
                if abs(conviction - 0.5) < 0.01:
                    signals[tk] = 0.5
                    continue
                
                # Regime
                if self.use_regime:
                    label = self.regime_engine.infer(df_slice)
                    if label == "high_vol": conviction = 0.5 + (conviction - 0.5) * 0.5
                    elif label == "neutral": conviction = 0.5 + (conviction - 0.5) * 0.8
                    
                # Wyckoff
                if self.use_wyckoff:
                    # FIX: Cache result, don't call multiple times!
                    wyckoff_s = structure_filter(df_slice)
                    if conviction > 0.5 and not wyckoff_s['allow_long']: conviction = 0.5
                    if conviction < 0.5 and not wyckoff_s['allow_short']: conviction = 0.5

                # Auction
                if self.use_auction:
                    auc = self.auction_engine.compute_confidence(
                        df_slice["Close"], df_slice["Volume"], df_slice["High"], df_slice["Low"]
                    )
                    conviction = 0.5 + (conviction - 0.5) * auc

                # Market Profile
                if self.use_market_profile:
                    mp = compute_market_profile(df_slice)
                    if mp['inside_value_area']: conviction = 0.5 + (conviction - 0.5) * 0.5
                    
                # VPIN (Skipped for now or stub)
                
                signals[tk] = conviction
                
            except Exception as e:
                # ERROR SAFETY: Catch all, log only once per 100 errors to avoid flooding
                # signals[tk] = 0.5 is the safe default
                if self._log_throttle % 100 == 0:
                     logger.error(f"Signal failed {tk}: {e}")
                self._log_throttle += 1
                signals[tk] = 0.5

        t_end = time.perf_counter()
        duration_ms = (t_end - t_start) * 1000.0
        
        # Performance Monitoring (Throttled)
        if duration_ms > 50:
             self._log_throttle += 1
             if self._log_throttle % 50 == 0: # Log only every 50th slow occurrence
                 logger.warning(f"SLOW SIGNAL GENERATION: {duration_ms:.2f}ms")
             
        return pd.DataFrame([signals], index=[ts])

    def calculate_risk(self, signals: pd.DataFrame, market_data: pd.DataFrame) -> pd.DataFrame:
        """
        Pass-through for risk calculation.
        Actual risk logic is handled by RiskManager in the Engine, but Strategy contract requires this.
        """
        return signals

    def train_models(self, data: pd.DataFrame):
        """Train independent ML models."""
        for tk in self.tickers:
            if tk in data.columns.levels[0]:
                df_tk = data[tk].dropna()
                if not df_tk.empty:
                    ml_model = self.ml_models.get(tk)
                    if ml_model:
                        ml_model.train(df_tk)
