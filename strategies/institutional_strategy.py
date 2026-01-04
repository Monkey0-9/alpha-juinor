# strategies/institutional_strategy.py
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import logging
from .base import BaseStrategy
from .alpha import CompositeAlpha, TrendAlpha, MeanReversionAlpha, RSIAlpha, MACDAlpha, BollingerBandAlpha
from .ml_models.ml_alpha import MLAlpha
from data.processors.features import FeatureEngineer

logger = logging.getLogger(__name__)

class InstitutionalStrategy(BaseStrategy):
    """
    Maximized High-Frequency Alpha Strategy.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.tickers = config.get('tickers', [])
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

    def generate_signals(self, market_data: pd.DataFrame) -> pd.DataFrame:
        signals = {}
        ts = market_data.index[-1]
        for tk in self.tickers:
            try:
                if tk not in market_data.columns.levels[0]: continue
                
                # CLEANING: Drop NaNs from ticker-specific slice (Fixes crypto/equity alignment)
                df_tk = market_data[tk].dropna()
                if df_tk.empty:
                    signals[tk] = 0.5
                    continue
                
                tech_score = float(self.technical_alpha.compute(df_tk).iloc[-1])
                
                ml_model = self.ml_models.get(tk)
                if ml_model and ml_model.is_trained and ml_model.ml_ready:
                    ml_score = ml_model.predict_conviction(df_tk)
                    # 50/50 Fusion
                    conviction = 0.5 * tech_score + 0.5 * ml_score
                else:
                    # ML FALLBACK: 50% Confidence Penalty (Institutional Guard)
                    # We slightly pull the signal towards neutral (0.5) if ML is missing.
                    conviction = 0.5 + (tech_score - 0.5) * 0.7 
                    if tk not in getattr(self, '_logged_fallback', set()):
                        logger.info(f"{tk}: ML DISABLED (History < {self.ml_models[tk].train_window}) — Technical only (with penalty)")
                        if not hasattr(self, '_logged_fallback'): self._logged_fallback = set()
                        self._logged_fallback.add(tk)
                signals[tk] = conviction
            except Exception as e:
                logger.error(f"Signal failure for {tk}: {e}")
                signals[tk] = 0.5 
        return pd.DataFrame([signals], index=[ts])

    def calculate_risk(self, signals: pd.DataFrame, market_data: pd.DataFrame) -> pd.DataFrame:
        return signals

    def train_models(self, data: pd.DataFrame):
        """Train independent ML models for each asset in the universe."""
        for tk in self.tickers:
            if tk in data.columns.levels[0]:
                # CLEANING: Mandatory dropna for unaligned panel slices
                df_tk = data[tk].dropna()
                if not df_tk.empty:
                    ml_model = self.ml_models.get(tk)
                    if ml_model:
                        ml_model.train(df_tk)
                else:
                    logger.warning(f"Skipping ML training for {tk}: No valid price data after cleaning.")
