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
        fe = FeatureEngineer()
        self.ml_alpha = MLAlpha(feature_engineer=fe)

    def generate_signals(self, market_data: pd.DataFrame) -> pd.DataFrame:
        signals = {}
        ts = market_data.index[-1]
        for tk in self.tickers:
            try:
                if tk not in market_data.columns.levels[0]: continue
                df_tk = market_data[tk]
                tech_score = float(self.technical_alpha.compute(df_tk).iloc[-1])
                if self.ml_alpha and self.ml_alpha.is_trained:
                    ml_score = self.ml_alpha.predict_conviction(df_tk)
                    # 50/50 Fusion
                    conviction = 0.5 * tech_score + 0.5 * ml_score
                else:
                    conviction = tech_score
                signals[tk] = conviction
            except Exception as e:
                signals[tk] = 0.5 
        return pd.DataFrame([signals], index=[ts])

    def calculate_risk(self, signals: pd.DataFrame, market_data: pd.DataFrame) -> pd.DataFrame:
        return signals

    def train_models(self, data: pd.DataFrame):
        for tk in self.tickers:
            if tk in data.columns.levels[0]:
                self.ml_alpha.train(data[tk])
