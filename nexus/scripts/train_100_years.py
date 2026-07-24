import os
import sys
import logging
import pandas as pd
import numpy as np

# Setup path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from nexus.data.historical_data_loader import HistoricalDataLoader
from nexus.core.ml_brain import AdvancedMLBrain
from nexus.core.strategies import StrategyFactory

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def train_100_years_model():
    """
    Downloads 100 years of index data, generates historical signals across all strategies,
    and trains the AdvancedMLBrain (XGBoost) to navigate all macro regimes.
    """
    loader = HistoricalDataLoader()
    # Pull S&P 500 max history (back to 1927)
    df = loader.load_macro_data("^GSPC")
    
    if df.empty or len(df) < 100:
        logger.error("Failed to load sufficient historical data for training.")
        return
        
    logger.info(f"Loaded {len(df)} days of historical data from {df.index.min().date()} to {df.index.max().date()}")
    
    strategies = StrategyFactory.all_strategies()
    logger.info(f"Generating historical features for {len(strategies)} strategies...")
    
    features_list = []
    targets = []
    
    # We will simulate stepping through history. 
    # To keep this fast, we'll sample every 5 days.
    step = 5
    for i in range(100, len(df) - 5, step):
        history_window = df.iloc[i-100:i]
        future_return = df['returns'].iloc[i:i+5].sum()  # predict 5-day forward return
        
        current_regime = "BULL" if history_window['regime'].iloc[-1] == 1 else "BEAR"
        regime_probs = {"BULL": 1.0 if current_regime=="BULL" else 0.0,
                        "BEAR": 1.0 if current_regime=="BEAR" else 0.0,
                        "SIDEWAYS": 0.0}
        
        alpha_proxy = float(np.tanh(history_window['returns'].mean() * 10))
        
        feature_row = {
            "raw_alpha": alpha_proxy,
            "regime_prob_bull": regime_probs["BULL"],
            "regime_prob_bear": regime_probs["BEAR"],
            "regime_prob_sideways": regime_probs["SIDEWAYS"]
        }
        
        for strat in strategies:
            # We mock alpha as the raw moving average momentum for the historical deep test
            score = strat.score("^GSPC", alpha_proxy, history_window, current_regime)
            feature_row[f"strat_{strat.name}"] = float(score)
            
        features_list.append(feature_row)
        targets.append(future_return)
        
        if len(features_list) % 1000 == 0:
            logger.info(f"Processed {len(features_list)} training samples...")
            
    features_df = pd.DataFrame(features_list)
    targets_series = pd.Series(targets)
    
    # Map target returns to a continuous signal score [-1, 1]
    # High return = 1.0, Large loss = -1.0
    targets_normalized = np.tanh(targets_series * 20.0) 
    
    ml_brain = AdvancedMLBrain(model_path="nexus/models/xgboost_brain.json")
    ml_brain.train(features_df, targets_normalized)
    
    logger.info("🎉 100-Year Machine Learning Training Complete!")

if __name__ == "__main__":
    train_100_years_model()
