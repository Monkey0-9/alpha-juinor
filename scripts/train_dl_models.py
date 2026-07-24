# train_dl_models.py has been unified with train_100_years.py
# to ensure Deep Learning models are trained on real 100-year market data
# instead of synthetic random data.

import os
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Redirecting Deep Learning Training to the unified 100-year real-data pipeline...")
    # Import and run the main 100-year pipeline which handles XGBoost, LSTM, Transformer, and PPO
    import train_100_years
    train_100_years.main()
