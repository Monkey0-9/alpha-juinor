
import sqlite3
import pandas as pd
from typing import Dict, List, Optional
import logging

DB = "runtime/institutional_trading.db"
logger = logging.getLogger(__name__)

from database.manager import DatabaseManager
from data.collectors.data_router import DataRouter

def load_market_data(symbols: List[str], lookback: int = 252) -> Dict[str, pd.DataFrame]:
    """ Institutional Wrapper for canonical data loading (Objective 3). """
    db = DatabaseManager()
    router = DataRouter()
    return router.load_market_data(symbols, db)
