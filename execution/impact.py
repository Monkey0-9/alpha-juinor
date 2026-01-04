
import pandas as pd
from typing import Dict

def calc_implementation_shortfall(orders: pd.DataFrame, executions: pd.DataFrame) -> pd.DataFrame:
    """
    implementation_shortfall = paper_return - actual_return
    """
    return pd.DataFrame() # Stub for impact model
