
import numpy as np
import pandas as pd
from typing import List

def calculate_portfolio_cvar(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """
    Computes Conditional Value at Risk (Expected Shortfall) at confidence level.
    Returns positive float representing loss.
    """
    if returns.empty or len(returns) < 20:
        return 0.0

    # Invert returns to losses
    losses = -returns.dropna()

    # Sort
    sorted_losses = np.sort(losses)

    # Calculate VaR index
    index = int((1 - confidence_level) * len(sorted_losses))
    if index >= len(sorted_losses):
        index = len(sorted_losses) - 1

    var = sorted_losses[index]

    cutoff_index = int(len(sorted_losses) * confidence_level)
    tail_losses = sorted_losses[cutoff_index:]

    if len(tail_losses) == 0:
        return var

    cvar = tail_losses.mean()
    return cvar

# Alias as compute_cvar for compatibility with risk/engine.py
compute_cvar = calculate_portfolio_cvar

class CVaRGate:
    def __init__(self, limit: float = 0.05):
        self.limit = limit

    def check_portfolio(self, current_returns: pd.Series, new_position_return: pd.Series = None) -> bool:
        """
        Returns False if CVaR > Limit.
        """
        if new_position_return is not None:
             # Simulation of adding position? (Requires complex portfolio modeling)
             # For now, just check total series
             combined = current_returns # Placeholder
             cvar = calculate_portfolio_cvar(combined)
        else:
             cvar = calculate_portfolio_cvar(current_returns)

        return cvar <= self.limit
