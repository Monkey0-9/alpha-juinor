
import logging
from typing import Dict, Any, List
from mini_quant_fund.portfolio.allocator import InstitutionalAllocator
from mini_quant_fund.portfolio.elite_optimizer import get_elite_optimizer

logger = logging.getLogger("PORTFOLIO_MANAGER")

class PortfolioManager:
    """
    Handles position sizing and portfolio optimization.
    """
    def __init__(self, nav: float = 1000000.0):
        self.nav = nav
        self.allocator = InstitutionalAllocator()
        self.optimizer = get_elite_optimizer()
        logger.info(f"PortfolioManager initialized with NAV: ${nav:,.2f}")

    def optimize_allocations(self, signals: Dict[str, Any], current_positions: Dict[str, float]) -> Dict[str, float]:
        """
        Convert signals to target weights and then to share quantities.
        """
        try:
            # 1. Generate Target Weights
            weights = self.allocator.calculate_weights(signals)
            
            # 2. Refine via Optimizer (Risk Parity / Mean-Var)
            optimized_weights = self.optimizer.optimize(weights, current_positions)
            
            # 3. Convert to share counts (placeholder logic for share calculation)
            target_positions = {ticker: (weight * self.nav / 150.0) for ticker, weight in optimized_weights.items()}
            
            return target_positions
        except Exception as e:
            logger.error(f"Portfolio optimization failed: {e}", exc_info=True)
            return current_positions

def get_portfolio_manager(nav: float) -> PortfolioManager:
    return PortfolioManager(nav)
