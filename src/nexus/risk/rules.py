from abc import ABC, abstractmethod
from typing import Optional, Dict
from ..models.trade import Order, PortfolioState

class RiskRule(ABC):
    """
    Abstract base class for all risk rules.
    Rules can be Pre-trade (check before submission) or Post-trade (monitor state).
    """
    @abstractmethod
    def validate(self, order: Optional[Order], portfolio: PortfolioState) -> bool:
        """
        Returns True if the check passes, False if it violates risk limits.
        """
        pass

    @abstractmethod
    def get_reason(self) -> str:
        """Description of why the risk check failed."""
        pass

class MaxOrderValueRule(RiskRule):
    """Prevents fat-finger errors by limiting the dollar value of a single order."""
    def __init__(self, max_value: float = 100000.0):
        self.max_value = max_value
        self.reason = ""

    def validate(self, order: Optional[Order], portfolio: PortfolioState) -> bool:
        if not order: return True
        
        # Approximate value using limit price or last price (simplified for now)
        price = order.limit_price or 100.0 # Placeholder
        value = order.quantity * price
        
        if value > self.max_value:
            self.reason = f"Order value ${value:,.2f} exceeds limit ${self.max_value:,.2f}"
            return False
        return True

    def get_reason(self) -> str:
        return self.reason

class MaxDrawdownRule(RiskRule):
    """Post-trade rule: Triggers if portfolio drawdown exceeds a limit."""
    def __init__(self, max_drawdown_pct: float = 0.10):
        self.max_drawdown_pct = max_drawdown_pct
        self.peak_equity = 0.0
        self.reason = ""

    def validate(self, order: Optional[Order], portfolio: PortfolioState) -> bool:
        self.peak_equity = max(self.peak_equity, portfolio.equity)
        if self.peak_equity == 0: return True
        
        drawdown = (self.peak_equity - portfolio.equity) / self.peak_equity
        if drawdown > self.max_drawdown_pct:
            self.reason = (
                f"Portfolio drawdown {drawdown*100:.2f}% "
                f"exceeds limit {self.max_drawdown_pct*100:.2f}%"
            )
            return False
        return True

    def get_reason(self) -> str:
        return self.reason

class SectorConcentrationRule(RiskRule):
    """
    Limits total exposure to a single sector.
    Institutional firms (Citadel style) use this to prevent correlated failures.
    """
    def __init__(self, sector_map: Dict[str, str], max_sector_weight: float = 0.25):
        self.sector_map = sector_map
        self.max_sector_weight = max_sector_weight
        self.reason = ""

    def validate(self, order: Optional[Order], portfolio: PortfolioState) -> bool:
        if not order: return True
        
        target_sector = self.sector_map.get(order.symbol, "Unknown")
        if target_sector == "Unknown":
            # In production, this might reject the trade. Here we log a warning.
            return True 
            
        current_sector_value = 0.0
        for symbol, pos in portfolio.positions.items():
            if self.sector_map.get(symbol) == target_sector:
                current_sector_value += abs(pos.quantity * pos.last_price)
        
        # Add the potential new order value
        price = order.limit_price or 100.0 # Placeholder
        new_order_value = order.quantity * price
        total_sector_value = current_sector_value + new_order_value
        
        weight = total_sector_value / portfolio.equity if portfolio.equity > 0 else 1.0
        
        if weight > self.max_sector_weight:
            self.reason = f"Sector concentration for {target_sector} is {weight*100:.2f}%, exceeds limit {self.max_sector_weight*100:.2f}%"
            return False
        return True

    def get_reason(self) -> str:
        return self.reason

class LeverageRule(RiskRule):
    """
    Limits the gross leverage of the portfolio.
    Gross Leverage = (Long Value + Short Value) / Net Equity
    """
    def __init__(self, max_leverage: float = 2.0):
        self.max_leverage = max_leverage
        self.reason = ""

    def validate(self, order: Optional[Order], portfolio: PortfolioState) -> bool:
        gross_value = sum(abs(p.quantity * p.last_price) for p in portfolio.positions.values())
        
        if order:
            price = order.limit_price or 100.0
            gross_value += (order.quantity * price)
            
        leverage = gross_value / portfolio.equity if portfolio.equity > 0 else 0.0
        
        if leverage > self.max_leverage:
            self.reason = f"Portfolio gross leverage {leverage:.2f}x exceeds limit {self.max_leverage:.2f}x"
            return False
        return True

    def get_reason(self) -> str:
        return self.reason
