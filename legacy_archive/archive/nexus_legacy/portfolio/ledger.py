import pandas as pd
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

class EventType(Enum):
    CASH_DEPOSIT = "CASH_DEPOSIT"
    CASH_WITHDRAWAL = "CASH_WITHDRAWAL"
    ORDER_FILLED = "ORDER_FILLED"
    DIVIDEND = "DIVIDEND"
    FEE = "FEE"
    MARK_TO_MARKET = "MARK_TO_MARKET" # Unrealized PnL update

@dataclass
class PortfolioEvent:
    timestamp: datetime
    event_type: EventType
    ticker: Optional[str] = None
    quantity: float = 0.0
    price: float = 0.0
    amount: float = 0.0 # Cash impact
    commission: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

class CashBook:
    def __init__(self, initial_cash: float = 0.0):
        self.balance = initial_cash
        self.history: List[Dict] = []

    def update(self, amount: float, timestamp: datetime, description: str):
        self.balance += amount
        self.history.append({
            "timestamp": timestamp,
            "change": amount,
            "balance": self.balance,
            "description": description
        })

    def to_dict(self) -> Dict:
        return {
            "balance": self.balance,
            "history": self.history
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'CashBook':
        book = cls(data["balance"])
        book.history = data.get("history", [])
        return book

class PositionBook:
    def __init__(self):
        self.positions: Dict[str, float] = {} # ticker -> quantity
        self.cost_basis: Dict[str, float] = {} # ticker -> total cost paid for current quantity

    def update(self, ticker: str, quantity: float, price: float, timestamp: datetime):
        current_qty = self.positions.get(ticker, 0.0)
        new_qty = current_qty + quantity
        
        # Calculate PnL if selling
        realized_pnl = 0.0
        if quantity < 0 and current_qty > 0:
            # Simple FIFO or average cost basis? Let's use average cost for now.
            avg_cost = self.cost_basis.get(ticker, 0.0) / current_qty if current_qty != 0 else 0
            # sell_qty = min(abs(quantity), current_qty)
            realized_pnl = (price - avg_cost) * abs(quantity)
            
        # Update cost basis
        if quantity > 0:
            self.cost_basis[ticker] = self.cost_basis.get(ticker, 0.0) + (quantity * price)
        elif new_qty > 0:
            # Proportionally reduce cost basis
            ratio = new_qty / current_qty
            self.cost_basis[ticker] *= ratio
        else:
            self.cost_basis[ticker] = 0.0

        self.positions[ticker] = new_qty
        if abs(self.positions[ticker]) < 1e-8:
            del self.positions[ticker]
            if ticker in self.cost_basis:
                del self.cost_basis[ticker]
                
        return realized_pnl

    def to_dict(self) -> Dict:
        return {
            "positions": self.positions,
            "cost_basis": self.cost_basis
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'PositionBook':
        book = cls()
        book.positions = data.get("positions", {})
        book.cost_basis = data.get("cost_basis", {})
        return book

class PnLBook:
    def __init__(self):
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.fees_paid = 0.0

    def add_realized(self, amount: float):
        self.realized_pnl += amount

    def add_fee(self, amount: float):
        self.fees_paid += amount

    def to_dict(self) -> Dict:
        return {
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "fees_paid": self.fees_paid
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'PnLBook':
        book = cls()
        book.realized_pnl = data.get("realized_pnl", 0.0)
        book.unrealized_pnl = data.get("unrealized_pnl", 0.0)
        book.fees_paid = data.get("fees_paid", 0.0)
        return book

class PortfolioLedger:
    def __init__(self, initial_capital: float = 1_000_000):
        self.cash_book = CashBook(initial_capital)
        self.position_book = PositionBook()
        self.pnl_book = PnLBook()
        self.events: List[PortfolioEvent] = []
        self.equity_curve: List[Dict] = []

    def record_event(self, event: PortfolioEvent):
        self.events.append(event)
        
        if event.event_type == EventType.ORDER_FILLED:
            # Double entry logic: 
            # 1. Cash decreases by (qty * price) + commission
            # 2. Inventory increases by qty
            trade_cost = event.quantity * event.price
            self.cash_book.update(-(trade_cost + event.commission), event.timestamp, f"Fill {event.ticker}")
            self.pnl_book.add_fee(event.commission)
            
            realized = self.position_book.update(event.ticker, event.quantity, event.price, event.timestamp)
            self.pnl_book.add_realized(realized)

        elif event.event_type == EventType.CASH_DEPOSIT:
            self.cash_book.update(event.amount, event.timestamp, "Deposit")
        
        elif event.event_type == EventType.CASH_WITHDRAWAL:
            self.cash_book.update(-event.amount, event.timestamp, "Withdrawal")

    def create_snapshot(self, timestamp: datetime, current_prices: Dict[str, float]):
        market_value = 0.0
        unrealized_pnl = 0.0
        
        for ticker, qty in self.position_book.positions.items():
            price = current_prices.get(ticker)
            if price is None or not isinstance(price, (int, float)):
                 # STRICT RULE: Missing prices → last known price ONLY. 
                 # But if even that is missing, we cannot compute equity.
                 raise ValueError(f"Cannot compute equity: Missing price for position {ticker}")
            
            mv = qty * price
            market_value += mv
            
            # Unrealized PnL = Market Value - Cost Basis
            cost = self.position_book.cost_basis.get(ticker, 0.0)
            unrealized_pnl += (mv - cost)
                
        self.pnl_book.unrealized_pnl = unrealized_pnl
        total_equity = self.cash_book.balance + market_value
        

        snapshot = {
            "timestamp": timestamp,
            "equity": total_equity,
            "cash": self.cash_book.balance,
            "market_value": market_value,
            "realized_pnl": self.pnl_book.realized_pnl,
            "unrealized_pnl": unrealized_pnl,
            "fees": self.pnl_book.fees_paid,
            "positions": self.position_book.positions.copy()
        }
        self.equity_curve.append(snapshot)
        return snapshot

    def to_dict(self) -> Dict:
        return {
            "cash_book": self.cash_book.to_dict(),
            "position_book": self.position_book.to_dict(),
            "pnl_book": self.pnl_book.to_dict(),
            "equity_curve": self.equity_curve
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'PortfolioLedger':
        ledger = cls()
        ledger.cash_book = CashBook.from_dict(data["cash_book"])
        ledger.position_book = PositionBook.from_dict(data["position_book"])
        ledger.pnl_book = PnLBook.from_dict(data["pnl_book"])
        ledger.equity_curve = data.get("equity_curve", [])
        return ledger

    def get_equity_curve_df(self) -> pd.DataFrame:
        if not self.equity_curve:
            return pd.DataFrame()
        df = pd.DataFrame(self.equity_curve)
        df.set_index("timestamp", inplace=True)
        return df
