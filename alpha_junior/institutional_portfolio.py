#!/usr/bin/env python3
"""
Alpha Junior - INSTITUTIONAL PORTFOLIO MANAGEMENT
Hedge Fund-Grade Risk Management & Position Sizing
Used by top 1% of institutional traders
"""

import numpy as np
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class Position:
    symbol: str
    shares: int
    entry_price: float
    current_price: float
    stop_loss: float
    take_profit: float
    strategy: str
    entry_time: datetime
    risk_amount: float
    portfolio_weight: float
    
    @property
    def market_value(self) -> float:
        return self.shares * self.current_price
    
    @property
    def unrealized_pnl(self) -> float:
        return self.shares * (self.current_price - self.entry_price)
    
    @property
    def unrealized_pnl_pct(self) -> float:
        return ((self.current_price - self.entry_price) / self.entry_price) * 100
    
    @property
    def days_held(self) -> int:
        return (datetime.now() - self.entry_time).days

class InstitutionalPortfolioManager:
    """
    Institutional-grade portfolio management
    Risk parity, position sizing, correlation management
    """
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        
        # Portfolio constraints (institutional standards)
        self.constraints = {
            'max_portfolio_risk': 0.02,        # 2% daily VaR limit
            'max_position_size': 0.15,          # 15% max single position
            'max_sector_exposure': 0.30,        # 30% max sector
            'max_correlated_positions': 5,      # Max correlated pairs
            'min_cash_reserve': 0.10,           # 10% minimum cash
            'max_drawdown_limit': 0.15,         # 15% max drawdown
            'target_beta': 0.80,                # Target 0.8 market beta
            'max_leverage': 1.0,                # No leverage for safety
        }
        
        # Current portfolio state
        self.positions: Dict[str, Position] = {}
        self.cash = 100000.0
        self.total_value = 100000.0
        self.peak_value = 100000.0
        self.current_drawdown = 0.0
        
        # Risk tracking
        self.daily_pnl = []
        self.var_95 = 0.0  # Value at Risk (95% confidence)
        self.sharpe_ratio = 0.0
        self.max_drawdown = 0.0
        
        # Sector allocations
        self.sectors = {
            'technology': 0.0,
            'healthcare': 0.0,
            'financial': 0.0,
            'energy': 0.0,
            'consumer': 0.0,
            'industrial': 0.0,
            'materials': 0.0,
            'utilities': 0.0,
            'realestate': 0.0,
            'cash': 1.0
        }
        
        # Sector mappings
        self.sector_map = {
            # Tech
            'AAPL': 'technology', 'MSFT': 'technology', 'GOOGL': 'technology',
            'AMZN': 'technology', 'NVDA': 'technology', 'META': 'technology',
            'NFLX': 'technology', 'AMD': 'technology', 'CRM': 'technology',
            'ADBE': 'technology', 'ORCL': 'technology', 'INTC': 'technology',
            'PLTR': 'technology', 'SNOW': 'technology', 'ZM': 'technology',
            'UBER': 'technology', 'ABNB': 'technology', 'SQ': 'technology',
            'CRWD': 'technology', 'NET': 'technology', 'DDOG': 'technology',
            
            # Healthcare
            'JNJ': 'healthcare', 'PFE': 'healthcare', 'UNH': 'healthcare',
            'ABBV': 'healthcare', 'MRK': 'healthcare', 'TMO': 'healthcare',
            'ABT': 'healthcare', 'DHR': 'healthcare', 'BMY': 'healthcare',
            'LLY': 'healthcare', 'MRNA': 'healthcare', 'REGN': 'healthcare',
            'VRTX': 'healthcare', 'GILD': 'healthcare', 'AMGN': 'healthcare',
            'BIIB': 'healthcare', 'CRSP': 'healthcare', 'EDIT': 'healthcare',
            
            # Financial
            'JPM': 'financial', 'BAC': 'financial', 'WFC': 'financial',
            'GS': 'financial', 'MS': 'financial', 'C': 'financial',
            'BLK': 'financial', 'AXP': 'financial', 'USB': 'financial',
            'PNC': 'financial', 'TFC': 'financial', 'COF': 'financial',
            'SCHW': 'financial', 'V': 'financial', 'MA': 'financial',
            'PYPL': 'financial', 'SQ': 'financial', 'SOFI': 'financial',
            'UPST': 'financial', 'AFRM': 'financial', 'COIN': 'financial',
            
            # Energy
            'XOM': 'energy', 'CVX': 'energy', 'COP': 'energy',
            'EOG': 'energy', 'SLB': 'energy', 'OXY': 'energy',
            'MPC': 'energy', 'VLO': 'energy', 'PSX': 'energy',
            'WMB': 'energy', 'KMI': 'energy', 'EPD': 'energy',
            'ENPH': 'energy', 'SEDG': 'energy', 'FSLR': 'energy',
            'RUN': 'energy', 'PLUG': 'energy', 'BE': 'energy',
            
            # Consumer
            'AMZN': 'consumer', 'TSLA': 'consumer', 'HD': 'consumer',
            'MCD': 'consumer', 'NKE': 'consumer', 'LOW': 'consumer',
            'TGT': 'consumer', 'COST': 'consumer', 'SBUX': 'consumer',
            'TJX': 'consumer', 'BKNG': 'consumer', 'MAR': 'consumer',
            'CMG': 'consumer', 'YUM': 'consumer', 'DPZ': 'consumer',
            
            # Industrial
            'CAT': 'industrial', 'GE': 'industrial', 'BA': 'industrial',
            'HON': 'industrial', 'UNP': 'industrial', 'UPS': 'industrial',
            'RTX': 'industrial', 'LMT': 'industrial', 'MMM': 'industrial',
            'DE': 'industrial', 'CSX': 'industrial', 'NSC': 'industrial',
            'FDX': 'industrial', 'ITW': 'industrial', 'EMR': 'industrial',
            
            # Materials
            'LIN': 'materials', 'APD': 'materials', 'SHW': 'materials',
            'FCX': 'materials', 'NEM': 'materials', 'DOW': 'materials',
            'DD': 'materials', 'ECL': 'materials', 'NUE': 'materials',
            
            # Utilities
            'NEE': 'utilities', 'SO': 'utilities', 'DUK': 'utilities',
            'AEP': 'utilities', 'EXC': 'utilities', 'XEL': 'utilities',
            
            # Real Estate
            'AMT': 'realestate', 'PLD': 'realestate', 'CCI': 'realestate',
            'EQIX': 'realestate', 'PSA': 'realestate', 'O': 'realestate',
            'DLR': 'realestate', 'SBAC': 'realestate', 'WELL': 'realestate',
            
            # Default
            'SPY': 'etf', 'QQQ': 'etf', 'IWM': 'etf'
        }
        
        # Default sector for unknown stocks
        self.default_sector = 'technology'
    
    def get_sector(self, symbol: str) -> str:
        """Get sector for a stock"""
        return self.sector_map.get(symbol, self.default_sector)
    
    def calculate_kelly_criterion(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Kelly Criterion for optimal position sizing
        f* = (bp - q) / b
        where: b = avg win / avg loss, p = win rate, q = 1-p
        """
        if avg_loss == 0 or avg_win == 0:
            return 0.02  # Conservative default
        
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - p
        
        kelly = (b * p - q) / b
        
        # Use half-Kelly for safety (institutional practice)
        half_kelly = kelly * 0.5
        
        # Cap between 1% and 15%
        return max(0.01, min(0.15, half_kelly))
    
    def calculate_position_size(self, signal, portfolio_value: float,
                                  volatility: float, correlation: float = 0.5) -> int:
        """
        Institutional position sizing with multiple factors
        """
        symbol = signal.symbol
        entry_price = signal.entry_price
        stop_loss = signal.stop_loss
        
        # 1. Risk-based sizing (1% risk per trade)
        risk_per_trade = portfolio_value * 0.01
        
        # Calculate risk per share
        risk_per_share = entry_price - stop_loss
        if risk_per_share <= 0:
            risk_per_share = entry_price * 0.02  # Default 2%
        
        # Base position size
        base_shares = int(risk_per_trade / risk_per_share)
        
        # 2. Volatility adjustment
        vol_adjustment = 1.0
        if volatility > 50:  # High volatility
            vol_adjustment = 0.5
        elif volatility > 30:
            vol_adjustment = 0.75
        elif volatility < 15:  # Low volatility
            vol_adjustment = 1.25
        
        # 3. Signal confidence adjustment
        confidence_adj = signal.confidence / 100
        
        # 4. Correlation adjustment (reduce if correlated positions exist)
        correlation_penalty = 1.0 - (correlation * 0.5)
        
        # Calculate final size
        adjusted_shares = int(base_shares * vol_adjustment * confidence_adj * correlation_penalty)
        
        # 5. Check position constraints
        position_value = adjusted_shares * entry_price
        max_position_value = portfolio_value * self.constraints['max_position_size']
        
        if position_value > max_position_value:
            adjusted_shares = int(max_position_value / entry_price)
        
        # 6. Minimum position size
        if position_value < 500:  # Min $500
            return 0
        
        return max(1, adjusted_shares)
    
    def check_sector_exposure(self, symbol: str, position_value: float) -> bool:
        """Check if adding position would exceed sector limits"""
        sector = self.get_sector(symbol)
        
        # Calculate current sector exposure
        current_sector_value = sum(
            pos.market_value
            for pos in self.positions.values()
            if self.get_sector(pos.symbol) == sector
        )
        
        new_sector_exposure = (current_sector_value + position_value) / self.total_value
        
        return new_sector_exposure <= self.constraints['max_sector_exposure']
    
    def calculate_portfolio_risk(self) -> Dict:
        """Calculate portfolio-level risk metrics"""
        if not self.positions:
            return {'var_95': 0, 'portfolio_volatility': 0}
        
        # Calculate portfolio P&L volatility
        position_pnl_pcts = [pos.unrealized_pnl_pct for pos in self.positions.values()]
        
        if len(position_pnl_pcts) > 1:
            portfolio_volatility = np.std(position_pnl_pcts)
        else:
            portfolio_volatility = 0
        
        # Simplified VaR (Value at Risk)
        # In real implementation, use historical simulation or Monte Carlo
        var_95 = portfolio_volatility * 1.645  # 95% confidence
        
        return {
            'var_95': var_95,
            'var_95_amount': self.total_value * (var_95 / 100),
            'portfolio_volatility': portfolio_volatility,
            'max_position_pnl_pct': max(position_pnl_pcts) if position_pnl_pcts else 0,
            'min_position_pnl_pct': min(position_pnl_pcts) if position_pnl_pcts else 0,
        }
    
    def check_drawdown(self) -> bool:
        """Check if portfolio is within drawdown limits"""
        if self.total_value > self.peak_value:
            self.peak_value = self.total_value
        
        current_drawdown = (self.peak_value - self.total_value) / self.peak_value
        self.current_drawdown = current_drawdown
        
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
        # If approaching limit, reduce risk
        return current_drawdown < self.constraints['max_drawdown_limit']
    
    def rebalance_portfolio(self):
        """Rebalance portfolio to maintain targets"""
        self.logger.info("🔄 Rebalancing portfolio...")
        
        # Check sector allocations
        for sector in self.sectors.keys():
            if sector == 'cash':
                continue
            
            sector_value = sum(
                pos.market_value
                for pos in self.positions.values()
                if self.get_sector(pos.symbol) == sector
            )
            
            sector_weight = sector_value / self.total_value
            
            if sector_weight > self.constraints['max_sector_exposure']:
                self.logger.warning(
                    f"⚠️ Sector {sector} overweight: {sector_weight:.1%} > "
                    f"{self.constraints['max_sector_exposure']:.1%}"
                )
    
    def get_position_to_reduce(self) -> Optional[str]:
        """Get position to reduce if risk limits exceeded"""
        if not self.positions:
            return None
        
        # Find worst performing position
        worst_pos = min(self.positions.values(), key=lambda p: p.unrealized_pnl_pct)
        
        if worst_pos.unrealized_pnl_pct < -8:  # 8% loss
            return worst_pos.symbol
        
        return None
    
    def can_add_position(self, symbol: str, value: float) -> Tuple[bool, str]:
        """Check if new position can be added"""
        # Check cash
        min_cash = self.total_value * self.constraints['min_cash_reserve']
        if self.cash - value < min_cash:
            return False, "Insufficient cash reserves"
        
        # Check sector limit
        if not self.check_sector_exposure(symbol, value):
            return False, "Sector exposure limit"
        
        # Check drawdown
        if not self.check_drawdown():
            return False, "Drawdown limit reached - reducing risk"
        
        # Check portfolio risk
        risk = self.calculate_portfolio_risk()
        if risk['var_95'] > self.constraints['max_portfolio_risk'] * 100:
            return False, "Portfolio risk limit"
        
        return True, "OK"
    
    def update_portfolio_value(self, prices: Dict[str, float]):
        """Update portfolio with current prices"""
        for symbol, price in prices.items():
            if symbol in self.positions:
                self.positions[symbol].current_price = price
        
        # Recalculate total
        positions_value = sum(pos.market_value for pos in self.positions.values())
        self.total_value = positions_value + self.cash
        
        # Check drawdown
        self.check_drawdown()
    
    def get_portfolio_summary(self) -> Dict:
        """Get institutional-grade portfolio summary"""
        risk = self.calculate_portfolio_risk()
        
        return {
            'total_value': self.total_value,
            'cash': self.cash,
            'positions_value': self.total_value - self.cash,
            'cash_pct': (self.cash / self.total_value) * 100,
            'num_positions': len(self.positions),
            'current_drawdown': self.current_drawdown * 100,
            'max_drawdown': self.max_drawdown * 100,
            'var_95_pct': risk['var_95'],
            'var_95_amount': risk['var_95_amount'],
            'portfolio_volatility': risk['portfolio_volatility'],
            'sector_allocation': self._get_sector_allocation(),
            'positions': [
                {
                    'symbol': p.symbol,
                    'shares': p.shares,
                    'entry': p.entry_price,
                    'current': p.current_price,
                    'value': p.market_value,
                    'pnl': p.unrealized_pnl,
                    'pnl_pct': p.unrealized_pnl_pct,
                    'days': p.days_held,
                    'sector': self.get_sector(p.symbol),
                    'strategy': p.strategy
                }
                for p in self.positions.values()
            ]
        }
    
    def _get_sector_allocation(self) -> Dict[str, float]:
        """Get current sector allocation"""
        allocation = {sector: 0.0 for sector in self.sectors.keys()}
        
        for pos in self.positions.values():
            sector = self.get_sector(pos.symbol)
            allocation[sector] += pos.market_value / self.total_value
        
        allocation['cash'] = self.cash / self.total_value
        
        return allocation
    
    def get_rebalancing_recommendations(self) -> List[str]:
        """Get recommendations for portfolio rebalancing"""
        recommendations = []
        
        # Check sector overconcentration
        for sector, weight in self._get_sector_allocation().items():
            if weight > self.constraints['max_sector_exposure']:
                recommendations.append(
                    f"Reduce {sector} exposure from {weight:.1%} to "
                    f"{self.constraints['max_sector_exposure']:.1%}"
                )
        
        # Check cash level
        if self.cash / self.total_value < self.constraints['min_cash_reserve']:
            recommendations.append(
                f"Increase cash reserves to {self.constraints['min_cash_reserve']:.1%}"
            )
        
        # Check for position concentration
        for symbol, pos in self.positions.items():
            weight = pos.market_value / self.total_value
            if weight > self.constraints['max_position_size']:
                recommendations.append(
                    f"Reduce {symbol} position from {weight:.1%} to "
                    f"{self.constraints['max_position_size']:.1%}"
                )
        
        return recommendations

# Singleton
portfolio_manager = None

def get_portfolio_manager(logger: logging.Logger = None) -> InstitutionalPortfolioManager:
    """Get portfolio manager instance"""
    global portfolio_manager
    if portfolio_manager is None:
        if logger is None:
            logger = logging.getLogger('PortfolioManager')
        portfolio_manager = InstitutionalPortfolioManager(logger)
    return portfolio_manager
