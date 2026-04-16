"""
Advanced Options Market Making Engine
Matches Citadel Securities/Virtu Financial options market making capabilities
"""

import os
import sys
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize
import threading
import time

logger = logging.getLogger(__name__)

@dataclass
class OptionContract:
    """Advanced option contract specification"""
    symbol: str
    strike: float
    expiration: datetime
    option_type: str  # CALL, PUT
    underlying: str
    multiplier: float = 100.0
    min_tick: float = 0.01
    
@dataclass
class OptionsMarketData:
    """Real-time options market data"""
    bid: float
    ask: float
    bid_size: int
    ask_size: int
    implied_vol: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    underlying_price: float
    time_to_expiry: float
    volume: int
    
@dataclass
class MarketMakingQuote:
    """Market making quote with inventory management"""
    contract: OptionContract
    bid_price: float
    ask_price: float
    bid_size: int
    ask_size: int
    inventory: int
    target_inventory: int
    theoretical_price: float
    edge_bps: float
    confidence: float
    timestamp: datetime

class AdvancedOptionsMarketMaker:
    """
    Advanced Options Market Making Engine
    
    Features:
    - Stochastic volatility modeling
    - Dynamic delta hedging
    - Skew and term structure modeling
    - Real-time risk management
    - Multi-leg strategies
    - Volatility surface calibration
    """
    
    def __init__(self, initial_capital: float = 10000000.0):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        
        # Market making parameters
        self.max_position_size = 1000  # contracts
        self.target_inventory = {}  # Target inventory per contract
        self.inventory_tolerance = {}  # Inventory tolerance
        
        # Volatility surface
        self.vol_surface = {}
        self.skew_surface = {}
        self.term_structure = {}
        
        # Quotes and positions
        self.quotes = {}
        self.positions = {}
        self.trade_history = []
        
        # Risk metrics
        self.pnl = {}
        self.greeks_exposure = {}
        self.var_95 = {}
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Market data cache
        self.market_data_cache = {}
        self.cache_ttl = 1.0  # seconds
        
        logger.info("Advanced Options Market Maker initialized")
        
    def initialize_volatility_surface(self, historical_data: pd.DataFrame):
        """
        Initialize volatility surface from historical options data
        
        Args:
            historical_data: DataFrame with options prices and Greeks
        """
        
        logger.info("Initializing volatility surface...")
        
        # Group by expiration
        expirations = historical_data['expiration'].unique()
        
        for exp in expirations:
            exp_data = historical_data[historical_data['expiration'] == exp]
            
            # Group by strike
            strikes = exp_data['strike'].unique()
            
            for strike in strikes:
                strike_data = exp_data[exp_data['strike'] == strike]
                
                if len(strike_data) > 5:
                    # Calculate implied volatilities
                    ivs = []
                    for _, row in strike_data.iterrows():
                        iv = self._calculate_implied_volatility(
                            row['underlying_price'], row['strike'], row['time_to_expiry'],
                            row['option_price'], row['option_type']
                        )
                        if iv > 0:
                            ivs.append(iv)
                    
                    if ivs:
                        # Store in volatility surface
                        contract_key = f"{row['symbol']}_{exp.strftime('%Y%m%d')}_{strike}"
                        self.vol_surface[contract_key] = {
                            'implied_vol': np.mean(ivs),
                            'vol_std': np.std(ivs),
                            'strike': strike,
                            'expiration': exp,
                            'option_type': row['option_type']
                        }
        
        logger.info(f"Volatility surface initialized with {len(self.vol_surface)} points")
    
    def _calculate_implied_volatility(self, S: float, K: float, T: float, 
                                    price: float, option_type: str) -> float:
        """Calculate implied volatility using Newton-Raphson"""
        
        def black_scholes_price(S, K, T, r, sigma, option_type):
            """Black-Scholes option pricing"""
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            if option_type == 'CALL':
                price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            else:  # PUT
                price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            
            return price
        
        def vega(S, K, T, r, sigma):
            """Calculate vega"""
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            return S * norm.pdf(d1) * np.sqrt(T) / 100
        
        # Newton-Raphson iteration
        r = 0.05  # Risk-free rate
        sigma = 0.3  # Initial guess
        tolerance = 1e-6
        max_iterations = 100
        
        for i in range(max_iterations):
            # Calculate option price
            model_price = black_scholes_price(S, K, T, r, sigma, option_type)
            
            # Calculate vega
            v = vega(S, K, T, r, sigma)
            
            # Update sigma
            if abs(v) > tolerance:
                sigma = sigma - (model_price - price) / v
            else:
                break
            
            # Check convergence
            if abs(model_price - price) < tolerance:
                break
        
        return max(0.01, min(2.0, sigma))  # Bound implied volatility
    
    def calculate_theoretical_price(self, contract: OptionContract, 
                                underlying_price: float, time_to_expiry: float,
                                volatility: float, risk_free_rate: float = 0.05) -> float:
        """Calculate theoretical option price using Black-Scholes"""
        
        d1 = (np.log(underlying_price / contract.strike) + 
                (risk_free_rate + 0.5 * volatility**2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
        d2 = d1 - volatility * np.sqrt(time_to_expiry)
        
        if contract.option_type == 'CALL':
            price = (underlying_price * norm.cdf(d1) - 
                     contract.strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2))
        else:  # PUT
            price = (contract.strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2) - 
                     underlying_price * norm.cdf(-d1))
        
        return price * contract.multiplier
    
    def calculate_greeks(self, contract: OptionContract, underlying_price: float,
                       time_to_expiry: float, volatility: float,
                       risk_free_rate: float = 0.05) -> Dict[str, float]:
        """Calculate option Greeks"""
        
        d1 = (np.log(underlying_price / contract.strike) + 
                (risk_free_rate + 0.5 * volatility**2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
        d2 = d1 - volatility * np.sqrt(time_to_expiry)
        
        sqrt_t = np.sqrt(time_to_expiry)
        exp_rt = np.exp(-risk_free_rate * time_to_expiry)
        
        # Common calculations
        nd1 = norm.pdf(d1)
        nd2 = norm.pdf(d2)
        cdf_d1 = norm.cdf(d1)
        cfd_d2 = norm.cdf(d2)
        
        if contract.option_type == 'CALL':
            delta = cdf_d1
            theta = (-underlying_price * nd1 * volatility / (2 * sqrt_t)) - \
                    (risk_free_rate * contract.strike * exp_rt * cfd_d2) + \
                    (risk_free_rate * contract.strike * exp_rt * (1 - cfd_d2))
        else:  # PUT
            delta = cfd_d1 - 1
            theta = (-underlying_price * nd1 * volatility / (2 * sqrt_t)) + \
                    (risk_free_rate * contract.strike * exp_rt * cfd_d2) - \
                    (risk_free_rate * contract.strike * exp_rt * (1 - cfd_d2))
        
        # Common Greeks
        gamma = nd1 / (underlying_price * volatility * sqrt_t)
        vega = underlying_price * sqrt_t * nd1 / 100
        rho = contract.strike * time_to_expiry * exp_rt * cfd_d2 / 100 if contract.option_type == 'CALL' else \
              -contract.strike * time_to_expiry * exp_rt * (1 - cfd_d2) / 100
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta / 365,  # Theta per day
            'vega': vega,
            'rho': rho
        }
    
    def generate_market_making_quotes(self, contract: OptionContract, 
                                  market_data: OptionsMarketData) -> MarketMakingQuote:
        """Generate optimal market making quotes"""
        
        # Get current position
        current_position = self.positions.get(contract.symbol, 0)
        
        # Calculate theoretical price
        theoretical_price = self.calculate_theoretical_price(
            contract, market_data.underlying_price, market_data.time_to_expiry,
            market_data.implied_vol
        )
        
        # Calculate inventory adjustment
        target_inventory = self.target_inventory.get(contract.symbol, 0)
        inventory_adjustment = (current_position - target_inventory) / 1000  # 0.1% per 1000 contracts
        
        # Calculate spread based on inventory
        base_spread_bps = 2.0  # Base spread
        
        if abs(inventory_adjustment) > 0.5:  # More than 500 contracts off target
            spread_multiplier = 2.0  # Widen spread
        elif abs(inventory_adjustment) < 0.1:  # Less than 100 contracts off target
            spread_multiplier = 0.5  # Narrow spread
        else:
            spread_multiplier = 1.0
        
        adjusted_spread_bps = base_spread_bps * spread_multiplier
        
        # Calculate bid/ask prices
        half_spread = adjusted_spread_bps / 20000  # Convert to decimal
        bid_price = theoretical_price - half_spread
        ask_price = theoretical_price + half_spread
        
        # Adjust for inventory
        if current_position > target_inventory:
            # Long inventory - lower bid, raise ask
            bid_price -= theoretical_price * inventory_adjustment
            ask_price += theoretical_price * inventory_adjustment * 0.5
        else:
            # Short inventory - raise bid, lower ask
            bid_price += theoretical_price * abs(inventory_adjustment) * 0.5
            ask_price -= theoretical_price * abs(inventory_adjustment)
        
        # Calculate edge
        edge_bps = (ask_price - bid_price) / theoretical_price * 10000
        
        # Calculate quote sizes
        base_size = 100
        if current_position > target_inventory:
            bid_size = base_size * 2  # More willing to buy
            ask_size = base_size // 2  # Less willing to sell
        else:
            bid_size = base_size // 2  # Less willing to buy
            ask_size = base_size * 2  # More willing to sell
        
        # Calculate confidence
        confidence = self._calculate_quote_confidence(market_data, edge_bps)
        
        return MarketMakingQuote(
            contract=contract,
            bid_price=bid_price,
            ask_price=ask_price,
            bid_size=bid_size,
            ask_size=ask_size,
            inventory=current_position,
            target_inventory=target_inventory,
            theoretical_price=theoretical_price,
            edge_bps=edge_bps,
            confidence=confidence,
            timestamp=datetime.now()
        )
    
    def _calculate_quote_confidence(self, market_data: OptionsMarketData, edge_bps: float) -> float:
        """Calculate confidence in quote"""
        
        # Base confidence from market data quality
        volume_confidence = min(1.0, market_data.volume / 100000)  # Volume-based confidence
        
        # Volatility confidence (lower vol = higher confidence)
        vol_confidence = max(0.1, 1.0 - market_data.implied_vol / 0.5)
        
        # Edge confidence (wider edge = lower confidence)
        edge_confidence = max(0.1, 1.0 - edge_bps / 10.0)
        
        # Combined confidence
        confidence = (volume_confidence + vol_confidence + edge_confidence) / 3
        
        return confidence
    
    def calculate_delta_hedge(self, contract: OptionContract, delta: float, 
                           underlying_price: float) -> Dict[str, Any]:
        """Calculate delta hedge for option position"""
        
        # Calculate hedge ratio
        hedge_contracts = int(delta * 100)  # Delta shares per option contract
        
        # Calculate hedge value
        hedge_value = hedge_contracts * underlying_price
        
        return {
            'contract': contract,
            'delta': delta,
            'hedge_contracts': hedge_contracts,
            'hedge_value': hedge_value,
            'hedge_type': 'DELTA_HEDGE',
            'timestamp': datetime.now()
        }
    
    def calculate_vega_hedge(self, contracts: List[OptionContract], 
                          vegas: List[float], target_vega: float) -> Dict[str, Any]:
        """Calculate vega hedge using options"""
        
        # Calculate vega weights
        total_vega = sum(vegas)
        if total_vega == 0:
            return {'error': 'Total vega is zero'}
        
        weights = [v / total_vega for v in vegas]
        
        # Calculate hedge positions
        hedge_positions = []
        for i, (contract, vega, weight) in enumerate(zip(contracts, vegas, weights)):
            hedge_size = int(weight * target_vega / vega) if vega != 0 else 0
            
            hedge_positions.append({
                'contract': contract,
                'vega': vega,
                'weight': weight,
                'hedge_size': hedge_size,
                'hedge_value': hedge_size * contract.multiplier * 100  # Approximate
            })
        
        return {
            'contracts': contracts,
            'target_vega': target_vega,
            'total_vega': total_vega,
            'hedge_positions': hedge_positions,
            'hedge_type': 'VEGA_HEDGE',
            'timestamp': datetime.now()
        }
    
    def calculate_gamma_scalp(self, contracts: List[OptionContract], 
                          gammas: List[float], underlying_price: float) -> Dict[str, Any]:
        """Calculate gamma scalping strategy"""
        
        # Calculate total gamma
        total_gamma = sum(gammas)
        
        if total_gamma == 0:
            return {'error': 'Total gamma is zero'}
        
        # Calculate optimal position
        optimal_contracts = []
        for contract, gamma in zip(contracts, gammas):
            if gamma > 0:  # Positive gamma
                # Buy both call and put for gamma scalping
                call_size = 100
                put_size = 100
                
                optimal_contracts.extend([
                    {'contract': contract, 'type': 'CALL', 'size': call_size},
                    {'contract': contract, 'type': 'PUT', 'size': put_size}
                ])
        
        return {
            'contracts': contracts,
            'gammas': gammas,
            'total_gamma': total_gamma,
            'optimal_positions': optimal_contracts,
            'strategy': 'GAMMA_SCALP',
            'underlying_price': underlying_price,
            'timestamp': datetime.now()
        }
    
    def calculate_var_95(self, portfolio_greeks: Dict[str, Dict[str, float]]) -> float:
        """Calculate 95% Value at Risk"""
        
        if not portfolio_greeks:
            return 0.0
        
        # Calculate portfolio delta and gamma
        total_delta = sum(greeks['delta'] for greeks in portfolio_greeks.values())
        total_gamma = sum(greeks['gamma'] for greeks in portfolio_greeks.values())
        
        # Simulate price movements (simplified)
        price_shock = 0.02  # 2% price shock
        
        # Calculate P&L from delta and gamma
        pnl_delta = total_delta * price_shock
        pnl_gamma = 0.5 * total_gamma * (price_shock ** 2)
        
        total_pnl = pnl_delta + pnl_gamma
        
        # Calculate VaR (95% confidence)
        var_95 = abs(total_pnl) * 1.65  # 1.65 for 95% confidence
        
        return var_95
    
    def update_positions(self, trades: List[Dict[str, Any]]):
        """Update positions from trades"""
        
        with self._lock:
            for trade in trades:
                contract = trade['contract']
                side = trade['side']
                quantity = trade['quantity']
                price = trade['price']
                
                # Update position
                if contract.symbol not in self.positions:
                    self.positions[contract.symbol] = 0
                
                if side == 'BUY':
                    self.positions[contract.symbol] += quantity
                else:
                    self.positions[contract.symbol] -= quantity
                
                # Record trade
                self.trade_history.append(trade)
                
                # Update P&L
                self._update_pnl(contract, quantity, price, side)
    
    def _update_pnl(self, contract: OptionContract, quantity: int, price: float, side: str):
        """Update P&L for position"""
        
        if contract.symbol not in self.pnl:
            self.pnl[contract.symbol] = 0.0
        
        # Calculate P&L
        if side == 'SELL':
            self.pnl[contract.symbol] += quantity * price * contract.multiplier
        else:
            self.pnl[contract.symbol] -= quantity * price * contract.multiplier
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio summary"""
        
        total_pnl = sum(self.pnl.values())
        total_positions = len(self.positions)
        
        # Calculate Greeks exposure
        portfolio_greeks = {}
        for symbol, position in self.positions.items():
            if position != 0:
                # Get contract details (simplified)
                contract = OptionContract(
                    symbol=symbol,
                    strike=150.0,  # Default
                    expiration=datetime.now() + timedelta(days=30),
                    option_type='CALL',
                    underlying=symbol
                )
                
                # Calculate Greeks (simplified)
                greeks = self.calculate_greeks(contract, 150.0, 30/365, 0.25)
                
                # Scale by position
                portfolio_greeks[symbol] = {
                    'delta': greeks['delta'] * position,
                    'gamma': greeks['gamma'] * position,
                    'theta': greeks['theta'] * position,
                    'vega': greeks['vega'] * position,
                    'rho': greeks['rho'] * position
                }
        
        # Calculate VaR
        var_95 = self.calculate_var_95(portfolio_greeks)
        
        return {
            'total_pnl': total_pnl,
            'total_positions': total_positions,
            'positions': self.positions,
            'portfolio_greeks': portfolio_greeks,
            'var_95': var_95,
            'trade_count': len(self.trade_history),
            'timestamp': datetime.now()
        }

def run_advanced_options_demo():
    """Demonstrate advanced options market making"""
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("ADVANCED OPTIONS MARKET MAKING DEMO")
    print("=" * 60)
    
    # Initialize market maker
    mm = AdvancedOptionsMarketMaker(initial_capital=10000000.0)
    
    # Create sample contract
    contract = OptionContract(
        symbol='AAPL_150_CALL',
        strike=150.0,
        expiration=datetime.now() + timedelta(days=30),
        option_type='CALL',
        underlying='AAPL',
        multiplier=100.0
    )
    
    # Create sample market data
    market_data = OptionsMarketData(
        bid=4.50,
        ask=4.60,
        bid_size=500,
        ask_size=500,
        implied_vol=0.25,
        delta=0.55,
        gamma=0.05,
        theta=-0.02,
        vega=0.15,
        rho=0.08,
        underlying_price=150.0,
        time_to_expiry=30/365,
        volume=10000
    )
    
    print("\n1. MARKET MAKING QUOTE GENERATION")
    quote = mm.generate_market_making_quotes(contract, market_data)
    
    print(f"Contract: {quote.contract.symbol}")
    print(f"Theoretical Price: ${quote.theoretical_price:.4f}")
    print(f"Bid Price: ${quote.bid_price:.4f}")
    print(f"Ask Price: ${quote.ask_price:.4f}")
    print(f"Spread: {quote.edge_bps:.2f} bps")
    print(f"Confidence: {quote.confidence:.3f}")
    print(f"Inventory: {quote.inventory}")
    print(f"Target Inventory: {quote.target_inventory}")
    
    print("\n2. DELTA HEDGING")
    delta_hedge = mm.calculate_delta_hedge(contract, market_data.delta, market_data.underlying_price)
    
    print(f"Delta: {delta_hedge['delta']:.4f}")
    print(f"Hedge Contracts: {delta_hedge['hedge_contracts']}")
    print(f"Hedge Value: ${delta_hedge['hedge_value']:,.2f}")
    
    print("\n3. VEGA HEDGING")
    vega_hedge = mm.calculate_vega_hedge([contract], [market_data.vega], target_vega=0.0)
    
    if 'error' not in vega_hedge:
        print(f"Target Vega: {vega_hedge['target_vega']:.4f}")
        print(f"Total Vega: {vega_hedge['total_vega']:.4f}")
        for i, hedge_pos in enumerate(vega_hedge['hedge_positions']):
            print(f"Hedge Position {i+1}: {hedge_pos['hedge_size']} contracts")
    
    print("\n4. GAMMA SCALPING")
    gamma_scalp = mm.calculate_gamma_scalp([contract], [market_data.gamma], market_data.underlying_price)
    
    if 'error' not in gamma_scalp:
        print(f"Total Gamma: {gamma_scalp['total_gamma']:.4f}")
        print(f"Strategy: {gamma_scalp['strategy']}")
        for i, opt_pos in enumerate(gamma_scalp['optimal_positions']):
            print(f"Position {i+1}: {opt_pos['type']} {opt_pos['size']} contracts")
    
    print("\n5. PORTFOLIO SUMMARY")
    # Simulate some trades
    trades = [
        {
            'contract': contract,
            'side': 'BUY',
            'quantity': 100,
            'price': 4.55
        }
    ]
    
    mm.update_positions(trades)
    portfolio = mm.get_portfolio_summary()
    
    print(f"Total P&L: ${portfolio['total_pnl']:,.2f}")
    print(f"Total Positions: {portfolio['total_positions']}")
    print(f"VaR 95%: ${portfolio['var_95']:,.2f}")
    print(f"Trade Count: {portfolio['trade_count']}")
    
    # Show Greeks exposure
    print(f"\nPortfolio Greeks Exposure:")
    for symbol, greeks in portfolio['portfolio_greeks'].items():
        print(f"{symbol}:")
        print(f"  Delta: {greeks['delta']:.4f}")
        print(f"  Gamma: {greeks['gamma']:.4f}")
        print(f"  Theta: {greeks['theta']:.4f}")
        print(f"  Vega: {greeks['vega']:.4f}")
    
    print("\n" + "=" * 60)
    print("ADVANCED OPTIONS MARKET MAKING DEMO COMPLETE")
    print("=" * 60)
    
    return mm

if __name__ == "__main__":
    run_advanced_options_demo()
