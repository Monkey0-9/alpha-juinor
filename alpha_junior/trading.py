#!/usr/bin/env python3
"""
Alpha Junior - Alpaca Trading Module
Paper Trading Integration for High Returns
"""

from flask import Blueprint, request, jsonify
import requests
import json
from datetime import datetime, timedelta
import sqlite3
import os
from typing import Dict, List, Optional

# Alpaca API Configuration
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"  # Paper trading URL
ALPACA_DATA_URL = "https://data.alpaca.markets"

# Blueprint for trading routes
trading_bp = Blueprint('trading', __name__, url_prefix='/api/trading')

def get_alpaca_credentials():
    """Get Alpaca API credentials from environment"""
    api_key = os.getenv('ALPACA_API_KEY', '')
    secret_key = os.getenv('ALPACA_SECRET_KEY', '')
    return api_key, secret_key

def alpaca_headers():
    """Generate Alpaca API headers"""
    api_key, secret_key = get_alpaca_credentials()
    return {
        'APCA-API-KEY-ID': api_key,
        'APCA-API-SECRET-KEY': secret_key,
        'Content-Type': 'application/json'
    }

# ==================== ALPACA ACCOUNT & BALANCE ====================

@trading_bp.route('/account', methods=['GET'])
def get_account():
    """Get Alpaca paper trading account info"""
    try:
        response = requests.get(
            f'{ALPACA_BASE_URL}/v2/account',
            headers=alpaca_headers()
        )
        
        if response.status_code == 200:
            account = response.json()
            return jsonify({
                'success': True,
                'account': {
                    'id': account.get('id'),
                    'account_number': account.get('account_number'),
                    'status': account.get('status'),
                    'currency': account.get('currency'),
                    'cash': float(account.get('cash', 0)),
                    'portfolio_value': float(account.get('portfolio_value', 0)),
                    'equity': float(account.get('equity', 0)),
                    'buying_power': float(account.get('buying_power', 0)),
                    'daytrading_buying_power': float(account.get('daytrading_buying_power', 0)),
                    'initial_margin': float(account.get('initial_margin', 0)),
                    'maintenance_margin': float(account.get('maintenance_margin', 0)),
                    'last_equity': float(account.get('last_equity', 0)),
                    'long_market_value': float(account.get('long_market_value', 0)),
                    'short_market_value': float(account.get('short_market_value', 0)),
                }
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Alpaca API error: {response.status_code}',
                'details': response.text
            }), 400
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ==================== POSITIONS & HOLDINGS ====================

@trading_bp.route('/positions', methods=['GET'])
def get_positions():
    """Get all open positions"""
    try:
        response = requests.get(
            f'{ALPACA_BASE_URL}/v2/positions',
            headers=alpaca_headers()
        )
        
        if response.status_code == 200:
            positions = response.json()
            formatted = []
            for pos in positions:
                formatted.append({
                    'symbol': pos.get('symbol'),
                    'qty': float(pos.get('qty', 0)),
                    'avg_entry_price': float(pos.get('avg_entry_price', 0)),
                    'current_price': float(pos.get('current_price', 0)),
                    'market_value': float(pos.get('market_value', 0)),
                    'unrealized_pl': float(pos.get('unrealized_pl', 0)),
                    'unrealized_plpc': float(pos.get('unrealized_plpc', 0)),
                    'side': pos.get('side'),
                    'change_today': float(pos.get('change_today', 0)),
                })
            
            return jsonify({
                'success': True,
                'positions': formatted,
                'count': len(formatted)
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Alpaca API error: {response.status_code}'
            }), 400
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@trading_bp.route('/position/<symbol>', methods=['DELETE'])
def close_position(symbol):
    """Close a specific position"""
    try:
        response = requests.delete(
            f'{ALPACA_BASE_URL}/v2/positions/{symbol}',
            headers=alpaca_headers()
        )
        
        if response.status_code in [200, 204]:
            return jsonify({
                'success': True,
                'message': f'Position {symbol} closed successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Failed to close position: {response.text}'
            }), 400
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ==================== ORDERS & TRADING ====================

@trading_bp.route('/orders', methods=['GET'])
def get_orders():
    """Get all orders (open and closed)"""
    try:
        status = request.args.get('status', 'all')
        
        params = {}
        if status != 'all':
            params['status'] = status
        
        response = requests.get(
            f'{ALPACA_BASE_URL}/v2/orders',
            headers=alpaca_headers(),
            params=params
        )
        
        if response.status_code == 200:
            orders = response.json()
            formatted = []
            for order in orders:
                formatted.append({
                    'id': order.get('id'),
                    'symbol': order.get('symbol'),
                    'qty': float(order.get('qty', 0)),
                    'side': order.get('side'),
                    'type': order.get('type'),
                    'status': order.get('status'),
                    'filled_qty': float(order.get('filled_qty', 0)),
                    'filled_avg_price': float(order.get('filled_avg_price', 0)) if order.get('filled_avg_price') else None,
                    'created_at': order.get('created_at'),
                    'updated_at': order.get('updated_at'),
                })
            
            return jsonify({
                'success': True,
                'orders': formatted,
                'count': len(formatted)
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Alpaca API error: {response.status_code}'
            }), 400
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@trading_bp.route('/order', methods=['POST'])
def place_order():
    """Place a new order"""
    try:
        data = request.get_json()
        
        # Required fields
        symbol = data.get('symbol')
        qty = data.get('qty')
        side = data.get('side')  # buy or sell
        order_type = data.get('type', 'market')  # market, limit, stop, stop_limit
        time_in_force = data.get('time_in_force', 'day')  # day, gtc, opg, cls, ioc, fok
        
        if not symbol or not qty or not side:
            return jsonify({
                'success': False,
                'error': 'Missing required fields: symbol, qty, side'
            }), 400
        
        order_data = {
            'symbol': symbol.upper(),
            'qty': str(qty),
            'side': side.lower(),
            'type': order_type,
            'time_in_force': time_in_force
        }
        
        # Add limit price if applicable
        if order_type in ['limit', 'stop_limit']:
            limit_price = data.get('limit_price')
            if limit_price:
                order_data['limit_price'] = str(limit_price)
        
        # Add stop price if applicable
        if order_type in ['stop', 'stop_limit']:
            stop_price = data.get('stop_price')
            if stop_price:
                order_data['stop_price'] = str(stop_price)
        
        response = requests.post(
            f'{ALPACA_BASE_URL}/v2/orders',
            headers=alpaca_headers(),
            json=order_data
        )
        
        if response.status_code in [200, 201]:
            order = response.json()
            return jsonify({
                'success': True,
                'order': {
                    'id': order.get('id'),
                    'symbol': order.get('symbol'),
                    'qty': float(order.get('qty', 0)),
                    'side': order.get('side'),
                    'type': order.get('type'),
                    'status': order.get('status'),
                    'created_at': order.get('created_at'),
                },
                'message': f'{side.upper()} order placed for {qty} shares of {symbol}'
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Order failed: {response.text}'
            }), 400
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@trading_bp.route('/order/<order_id>', methods=['DELETE'])
def cancel_order(order_id):
    """Cancel an open order"""
    try:
        response = requests.delete(
            f'{ALPACA_BASE_URL}/v2/orders/{order_id}',
            headers=alpaca_headers()
        )
        
        if response.status_code in [200, 204]:
            return jsonify({
                'success': True,
                'message': f'Order {order_id} cancelled successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Failed to cancel order: {response.text}'
            }), 400
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ==================== MARKET DATA ====================

@trading_bp.route('/quote/<symbol>', methods=['GET'])
def get_quote(symbol):
    """Get latest quote for a symbol"""
    try:
        response = requests.get(
            f'{ALPACA_DATA_URL}/v2/stocks/{symbol.upper()}/quotes/latest',
            headers=alpaca_headers()
        )
        
        if response.status_code == 200:
            data = response.json()
            quote = data.get('quote', {})
            return jsonify({
                'success': True,
                'symbol': symbol.upper(),
                'quote': {
                    'ask_price': float(quote.get('ap', 0)),
                    'ask_size': float(quote.get('as', 0)),
                    'bid_price': float(quote.get('bp', 0)),
                    'bid_size': float(quote.get('bs', 0)),
                    'timestamp': quote.get('t'),
                }
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Failed to get quote: {response.status_code}'
            }), 400
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@trading_bp.route('/bars/<symbol>', methods=['GET'])
def get_bars(symbol):
    """Get historical price bars"""
    try:
        timeframe = request.args.get('timeframe', '1Day')  # 1Min, 5Min, 15Min, 1Hour, 1Day
        limit = request.args.get('limit', 100)
        
        response = requests.get(
            f'{ALPACA_DATA_URL}/v2/stocks/{symbol.upper()}/bars',
            headers=alpaca_headers(),
            params={
                'timeframe': timeframe,
                'limit': limit,
                'adjustment': 'raw'
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            bars = data.get('bars', [])
            return jsonify({
                'success': True,
                'symbol': symbol.upper(),
                'timeframe': timeframe,
                'bars': [
                    {
                        'timestamp': bar.get('t'),
                        'open': float(bar.get('o', 0)),
                        'high': float(bar.get('h', 0)),
                        'low': float(bar.get('l', 0)),
                        'close': float(bar.get('c', 0)),
                        'volume': int(bar.get('v', 0)),
                        'vwap': float(bar.get('vw', 0)) if bar.get('vw') else None,
                    }
                    for bar in bars
                ]
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Failed to get bars: {response.status_code}'
            }), 400
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ==================== TRADING STRATEGY ====================

@trading_bp.route('/strategy/execute', methods=['POST'])
def execute_strategy():
    """Execute automated trading strategy"""
    try:
        data = request.get_json()
        strategy_type = data.get('strategy', 'momentum')  # momentum, breakout, mean_reversion
        
        # Get watchlist symbols
        symbols = data.get('symbols', ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'])
        
        results = {
            'strategy': strategy_type,
            'timestamp': datetime.now().isoformat(),
            'signals': [],
            'orders_placed': []
        }
        
        # Simple momentum strategy
        for symbol in symbols:
            try:
                # Get recent bars
                response = requests.get(
                    f'{ALPACA_DATA_URL}/v2/stocks/{symbol}/bars',
                    headers=alpaca_headers(),
                    params={'timeframe': '1Day', 'limit': 20}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    bars = data.get('bars', [])
                    
                    if len(bars) >= 10:
                        closes = [float(bar['c']) for bar in bars[-10:]]
                        
                        # Calculate momentum (price change %)
                        momentum = ((closes[-1] - closes[0]) / closes[0]) * 100
                        
                        # Calculate RSI (simplified)
                        gains = []
                        losses = []
                        for i in range(1, len(closes)):
                            change = closes[i] - closes[i-1]
                            if change > 0:
                                gains.append(change)
                                losses.append(0)
                            else:
                                gains.append(0)
                                losses.append(abs(change))
                        
                        avg_gain = sum(gains) / len(gains) if gains else 0
                        avg_loss = sum(losses) / len(losses) if losses else 0.001
                        rs = avg_gain / avg_loss
                        rsi = 100 - (100 / (1 + rs))
                        
                        signal = 'neutral'
                        if momentum > 5 and rsi < 70:
                            signal = 'buy'
                        elif momentum < -5 or rsi > 80:
                            signal = 'sell'
                        
                        results['signals'].append({
                            'symbol': symbol,
                            'current_price': closes[-1],
                            'momentum_pct': round(momentum, 2),
                            'rsi': round(rsi, 2),
                            'signal': signal
                        })
                        
                        # Auto-trade if signal is strong
                        if signal == 'buy' and data.get('auto_trade', False):
                            order_response = requests.post(
                                f'{ALPACA_BASE_URL}/v2/orders',
                                headers=alpaca_headers(),
                                json={
                                    'symbol': symbol,
                                    'qty': '10',
                                    'side': 'buy',
                                    'type': 'market',
                                    'time_in_force': 'day'
                                }
                            )
                            if order_response.status_code in [200, 201]:
                                results['orders_placed'].append({
                                    'symbol': symbol,
                                    'side': 'buy',
                                    'qty': 10
                                })
                                
            except Exception as e:
                results['signals'].append({
                    'symbol': symbol,
                    'error': str(e)
                })
        
        return jsonify({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ==================== PORTFOLIO ANALYTICS ====================

@trading_bp.route('/portfolio/performance', methods=['GET'])
def get_portfolio_performance():
    """Get portfolio performance metrics"""
    try:
        # Get account info
        account_resp = requests.get(
            f'{ALPACA_BASE_URL}/v2/account',
            headers=alpaca_headers()
        )
        
        if account_resp.status_code != 200:
            return jsonify({'success': False, 'error': 'Failed to get account'}), 400
        
        account = account_resp.json()
        
        # Get positions
        positions_resp = requests.get(
            f'{ALPACA_BASE_URL}/v2/positions',
            headers=alpaca_headers()
        )
        
        positions = positions_resp.json() if positions_resp.status_code == 200 else []
        
        # Calculate metrics
        portfolio_value = float(account.get('portfolio_value', 0))
        last_equity = float(account.get('last_equity', portfolio_value))
        cash = float(account.get('cash', 0))
        
        # Daily return
        daily_return = 0
        daily_return_pct = 0
        if last_equity > 0:
            daily_return = portfolio_value - last_equity
            daily_return_pct = (daily_return / last_equity) * 100
        
        # Total unrealized P&L
        total_unrealized_pl = sum(float(p.get('unrealized_pl', 0)) for p in positions)
        
        # Position breakdown
        position_values = []
        for pos in positions:
            position_values.append({
                'symbol': pos.get('symbol'),
                'market_value': float(pos.get('market_value', 0)),
                'unrealized_pl': float(pos.get('unrealized_pl', 0)),
                'unrealized_plpc': float(pos.get('unrealized_plpc', 0)),
                'pct_of_portfolio': 0  # Will calculate below
            })
        
        # Calculate portfolio percentages
        if portfolio_value > 0:
            for pos in position_values:
                pos['pct_of_portfolio'] = round((pos['market_value'] / portfolio_value) * 100, 2)
        
        return jsonify({
            'success': True,
            'performance': {
                'portfolio_value': portfolio_value,
                'cash': cash,
                'market_value': portfolio_value - cash,
                'daily_return': round(daily_return, 2),
                'daily_return_pct': round(daily_return_pct, 2),
                'total_unrealized_pl': round(total_unrealized_pl, 2),
                'position_count': len(positions),
                'positions': position_values,
                'timestamp': datetime.now().isoformat()
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
