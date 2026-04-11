#!/usr/bin/env python3
"""
INSTITUTIONAL DATA FEEDS FOR TOP 1% TRADING
==========================================

Connect to real institutional data sources:
- Bloomberg Terminal API
- Refinitiv Eikon
- Direct exchange feeds
- Real-time market data
- Alternative data sources
"""

import asyncio
import time
import json
import os
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import requests
import websocket
import threading
from queue import Queue, Empty
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class InstitutionalDataSource:
    """Institutional data source configuration"""
    name: str
    provider: str
    data_type: str  # market_data, fundamental, alternative, news
    latency_ms: float
    coverage: List[str]
    
    # Connection details
    api_endpoint: str = ""
    websocket_endpoint: str = ""
    api_key: str = ""
    credentials: Dict[str, str] = field(default_factory=dict)
    
    # Status
    is_connected: bool = False
    last_update: datetime = field(default_factory=datetime.utcnow)
    error_count: int = 0
    data_points_received: int = 0


@dataclass
class MarketDataPoint:
    """Real-time market data point"""
    symbol: str
    timestamp: datetime
    price: float
    bid: float
    ask: float
    volume: int
    exchange: str
    source: str
    
    # Additional fields
    bid_size: int = 0
    ask_size: int = 0
    high: float = 0.0
    low: float = 0.0
    open: float = 0.0
    close: float = 0.0


class InstitutionalDataFeeds:
    """
    Connect to institutional data sources for top 1% trading.
    
    This provides real institutional-grade data feeds.
    """
    
    def __init__(self):
        self.data_sources: Dict[str, InstitutionalDataSource] = {}
        self.data_streams: Dict[str, Queue] = {}
        self.websocket_connections: Dict[str, Any] = {}
        
        # Performance metrics
        self.metrics = {
            'total_data_points': 0,
            'average_latency_ms': 0.0,
            'connected_sources': 0,
            'error_rate': 0.0
        }
        
        # Initialize data sources
        self._initialize_data_sources()
        
        logger.info("Institutional Data Feeds initialized")
    
    def _initialize_data_sources(self):
        """Initialize institutional data sources"""
        
        # Bloomberg Terminal API
        self.data_sources['bloomberg'] = InstitutionalDataSource(
            name='Bloomberg Terminal',
            provider='Bloomberg',
            data_type='market_data',
            latency_ms=1.0,
            coverage=['equities', 'fixed_income', 'fx', 'commodities', 'derivatives'],
            api_endpoint='https://api.bloomberg.com/v1',
            websocket_endpoint='wss://api.bloomberg.com/ws',
            credentials={
                'terminal_id': os.getenv('BLOOMBERG_TERMINAL_ID', ''),
                'api_key': os.getenv('BLOOMBERG_API_KEY', ''),
                'auth_token': os.getenv('BLOOMBERG_AUTH_TOKEN', '')
            }
        )
        
        # Refinitiv Eikon
        self.data_sources['refinitiv'] = InstitutionalDataSource(
            name='Refinitiv Eikon',
            provider='Refinitiv',
            data_type='market_data',
            latency_ms=2.0,
            coverage=['equities', 'fixed_income', 'fx', 'commodities', 'news'],
            api_endpoint='https://api.refinitiv.com/v1',
            websocket_endpoint='wss://api.refinitiv.com/ws',
            credentials={
                'app_key': os.getenv('REFINITIV_APP_KEY', ''),
                'username': os.getenv('REFINITIV_USERNAME', ''),
                'password': os.getenv('REFINITIV_PASSWORD', '')
            }
        )
        
        # Direct NYSE Feed
        self.data_sources['nyse'] = InstitutionalDataSource(
            name='NYSE Direct Feed',
            provider='NYSE',
            data_type='market_data',
            latency_ms=0.5,
            coverage=['equities', 'options'],
            websocket_endpoint='wss://feed.nyse.com/marketdata',
            credentials={
                'client_id': os.getenv('NYSE_CLIENT_ID', ''),
                'client_secret': os.getenv('NYSE_CLIENT_SECRET', '')
            }
        )
        
        # NASDAQ Direct Feed
        self.data_sources['nasdaq'] = InstitutionalDataSource(
            name='NASDAQ Direct Feed',
            provider='NASDAQ',
            data_type='market_data',
            latency_ms=0.3,
            coverage=['equities', 'options'],
            websocket_endpoint='wss://feed.nasdaq.com/marketdata',
            credentials={
                'client_id': os.getenv('NASDAQ_CLIENT_ID', ''),
                'client_secret': os.getenv('NASDAQ_CLIENT_SECRET', '')
            }
        )
        
        # CME Direct Feed
        self.data_sources['cme'] = InstitutionalDataSource(
            name='CME Direct Feed',
            provider='CME',
            data_type='market_data',
            latency_ms=0.8,
            coverage=['futures', 'options', 'fx'],
            websocket_endpoint='wss://feed.cmegroup.com/marketdata',
            credentials={
                'client_id': os.getenv('CME_CLIENT_ID', ''),
                'client_secret': os.getenv('CME_CLIENT_SECRET', '')
            }
        )
        
        # ICE Direct Feed
        self.data_sources['ice'] = InstitutionalDataSource(
            name='ICE Direct Feed',
            provider='ICE',
            data_type='market_data',
            latency_ms=0.6,
            coverage=['futures', 'options', 'fx'],
            websocket_endpoint='wss://feed.theice.com/marketdata',
            credentials={
                'client_id': os.getenv('ICE_CLIENT_ID', ''),
                'client_secret': os.getenv('ICE_CLIENT_SECRET', '')
            }
        )
        
        # Alternative Data: Satellite Imagery
        self.data_sources['satellite'] = InstitutionalDataSource(
            name='Satellite Imagery',
            provider='Planet Labs',
            data_type='alternative',
            latency_ms=3600000,  # 1 hour
            coverage=['satellite_imagery', 'geospatial'],
            api_endpoint='https://api.planet.com/v1',
            credentials={
                'api_key': os.getenv('PLANET_API_KEY', '')
            }
        )
        
        # Alternative Data: Credit Card Transactions
        self.data_sources['credit_cards'] = InstitutionalDataSource(
            name='Credit Card Data',
            provider='Yodlee',
            data_type='alternative',
            latency_ms=86400000,  # 1 day
            coverage=['consumer_spending', 'credit_card_transactions'],
            api_endpoint='https://api.yodlee.com/v1',
            credentials={
                'api_key': os.getenv('YODLEE_API_KEY', ''),
                'consumer_key': os.getenv('YODLEE_CONSUMER_KEY', '')
            }
        )
        
        # Alternative Data: Supply Chain
        self.data_sources['supply_chain'] = InstitutionalDataSource(
            name='Supply Chain Data',
            provider='Project44',
            data_type='alternative',
            latency_ms=300000,  # 5 minutes
            coverage=['logistics', 'supply_chain', 'shipping'],
            api_endpoint='https://api.project44.com/v1',
            credentials={
                'api_key': os.getenv('PROJECT44_API_KEY', ''),
                'client_id': os.getenv('PROJECT44_CLIENT_ID', '')
            }
        )
        
        # News: Dow Jones
        self.data_sources['dowjones'] = InstitutionalDataSource(
            name='Dow Jones News',
            provider='Dow Jones',
            data_type='news',
            latency_ms=5000,  # 5 seconds
            coverage=['news', 'headlines', 'sentiment'],
            api_endpoint='https://api.dowjones.com/v1',
            websocket_endpoint='wss://api.dowjones.com/ws',
            credentials={
                'api_key': os.getenv('DOWJONES_API_KEY', ''),
                'partner_id': os.getenv('DOWJONES_PARTNER_ID', '')
            }
        )
        
        logger.info(f"Initialized {len(self.data_sources)} institutional data sources")
    
    async def connect_all_sources(self) -> Dict[str, Any]:
        """Connect to all institutional data sources"""
        try:
            logger.info("Connecting to all institutional data sources")
            
            results = {}
            
            for source_name, source in self.data_sources.items():
                result = await self._connect_source(source_name)
                results[source_name] = result
                
                if result.get('success'):
                    source.is_connected = True
                    self.metrics['connected_sources'] += 1
                else:
                    source.error_count += 1
            
            logger.info(f"Connected to {self.metrics['connected_sources']}/{len(self.data_sources)} data sources")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to connect to data sources: {e}")
            return {'error': str(e)}
    
    async def _connect_source(self, source_name: str) -> Dict[str, Any]:
        """Connect to individual data source"""
        try:
            source = self.data_sources.get(source_name)
            if not source:
                return {'error': f'Source {source_name} not found'}
            
            logger.info(f"Connecting to {source.name}")
            
            if source.data_type == 'market_data':
                return await self._connect_market_data_source(source)
            elif source.data_type == 'alternative':
                return await self._connect_alternative_data_source(source)
            elif source.data_type == 'news':
                return await self._connect_news_source(source)
            else:
                return {'error': f'Unsupported data type: {source.data_type}'}
                
        except Exception as e:
            logger.error(f"Failed to connect to {source_name}: {e}")
            return {'error': str(e)}
    
    async def _connect_market_data_source(self, source: InstitutionalDataSource) -> Dict[str, Any]:
        """Connect to market data source"""
        try:
            if source.provider == 'Bloomberg':
                return await self._connect_bloomberg(source)
            elif source.provider == 'Refinitiv':
                return await self._connect_refinitiv(source)
            elif source.provider in ['NYSE', 'NASDAQ', 'CME', 'ICE']:
                return await self._connect_exchange_feed(source)
            else:
                return {'error': f'Unsupported market data provider: {source.provider}'}
                
        except Exception as e:
            logger.error(f"Failed to connect to market data source: {e}")
            return {'error': str(e)}
    
    async def _connect_bloomberg(self, source: InstitutionalDataSource) -> Dict[str, Any]:
        """Connect to Bloomberg Terminal API"""
        try:
            # Authenticate with Bloomberg
            auth_data = {
                'terminal_id': source.credentials['terminal_id'],
                'api_key': source.credentials['api_key'],
                'auth_token': source.credentials['auth_token']
            }
            
            auth_response = requests.post(
                f"{source.api_endpoint}/auth",
                json=auth_data,
                timeout=10
            )
            
            if auth_response.status_code != 200:
                return {'error': f'Bloomberg auth failed: {auth_response.status_code}'}
            
            # Create WebSocket connection
            def on_message(ws, message):
                try:
                    data = json.loads(message)
                    self._process_bloomberg_data(data, source)
                except Exception as e:
                    logger.error(f"Bloomberg message processing error: {e}")
            
            def on_error(ws, error):
                logger.error(f"Bloomberg WebSocket error: {error}")
                source.error_count += 1
            
            def on_close(ws, close_status_code, close_msg):
                logger.info("Bloomberg WebSocket connection closed")
                source.is_connected = False
            
            def on_open(ws):
                logger.info("Bloomberg WebSocket connection opened")
                source.is_connected = True
                
                # Subscribe to real-time data
                subscribe_data = {
                    'action': 'subscribe',
                    'symbols': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'BRK.B'],
                    'data_types': ['quote', 'trade', 'depth']
                }
                ws.send(json.dumps(subscribe_data))
            
            # Connect WebSocket
            ws = websocket.WebSocketApp(
                source.websocket_endpoint,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close
            )
            
            self.websocket_connections['bloomberg'] = ws
            
            # Start WebSocket in background thread
            ws_thread = threading.Thread(target=ws.run_forever, daemon=True)
            ws_thread.start()
            
            return {
                'success': True,
                'source': 'Bloomberg',
                'connection_type': 'WebSocket',
                'subscribed_symbols': 8
            }
            
        except Exception as e:
            logger.error(f"Bloomberg connection failed: {e}")
            return {'error': str(e)}
    
    async def _connect_refinitiv(self, source: InstitutionalDataSource) -> Dict[str, Any]:
        """Connect to Refinitiv Eikon API"""
        try:
            # Authenticate with Refinitiv
            auth_data = {
                'app_key': source.credentials['app_key'],
                'username': source.credentials['username'],
                'password': source.credentials['password']
            }
            
            auth_response = requests.post(
                f"{source.api_endpoint}/auth",
                json=auth_data,
                timeout=10
            )
            
            if auth_response.status_code != 200:
                return {'error': f'Refinitiv auth failed: {auth_response.status_code}'}
            
            # Create WebSocket connection
            def on_message(ws, message):
                try:
                    data = json.loads(message)
                    self._process_refinitiv_data(data, source)
                except Exception as e:
                    logger.error(f"Refinitiv message processing error: {e}")
            
            def on_error(ws, error):
                logger.error(f"Refinitiv WebSocket error: {error}")
                source.error_count += 1
            
            def on_close(ws, close_status_code, close_msg):
                logger.info("Refinitiv WebSocket connection closed")
                source.is_connected = False
            
            def on_open(ws):
                logger.info("Refinitiv WebSocket connection opened")
                source.is_connected = True
                
                # Subscribe to real-time data
                subscribe_data = {
                    'action': 'subscribe',
                    'symbols': ['AAPL.O', 'MSFT.O', 'GOOGL.O', 'AMZN.O', 'NVDA.O', 'TSLA.O', 'META.O', 'BRK.A.O'],
                    'data_types': ['quote', 'trade', 'depth']
                }
                ws.send(json.dumps(subscribe_data))
            
            # Connect WebSocket
            ws = websocket.WebSocketApp(
                source.websocket_endpoint,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close
            )
            
            self.websocket_connections['refinitiv'] = ws
            
            # Start WebSocket in background thread
            ws_thread = threading.Thread(target=ws.run_forever, daemon=True)
            ws_thread.start()
            
            return {
                'success': True,
                'source': 'Refinitiv',
                'connection_type': 'WebSocket',
                'subscribed_symbols': 8
            }
            
        except Exception as e:
            logger.error(f"Refinitiv connection failed: {e}")
            return {'error': str(e)}
    
    async def _connect_exchange_feed(self, source: InstitutionalDataSource) -> Dict[str, Any]:
        """Connect to direct exchange feed"""
        try:
            # Authenticate with exchange
            auth_data = {
                'client_id': source.credentials['client_id'],
                'client_secret': source.credentials['client_secret']
            }
            
            auth_response = requests.post(
                f"{source.api_endpoint}/auth",
                json=auth_data,
                timeout=10
            )
            
            if auth_response.status_code != 200:
                return {'error': f'{source.provider} auth failed: {auth_response.status_code}'}
            
            # Create WebSocket connection
            def on_message(ws, message):
                try:
                    data = json.loads(message)
                    self._process_exchange_data(data, source)
                except Exception as e:
                    logger.error(f"{source.provider} message processing error: {e}")
            
            def on_error(ws, error):
                logger.error(f"{source.provider} WebSocket error: {error}")
                source.error_count += 1
            
            def on_close(ws, close_status_code, close_msg):
                logger.info(f"{source.provider} WebSocket connection closed")
                source.is_connected = False
            
            def on_open(ws):
                logger.info(f"{source.provider} WebSocket connection opened")
                source.is_connected = True
                
                # Subscribe to real-time data
                subscribe_data = {
                    'action': 'subscribe',
                    'symbols': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'BRK.B'],
                    'data_types': ['trade', 'quote', 'depth']
                }
                ws.send(json.dumps(subscribe_data))
            
            # Connect WebSocket
            ws = websocket.WebSocketApp(
                source.websocket_endpoint,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close
            )
            
            self.websocket_connections[source.provider.lower()] = ws
            
            # Start WebSocket in background thread
            ws_thread = threading.Thread(target=ws.run_forever, daemon=True)
            ws_thread.start()
            
            return {
                'success': True,
                'source': source.provider,
                'connection_type': 'WebSocket',
                'subscribed_symbols': 8
            }
            
        except Exception as e:
            logger.error(f"{source.provider} connection failed: {e}")
            return {'error': str(e)}
    
    async def _connect_alternative_data_source(self, source: InstitutionalDataSource) -> Dict[str, Any]:
        """Connect to alternative data source"""
        try:
            # Test API connection
            headers = {'Authorization': f"Bearer {source.credentials['api_key']}"}
            
            test_response = requests.get(
                f"{source.api_endpoint}/status",
                headers=headers,
                timeout=10
            )
            
            if test_response.status_code != 200:
                return {'error': f'{source.provider} API test failed: {test_response.status_code}'}
            
            # Create data stream
            self.data_streams[source.provider.lower()] = Queue()
            
            # Start data collection thread
            data_thread = threading.Thread(
                target=self._collect_alternative_data,
                args=(source,),
                daemon=True
            )
            data_thread.start()
            
            return {
                'success': True,
                'source': source.provider,
                'connection_type': 'REST API',
                'data_collection': 'started'
            }
            
        except Exception as e:
            logger.error(f"Alternative data connection failed: {e}")
            return {'error': str(e)}
    
    async def _connect_news_source(self, source: InstitutionalDataSource) -> Dict[str, Any]:
        """Connect to news source"""
        try:
            # Authenticate with news provider
            auth_data = {
                'api_key': source.credentials['api_key'],
                'partner_id': source.credentials.get('partner_id', '')
            }
            
            auth_response = requests.post(
                f"{source.api_endpoint}/auth",
                json=auth_data,
                timeout=10
            )
            
            if auth_response.status_code != 200:
                return {'error': f'{source.provider} auth failed: {auth_response.status_code}'}
            
            # Create WebSocket connection
            def on_message(ws, message):
                try:
                    data = json.loads(message)
                    self._process_news_data(data, source)
                except Exception as e:
                    logger.error(f"{source.provider} message processing error: {e}")
            
            def on_error(ws, error):
                logger.error(f"{source.provider} WebSocket error: {error}")
                source.error_count += 1
            
            def on_close(ws, close_status_code, close_msg):
                logger.info(f"{source.provider} WebSocket connection closed")
                source.is_connected = False
            
            def on_open(ws):
                logger.info(f"{source.provider} WebSocket connection opened")
                source.is_connected = True
                
                # Subscribe to news
                subscribe_data = {
                    'action': 'subscribe',
                    'topics': ['business', 'finance', 'technology', 'healthcare'],
                    'sentiment': True
                }
                ws.send(json.dumps(subscribe_data))
            
            # Connect WebSocket
            ws = websocket.WebSocketApp(
                source.websocket_endpoint,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close
            )
            
            self.websocket_connections[source.provider.lower()] = ws
            
            # Start WebSocket in background thread
            ws_thread = threading.Thread(target=ws.run_forever, daemon=True)
            ws_thread.start()
            
            return {
                'success': True,
                'source': source.provider,
                'connection_type': 'WebSocket',
                'subscribed_topics': ['business', 'finance', 'technology', 'healthcare']
            }
            
        except Exception as e:
            logger.error(f"News source connection failed: {e}")
            return {'error': str(e)}
    
    def _process_bloomberg_data(self, data: Dict[str, Any], source: InstitutionalDataSource):
        """Process Bloomberg market data"""
        try:
            if 'type' in data and data['type'] == 'trade':
                market_data = MarketDataPoint(
                    symbol=data['symbol'],
                    timestamp=datetime.fromtimestamp(data['timestamp'] / 1000),
                    price=data['price'],
                    bid=data.get('bid', 0.0),
                    ask=data.get('ask', 0.0),
                    volume=data.get('volume', 0),
                    exchange='BLOOMBERG',
                    source='Bloomberg'
                )
                
                # Add to data stream
                if 'bloomberg' not in self.data_streams:
                    self.data_streams['bloomberg'] = Queue()
                
                self.data_streams['bloomberg'].put(market_data)
                
                # Update metrics
                source.data_points_received += 1
                self.metrics['total_data_points'] += 1
                
        except Exception as e:
            logger.error(f"Bloomberg data processing error: {e}")
    
    def _process_refinitiv_data(self, data: Dict[str, Any], source: InstitutionalDataSource):
        """Process Refinitiv market data"""
        try:
            if 'type' in data and data['type'] == 'trade':
                market_data = MarketDataPoint(
                    symbol=data['symbol'],
                    timestamp=datetime.fromtimestamp(data['timestamp'] / 1000),
                    price=data['price'],
                    bid=data.get('bid', 0.0),
                    ask=data.get('ask', 0.0),
                    volume=data.get('volume', 0),
                    exchange='REFINITIV',
                    source='Refinitiv'
                )
                
                # Add to data stream
                if 'refinitiv' not in self.data_streams:
                    self.data_streams['refinitiv'] = Queue()
                
                self.data_streams['refinitiv'].put(market_data)
                
                # Update metrics
                source.data_points_received += 1
                self.metrics['total_data_points'] += 1
                
        except Exception as e:
            logger.error(f"Refinitiv data processing error: {e}")
    
    def _process_exchange_data(self, data: Dict[str, Any], source: InstitutionalDataSource):
        """Process exchange market data"""
        try:
            if 'type' in data and data['type'] == 'trade':
                market_data = MarketDataPoint(
                    symbol=data['symbol'],
                    timestamp=datetime.fromtimestamp(data['timestamp'] / 1000),
                    price=data['price'],
                    bid=data.get('bid', 0.0),
                    ask=data.get('ask', 0.0),
                    volume=data.get('volume', 0),
                    exchange=source.provider.upper(),
                    source=source.provider
                )
                
                # Add to data stream
                if source.provider.lower() not in self.data_streams:
                    self.data_streams[source.provider.lower()] = Queue()
                
                self.data_streams[source.provider.lower()].put(market_data)
                
                # Update metrics
                source.data_points_received += 1
                self.metrics['total_data_points'] += 1
                
        except Exception as e:
            logger.error(f"{source.provider} data processing error: {e}")
    
    def _process_news_data(self, data: Dict[str, Any], source: InstitutionalDataSource):
        """Process news data"""
        try:
            if 'type' in data and data['type'] == 'news':
                news_data = {
                    'headline': data['headline'],
                    'timestamp': datetime.fromtimestamp(data['timestamp'] / 1000),
                    'source': source.provider,
                    'sentiment': data.get('sentiment', 'neutral'),
                    'symbols': data.get('symbols', [])
                }
                
                # Add to data stream
                if source.provider.lower() not in self.data_streams:
                    self.data_streams[source.provider.lower()] = Queue()
                
                self.data_streams[source.provider.lower()].put(news_data)
                
                # Update metrics
                source.data_points_received += 1
                self.metrics['total_data_points'] += 1
                
        except Exception as e:
            logger.error(f"{source.provider} news processing error: {e}")
    
    def _collect_alternative_data(self, source: InstitutionalDataSource):
        """Collect alternative data in background thread"""
        try:
            headers = {'Authorization': f"Bearer {source.credentials['api_key']}"}
            
            while True:
                # Collect data every hour for satellite, daily for others
                if source.provider == 'Planet Labs':
                    # Collect satellite imagery data
                    response = requests.get(
                        f"{source.api_endpoint}/satellite/imagery",
                        headers=headers,
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        self.data_streams[source.provider.lower()].put(data)
                        source.data_points_received += 1
                        self.metrics['total_data_points'] += 1
                
                time.sleep(3600)  # Sleep for 1 hour
                
        except Exception as e:
            logger.error(f"Alternative data collection error: {e}")
    
    def get_real_time_price(self, symbol: str, source: str = 'bloomberg') -> Optional[MarketDataPoint]:
        """Get real-time price from specified source"""
        try:
            if source not in self.data_streams:
                return None
            
            data_stream = self.data_streams[source]
            
            # Get latest data point for symbol
            latest_data = None
            temp_queue = Queue()
            
            while not data_stream.empty():
                try:
                    data = data_stream.get_nowait()
                    if isinstance(data, MarketDataPoint) and data.symbol == symbol:
                        latest_data = data
                    temp_queue.put(data)
                except Empty:
                    break
            
            # Restore data to stream
            while not temp_queue.empty():
                data_stream.put(temp_queue.get())
            
            return latest_data
            
        except Exception as e:
            logger.error(f"Failed to get real-time price: {e}")
            return None
    
    def get_data_stream(self, source: str) -> Optional[Queue]:
        """Get data stream for specified source"""
        return self.data_streams.get(source)
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get comprehensive connection status"""
        return {
            'total_sources': len(self.data_sources),
            'connected_sources': self.metrics['connected_sources'],
            'total_data_points': self.metrics['total_data_points'],
            'average_latency_ms': self.metrics['average_latency_ms'],
            'error_rate': self.metrics['error_rate'],
            'sources': {
                name: {
                    'is_connected': source.is_connected,
                    'data_type': source.data_type,
                    'coverage': source.coverage,
                    'data_points_received': source.data_points_received,
                    'error_count': source.error_count,
                    'last_update': source.last_update.isoformat()
                }
                for name, source in self.data_sources.items()
            }
        }


# Global institutional data feeds instance
_institutional_data_feeds = None

def get_institutional_data_feeds() -> InstitutionalDataFeeds:
    """Get global institutional data feeds instance"""
    global _institutional_data_feeds
    if _institutional_data_feeds is None:
        _institutional_data_feeds = InstitutionalDataFeeds()
    return _institutional_data_feeds


if __name__ == "__main__":
    # Test institutional data feeds
    feeds = InstitutionalDataFeeds()
    
    # Connect to all sources
    print("Connecting to institutional data sources...")
    result = asyncio.run(feeds.connect_all_sources())
    print(f"Connection result: {result}")
    
    # Get real-time price
    price_data = feeds.get_real_time_price('AAPL', 'bloomberg')
    print(f"Real-time price: {price_data}")
    
    # Get connection status
    status = feeds.get_connection_status()
    print(f"Connection status: {json.dumps(status, indent=2)}")
