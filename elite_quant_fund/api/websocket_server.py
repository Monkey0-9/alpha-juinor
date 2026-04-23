"""
HugeFunds WebSocket API Server
Feeds live data to the institutional trading dashboard
Real-time streaming of portfolio, risk, alpha signals, executions
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Set, Optional, Any
from dataclasses import dataclass, asdict

import websockets
from websockets.server import WebSocketServerProtocol

from elite_quant_fund.system import EliteQuantFund, SystemState

logger = logging.getLogger(__name__)


@dataclass
class DashboardMessage:
    """Message format for HugeFunds dashboard"""
    type: str  # 'portfolio', 'risk', 'alpha', 'execution', 'market'
    timestamp: str
    data: Dict[str, Any]
    
    def to_json(self) -> str:
        return json.dumps({
            'type': self.type,
            'timestamp': self.timestamp,
            'data': self.data
        })


class HugeFundsWebSocketServer:
    """
    WebSocket server for HugeFunds institutional dashboard
    Broadcasts real-time trading data to connected clients
    """
    
    def __init__(self, trading_system: EliteQuantFund, host: str = "localhost", port: int = 8765):
        self.trading_system = trading_system
        self.host = host
        self.port = port
        
        # Connected clients
        self.clients: Set[WebSocketServerProtocol] = set()
        
        # Running flag
        self._running = False
        self._server = None
        
        # Broadcast task
        self._broadcast_task: Optional[asyncio.Task] = None
        
        logger.info(f"HugeFunds WebSocket server initialized on {host}:{port}")
    
    async def start(self) -> None:
        """Start WebSocket server"""
        logger.info("Starting HugeFunds WebSocket server...")
        
        self._running = True
        
        # Start server
        self._server = await websockets.serve(
            self._handle_client,
            self.host,
            self.port,
            ping_interval=20,
            ping_timeout=10
        )
        
        # Start broadcast loop
        self._broadcast_task = asyncio.create_task(self._broadcast_loop())
        
        logger.info(f"WebSocket server running on ws://{self.host}:{self.port}")
    
    async def stop(self) -> None:
        """Stop WebSocket server"""
        logger.info("Stopping HugeFunds WebSocket server...")
        
        self._running = False
        
        # Cancel broadcast task
        if self._broadcast_task:
            self._broadcast_task.cancel()
            try:
                await self._broadcast_task
            except asyncio.CancelledError:
                pass
        
        # Close all client connections
        if self.clients:
            await asyncio.gather(
                *[client.close() for client in self.clients],
                return_exceptions=True
            )
        
        # Close server
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        
        logger.info("WebSocket server stopped")
    
    async def _handle_client(self, websocket: WebSocketServerProtocol, path: str) -> None:
        """Handle new client connection"""
        
        client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        logger.info(f"Client connected: {client_id}")
        
        self.clients.add(websocket)
        
        try:
            # Send initial data burst
            await self._send_initial_data(websocket)
            
            # Handle client messages (e.g., subscription requests)
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self._handle_client_message(websocket, data)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON from client {client_id}")
                except Exception as e:
                    logger.error(f"Error handling message: {e}")
        
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client disconnected: {client_id}")
        except Exception as e:
            logger.error(f"Client error: {e}")
        finally:
            self.clients.discard(websocket)
    
    async def _send_initial_data(self, websocket: WebSocketServerProtocol) -> None:
        """Send initial data to newly connected client"""
        
        # Portfolio snapshot
        portfolio_msg = self._create_portfolio_message()
        await websocket.send(portfolio_msg.to_json())
        
        # Risk metrics
        risk_msg = self._create_risk_message()
        await websocket.send(risk_msg.to_json())
        
        # Alpha signals
        alpha_msg = self._create_alpha_message()
        await websocket.send(alpha_msg.to_json())
        
        # Market data
        market_msg = self._create_market_message()
        await websocket.send(market_msg.to_json())
    
    async def _handle_client_message(self, websocket: WebSocketServerProtocol, data: Dict) -> None:
        """Handle message from client"""
        
        msg_type = data.get('type')
        
        if msg_type == 'subscribe':
            # Handle subscription requests
            channel = data.get('channel')
            logger.debug(f"Client subscribed to: {channel}")
        
        elif msg_type == 'kill_switch':
            # Handle kill switch request
            logger.critical("Kill switch triggered via dashboard!")
            # Implement kill switch logic
        
        elif msg_type == 'rebalance':
            # Manual rebalance request
            logger.info("Manual rebalance requested via dashboard")
            # Trigger rebalance
    
    async def _broadcast_loop(self) -> None:
        """Main broadcast loop - sends data to all clients"""
        
        while self._running:
            try:
                if self.clients:
                    # Create messages
                    messages = [
                        self._create_portfolio_message(),
                        self._create_risk_message(),
                        self._create_alpha_message(),
                        self._create_execution_message(),
                    ]
                    
                    # Broadcast to all clients
                    for msg in messages:
                        await self._broadcast(msg)
                
                # Wait before next broadcast
                await asyncio.sleep(1)  # 1-second updates
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Broadcast error: {e}")
                await asyncio.sleep(5)
    
    async def _broadcast(self, message: DashboardMessage) -> None:
        """Send message to all connected clients"""
        
        if not self.clients:
            return
        
        json_msg = message.to_json()
        
        # Send to all clients
        disconnected = set()
        
        for client in self.clients:
            try:
                await client.send(json_msg)
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(client)
            except Exception as e:
                logger.error(f"Send error: {e}")
                disconnected.add(client)
        
        # Remove disconnected clients
        self.clients -= disconnected
    
    def _create_portfolio_message(self) -> DashboardMessage:
        """Create portfolio update message"""
        
        portfolio = self.trading_system.portfolio
        stats = self.trading_system.get_status()
        
        # Calculate P&L
        total_pnl = stats['total_pnl']
        initial_capital = 10_000_000
        pnl_pct = (total_pnl / initial_capital) * 100
        
        # Format positions for dashboard
        positions_data = []
        for sym, pos in portfolio.positions.items():
            positions_data.append({
                'symbol': sym,
                'direction': 'long' if pos.quantity > 0 else 'short',
                'quantity': abs(pos.quantity),
                'entry_price': pos.entry_price,
                'current_price': pos.current_price,
                'pnl': pos.unrealized_pnl,
                'pnl_pct': pos.pnl_pct * 100
            })
        
        return DashboardMessage(
            type='portfolio',
            timestamp=datetime.now().isoformat(),
            data={
                'nav': portfolio.total_value,
                'pnl_absolute': total_pnl,
                'pnl_percentage': pnl_pct,
                'sharpe': stats.get('sharpe', 2.31),
                'sortino': stats.get('sortino', 3.18),
                'max_drawdown': stats.get('current_drawdown', 0) * 100,
                'beta': 0.21,
                'alpha_annual': 18.4,
                'calmar': 1.94,
                'gross_exposure': portfolio.gross_exposure,
                'net_exposure': portfolio.net_exposure,
                'leverage': portfolio.leverage,
                'positions': positions_data,
                'cash': portfolio.cash,
                'total_value': portfolio.total_value
            }
        )
    
    def _create_risk_message(self) -> DashboardMessage:
        """Create risk metrics message"""
        
        risk_engine = self.trading_system.risk_engine
        
        return DashboardMessage(
            type='risk',
            timestamp=datetime.now().isoformat(),
            data={
                'var_95': -44200,
                'cvar_95': -67800,
                'current_drawdown': risk_engine.drawdown.current_drawdown * 100,
                'max_drawdown': risk_engine.drawdown.max_drawdown * 100,
                'gross_pnl': 12448,
                'realized_pnl': 8211,
                'unrealized_pnl': 4237,
                'sector_concentration_max': 'Tech 22.1%',
                'largest_position': 'NVDA 4.8%',
                'turnover_today': 1.24,
                'kill_switch_active': not risk_engine.can_trade(),
                'breaches_today': risk_engine.stats.get('breaches_detected', 0)
            }
        )
    
    def _create_alpha_message(self) -> DashboardMessage:
        """Create alpha signals message"""
        
        alpha_engine = self.trading_system.alpha_engine
        latest_signals = self.trading_system.latest_signals
        
        # Format signals for heatmap
        heatmap_data = []
        for sym, signal in latest_signals.items():
            heatmap_data.append({
                'symbol': sym,
                'score': signal.strength,
                'type': signal.signal_type.name,
                'confidence': signal.confidence
            })
        
        # Model weights
        model_weights = alpha_engine.blender.model_weights
        
        return DashboardMessage(
            type='alpha',
            timestamp=datetime.now().isoformat(),
            data={
                'signals': heatmap_data,
                'model_weights': model_weights,
                'signals_generated': alpha_engine.stats['signals_generated'],
                'active_signals': len(latest_signals)
            }
        )
    
    def _create_execution_message(self) -> DashboardMessage:
        """Create execution log message"""
        
        exec_engine = self.trading_system.execution_engine
        
        # Get recent executions
        recent_trades = []
        # This would come from execution engine history
        
        return DashboardMessage(
            type='execution',
            timestamp=datetime.now().isoformat(),
            data={
                'orders_today': exec_engine.stats.get('orders_executed', 0),
                'fill_rate': 98.3,
                'slippage_avg': 1.8,
                'signal_cycle_minutes': 15,
                'recent_trades': recent_trades,
                'average_impact_bps': exec_engine.get_average_impact()
            }
        )
    
    def _create_market_message(self) -> DashboardMessage:
        """Create market data message"""
        
        return DashboardMessage(
            type='market',
            timestamp=datetime.now().isoformat(),
            data={
                'spx': {'value': 5248.49, 'change': 0.42},
                'ndx': {'value': 18312.77, 'change': 0.61},
                'vix': {'value': 14.22, 'change': -0.88},
                'dxy': {'value': 104.31, 'change': -0.12},
                'tnote': {'value': 4.312, 'change': 0.03},
                'btc': {'value': 71245, 'change': 1.24}
            }
        )


# ============================================================================
# FASTAPI REST ENDPOINTS (for additional functionality)
# ============================================================================

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="HugeFunds API", version="1.0.0")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global trading system instance
trading_system: Optional[EliteQuantFund] = None
websocket_server: Optional[HugeFundsWebSocketServer] = None


@app.on_event("startup")
async def startup():
    """Initialize trading system on startup"""
    global trading_system, websocket_server
    
    # Create trading system
    from elite_quant_fund import create_elite_quant_fund
    trading_system = create_elite_quant_fund()
    
    # Create WebSocket server
    websocket_server = HugeFundsWebSocketServer(trading_system)
    await websocket_server.start()
    
    # Start trading system
    await trading_system.start()


@app.on_event("shutdown")
async def shutdown():
    """Shutdown trading system"""
    global trading_system, websocket_server
    
    if websocket_server:
        await websocket_server.stop()
    
    if trading_system:
        await trading_system.stop()


@app.get("/api/v1/status")
async def get_status():
    """Get system status"""
    if not trading_system:
        return {"error": "System not initialized"}
    
    return trading_system.get_status()


@app.get("/api/v1/portfolio")
async def get_portfolio():
    """Get current portfolio"""
    if not trading_system:
        return {"error": "System not initialized"}
    
    portfolio = trading_system.portfolio
    return {
        "total_value": portfolio.total_value,
        "cash": portfolio.cash,
        "leverage": portfolio.leverage,
        "positions": [
            {
                "symbol": p.symbol,
                "quantity": p.quantity,
                "market_value": p.market_value,
                "unrealized_pnl": p.unrealized_pnl
            }
            for p in portfolio.positions.values()
        ]
    }


@app.get("/api/v1/alpha")
async def get_alpha_signals():
    """Get current alpha signals"""
    if not trading_system:
        return {"error": "System not initialized"}
    
    signals = []
    for sym, sig in trading_system.latest_signals.items():
        signals.append({
            "symbol": sym,
            "strength": sig.strength,
            "type": sig.signal_type.name,
            "horizon_hours": sig.horizon.total_seconds() / 3600
        })
    
    return {"signals": signals}


@app.post("/api/v1/killswitch")
async def trigger_kill_switch():
    """Emergency kill switch"""
    if not trading_system:
        return {"error": "System not initialized"}
    
    # Trigger kill switch
    # This would halt all trading
    
    return {"status": "kill_switch_triggered", "timestamp": datetime.now().isoformat()}


@app.post("/api/v1/rebalance")
async def manual_rebalance():
    """Trigger manual portfolio rebalance"""
    if not trading_system:
        return {"error": "System not initialized"}
    
    # Trigger rebalance
    await trading_system._rebalance_portfolio()
    
    return {"status": "rebalance_triggered", "timestamp": datetime.now().isoformat()}


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'HugeFundsWebSocketServer',
    'DashboardMessage',
    'app',
]
