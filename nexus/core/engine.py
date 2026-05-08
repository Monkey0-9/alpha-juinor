import asyncio
import logging
import time
import httpx
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional

from nexus.core.governance import GovernanceEngine
from nexus.core.alpha import AlphaEngine
from nexus.core.execution_ai import ExecutionAgent
from nexus.core.intelligence import MarketBrain
from nexus.core.monitoring import HealthMonitor
from nexus.math.risk import RiskEngine
from nexus.math.indicators import RegimeDetector
from nexus.math.optimization import PortfolioOptimizer, MultiFactorEngine
from nexus.math.models import NeuralODE
from nexus.math.governance import LatticeVoter, StrategySwitcher
from nexus.utils.config import Config
from nexus.utils.platform_logging import setup_logging
from nexus.utils.polyglot_bridge import PolyglotBridge

logger = logging.getLogger(__name__)


class NexusEngine:
    """Main orchestration engine for the Nexus institutional trading platform."""

    def __init__(self, backend_url: str = Config.BACKEND_URL):
        setup_logging()
        self.backend_url = backend_url

        self.governance = GovernanceEngine(
            single_position_limit=Config.MAX_POSITION_SIZE,
            max_drawdown_limit=Config.MAX_DRAWDOWN
        )
        self.alpha_engine = AlphaEngine(backend_url=backend_url)
        self.execution_agent = ExecutionAgent()
        self.market_brain = MarketBrain()
        self.risk_engine = RiskEngine()
        self.health_monitor = HealthMonitor()

        self.regime_detector = RegimeDetector()
        self.optimizer = PortfolioOptimizer()
        self.factor_engine = MultiFactorEngine()
        self.neural_ode = NeuralODE()
        self.lattice_voter = LatticeVoter()
        self.strategy_switcher = StrategySwitcher()

        self.symbols: List[str] = []
        self.market_regime = "SIDEWAYS"
        self.portfolio_value = Config.FALLBACK_EQUITY
        self.max_positions = Config.MAX_OPEN_POSITIONS
        self.last_universe_refresh = 0.0
        
        # PERF FIX: Persistent client for connection pooling
        self._client: Optional[httpx.AsyncClient] = None
        # Tracking Submitted Orders
        self._active_orders: Dict[str, str] = {}  # symbol -> order_id

    async def _get_client(self) -> httpx.AsyncClient:
        """Returns the shared AsyncClient for connection pooling."""
        if self._client is None or self._client.is_closed:
            # Add API Key header if configured
            headers = {}
            if Config.API_KEY:
                headers["X-API-Key"] = Config.API_KEY
            self._client = httpx.AsyncClient(timeout=30, headers=headers)
        return self._client

    async def close(self):
        """Cleanup connection pool."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
        logger.info("Nexus engine resources released.")

    async def initialize(self) -> bool:
        logger.info(f"Initializing Nexus engine with backend {self.backend_url}")
        
        client = await self._get_client()
        for attempt in range(1, 6):
            try:
                response = await client.get(
                    f"{self.backend_url}/api/alpaca/health"
                )
                if response.status_code == 200:
                    logger.info("Execution backend healthy.")
                    break
                else:
                    logger.warning(f"Backend returned status {response.status_code}. Retrying...")
                    await asyncio.sleep(min(30, 2 ** attempt))
            except Exception as exc:
                wait = min(30, 2 ** attempt)
                logger.warning(f"Backend health check failed ({attempt}): {exc}. Retrying...")
                await asyncio.sleep(wait)
        else:
            logger.error("Unable to reach execution backend.")
            return False

        # High-speed Go Audit
        audit = PolyglotBridge.audit_platform_go(self.backend_url)
        logger.info(f"Polyglot Audit Status: {audit.get('overall_health')}")
        
        self.health_monitor.record("backend", True, "connected")
        await self.refresh_universe()
        return True

    async def refresh_universe(self):
        """Dynamic universe re-scanner."""
        logger.info("Refreshing tradable universe assets...")
        client = await self._get_client()
        try:
            params = {
                "asset_class": "us_equity", 
                "status": "active", 
                "tradable": True, 
                "limit": Config.MAX_UNIVERSE_ASSETS
            }
            response = await client.get(
                f"{self.backend_url}/api/alpaca/assets", 
                params=params
            )
            if response.status_code == 200:
                symbols = response.json().get("symbols", [])
                if symbols:
                    self.symbols = symbols[: Config.CANDIDATE_POOL_SIZE]
                else:
                    self.symbols = ["AAPL", "MSFT", "QQQ", "SPY", "NVDA"]
                logger.info(f"Loaded universe with {len(self.symbols)} symbols.")
                self.last_universe_refresh = time.time()
            else:
                raise ValueError(f"Asset scan returned {response.status_code}")
        except Exception as exc:
            logger.warning(f"Asset universe fallback: {exc}")
            self.symbols = ["AAPL", "MSFT", "QQQ", "SPY", "NVDA", "IWM", "AMZN", "GOOG"]

    async def get_account_state(self) -> Dict[str, Any]:
        client = await self._get_client()
        try:
            response = await client.get(f"{self.backend_url}/api/alpaca/account")
            if response.status_code == 200:
                return response.json()
        except Exception as exc:
            logger.warning(f"Account fetch failure: {exc}")
        return {"equity": self.portfolio_value, "last_equity": self.portfolio_value}

    async def get_positions(self) -> List[Dict[str, Any]]:
        client = await self._get_client()
        try:
            response = await client.get(f"{self.backend_url}/api/alpaca/positions")
            if response.status_code == 200:
                return response.json().get("positions", [])
        except Exception as exc:
            logger.warning(f"Position fetch failure: {exc}")
        return []

    async def fetch_universe_history(self, symbols: List[str], timeframe: str = "1D", limit: int = 80) -> Dict[str, pd.DataFrame]:
        tasks = [self.alpha_engine.fetch_market_data(symbol, timeframe=timeframe, limit=limit) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        history: Dict[str, pd.DataFrame] = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, BaseException):
                continue
            if not result.empty:
                history[symbol] = result
        return history

    def build_portfolio_state(self, account: Dict[str, Any], positions: List[Dict[str, Any]]) -> Dict[str, Any]:
        total_value = float(account.get("equity", self.portfolio_value))
        last_equity = float(account.get("last_equity", total_value))
        if np.isnan(total_value) or total_value <= 0: total_value = self.portfolio_value
        drawdown = max(0.0, (last_equity - total_value) / max(total_value, 1.0))
        return {"total_value": total_value, "drawdown": drawdown, "positions": positions}

    async def _verify_order_fills(self):
        """Check status of active orders and clear them once filled/cancelled."""
        if not self._active_orders:
            return

        client = await self._get_client()
        to_remove = []
        for symbol, order_id in self._active_orders.items():
            try:
                response = await client.get(
                    f"{self.backend_url}/api/alpaca/orders/{order_id}"
                )
                if response.status_code == 200:
                    order_info = response.json()
                    status = order_info.get("status")
                    if status in {"filled", "canceled", "expired", "rejected"}:
                        logger.info(f"Order {order_id} for {symbol} finalized: {status}")
                        to_remove.append(symbol)
                elif response.status_code == 404:
                    logger.warning(f"Order {order_id} not found on backend.")
                    to_remove.append(symbol)
            except Exception as exc:
                logger.debug(f"Error verifying order {order_id}: {exc}")

        for s in to_remove:
            del self._active_orders[s]

    async def run_cycle(self):
        logger.info("Starting new trading cycle.")
        if time.time() - self.last_universe_refresh > Config.UNIVERSE_RESCAN_INTERVAL:
            await self.refresh_universe()

        # Phase 1: Verify fills from previous cycle
        await self._verify_order_fills()

        client = await self._get_client()
        try:
            response = await client.get(f"{self.backend_url}/api/alpaca/clock")
            clock_data = response.json() if response.status_code == 200 else {"is_open": False}
        except Exception:
            clock_data = {"is_open": False}

        await self.manage_positions()
        account = await self.get_account_state()
        current_positions = await self.get_positions()
        holdings = {p["symbol"]: p for p in current_positions}

        symbols = self.symbols[: Config.CANDIDATE_POOL_SIZE]
        raw_signals = await self.alpha_engine.get_batch_signals(symbols, timeframe="15Min")
        history = await self.fetch_universe_history(symbols, timeframe="1D", limit=100)
        benchmark_data = await self.alpha_engine.fetch_market_data("SPY", timeframe="1D", limit=120)

        market_insight = self.market_brain.analyze_market(benchmark_data, current_positions)
        self.market_regime = market_insight.get("regime", self.market_regime)
        selected_strategy = market_insight.get("selected_strategy", "Mean Reversion")
        
        portfolio_scores = self.market_brain.build_portfolio_signals(raw_signals, history, self.market_regime)
        ranked = self.factor_engine.rank_assets(portfolio_scores, history)
        top_targets = dict(list(ranked.items())[: self.max_positions])
        weights = self.optimizer.optimize_weights(list(top_targets.keys()), [top_targets[s] for s in top_targets])

        returns_list = benchmark_data["close"].pct_change().dropna().tolist() if not benchmark_data.empty else []
        risk_metrics = {"var": 0.0}
        if returns_list:
            rust_risk = PolyglotBridge.calculate_risk_rust(returns_list)
            risk_metrics["var"] = float(rust_risk.get("var", 0.0))

        multiplier = self.determine_risk_scale(market_insight, risk_metrics)
        weights = {s: w * multiplier for s, w in weights.items()}

        portfolio_state = self.build_portfolio_state(account, current_positions)
        self.health_monitor.record("market", not benchmark_data.empty, details=selected_strategy)
        self.health_monitor.record("risk", risk_metrics["var"] > -0.25, details=str(risk_metrics))
        self.health_monitor.heartbeat()

        current_symbols = set(holdings.keys())
        target_symbols = set(weights.keys())
        
        # Close tasks
        close_tasks = [self._close_position(s) for s in (current_symbols - target_symbols)]
        if close_tasks: await asyncio.gather(*close_tasks)
            
        # Trade tasks
        trade_tasks = []
        for symbol, weight in weights.items():
            # Skip if there's already an active order pending for this symbol
            if symbol in self._active_orders:
                logger.debug(f"Skipping trade for {symbol}: order pending.")
                continue

            current_qty = float(holdings.get(symbol, {}).get("qty", 0.0))
            trade_tasks.append(self._submit_trade(
                symbol, weight, current_qty, top_targets, history, portfolio_state,
                raw_signals, market_insight, selected_strategy, is_open=clock_data.get("is_open", False)
            ))
        if trade_tasks: await asyncio.gather(*trade_tasks)

        self.execution_agent.learn(reward=float(np.mean(list(portfolio_scores.values()) or [0.0])))

    async def _submit_trade(self, symbol: str, weight: float, current_qty: float, top_targets: Dict[str, float],
                            history: Dict[str, pd.DataFrame], portfolio_state: Dict[str, Any],
                            raw_signals: Dict[str, float], market_insight: Dict[str, Any],
                            selected_strategy: str, is_open: bool):
        if abs(weight) < 0.0001 or symbol not in history or history[symbol].empty: return
        current_price = float(history[symbol]["close"].iloc[-1])
        score = top_targets.get(symbol, 0.0)
        
        total_value = portfolio_state.get("total_value", self.portfolio_value)
        target_qty = int((total_value * abs(weight)) / max(current_price, 1.0))
        qty_diff = target_qty - abs(current_qty)
        
        if abs(qty_diff) < 1 or (abs(qty_diff) * current_price < 100): return

        side = "buy" if qty_diff > 0 else "sell"
        qty = abs(qty_diff)
        order_type, limit_price = ("market", None)

        if not is_open:
            order_type, limit_price = ("limit", current_price * (1.001 if side == "buy" else 0.999))

        trade_request = {"symbol": symbol, "qty": qty, "side": side, "price": current_price, "order_type": order_type, "strategy": selected_strategy}
        if not PolyglotBridge.validate_order_zig(trade_request).get("valid"): return
        approved, _ = self.governance.check_compliance(trade_request, portfolio_state, current_qty=current_qty)
        if not approved: return

        payload = {"symbol": symbol, "qty": qty, "side": side, "order_type": order_type, "strategy": selected_strategy}
        if limit_price: payload["limit_price"] = round(limit_price, 4)
        if not is_open: payload["extended_hours"] = True

        client = await self._get_client()
        try:
            response = await client.post(f"{self.backend_url}/api/alpaca/order", json=payload)
            if response.status_code == 200:
                order_data = response.json()
                if "id" in order_data:
                    self._active_orders[symbol] = order_data["id"]
                    logger.info(f"Order submitted for {symbol}: {order_data['id']}")
        except Exception as exc:
            logger.error(f"Order submission error for {symbol}: {exc}")

    async def _close_position(self, symbol: str):
        client = await self._get_client()
        try:
            await client.delete(f"{self.backend_url}/api/alpaca/positions/{symbol}")
        except Exception as exc:
            logger.error(f"Error closing position for {symbol}: {exc}")

    async def manage_positions(self):
        positions = await self.get_positions()
        if not positions: return
        client = await self._get_client()
        for pos in positions:
            symbol = pos.get("symbol")
            if not symbol: continue
            try:
                pnl = float(pos.get("unrealized_plpc", 0.0))
                if pnl >= Config.TAKE_PROFIT_THRESHOLD * 100 or pnl <= Config.STOP_LOSS_THRESHOLD * 100:
                    await client.delete(f"{self.backend_url}/api/alpaca/positions/{symbol}")
            except Exception: pass

    async def main_loop(self):
        while True:
            if not await self.initialize():
                await asyncio.sleep(10)
                continue
            while True:
                try:
                    await self.run_cycle()
                except Exception as exc:
                    logger.error(f"Engine cycle error: {exc}")
                await asyncio.sleep(Config.HEARTBEAT_INTERVAL)

    def determine_risk_scale(self, market_insight: Dict[str, Any], risk_metrics: Dict[str, float]) -> float:
        scale = 1.0
        regime = market_insight.get("regime")
        if regime == "TURBULENT": scale *= 0.35
        elif regime == "BEAR": scale *= 0.55
        return max(0.2, min(scale, 1.0))

if __name__ == "__main__":
    engine = NexusEngine()
    try:
        asyncio.run(engine.main_loop())
    except KeyboardInterrupt:
        pass
    finally:
        asyncio.run(engine.close())
