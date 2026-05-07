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

    async def initialize(self) -> bool:
        logger.info(f"Initializing Nexus engine with backend {self.backend_url}")

        async with httpx.AsyncClient() as client:
            for attempt in range(1, 6):
                try:
                    response = await client.get(
                        f"{self.backend_url}/api/alpaca/health", 
                        timeout=5
                    )
                    if response.status_code == 200:
                        logger.info("Execution backend healthy.")
                        break
                except (httpx.RequestError, httpx.TimeoutException) as exc:
                    wait = min(30, 2 ** attempt)
                    logger.warning(
                        f"Backend health check failed ({attempt}): {exc}. "
                        f"Retrying in {wait}s..."
                    )
                    await asyncio.sleep(wait)
            else:
                logger.error("Unable to reach execution backend after retries.")
                self.health_monitor.record(
                    "backend", False, "health-check-failure"
                )
                return False

            # High-speed Go Audit
            audit = PolyglotBridge.audit_platform_go()
            logger.info(f"Polyglot Audit Status: {audit.get('overall_health')}")
            
            self.health_monitor.record("backend", True, "connected")
            await self.refresh_universe(client)

        return True

    async def refresh_universe(self, client: Optional[httpx.AsyncClient] = None):
        """Dynamic universe re-scanner for 24/7 institutional trading."""
        logger.info("Refreshing tradable universe assets...")
        
        should_close = False
        if client is None:
            client = httpx.AsyncClient()
            should_close = True

        try:
            params = {
                "asset_class": "us_equity", 
                "status": "active", 
                "tradable": True, 
                "limit": Config.MAX_UNIVERSE_ASSETS
            }
            response = await client.get(
                f"{self.backend_url}/api/alpaca/assets", 
                params=params, 
                timeout=20
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
            logger.warning(
                f"Asset universe unavailable, using fallback: {exc}"
            )
            self.symbols = [
                "AAPL", "MSFT", "QQQ", "SPY", "NVDA", "IWM", "AMZN", "GOOG"
            ]
        finally:
            if should_close:
                await client.aclose()

        return True

    async def get_account_state(self) -> Dict[str, Any]:
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(f"{self.backend_url}/api/alpaca/account", timeout=10)
                if response.status_code == 200:
                    return response.json()
            except Exception as exc:
                logger.warning(f"Account fetch failure: {exc}")
        return {"equity": self.portfolio_value, "last_equity": self.portfolio_value}

    async def get_positions(self) -> List[Dict[str, Any]]:
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(f"{self.backend_url}/api/alpaca/positions", timeout=10)
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
                logger.warning(f"History load failed for {symbol}: {result}")
                continue
            if not result.empty:
                history[symbol] = result
        return history

    def build_portfolio_state(self, account: Dict[str, Any], positions: List[Dict[str, Any]]) -> Dict[str, Any]:
        total_value = float(account.get("equity", self.portfolio_value))
        last_equity = float(account.get("last_equity", total_value))

        # Handle NaN values
        if np.isnan(total_value) or total_value <= 0:
            total_value = self.portfolio_value
        if np.isnan(last_equity) or last_equity <= 0:
            last_equity = total_value

        drawdown = max(0.0, (last_equity - total_value) / max(total_value, 1.0))
        return {"total_value": total_value, "drawdown": drawdown, "positions": positions}

    def determine_risk_scale(self, market_insight: Dict[str, Any], risk_metrics: Dict[str, float]) -> float:
        regime = market_insight.get("regime", "SIDEWAYS")
        sentiment = market_insight.get("market_sentiment", 0.5)
        risk_var = risk_metrics.get("var", 0.0)

        scale = 1.0
        if regime == "TURBULENT":
            scale *= 0.35
        elif regime == "BEAR":
            scale *= 0.55
        elif regime == "SIDEWAYS":
            scale *= 0.75

        if sentiment < 0.35:
            scale *= 0.8
        if risk_var < -0.08:
            scale *= 0.7
        return max(0.2, min(scale, 1.0))

    def scale_weights(self, weights: Dict[str, float], multiplier: float) -> Dict[str, float]:
        return {symbol: weight * multiplier for symbol, weight in weights.items()}

    async def run_cycle(self):
        logger.info("Starting new trading cycle.")

        # Check for dynamic universe re-scan
        if time.time() - self.last_universe_refresh > Config.UNIVERSE_RESCAN_INTERVAL:
            await self.refresh_universe()

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(f"{self.backend_url}/api/alpaca/clock", timeout=10)
                clock_data = response.json() if response.status_code == 200 else {"is_open": False}
            except Exception as exc:
                logger.warning(f"Market clock unavailable: {exc}")
                clock_data = {"is_open": False}

        if not clock_data.get("is_open", False):
            logger.info("Market is closed, but 24/7 trading is enabled. Continuing execution.")
            self.health_monitor.record("market_session", True, details="24/7-override")

        await self.manage_positions()
        account = await self.get_account_state()
        current_positions = await self.get_positions()
        
        # Build symbol map of current holdings for quick lookup
        holdings = {p["symbol"]: p for p in current_positions}

        symbols = self.symbols[: Config.CANDIDATE_POOL_SIZE]
        raw_signals = await self.alpha_engine.get_batch_signals(symbols, timeframe="15Min")
        history = await self.fetch_universe_history(symbols, timeframe="1D", limit=100)
        benchmark_data = await self.alpha_engine.fetch_market_data("SPY", timeframe="1D", limit=120)

        market_insight = self.market_brain.analyze_market(benchmark_data, current_positions)
        self.market_regime = market_insight.get("regime", self.market_regime)
        selected_strategy = market_insight.get("selected_strategy", "Mean Reversion")
        
        logger.info("-" * 40)
        logger.info(f"MARKET STATUS: {self.market_regime} | STRATEGY: {selected_strategy}")
        logger.info("-" * 40)

        portfolio_scores = self.market_brain.build_portfolio_signals(raw_signals, history, self.market_regime)
        ranked = self.factor_engine.rank_assets(portfolio_scores, history)
        
        # Log top candidates for user visibility
        top_5 = list(ranked.items())[:5]
        logger.info("TOP BRAIN-FIRST CANDIDATES:")
        for sym, score in top_5:
            logger.info(f" -> {sym}: Score {score:.4f}")
            
        top_targets = dict(list(ranked.items())[: self.max_positions])
        weights = self.optimizer.optimize_weights(list(top_targets.keys()), [top_targets[symbol] for symbol in top_targets])

        # Ultra-fast Rust Risk Assessment
        returns_list: List[float] = benchmark_data["close"].pct_change().dropna().tolist() if not benchmark_data.empty else []
        risk_metrics: Dict[str, float] = {"var": 0.0}
        if returns_list:
            rust_risk = PolyglotBridge.calculate_risk_rust(returns_list)
            logger.info(f"Rust Risk Metrics: VaR={rust_risk.get('var')}, ES={rust_risk.get('expected_shortfall')}")
            risk_metrics["var"] = float(rust_risk.get("var", 0.0))

        risk_scale = self.determine_risk_scale(market_insight, risk_metrics)
        weights = self.scale_weights(weights, risk_scale)

        portfolio_state = self.build_portfolio_state(account, current_positions)
        self.health_monitor.record("market", benchmark_data.shape[0] > 0, details=selected_strategy)
        self.health_monitor.record("risk", risk_metrics.get("var", 0.0) > -0.25, details=str(risk_metrics))
        self.health_monitor.heartbeat()

        # Rebalancing Logic
        target_symbols = set(weights.keys())
        current_symbols = set(holdings.keys())
        
        # 1. Close positions no longer desired by strategy
        to_close = current_symbols - target_symbols
        close_tasks = []
        for symbol in to_close:
            logger.info(f"Exiting position for {symbol} - no longer in target universe.")
            close_tasks.append(self._close_position(symbol))
        if close_tasks:
            await asyncio.gather(*close_tasks)
            
        # 2. Rebalance existing or enter new positions
        trade_tasks = []
        for symbol, weight in weights.items():
            current_qty = float(holdings.get(symbol, {}).get("qty", 0.0))
            trade_tasks.append(self._submit_trade(
                symbol, weight, current_qty, top_targets, history, portfolio_state,
                raw_signals, market_insight, selected_strategy, is_open=clock_data.get("is_open", False)
            ))

        if trade_tasks:
            await asyncio.gather(*trade_tasks)

        self.execution_agent.learn(
            reward=float(np.mean(list(portfolio_scores.values()) or [0.0]))
        )

    async def _submit_trade(
        self, symbol: str, weight: float, current_qty: float, top_targets: Dict[str, float],
        history: Dict[str, pd.DataFrame], portfolio_state: Dict[str, Any],
        raw_signals: Dict[str, float], market_insight: Dict[str, Any],
        selected_strategy: str, is_open: bool = True
    ):
        """Helper to process and submit an individual trade concurrently."""
        # Lowered threshold to 0.0001 to allow high-frequency institutional trading
        if abs(weight) < 0.0001:
            return
        if symbol not in history or history[symbol].empty:
            return

        current_price = float(history[symbol]["close"].iloc[-1])
        score = top_targets.get(symbol, 0.0)
        side = "buy" if score > 0 else "sell"

        # Safe quantity calculation with NaN handling
        total_value = portfolio_state.get("total_value", self.portfolio_value)
        if not isinstance(total_value, (int, float)) or np.isnan(total_value):
            total_value = self.portfolio_value

        if not isinstance(weight, (int, float)) or np.isnan(weight):
            weight = 0.0

        if (not isinstance(current_price, (int, float)) or 
                np.isnan(current_price) or current_price <= 0):
            current_price = 1.0

        position_value = total_value * abs(weight)
        target_qty = int(position_value / current_price)
        
        # Calculate difference (target - current)
        qty_diff = target_qty - abs(current_qty)
        
        # Professional Threshold: Only trade if difference is meaningful (>1 share and >$100 value)
        if abs(qty_diff) < 1 or (abs(qty_diff) * current_price < 100):
            return

        side = "buy" if qty_diff > 0 else "sell"
        qty = abs(qty_diff)

        order_type = "market"
        time_in_force = "day"
        limit_price = None

        if self.market_regime == "TURBULENT":
            order_type = "limit"
            limit_price = current_price * (1.002 if side == "buy" else 0.998)
        elif self.market_regime == "BEAR":
            order_type = "limit"
            limit_price = current_price * (0.995 if side == "buy" else 1.002)
        elif self.market_regime == "BULL":
            order_type = "limit"
            limit_price = current_price * (0.998 if side == "buy" else 1.002)

        # Extended Hours Support for Professional 24/7 Trading
        extended_hours = False
        if not is_open:
            extended_hours = True
            order_type = "limit" # Alpaca requires limit orders for extended hours
            if limit_price is None:
                limit_price = current_price * (1.001 if side == "buy" else 0.999)

        trade_request = {
            "symbol": symbol,
            "qty": qty,
            "side": side,
            "price": current_price,
            "order_type": order_type,
            "time_in_force": time_in_force,
            "limit_price": limit_price,
            "regime": self.market_regime,
            "strategy": selected_strategy
        }

        # Hardware-level Zig Validation
        validation = PolyglotBridge.validate_order_zig(trade_request)
        if not validation.get("valid"):
            logger.error(f"Zig Validation failed for {symbol}: {validation.get('error')}")
            return

        # Compliance Check with Current Positions
        approved, violations = self.governance.check_compliance(
            trade_request, portfolio_state, current_qty=current_qty
        )
        if not approved:
            logger.warning(f"Trade rejected for {symbol}: {violations}")
            return

        # Classify strategy
        alpha = raw_signals.get(symbol, 0.0)
        strategy_class = self.market_brain.classify_strategy(
            symbol, alpha, history.get(symbol, pd.DataFrame()),
            self.market_regime, market_insight.get("macro_profile", {})
        )

        action = self.execution_agent.get_action(np.random.rand(10))
        logger.info(
            f"Executing {side} order for {symbol}: qty={qty}, "
            f"type={order_type}, action={action}, strategy={strategy_class}"
        )

        payload = {
            "symbol": symbol,
            "qty": qty,
            "side": side,
            "order_type": order_type,
            "time_in_force": time_in_force,
            "asset_class": "equity",
            "strategy": strategy_class
        }
        if limit_price is not None:
            payload["limit_price"] = round(limit_price, 4)
        
        if extended_hours:
            payload["extended_hours"] = True

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.backend_url}/api/alpaca/order", 
                    json=payload, 
                    timeout=15
                )
                if response.status_code not in {200, 201}:
                    try:
                        error_text = response.text
                    except Exception:
                        error_text = "Unknown error"
                    logger.error(
                        f"Order failed for {symbol}: "
                        f"{response.status_code} {error_text}"
                    )
            except Exception as exc:
                logger.error(f"Order submission error for {symbol}: {exc}")

    async def _close_position(self, symbol: str):
        """Force close a position via API."""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.delete(
                    f"{self.backend_url}/api/alpaca/positions/{symbol}", 
                    timeout=15
                )
                if response.status_code in {200, 201}:
                    logger.info(f"Successfully closed position for {symbol}")
                else:
                    logger.error(f"Failed to close position for {symbol}: {response.text}")
            except Exception as exc:
                logger.error(f"Error closing position for {symbol}: {exc}")


    async def manage_positions(self):
        logger.info("Running risk management and position maintenance.")
        positions = await self.get_positions()
        if not positions:
            return

        async with httpx.AsyncClient() as client:
            for pos in positions:
                symbol = pos.get("symbol")
                if not symbol:
                    continue
                try:
                    unrealized_plpc = float(pos.get("unrealized_plpc", 0.0))
                    if unrealized_plpc >= Config.TAKE_PROFIT_THRESHOLD * 100:
                        logger.info(f"Take-profit triggered for {symbol} ({unrealized_plpc:.2f}%)")
                        await client.delete(f"{self.backend_url}/api/alpaca/positions/{symbol}", timeout=10)
                    elif unrealized_plpc <= Config.STOP_LOSS_THRESHOLD * 100:
                        logger.warning(f"Stop-loss triggered for {symbol} ({unrealized_plpc:.2f}%)")
                        await client.delete(f"{self.backend_url}/api/alpaca/positions/{symbol}", timeout=10)
                except Exception as exc:
                    logger.error(f"Position management failed for {symbol}: {exc}")

    async def main_loop(self):
        retry_delay = 10
        while True:
            if not await self.initialize():
                logger.warning("Engine initialization failed. Backing off before retry.")
                await asyncio.sleep(retry_delay)
                retry_delay = min(60, retry_delay * 2)
                continue
            retry_delay = 10

            while True:
                try:
                    await self.run_cycle()
                except Exception as exc:
                    logger.error(f"Engine cycle error: {exc}")
                await asyncio.sleep(Config.HEARTBEAT_INTERVAL)


if __name__ == "__main__":
    engine = NexusEngine()
    try:
        asyncio.run(engine.main_loop())
    except KeyboardInterrupt:
        logger.info("Nexus engine shutdown completed.")
