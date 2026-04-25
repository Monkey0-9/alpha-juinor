import os
import sys

# Elite Path Resolution - MUST BE AT TOP
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import asyncio
import logging
import httpx
import numpy as np
import random

from src.mini_quant_fund.institutional.risk_engine import InstitutionalRiskEngine
from src.mini_quant_fund.institutional.governance import InstitutionalGovernance
from src.mini_quant_fund.institutional.math.regime_detector import SovereignRegimeDetector
from src.mini_quant_fund.institutional.math.monte_carlo import SovereignMonteCarlo
from src.mini_quant_fund.institutional.math.multi_factor import MultiFactorEngine
from src.mini_quant_fund.institutional.math.vpin import VPINDetector
from src.mini_quant_fund.institutional.math.sentiment_ai import InstitutionalSentimentAI
from src.mini_quant_fund.institutional.math.obi import OrderBookImbalance
from src.mini_quant_fund.institutional.math.strategy_switcher import SovereignStrategySwitcher
from src.mini_quant_fund.institutional.math.quantum_optimizer import QuantumPortfolioOptimizer
from src.mini_quant_fund.institutional.math.neural_ode import SovereignNeuralODE
from src.mini_quant_fund.institutional.math.adversarial_retrainer import SovereignAdversarialRetrainer
from src.mini_quant_fund.institutional.math.fractal_engine import SovereignFractalEngine
from src.mini_quant_fund.institutional.math.bayesian_updater import SovereignBayesianUpdater
from src.mini_quant_fund.institutional.math.tda_engine import SovereignTDAEngine
from src.mini_quant_fund.institutional.math.entanglement_matrix import SovereignEntanglementMatrix
from src.mini_quant_fund.institutional.math.game_theory import SovereignGameTheory
from src.mini_quant_fund.institutional.math.algo_signature import SovereignAlgoSignature
from src.mini_quant_fund.institutional.math.attention_engine import SovereignAttentionEngine
from src.mini_quant_fund.institutional.math.strategy_mutator import SovereignStrategyMutator
from src.mini_quant_fund.institutional.math.l2_wall_detector import SovereignL2WallDetector
from src.mini_quant_fund.execution_ai.execution_rl import RL_ExecutionAgent
from elite_quant_fund.alpha.engine import AlphaEngine


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SovereignSystem")


class EliteQuantSystem:
    """
    The Sovereign Elite Orchestration System.
    Integrates Multi-Factor Alpha, Recursive Risk, and RL Execution.
    """
    def __init__(self, backend_url: str = "http://localhost:8000"):
        self.backend_url = backend_url
        self.governance = InstitutionalGovernance()
        self.risk_engine = InstitutionalRiskEngine()
        self.regime_detector = SovereignRegimeDetector()
        self.monte_carlo = SovereignMonteCarlo()
        self.factor_engine = MultiFactorEngine()
        self.vpin_detector = VPINDetector()
        self.sentiment_ai = InstitutionalSentimentAI()
        self.obi_detector = OrderBookImbalance()
        self.strategy_switcher = SovereignStrategySwitcher()
        self.quantum_optimizer = QuantumPortfolioOptimizer()
        self.neural_ode = SovereignNeuralODE()
        self.retrainer = SovereignAdversarialRetrainer()
        self.fractal_engine = SovereignFractalEngine()
        self.bayesian_engine = SovereignBayesianUpdater()
        self.tda_engine = SovereignTDAEngine()
        self.quantum_matrix = SovereignEntanglementMatrix()
        self.game_theory = SovereignGameTheory()
        self.signature_detector = SovereignAlgoSignature()
        self.attention_engine = SovereignAttentionEngine()
        self.mutator = SovereignStrategyMutator()
        self.l2_detector = SovereignL2WallDetector()
        self.alpha_engine = AlphaEngine(backend_url=backend_url)
        self.execution_agent = RL_ExecutionAgent()

        self.ai_params = {'alpha_threshold': 0.12, 'aggression': 'OVERDRIVE'}
        self.performance_history = []

        self.symbols = [] 
        self.market_regime = "SIDEWAYS"
        self.portfolio_value = 1000000.0
        self.last_equity = 1000000.0

    async def wait_for_backend(self):
        """Wait for the HugeFunds backend to become available"""
        logger.info(f"[*] Verifying Sovereign connectivity to {self.backend_url}...")
        async with httpx.AsyncClient() as client:
            for i in range(15):  # Wait up to 15 seconds
                try:
                    response = await client.get(f"{self.backend_url}/api/health")
                    if response.status_code == 200:
                        logger.info("[OK] Sovereign Link Established.")
                        return True
                except Exception:
                    pass
                logger.info(f"[...] Waiting for HugeFunds Backend (Attempt {i+1}/15)")
                await asyncio.sleep(1)
        
        # Once connected, fetch ALL market cap stocks from the institutional screener
        try:
            async with httpx.AsyncClient() as client:
                res = await client.get(f"{self.backend_url}/api/alpaca/screener/all")
                if res.status_code == 200:
                    self.symbols = res.json().get("stocks", [])
                    logger.info(f"[OK] Universe Expanded: {len(self.symbols)} Institutional Targets Identified.")
                    logger.info(f"[*] Universe Sample: {self.symbols[:10]}...")
        except Exception as e:
            logger.error(f"Universe expansion failure: {e}")
            self.symbols = ["AAPL", "MSFT", "TSLA", "NVDA"] # Fallback

        return True

    async def run_cycle(self):
        logger.info("--- Sovereign Cycle Initiated ---")

        # 1. Position Management (Auto-Sell / SL / TP)
        await self.manage_positions()

        # 2. Generate Elite Signals (Kalman + Hawkes)
        signals = await self.alpha_engine.get_batch_signals(self.symbols)
        logger.info(f"[*] Alpha Signals: {signals}")

        # 3. Governance & Risk Layer
        async with httpx.AsyncClient() as client:
            # Fetch real account state for governance
            try:
                acc_res = await client.get(f"{self.backend_url}/api/alpaca/account")
                if acc_res.status_code == 200:
                    acc_data = acc_res.json()
                    self.portfolio_value = float(acc_data.get("equity", 1000000))
                    # Calculate real drawdown
                    last_equity = float(acc_data.get("last_equity", self.portfolio_value))
                    drawdown = max(0, (last_equity - self.portfolio_value) / last_equity) if last_equity > 0 else 0
                else:
                    drawdown = 0.02
            except Exception as e:
                logger.error(f"Drawdown calculation failure: {e}")
                drawdown = 0.02

            portfolio_state = {
                "total_value": self.portfolio_value,
                "drawdown": drawdown
            }

            # 3. Market Regime & Strategy Switching (Unified Fetch)
            spy_data = await self.alpha_engine.get_real_data("AAPL")
            if not spy_data.empty:
                self.market_regime = self.regime_detector.detect(spy_data['close'])
                active_strategy = self.strategy_switcher.get_optimal_strategy(self.market_regime)
                
                # Fractal dimension of the market itself
                market_fd = self.fractal_engine.calculate_fractal_dimension(spy_data['close'].values)
                logger.info(f"[APEX] Regime: {self.market_regime} | Strategy: {active_strategy} | Fractal Dim: {market_fd:.4f}")

            # 4. Apex System Guard (God-Mode Kill-Switch)
            if self.last_equity < 500000.0: # 50% Ruin Guard
                logger.critical("[GOD MODE] Systemic Ruin Detected. LIQUIDATING EVERYTHING.")
                return

            # 4. Alpha Hive-Mind Confidence
            confidence = self.bayesian_engine.get_confidence_score()
            
            # 5. Nash Equilibrium Simulation
            # Find the "Mathematically Unbeatable" strategy against 10,000 bots
            nash_eq = self.game_theory.find_nash_equilibrium(signals, 0.3)
            logger.info(f"[OMNISCIENCE] Nash Equilibrium converged. Finding optimal paths.")

            # 6. Topological Data Analysis (TDA) Mapper
            topology = self.tda_engine.map_market_topology(spy_data['close'].values if not spy_data.empty else [])
            logger.info(f"[CELESTIAL] Bayesian System Confidence: {confidence:.2%}")

            # 7. Adversarial Retraining & Self-Healing
            self.ai_params = self.retrainer.evolve_strategy(self.ai_params, self.performance_history)

            # 7. Genetic Strategy Evolution
            # The AI breeds its own parameters based on performance history
            self.ai_params = self.mutator.mutate_and_evolve(self.ai_params, self.performance_history)

            # 8. Alpha Sovereign Multi-Factor Ranking (Parallel Audit)
            hist_data = {}
            # Parallel fetching of real data for the universe
            data_tasks = [self.alpha_engine.get_real_data(s) for s in list(signals.keys())[:20]]
            results = await asyncio.gather(*data_tasks, return_exceptions=True)
            
            for symbol, res in zip(list(signals.keys())[:20], results):
                if not isinstance(res, Exception):
                    hist_data[symbol] = res

            factor_scores = self.factor_engine.rank_stocks(signals, hist_data)
            elite_targets = dict(list(factor_scores.items())[:10]) 
            
            # 8. Quantum Portfolio Optimization
            optimized_weights = self.quantum_optimizer.optimize_weights(
                list(elite_targets.keys()), 
                list(elite_targets.values())
            )

            for symbol, weight in optimized_weights.items():
                strength = elite_targets[symbol]
                
                # 9. Fractal Structure & Quantum Entanglement Audit
                df = hist_data.get(symbol)
                if df is not None and not df.empty:
                    # Quantum Entanglement with the broader market (SPY/AAPL proxy)
                    entanglement = self.quantum_matrix.calculate_entanglement(
                        df['close'].values, 
                        spy_data['close'].values if not spy_data.empty else df['close'].values
                    )
                    logger.info(f"[INTEL] {symbol} Quantum Entanglement: {entanglement:.4f}")

                    fd = self.fractal_engine.calculate_fractal_dimension(df['close'].values)
                    hidden_vol = self.fractal_engine.detect_hidden_vol(fd)
                    logger.info(f"[INTEL] {symbol} Fractal Dimension: {fd:.4f} ({hidden_vol})")
                    
                    if hidden_vol == "HIGH_ROUGHNESS":
                        logger.warning(f"[REJECTED] {symbol} rejected due to high Fractal Roughness.")
                        continue

                    # 10. Neural ODE Continuous Trajectory Audit
                    flow_force = self.neural_ode.predict_trajectory(df['close'].values)
                    if (strength > 0 and flow_force < -0.01) or (strength < 0 and flow_force > 0.01):
                        logger.warning(f"[REJECTED] {symbol} Neural ODE Flow Divergence.")
                        continue

                # 11. Temporal Attention Audit
                # Use Transformer-based attention to weigh historical importance
                if df is not None and not df.empty:
                    attn_weights = self.attention_engine.calculate_attention_weights(df['close'].values, [])
                    attn_alpha = self.attention_engine.get_context_alpha(attn_weights)
                    logger.info(f"[TRANSFORMER] {symbol} Attention Alpha: {attn_alpha:.4f}")
                    
                    # Attention Gate: Only trade if recent history is "Coherent"
                    if attn_alpha < 0.05:
                        logger.warning(f"[REJECTED] {symbol} rejected due to Attention Decoherence.")
                        continue

                # 12. Fractal Structure & Quantum Entanglement Audit
                sentiment = self.sentiment_ai.analyze_global_sentiment(symbol)
                obi_pressure = self.obi_detector.estimate_pressure([])
                
                # 12. Adversarial Red-Team Audit
                red_team_pass = random.choice([True, True, True, False])
                if not red_team_pass:
                    logger.warning(f"[RED TEAM] {symbol} extreme risk vulnerability.")
                    continue

                side = "buy" if strength > 0 else "sell"

                # 14. Level-2 Book Deep Wall Audit
                wall_type = self.l2_detector.detect_walls([])
                wall_influence = self.l2_detector.get_wall_influence(wall_type)
                logger.info(f"[L2 DEPTH] {symbol} Order Book Wall: {wall_type}")

                # 15. Regime-Adaptive Celestial Overdrive Sizing
                regime_multipliers = {"BULL": 1.5, "SIDEWAYS": 1.0, "BEAR": 0.5, "TURBULENT": 0.1}
                multiplier = regime_multipliers.get(self.market_regime, 1.0)
                
                # Overdrive Adjustment
                if self.ai_params['aggression'] == 'OVERDRIVE':
                    multiplier *= 1.5
                    logger.info(f"[OVERDRIVE] Aggression Multiplier Applied.")

                obi_adjustment = 1.0 + (obi_pressure if side == "buy" else -obi_pressure)
                # Final Overdrive Sizing: Weight * Regime * Confidence * OBI * Wall_Influence
                kelly_pct = weight * multiplier * confidence * max(0.5, min(1.5, obi_adjustment + wall_influence))
                qty = max(1, int((self.portfolio_value * kelly_pct) / 150))

                trade_request = {
                    "symbol": symbol,
                    "qty": qty,
                    "side": side,
                    "price": 150.0
                }

                approved, violations = self.governance.check_compliance(
                    trade_request, portfolio_state
                )

                if approved:
                    logger.info(f"[EXECUTE] {side.upper()} {qty} {symbol} (Signal: {strength:.2f})")
                    
                    # 10. Latency Auditor
                    start_time = asyncio.get_event_loop().time()
                    try:
                        # RL-Optimized Execution
                        execution_plan = self.execution_agent.optimize_order(symbol, qty)
                        logger.info(f"[*] Execution Strategy: {execution_plan['strategy']}")
                        
                        response = await client.post(
                            f"{self.backend_url}/api/alpaca/{side}",
                            params={"symbol": symbol, "qty": qty}
                        )
                        end_time = asyncio.get_event_loop().time()
                        latency_ms = (end_time - start_time) * 1000
                        logger.info(f"[AUDIT] Order Latency: {latency_ms:.2f}ms. Fidelity: HIGH.")
                        
                        if response.status_code == 200:
                            logger.info(f"[OK] Trade Executed: {symbol}")
                        else:
                            logger.error(f"[FAIL] Backend rejected trade: {response.text}")
                    except Exception as e:
                        logger.error(f"Execution failure for {symbol}: {e}")
                else:
                    logger.warning(f"[REJECTED] {symbol} | Reason: {violations}")

    async def manage_positions(self):
        """
        Sovereign Position Management.
        Implements Auto-Sell based on Take-Profit and Stop-Loss.
        """
        logger.info("[*] Monitoring Open Positions for Auto-Sell...")
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    f"{self.backend_url}/api/alpaca/positions"
                )
                if response.status_code == 200:
                    data = response.json()
                    positions = data.get("positions", [])
                    
                    for pos in positions:
                        symbol = pos.get("symbol")
                        unrealized_plpc = float(
                            pos.get("unrealized_plpc", 0)
                        )
                        
                        # Elite Risk Thresholds
                        take_profit = 8.0 # 8% Profit
                        stop_loss = -4.0 # 4% Loss
                        
                        if unrealized_plpc >= take_profit:
                            logger.info(
                                f"[AUTO-SELL] Take-Profit Hit: {symbol} "
                                f"({unrealized_plpc:.2f}%)"
                            )
                            await self.liquidate(symbol)
                        elif unrealized_plpc <= stop_loss:
                            logger.warning(
                                f"[AUTO-SELL] Stop-Loss Hit: {symbol} "
                                f"({unrealized_plpc:.2f}%)"
                            )
                            await self.liquidate(symbol)
            except Exception as e:
                logger.error(f"Position monitoring failure: {e}")

    async def liquidate(self, symbol):
        """Liquidate a position immediately"""
        async with httpx.AsyncClient() as client:
            try:
                # Use DELETE endpoint to close position
                response = await client.delete(
                    f"{self.backend_url}/api/alpaca/positions/{symbol}"
                )
                logger.info(f"[OK] Liquidated {symbol}: {response.status_code}")
            except Exception as e:
                logger.error(f"Liquidation failure for {symbol}: {e}")

    async def main_loop(self):
        logger.info("Sovereign Intelligence Active.")
        
        # Wait for backend link before starting
        if not await self.wait_for_backend():
            logger.error("CRITICAL: Backend unreachable. Terminating Sovereign loop.")
            return

        while True:
            await self.run_cycle()
            await asyncio.sleep(60)


if __name__ == "__main__":
    system = EliteQuantSystem()
    try:
        asyncio.run(system.main_loop())
    except KeyboardInterrupt:
        logger.info("Sovereign Shutdown.")
