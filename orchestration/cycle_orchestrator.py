
import logging
import uuid
import time
import yaml
import os
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

from contracts import DecisionRecord
# from contracts import Decision, decision_enum # Removed/Replaced
from orchestration.symbol_worker import SymbolWorker
from data.universe_manager import UnifiedUniverseManager
from data.collectors.data_router import DataRouter
from alpha_agents.registry import AlphaRegistry
from meta_intelligence.pm_brain import PMBrain
from risk.engine import RiskManager
from audit.decision_log import write_audit, SystemHalt
from data_intelligence.provider_bandit import ProviderBandit
from data_intelligence.provider_health import ProviderCircuitBreaker
from data_intelligence.quality_agent import QualityAgent
from data_intelligence.confidence_agent import ConfidenceAgent
from portfolio.allocator import InstitutionalAllocator
from strategies.stat_arb.engine import StatArbEngine
from monitoring.cycle_summary import print_cycle_summary

logger = logging.getLogger(__name__)

class CycleOrchestrator:
    """
    Main Engine Class.
    Runs the Map-Reduce cycle across the universe with full data governance.

    GUARANTEES:
    - 100% decision coverage (len(results) == len(universe))
    - SystemHalt on coverage mismatch or audit failure
    - Deterministic ordering for audit
    - Provider circuit breaking with 1-attempt-per-symbol
    """
    def __init__(self, mode: str = "paper", pm_config: dict = None):
        self.mode = mode
        self.cycle_id = str(uuid.uuid4())

        # Initialize Sub-Systems
        self.universe_manager = UnifiedUniverseManager()
        self.data_router = DataRouter()
        self.pm_brain = PMBrain(config=pm_config)
        self.risk_manager = RiskManager()
        self.statarb_engine = StatArbEngine()
        self.allocator = InstitutionalAllocator()

        # Agents Registry (50 Agents)
        self.agents = AlphaRegistry.get_all_agents()

        # Data Intelligence
        self.quality_agent = QualityAgent()
        self.confidence_agent = ConfidenceAgent()

        # Provider Bandit & Circuit Breaker
        self.provider_bandit = self._initialize_provider_bandit()
        self.circuit_breaker = ProviderCircuitBreaker(
            failure_threshold=0.5,
            min_attempts=5,
            cooldown_seconds=300.0,
            consecutive_failures_limit=3
        )

        # Register providers in circuit breaker
        if self.provider_bandit:
            for provider in self.provider_bandit.providers:
                self.circuit_breaker.register_provider(provider)

        # Provider Audit Stats
        self.providers_tally = {}
        self.data_quality_stats = {
            "total_fetches": 0,
            "quality_pass": 0,
            "quality_fail": 0,
            "avg_quality_score": 0.0
        }

        # Worker
        self.worker = SymbolWorker(
            self.data_router,
            self.agents,
            self.pm_brain,
            self.risk_manager,
            provider_bandit=self.provider_bandit,
            circuit_breaker=self.circuit_breaker,
            quality_agent=self.quality_agent,
            confidence_agent=self.confidence_agent,
            allocator=self.allocator
        )

    def _initialize_provider_bandit(self) -> ProviderBandit:
        """Initialize provider bandit from config"""
        config_path = "configs/providers.yaml"

        if not os.path.exists(config_path):
            logger.warning(f"Provider config not found at {config_path}, using defaults")
            return ProviderBandit(providers=["yahoo", "polygon", "alpha_vantage"])

        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            providers = list(config.get('providers', {}).keys())
            exploration_factor = config.get('selection', {}).get('exploration_factor', 2.0)

            logger.info(f"Initialized ProviderBandit with {len(providers)} providers: {providers}")
            return ProviderBandit(providers=providers, exploration_factor=exploration_factor)

        except Exception as e:
            logger.error(f"Failed to load provider config: {e}, using defaults")
            return ProviderBandit(providers=["yahoo", "polygon", "alpha_vantage"])

    def run_cycle(self):
        """
        Execute one full trading cycle.

        GUARANTEES:
        - 100% decision coverage
        - SystemHalt on coverage mismatch or audit failure
        - Deterministic ordering

        Returns:
            List[Decision]: Decisions for all symbols

        Raises:
            SystemHalt: On coverage mismatch or audit failure
        """
        self.cycle_id = str(uuid.uuid4())
        cycle_start = time.time()
        logger.info(f"╔══════════════════════════════════════════════════════════════════════════════╗")
        logger.info(f"║ CYCLE STARTED: {self.cycle_id[:40]:<40}                        ║")
        logger.info(f"╚══════════════════════════════════════════════════════════════════════════════╝")

        # 1. Load Universe (deterministic ordering)
        universe = sorted(self.universe_manager.get_active_universe())  # Sorted for determinism
        universe_size = len(universe)
        logger.info(f"Target Universe Size: {universe_size}")

        # Start circuit breaker cycle
        if self.circuit_breaker:
            self.circuit_breaker.start_cycle(self.cycle_id)

        results: List[DecisionRecord] = []

        # Reset stats
        self.providers_tally = {}
        self.data_quality_stats = {
            "total_fetches": 0,
            "quality_pass": 0,
            "quality_fail": 0,
            "avg_quality_score": 0.0
        }

        # 1.5 GLOBAL PRE-SCAN (StatArb)
        statarb_signals = pd.DataFrame()
        try:
            # Fetch small sample for all to get latest closes
            logger.info("StatArb: Performing global scan for pairs discovery...")
            panel_data = {}
            for sym in universe:
                # Get enough for cointegration check
                panel_data[sym] = self.data_router.get_price_history(sym, start_date="2022-01-01")

            # Convert to wide format
            all_closes = pd.DataFrame()
            for sym, df in panel_data.items():
                if not df.empty and 'Close' in df.columns:
                    all_closes[sym] = df['Close']

            if not all_closes.empty:
                statarb_signals = self.statarb_engine.generate_signals(all_closes)
                logger.info(f"StatArb: Found {len(statarb_signals)} pair signals")
            self.all_closes = all_closes # Store for optimization
        except Exception as e:
            logger.error(f"Global StatArb scan failed: {e}")
            self.all_closes = pd.DataFrame()

        # 2. Parallel Map Phase
        with ThreadPoolExecutor(max_workers=50) as executor:
            future_to_symbol = {
                executor.submit(self.worker.process_symbol, self.cycle_id, sym, statarb_signals=statarb_signals): sym
                for sym in universe
            }

            for future in as_completed(future_to_symbol):
                sym = future_to_symbol[future]
                try:
                    decision = future.result()
                    results.append(decision)

                    # Provider Tally
                    provider_meta = decision.data_providers if hasattr(decision, 'data_providers') else {}
                    # Assuming data_providers is Dict or string, simplified tally logic
                    # If we can't extract cleanly, we skip or use default

                    # Log quality check
                    # DecisionRecord doesn't have metadata field directly for provider extraction in same way potentially
                    # But we trust the worker returns a valid DecisionRecord

                    self.data_quality_stats["total_fetches"] += 1

                except SystemHalt:
                    # Re-raise SystemHalt immediately
                    raise
                except Exception as e:
                    logger.critical(f"Orchestrator failed to retrieve result for {sym}: {e}")
                    # Create fail-safe audit record
                    fail_decision = DecisionRecord(
                        cycle_id=self.cycle_id,
                        symbol=sym,
                        timestamp=pd.Timestamp.utcnow().isoformat() + "Z",
                        final_decision="ERROR",
                        reason_codes=["ORCHESTRATOR_RETRIEVAL_FAIL", str(e)]
                    )

                    try:
                        write_audit(fail_decision)
                    except Exception as audit_error:
                        logger.critical(f"CRITICAL: Audit write failed for fail-safe decision: {audit_error}")
                        raise SystemHalt(f"Cannot write fail-safe audit for {sym}: {audit_error}")

                    results.append(fail_decision)

        # 3. Validate 100% Coverage (CRITICAL)
        if len(results) != universe_size:
            error_msg = (f"CRITICAL COVERAGE MISMATCH: Expected {universe_size} decisions, "
                        f"got {len(results)}. Missing {universe_size - len(results)} symbols.")
            logger.critical(error_msg)
            logger.critical("SYSTEM HALT: Cannot proceed with incomplete coverage")
            raise SystemHalt(error_msg)

        logger.info(f"✓ 100% Decision Coverage Validated: {len(results)}/{universe_size}")

        # ═══════════════════════════════════════════════════════════
        # 3.5. GLOBAL OPTIMIZATION (PM BRAIN)
        # ═══════════════════════════════════════════════════════════
        candidates = [d for d in results if d.final_decision == "CANDIDATE"]

        if candidates:
            logger.info(f"Optimizing portfolio across {len(candidates)} candidates...")

            # Run Crown Jewel Optimizer
            opt_config = {
                "max_position_size": 0.10,
                "risk_aversion": 2.0,
                "cvar_weight": 5.0
            }
            # Prepare historical returns for Ledoit-Wolf
            historical_returns = None
            if not self.all_closes.empty:
                candidate_symbols = [c.symbol for c in candidates]
                # Filter all_closes to candidates
                present_symbols = [s for s in candidate_symbols if s in self.all_closes.columns]
                if present_symbols:
                    historical_returns = self.all_closes[present_symbols].pct_change().dropna()

            opt_result = self.pm_brain.optimize_cycle(
                candidates,
                config=opt_config,
                historical_returns=historical_returns
            )

            # Map results back to symbols
            weights_map = {opt_result['symbols'][i]: opt_result['w'][i] for i in range(len(opt_result['symbols']))}
            reject_map = {r['symbol']: r['reason'] for r in opt_result.get('rejected_assets_named', [])}

            # Process Candidates -> Final Decisions
            total_capital = self.risk_manager.initial_capital

            for d in results:
                if d.final_decision == "CANDIDATE":
                    symbol = d.symbol
                    weight = weights_map.get(symbol, 0.0)

                    if symbol in reject_map:
                        # OPTIMIZER REJECT
                        d.final_decision = "REJECT"
                        d.decision = "REJECT"
                        d.reason_codes.append(f"OPTIMIZER: {reject_map[symbol]}")
                        d.pm_override = "REJECT"
                    elif weight <= 1e-4:
                        # ZERO WEIGHT
                        d.final_decision = "REJECT"
                        d.decision = "REJECT"
                        d.reason_codes.append("OPTIMIZER_ZERO_WEIGHT")
                        d.pm_override = "REJECT"
                    else:
                        # EXECUTE
                        d.final_decision = "EXECUTE"
                        d.decision = "EXECUTE"
                        d.pm_override = "ALLOW"

                        # Calculate Quantity
                        price = d.price if d.price > 0 else 1.0 # Safety
                        quantity = (weight * total_capital) / price

                        d.order = {
                            "symbol": symbol,
                            "side": "BUY",
                            "quantity": quantity,
                            "target_weight": weight,
                            "order_type": "MARKET"
                        }

                    # WRITE AUDIT FOR FINALIZED CANDIDATE
                    try:
                        write_audit(d)
                    except Exception as e:
                        logger.critical(f"Failed to write audit for optimized candidate {d.symbol}: {e}")
                        raise SystemHalt(f"Audit write fail: {e}")

        # 4. Reduce Phase (Metrics)
        execute_count = sum(1 for d in results if d.final_decision == "EXECUTE")
        hold_count = sum(1 for d in results if d.final_decision == "HOLD")
        reject_count = sum(1 for d in results if d.final_decision == "REJECT")
        error_count = sum(1 for d in results if d.final_decision == "ERROR")

        cycle_duration = time.time() - cycle_start

        # 5. PM-Style Terminal Summary
        print_cycle_summary(
            cycle_id=self.cycle_id,
            results=results,
            universe_size=universe_size,
            duration_sec=cycle_duration,
            provider_stats=self.providers_tally,
            quality_stats=self.data_quality_stats,
            bandit_stats=self.provider_bandit.get_stats() if self.provider_bandit else {}
        )

        # 6. Circuit Breaker Health Report
        if self.circuit_breaker:
            health_report = self.circuit_breaker.get_health_report()
            logger.info(f"Provider Health Report: {health_report}")

        return results
