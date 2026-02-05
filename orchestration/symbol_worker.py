
import logging
import traceback
import pandas as pd
import time
from typing import List, Dict, Any, Optional
from enum import Enum
from contracts import DecisionRecord, OrderInfo, AllocationRequest
from data.collectors.data_router import DataRouter
from data_intelligence.quality_agent import QualityAgent
from data_intelligence.confidence_agent import ConfidenceAgent
from data_intelligence.provider_bandit import ProviderBandit
from data_intelligence.provider_health import ProviderCircuitBreaker
from portfolio.allocator import InstitutionalAllocator
from audit.decision_log import write_audit, SystemHalt

logger = logging.getLogger(__name__)


class SymbolState(Enum):
    """State machine states for symbol processing"""
    INIT = "INIT"
    FETCH = "FETCH"
    QUALITY = "QUALITY"
    ALPHAS = "ALPHAS"
    RISK = "RISK"
    PM = "PM"
    ALLOCATE = "ALLOCATE"
    DECISION = "DECISION"
    AUDIT = "AUDIT"
    DONE = "DONE"
    ERROR = "ERROR"


class SymbolWorker:
    """
    Worker node that processes a single symbol through deterministic state machine.
    """
    def __init__(
        self,
        data_router: DataRouter,
        agents: List[Any],
        pm_brain: Any,
        risk_manager: Any,
        provider_bandit: Optional[ProviderBandit] = None,
        circuit_breaker: Optional[ProviderCircuitBreaker] = None,
        quality_agent: Optional[QualityAgent] = None,
        confidence_agent: Optional[ConfidenceAgent] = None,
        allocator: Optional[InstitutionalAllocator] = None
    ):
        self.data_router = data_router
        self.agents = agents
        self.pm_brain = pm_brain
        self.risk_manager = risk_manager

        # Data Intelligence
        self.provider_bandit = provider_bandit
        self.circuit_breaker = circuit_breaker
        self.quality_agent = quality_agent or QualityAgent()
        self.confidence_agent = confidence_agent or ConfidenceAgent()

        # Allocation
        self.allocator = allocator or InstitutionalAllocator()

    def process_symbol(self, cycle_id: str, symbol: str, regime: str = "UNCERTAIN", statarb_signals: Optional[pd.DataFrame] = None) -> DecisionRecord:
        """
        Process symbol through state machine.
        Returns: DecisionRecord object (guaranteed)
        """
        state = SymbolState.INIT
        decision = None
        provider_name = "Unknown"
        provider_confidence = 0.5
        quality_score = 0.0
        latency_ms = 0.0

        try:
            # ═══════════════════════════════════════════════════════════
            # STATE: FETCH
            # ═══════════════════════════════════════════════════════════
            state = SymbolState.FETCH
            start_time = time.time()

            if self.provider_bandit and self.circuit_breaker:
                available_providers = list(self.provider_bandit.providers.keys())
                data = pd.DataFrame()
                for p_name in available_providers:
                    can_attempt, reason = self.circuit_breaker.can_attempt(p_name, symbol)
                    if not can_attempt:
                        continue
                    try:
                        fetch_start = time.time()
                        # Allow long history for alpha calculation in paper/live loop
                        data = self.data_router.get_price_history(symbol, start_date="2020-01-01", allow_long_history=True)
                        latency_ms = (time.time() - fetch_start) * 1000
                        success = not data.empty
                        self.circuit_breaker.record_attempt(p_name, symbol, success, latency_ms)
                        if success:
                            break
                    except Exception as e:
                        self.circuit_breaker.record_attempt(p_name, symbol, False, 0.0)
                        continue
            else:
                data = self.data_router.get_price_history(symbol, start_date="2020-01-01", allow_long_history=True)
                latency_ms = (time.time() - start_time) * 1000

            if hasattr(data, 'attrs') and 'provider' in data.attrs:
                provider_name = data.attrs['provider']

            if data.empty:
                decision = DecisionRecord(
                    cycle_id=cycle_id, symbol=symbol, timestamp=pd.Timestamp.utcnow().isoformat()+"Z",
                    final_decision="REJECT",
                    reason_codes=["NO_DATA"],
                    data_providers={"name": provider_name, "state": state.value}
                )
                state = SymbolState.AUDIT
                self._write_audit_safe(decision)
                return decision

            # ═══════════════════════════════════════════════════════════
            # STATE: QUALITY
            # ═══════════════════════════════════════════════════════════
            state = SymbolState.QUALITY
            qc_result = self.quality_agent.check_quality(symbol, data)
            is_usable = qc_result.is_usable
            quality_score = qc_result.quality_score
            quality_reasons = qc_result.reasons

            if not is_usable:
                decision = DecisionRecord(
                    cycle_id=cycle_id, symbol=symbol, timestamp=pd.Timestamp.utcnow().isoformat()+"Z",
                    final_decision="REJECT",
                    reason_codes=["DATA_QUALITY_FAIL"] + quality_reasons,
                    data_providers={"name": provider_name, "quality_score": quality_score, "state": state.value}
                )
                state = SymbolState.AUDIT
                self._write_audit_safe(decision)
                return decision

            provider_confidence = self.confidence_agent.get_provider_confidence(provider_name)
            self.confidence_agent.update_confidence(provider_name, success=True, latency_ms=latency_ms)

            if provider_confidence < 0.5:
                decision = DecisionRecord(
                    cycle_id=cycle_id, symbol=symbol, timestamp=pd.Timestamp.utcnow().isoformat()+"Z",
                    final_decision="REJECT",
                    reason_codes=[f"LOW_PROVIDER_CONFIDENCE_{provider_confidence:.2f}"],
                    data_providers={"name": provider_name, "confidence": provider_confidence, "state": state.value}
                )
                state = SymbolState.AUDIT
                self._write_audit_safe(decision)
                return decision

            # ═══════════════════════════════════════════════════════════
            # STATE: ALPHAS
            # ═══════════════════════════════════════════════════════════
            state = SymbolState.ALPHAS
            agent_results = []
            for agent in self.agents:
                try:
                    # Pass extra context to agents (statarb)
                    res = agent.evaluate(symbol, data, statarb_signals=statarb_signals)
                    agent_results.append(res)
                except Exception as e:
                    logger.error(f"Agent {agent.name} failed on {symbol}: {e}")

            if not agent_results:
                decision = DecisionRecord(
                    cycle_id=cycle_id, symbol=symbol, timestamp=pd.Timestamp.utcnow().isoformat()+"Z",
                    final_decision="REJECT",
                    reason_codes=["ALL_AGENTS_FAILED"],
                    data_providers={"name": provider_name, "confidence": provider_confidence, "state": state.value}
                )
                state = SymbolState.AUDIT
                self._write_audit_safe(decision)
                return decision

            # ═══════════════════════════════════════════════════════════
            # STATE: PM
            # ═══════════════════════════════════════════════════════════
            state = SymbolState.PM

            allocation_request, final_decision, pm_reasons, pm_metadata = self.pm_brain.aggregate(
                symbol=symbol,
                results=agent_results,
                cycle_id=cycle_id,
                regime=regime,
                liquidity_usd=1e6
            )

            alphas_dict = {res.agent_name: res.score for res in agent_results if hasattr(res, 'score')}

            if final_decision != "EXECUTE" or allocation_request is None:
                decision = DecisionRecord(
                    cycle_id=cycle_id, symbol=symbol, timestamp=pd.Timestamp.utcnow().isoformat()+"Z",
                    final_decision=final_decision if isinstance(final_decision, str) else final_decision.value,
                    reason_codes=pm_reasons,
                    conviction=pm_metadata.get("pm_score", 0.0),
                    conviction_zscore=pm_metadata.get("conviction_zscore", 0.0),
                    alphas=alphas_dict,
                    data_providers={"name": provider_name, "quality_score": quality_score, "state": state.value, "pm_metadata": pm_metadata},
                    pm_override="REJECT"
                )
                state = SymbolState.AUDIT
                self._write_audit_safe(decision)
                return decision

            # ═══════════════════════════════════════════════════════════
            # STATE: PENDING OPTIMIZATION (CANDIDATE)
            # ═══════════════════════════════════════════════════════════
            state = SymbolState.DECISION

            # Extract mu/sigma for optimizer
            mu_hat = allocation_request.mu
            sigma_hat = allocation_request.sigma

            # Extract current price for execution sizing
            current_price = 0.0
            try:
                if not data.empty:
                    # Try Close, then Adj Close, then first column
                    if 'Close' in data.columns:
                        current_price = float(data['Close'].iloc[-1])
                    elif 'Adj Close' in data.columns:
                        current_price = float(data['Adj Close'].iloc[-1])
                    else:
                        current_price = float(data.iloc[-1, 0])
            except Exception:
                pass

            # Create CANDIDATE record (No Order yet)
            decision = DecisionRecord(
                cycle_id=cycle_id, symbol=symbol, timestamp=pd.Timestamp.utcnow().isoformat()+"Z",
                final_decision="CANDIDATE",
                reason_codes=pm_reasons + ["AWAITING_OPTIMIZER"],
                conviction=pm_metadata.get("pm_score", 0.0),
                conviction_zscore=pm_metadata.get("conviction_zscore", 0.0),
                mu=mu_hat,
                sigma=sigma_hat,
                price=current_price,
                alphas=alphas_dict,
                order=None, # No order yet
                data_providers={"name": provider_name, "quality_score": quality_score, "state": state.value, "pm_metadata": pm_metadata},
                pm_override="ALLOW"
            )

            # DO NOT WRITE AUDIT LOG FOR CANDIDATE - Defer to Orchestrator/Optimizer
            # state = SymbolState.AUDIT
            # self._write_audit_safe(decision)

            state = SymbolState.DONE
            return decision

        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"CRITICAL WORKER FAILURE on {symbol} at state {state.value}: {e}")
            decision = DecisionRecord(
                cycle_id=cycle_id, symbol=symbol, timestamp=pd.Timestamp.utcnow().isoformat()+"Z",
                final_decision="ERROR",
                reason_codes=["WORKER_CRASH", f"STATE_{state.value}", str(e)],
                raw_traceback=tb,
                data_providers={"name": provider_name, "state": state.value}
            )
            self._write_audit_safe(decision)
            return decision

    def _write_audit_safe(self, decision: DecisionRecord):
        """
        Write audit with SystemHalt on failure.
        """
        try:
            write_audit(decision)
        except Exception as audit_error:
            logger.critical(f"AUDIT WRITE FAILED for {decision.symbol}: {audit_error}")
            raise SystemHalt(f"Cannot write audit for {decision.symbol}: {audit_error}")
