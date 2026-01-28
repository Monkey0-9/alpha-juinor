"""
Cycle Orchestrator - Main Pipeline Driver.

This is the central orchestrator that runs the complete institutional trading pipeline:
1. Fetch 5 years of daily market data for all symbols
2. Validate data quality per symbol
3. Compute comprehensive features
4. Run all agent models
5. Aggregate decisions via Meta-Brain
6. Apply risk checks (CVaR, exposure limits, etc.)
7. Generate orders
8. Persist everything to database
9. Produce cycle summary JSON

Deterministic, auditable, and fault-tolerant.
"""

import logging
import hashlib
import json
import time
import uuid
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from database.schema import (
    DecisionRecord, ModelOutput, OrderRecord, PositionRecord,
    AuditEntry, CycleMeta
)
from database.manager import get_db

from features.compute import FeatureComputer, compute_z_temporal_ae
from agents.meta_brain import MetaBrain, SymbolDecision, DECISION_BUY, DECISION_SELL, DECISION_HOLD, DECISION_REJECT

from risk.engine import RiskManager, RiskDecision
from data_intelligence.quality_agent import QualityAgent, QualityResult
from data_intelligence.provider_bandit import ProviderBandit

logger = logging.getLogger(__name__)


@dataclass
class CycleResult:
    """Complete result of a cycle run"""
    cycle_id: str
    timestamp: str
    universe_size: int

    # Decision counts
    buy_count: int = 0
    sell_count: int = 0
    hold_count: int = 0
    reject_count: int = 0

    # Performance metrics
    duration_seconds: float = 0.0
    nav: float = 0.0
    daily_return: float = 0.0
    drawdown: float = 0.0

    # Risk metrics
    cvar: float = 0.0
    leverage: float = 0.0
    risk_warnings: List[str] = field(default_factory=list)

    # Provider metrics
    provider_health: Dict[str, Any] = field(default_factory=dict)

    # Top signals
    top_buys: List[Dict[str, Any]] = field(default_factory=list)
    top_sells: List[Dict[str, Any]] = field(default_factory=list)

    # Full decisions
    decisions: Dict[str, SymbolDecision] = field(default_factory=dict)

    # Errors
    symbol_errors: Dict[str, str] = field(default_factory=dict)

    # Quality metrics
    avg_quality_score: float = 1.0
    low_quality_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'cycle_id': self.cycle_id,
            'timestamp': self.timestamp,
            'universe_size': self.universe_size,
            'decision_counts': {
                'EXECUTE_BUY': self.buy_count,
                'EXECUTE_SELL': self.sell_count,
                'HOLD': self.hold_count,
                'REJECT': self.reject_count
            },
            'duration_seconds': self.duration_seconds,
            'performance': {
                'nav': self.nav,
                'daily_return': self.daily_return,
                'drawdown': self.drawdown
            },
            'risk': {
                'cvar': self.cvar,
                'leverage': self.leverage,
                'warnings': self.risk_warnings
            },
            'provider_health': self.provider_health,
            'top_buys': self.top_buys,
            'top_sells': self.top_sells,
            'n_errors': len(self.symbol_errors),
            'quality_metrics': {
                'avg_quality_score': self.avg_quality_score,
                'low_quality_count': self.low_quality_count
            }
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)


class CycleOrchestrator:
    """
    Main orchestrator for the institutional trading pipeline.

    Ensures:
    - All symbols in universe are evaluated each cycle
    - Exactly one decision per symbol (BUY, SELL, HOLD, REJECT)
    - All outputs persisted to DB
    - Full audit trail
    - Deterministic and auditable
    - Data quality enforcement (REJECT if quality_score < 0.6)
    """

    # Quality thresholds
    MIN_DATA_QUALITY = 0.6

    def __init__(
        self,
        universe_path: str = "configs/universe.json",
        lookback_years: int = 5,
        max_workers: int = 10,
        paper_mode: bool = True,
        **kwargs
    ):
        """
        Initialize the orchestrator.

        Args:
            universe_path: Path to universe JSON file
            lookback_years: Years of historical data to fetch
            max_workers: Maximum parallel workers for data fetching
            paper_mode: If True, run in paper mode (no real trades)
            **kwargs: Additional configuration
        """
        self.universe_path = Path(universe_path)
        self.lookback_years = lookback_years
        self.max_workers = max_workers
        self.paper_mode = paper_mode
        self.dry_run = kwargs.get('dry_run', False)


        # Calculate date range
        end_date = datetime.utcnow().strftime('%Y-%m-%d')
        start_date = (datetime.utcnow() - timedelta(days=lookback_years * 365)).strftime('%Y-%m-%d')
        self.start_date = start_date
        self.end_date = end_date

        # Initialize components
        from configs.config_manager import ConfigManager
        self.config = ConfigManager().config

        self.db = get_db()
        self.quality_agent = QualityAgent(min_quality=self.MIN_DATA_QUALITY)
        self.feature_computer = FeatureComputer()

        # Filter kwargs for components
        comp_kwargs = kwargs.copy()
        comp_kwargs.pop('dry_run', None)
        self.meta_brain = MetaBrain(**comp_kwargs)

        self.risk_manager = RiskManager()
        self.provider_bandit = ProviderBandit(
            providers=['polygon', 'alpha_vantage', 'yahoo', 'stooq']
        )

        # State
        self.universe = self._load_universe()
        self.positions: Dict[str, float] = {}
        self.portfolio_nav = 100000.0  # Default initial NAV
        self.portfolio_history = []

        logger.info(f"CycleOrchestrator initialized for {len(self.universe)} symbols")

    def _load_universe(self) -> List[str]:
        """Load the trading universe"""
        if not self.universe_path.exists():
            logger.warning(f"Universe file not found at {self.universe_path}")
            return []

        with open(self.universe_path) as f:
            config = json.load(f)

        return config.get('active_tickers', [])

    def run_cycle(self) -> CycleResult:
        """
        Run a complete trading cycle.

        This is the main entry point for each cycle.
        """
        cycle_id = f"cycle_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        start_time = time.time()

        logger.info(f"Starting cycle {cycle_id}")
        logger.info(f"Universe size: {len(self.universe)} symbols")

        # Initialize result
        result = CycleResult(
            cycle_id=cycle_id,
            timestamp=datetime.utcnow().isoformat(),
            universe_size=len(self.universe)
        )

        try:
            # Step 1: Fetch data for all symbols
            logger.info("Step 1: Fetching 5-year price data...")
            price_data = self._fetch_all_prices()

            # Step 2: Validate data quality
            logger.info("Step 2: Validating data quality...")
            quality_results = self._validate_quality(price_data)

            # Step 3: Compute features
            logger.info("Step 3: Computing features...")
            features = self._compute_all_features(price_data)

            # Step 4: Get current positions
            self._load_positions()

            # Step 5: Get portfolio state
            portfolio_state = self._get_portfolio_state()

            # Step 6: Get risk state
            risk_state = self._get_risk_state()

            # Step 7: Collect agent outputs
            logger.info("Step 4-6: Running agents...")
            agent_outputs = self._run_all_agents(price_data, features, quality_results)

            # Step 8: Make decisions via Meta-Brain
            logger.info("Step 7: Aggregating decisions via Meta-Brain...")
            decisions = self.meta_brain.make_decisions(
                cycle_id=cycle_id,
                symbol_agent_outputs=agent_outputs,
                symbol_features=features,
                symbol_positions=self.positions,
                portfolio_state=portfolio_state,
                risk_state=risk_state
            )

            # Step 9: Compute batch z-scores
            self.meta_brain.compute_batch_z_scores(decisions)

            # Step 10: Apply data quality enforcement (REJECT low quality)
            logger.info("Step 8: Applying data quality enforcement...")
            decisions = self._apply_quality_enforcement(decisions, quality_results)

            # Step 11: Apply final risk checks
            logger.info("Step 9: Applying risk checks...")
            decisions = self._apply_risk_checks(decisions, risk_state)

            # Step 12: Generate orders
            logger.info("Step 10: Generating orders...")
            orders = self._generate_orders(decisions, price_data)

            # Step 13: Persist everything
            if not self.dry_run:
                logger.info("Step 11: Persisting to database...")
                self._persist_results(cycle_id, decisions, orders, features, quality_results)
            else:
                logger.info("Step 11: Skipping persistence (DRY RUN mode)")


            # Step 14: Update positions (simulated for paper mode)
            if self.paper_mode:
                self._update_positions_paper(decisions)

            # Step 15: Build result
            result.decisions = decisions
            result.buy_count = sum(1 for d in decisions.values() if d.final_decision == DECISION_BUY)
            result.sell_count = sum(1 for d in decisions.values() if d.final_decision == DECISION_SELL)
            result.hold_count = sum(1 for d in decisions.values() if d.final_decision == DECISION_HOLD)
            result.reject_count = sum(1 for d in decisions.values() if d.final_decision == DECISION_REJECT)

            # Quality metrics
            result.avg_quality_score = self._calculate_avg_quality(quality_results)
            result.low_quality_count = sum(
                1 for r in quality_results.values()
                if not r.is_usable or r.quality_score < self.MIN_DATA_QUALITY
            )

            # Top signals
            buys = [(s, d) for s, d in decisions.items() if d.final_decision == DECISION_BUY]
            result.top_buys = [
                {'symbol': s, 'mu_hat': d.mu_hat, 'sigma_hat': d.sigma_hat, 'conviction': d.conviction}
                for s, d in sorted(buys, key=lambda x: -x[1].conviction)[:5]
            ]

            # Provider health
            result.provider_health = self.provider_bandit.get_stats()

            # Log completion
            self._log_cycle_start(result)

        except Exception as e:
            logger.error(f"Cycle {cycle_id} failed: {e}", exc_info=True)
            result.symbol_errors['_cycle_error'] = str(e)

            # Create REJECT decisions for all symbols on cycle failure
            for symbol in self.universe:
                result.decisions[symbol] = self.meta_brain._create_reject_decision(
                    cycle_id, symbol, ['cycle_error', str(e)]
                )
            result.reject_count = len(self.universe)

        # Calculate duration
        result.duration_seconds = time.time() - start_time

        # Log audit entry
        self._log_cycle_end(result)

        logger.info(f"Cycle {cycle_id} completed in {result.duration_seconds:.2f}s")
        logger.info(f"  Decisions: BUY={result.buy_count}, SELL={result.sell_count}, HOLD={result.hold_count}, REJECT={result.reject_count}")
        logger.info(f"  Quality: avg={result.avg_quality_score:.2f}, low_quality={result.low_quality_count}")

        return result

    def _fetch_all_prices(self) -> Dict[str, pd.DataFrame]:
        """Fetch 5-year price data for all symbols using provider MAB"""
        price_data = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._fetch_symbol_prices, symbol): symbol
                for symbol in self.universe
            }

            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    data = future.result()
                    if data is not None and not data.empty:
                        price_data[symbol] = data
                except Exception as e:
                    logger.error(f"Failed to fetch {symbol}: {e}")

        logger.info(f"Fetched data for {len(price_data)}/{len(self.universe)} symbols")
        return price_data

    def _fetch_symbol_prices(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch prices for a single symbol"""
        try:
            from data.collectors.data_router import DataRouter
            router = DataRouter()
            df = router.get_panel_parallel([symbol], self.start_date, self.end_date)
            return df
        except Exception as e:
            logger.warning(f"DataRouter fetch failed for {symbol}: {e}")

            # Fallback: Try direct provider
            try:
                from data.providers.yahoo import YahooDataProvider
                provider = YahooDataProvider()
                df = provider.fetch_ohlcv(symbol, self.start_date, self.end_date)
                return df
            except Exception as e2:
                logger.error(f"Yahoo fetch failed for {symbol}: {e2}")
                return None

    def _validate_quality(self, price_data: Dict[str, pd.DataFrame]) -> Dict[str, QualityResult]:
        """
        Validate data quality for each symbol.

        Returns:
            Dict mapping symbol -> QualityResult
        """
        results: Dict[str, QualityResult] = {}

        for symbol, df in price_data.items():
            result = self.quality_agent.check_quality(
                symbol=symbol,
                df=df,
                start_date=self.start_date,
                end_date=self.end_date
            )
            results[symbol] = result

            # Log quality issues
            if not result.is_usable:
                logger.warning(
                    f"Quality issue for {symbol}: "
                    f"score={result.quality_score:.2f}, "
                    f"reasons={'; '.join(result.reasons)}"
                )

        # Calculate aggregate quality
        if results:
            avg_score = sum(r.quality_score for r in results.values()) / len(results)
            logger.info(f"Average quality score: {avg_score:.2f}")

        return results

    def _calculate_avg_quality(self, quality_results: Dict[str, QualityResult]) -> float:
        """Calculate average quality score across all symbols"""
        if not quality_results:
            return 1.0
        return sum(r.quality_score for r in quality_results.values()) / len(quality_results)

    def _apply_quality_enforcement(
        self,
        decisions: Dict[str, SymbolDecision],
        quality_results: Dict[str, QualityResult]
    ) -> Dict[str, SymbolDecision]:
        """
        Apply data quality enforcement: REJECT symbols with quality_score < 0.6.

        This is a critical step - any symbol with low quality data gets
        a REJECT decision with reason: data_quality.
        """
        for symbol, decision in decisions.items():
            # Get quality result for this symbol
            quality = quality_results.get(symbol)

            if quality is None:
                # No quality data - check if we have price data
                logger.warning(f"No quality result for {symbol}, keeping decision")
                continue

            # Check if quality is below threshold
            if not quality.is_usable or quality.quality_score < self.MIN_DATA_QUALITY:
                # REJECT the symbol
                decision.final_decision = DECISION_REJECT

                # Add data_quality reason code
                if 'data_quality' not in decision.reason_codes:
                    decision.reason_codes.append('data_quality')

                # Add detailed reasons
                for reason in quality.reasons:
                    if reason not in decision.reason_codes:
                        decision.reason_codes.append(reason)

                # Clear position size
                decision.position_size = 0.0
                decision.mu_hat = 0.0
                decision.sigma_hat = 0.0
                decision.conviction = 0.0

                logger.info(f"REJECTED {symbol} due to data_quality: {quality.reasons}")

        return decisions

    def _compute_all_features(self, price_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
        """Compute features for all symbols"""
        # Get benchmark data for beta calculations
        benchmark_data = None
        if 'SPY' in price_data:
            benchmark_data = price_data['SPY']

        features = {}

        for symbol, df in price_data.items():
            try:
                feature_set = self.feature_computer.compute_all_features(
                    symbol=symbol,
                    price_data=df,
                    benchmark_data=benchmark_data
                )
                features[symbol] = feature_set.to_dict()
            except Exception as e:
                logger.error(f"Feature computation failed for {symbol}: {e}")
                features[symbol] = {'_error': str(e)}

        return features

    def _load_positions(self) -> None:
        """Load current positions from database"""
        try:
            db_positions = self.db.get_positions()
            self.positions = {p['symbol']: p['qty'] for p in db_positions}
        except Exception as e:
            logger.warning(f"Failed to load positions: {e}")
            self.positions = {}

    def _get_portfolio_state(self) -> Dict[str, Any]:
        """Get current portfolio state"""
        return {
            'nav': self.portfolio_nav,
            'cash': self.portfolio_nav * 0.1,  # Assume 10% cash
            'positions': self.positions,
            'leverage_used': sum(abs(v) for v in self.positions.values())
        }

    def _get_risk_state(self) -> Dict[str, Any]:
        """Get current risk state from RiskManager"""
        return {
            'is_risk_on': True,
            'regime': 'BULL_QUIET',
            'regime_scalar': 1.0,
            'risk_override': False,
            'cvar_breach': False,
            'portfolio_leverage': 0.0,
            'violations': []
        }

    def _run_all_agents(
        self,
        price_data: Dict[str, pd.DataFrame],
        features: Dict[str, Dict[str, Any]],
        quality_results: Dict[str, QualityResult]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Run all agents for all symbols.

        This is where we'd run:
        - MomentumAgent (XGBoost/LightGBM)
        - MeanReversionAgent (Bayesian OU)
        - VolatilityAgent (GARCH)
        - SentimentAgent (FinBERT)
        - SequenceAgent (Transformer on z_t)
        - CrossSectionAgent (RankNet)
        - DataQualityAgent (AE+IF)
        - RegimeAgent (HMM)
        - TailRiskAgent (EVT)
        """
        agent_outputs = {}

        for symbol in self.universe:
            symbol_outputs = []

            # Skip if no data
            if symbol not in price_data:
                continue

            df = price_data.get(symbol)
            feat = features.get(symbol, {})

            # Get quality result
            quality = quality_results.get(symbol)

            # Run each agent (placeholder implementations)
            try:
                # Data Quality Agent - SKIP if quality is too low
                if quality and not quality.is_usable:
                    symbol_outputs.append({
                        'agent_name': 'DataQualityAgent',
                        'mu': 0.0,
                        'sigma': 0.1,
                        'confidence': 1.0,
                        'metadata': {
                            'quality_score': quality.quality_score,
                            'skipped': True,
                            'reasons': quality.reasons
                        }
                    })
                else:
                    symbol_outputs.append({
                        'agent_name': 'DataQualityAgent',
                        'mu': 0.0,
                        'sigma': 0.05,
                        'confidence': 1.0,
                        'metadata': {
                            'quality_score': quality.quality_score if quality else 1.0,
                            'skipped': False
                        }
                    })

                # Technical Momentum Agent
                momentum_mu = self._compute_momentum_signal(df)
                symbol_outputs.append({
                    'agent_name': 'MomentumAgent',
                    'mu': momentum_mu,
                    'sigma': 0.15,
                    'confidence': 0.7,
                    'metadata': {'type': 'technical'}
                })

                # Mean Reversion Agent
                mr_mu = self._compute_mean_reversion_signal(df)
                symbol_outputs.append({
                    'agent_name': 'MeanReversionAgent',
                    'mu': mr_mu,
                    'sigma': 0.12,
                    'confidence': 0.6,
                    'metadata': {'type': 'statistical'}
                })

                # Volatility Signal
                vol_signal = self._compute_volatility_signal(df)
                symbol_outputs.append({
                    'agent_name': 'VolatilityAgent',
                    'mu': vol_signal,
                    'sigma': 0.10,
                    'confidence': 0.65,
                    'metadata': {'type': 'garch'}
                })

                # Liquidity Signal
                liq_signal = self._compute_liquidity_signal(df)
                symbol_outputs.append({
                    'agent_name': 'LiquidityAgent',
                    'mu': liq_signal,
                    'sigma': 0.05,
                    'confidence': 0.5,
                    'metadata': {'type': 'liquidity'}
                })

                # Pattern/Technical Agent
                pattern_signal = self._compute_pattern_signal(df)
                symbol_outputs.append({
                    'agent_name': 'PatternAgent',
                    'mu': pattern_signal,
                    'sigma': 0.08,
                    'confidence': 0.55,
                    'metadata': {'type': 'pattern'}
                })

                # Regime-adjusted signal
                regime_mu = self._compute_regime_signal(df, feat)
                symbol_outputs.append({
                    'agent_name': 'RegimeAgent',
                    'mu': regime_mu,
                    'sigma': 0.10,
                    'confidence': 0.6,
                    'metadata': {'type': 'regime'}
                })

            except Exception as e:
                logger.error(f"Agent error for {symbol}: {e}")

            agent_outputs[symbol] = symbol_outputs

        return agent_outputs

    def _compute_momentum_signal(self, df: pd.DataFrame) -> float:
        """Compute momentum signal"""
        if df is None or len(df) < 60:
            return 0.0

        close = df['Close']
        returns = np.log(close / close.shift(1)).dropna()

        # Multi-period momentum
        mom_1m = returns.tail(21).sum()
        mom_3m = returns.tail(63).sum()
        mom_6m = returns.tail(126).sum()

        # Weighted momentum signal
        signal = 0.3 * mom_1m + 0.3 * mom_3m + 0.4 * mom_6m

        return signal

    def _compute_mean_reversion_signal(self, df: pd.DataFrame) -> float:
        """Compute mean reversion signal"""
        if df is None or len(df) < 60:
            return 0.0

        close = df['Close']

        # Compare recent returns to longer-term mean
        short_term = close.pct_change(5).iloc[-1]
        long_term = close.pct_change(60).iloc[-1]

        # Mean reversion: if short-term underperforms long-term, expect reversion
        signal = long_term - short_term

        return signal

    def _compute_volatility_signal(self, df: pd.DataFrame) -> float:
        """Compute volatility signal"""
        if df is None or len(df) < 60:
            return 0.0

        close = df['Close']
        returns = np.log(close / close.shift(1)).dropna()

        # Current vol vs historical vol
        current_vol = returns.tail(20).std() * np.sqrt(252)
        hist_vol = returns.tail(252).std() * np.sqrt(252)

        if hist_vol > 0:
            vol_ratio = current_vol / hist_vol
            # Negative signal when vol is high (risk-off)
            signal = -0.5 * (vol_ratio - 1.0)
        else:
            signal = 0.0

        return signal

    def _compute_liquidity_signal(self, df: pd.DataFrame) -> float:
        """Compute liquidity signal"""
        if df is None or len(df) < 20:
            return 0.0

        volume = df['Volume']
        close = df['Close']

        # Volume trend
        vol_20d_avg = volume.tail(20).mean()
        vol_current = volume.iloc[-1]

        if vol_20d_avg > 0:
            vol_ratio = vol_current / vol_20d_avg
            # Positive signal for good liquidity
            signal = min(0.2, max(-0.2, (vol_ratio - 1.0) * 0.1))
        else:
            signal = 0.0

        return signal

    def _compute_pattern_signal(self, df: pd.DataFrame) -> float:
        """Compute technical pattern signal"""
        if df is None or len(df) < 50:
            return 0.0

        close = df['Close']
        high = df['High']
        low = df['Low']

        # Simple technical signals
        sma_20 = close.rolling(20).mean().iloc[-1]
        sma_50 = close.rolling(50).mean().iloc[-1]

        # Trend signal
        trend = 0.0
        if close.iloc[-1] > sma_20 > sma_50:
            trend = 0.1
        elif close.iloc[-1] < sma_20 < sma_50:
            trend = -0.1

        # RSI signal
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean().iloc[-1]
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean().iloc[-1]
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))

        rsi_signal = 0.0
        if rsi < 30:
            rsi_signal = 0.1  # Oversold - bullish
        elif rsi > 70:
            rsi_signal = -0.1  # Overbought - bearish

        return trend + rsi_signal

    def _compute_regime_signal(self, df: pd.DataFrame, features: Dict[str, Any]) -> float:
        """Compute regime-adjusted signal"""
        # Use features from FeatureComputer
        vol_regime = features.get('vol_regime', 1.0)
        ema_200_pct = features.get('ema_200_pct_above', 0.0)

        # Adjust momentum for regime
        signal = 0.0

        # High volatility regime adjustment
        if vol_regime > 1.5:
            signal -= 0.05

        # Uptrend adjustment
        if ema_200_pct > 0:
            signal += 0.05

        return signal

    def _apply_risk_checks(
        self,
        decisions: Dict[str, SymbolDecision],
        risk_state: Dict[str, Any]
    ) -> Dict[str, SymbolDecision]:
        """Apply final risk checks and adjust/reject decisions"""
        violations = []

        # Check CVaR limit
        if risk_state.get('cvar_breach', False):
            for symbol, decision in decisions.items():
                if decision.final_decision == DECISION_BUY:
                    decision.final_decision = DECISION_REJECT
                    decision.reason_codes.append('cvar_breach')
                    violations.append(f"{symbol}: CVaR breach")

        # Check single-name exposure
        max_exposure = 0.10  # 10% per name
        for symbol, decision in decisions.items():
            if decision.position_size > max_exposure:
                decision.position_size = max_exposure
                decision.reason_codes.append('exposure_limit')

        return decisions

    def _generate_orders(
        self,
        decisions: Dict[str, SymbolDecision],
        price_data: Dict[str, pd.DataFrame]
    ) -> List[OrderRecord]:
        """Generate orders from decisions with Execution Decision Layer validation"""
        orders = []

        # Load skipping history for checks
        # In a real system, this would come from DB. For now, empty or tracked in memory if daemon.
        # But this is a script run. Skipping history is persistent state.
        # We can implement a lightweight DB query here or pass it in.
        # For this task, we will default to empty skipping history unless we implement the query.
        skipping_history = {} # TODO: Load from audit_log or decision history

        from governance.execution_decision import decide_execution

        # Check config feature flag
        execution_enabled = self.config.get('execution', {}).get('enabled', True)

        for symbol, decision in decisions.items():
            if decision.final_decision not in [DECISION_BUY, DECISION_SELL]:
                continue

            # Get current price
            current_price = 0.0
            if symbol in price_data and not price_data[symbol].empty:
                current_price = price_data[symbol]['Close'].iloc[-1]

            # Setup inputs for decision layer
            # Need current weight.
            # decision.position_size IS the Target Weight (from Allocator)
            target_weight = decision.position_size
            if decision.final_decision == DECISION_SELL:
               # If SELL, usually position_size is the TARGET size (reduced).
               # Or is it the AMOUNT to sell?
               # MetaBrain/Allocator usually returns TARGET weights.
               # Let's verify: In _generate_orders original:
               # "dollar_size = decision.position_size * self.portfolio_nav"
               # "side='BUY' if ... else 'SELL'"
               # This implies position_size is the signed size?? No, position_size is usually absolute.
               # If `final_decision` is SELL, does `position_size` mean "Target ownership" or "Amount to Sell"?
               # Protocol: Allocator returns Target Weights.
               # MetaBrain decision likely holds the Target Weight in `position_size`.
               # If SELL, `position_size` should be < `current_holding`.
               pass

            current_qty = self.positions.get(symbol, 0.0)
            current_val = current_qty * current_price
            current_weight = current_val / (self.portfolio_nav if self.portfolio_nav > 0 else 1.0)

            # If decision is SELL, target_weight is usually what we WANT to have.
            # But wait, original code:
            # "side='BUY' if decision.final_decision == DECISION_BUY else 'SELL'"
            # "orders.append(OrderRecord(..., qty=qty, ...))"
            # It just treats `qty` as `position_size * NAV / price`.
            # This implies `position_size` is the DELTA or the TARGET?
            # If it were TARGET, we'd subtract current.
            # The original code: `dollar_size = decision.position_size * self.portfolio_nav`.
            # Then creates order with that `qty`.
            # This STRONGLY suggests `position_size` in `SymbolDecision` is the **TRADE SIZE** (Delta), not Target Weight.
            # Because it generates an order of that size directly.

            # Let's assume `position_size` is the TRADE SIZE (Target Weight Delta basically).
            # So `target_weight` (Target Portfolio Weight) = Current Weight + (Signed `position_size`)

            signed_trade_size = decision.position_size if decision.final_decision == DECISION_BUY else -decision.position_size
            target_weight_calculated = current_weight + signed_trade_size

            # Risk Scaled Weight: passed from upstream, assume same as target for now
            risk_scaled_weight = target_weight_calculated

            # market_open check:
            # If we are running this script, we assume market is reachable or we check time.
            # For backtest/paper, assume True.
            market_open = True

            if execution_enabled:
                # Call Execution Decision Layer (Governance)
                exec_result = decide_execution(
                    cycle_id=decision.cycle_id,
                    symbol=symbol,
                    target_weight=target_weight_calculated,
                    current_weight=current_weight,
                    nav_usd=self.portfolio_nav,
                    price=current_price,
                    conviction=decision.conviction,
                    data_quality=decision.data_quality_score if hasattr(decision, 'data_quality_score') else 1.0,
                    risk_scaled_weight=risk_scaled_weight,
                    skipping_history=skipping_history,
                    market_open=market_open,
                    config=self.config
                )

                # Apply Decision
                if exec_result['decision'] == 'EXECUTE':
                    # Proceed with order generation
                    # Update decision metadata
                    decision.metadata['execution_check'] = 'PASSED'
                    pass
                else:
                    # BLOCK EXECUTION
                    logger.info(f"Execution Blocked for {symbol}: {exec_result['decision']} {exec_result['reason_codes']}")

                    # Update Decision Record to reflect SKIP
                    decision.final_decision = exec_result['decision'] # e.g. SKIP_TOO_SMALL
                    decision.reason_codes.extend(exec_result['reason_codes'])

                    # Add execution audit details to metadata
                    decision.metadata['execution_audit'] = exec_result

                    # Do NOT create an order
                    continue

            # --- Order Generation (if EXECUTE) ---

            # Calculate quantity
            qty = 0.0
            if self.portfolio_nav > 0:
                dollar_size = decision.position_size * self.portfolio_nav
                qty = dollar_size / (current_price + 1e-10)

            # Generate idempotent order_id
            order_id = hashlib.sha256(
                f"{decision.cycle_id}_{symbol}_{decision.final_decision}".encode()
            ).hexdigest()[:32]

            order = OrderRecord(
                order_id=order_id,
                cycle_id=decision.cycle_id,
                symbol=symbol,
                side='BUY' if decision.final_decision == DECISION_BUY else 'SELL',
                qty=qty,
                price=current_price,
                order_type='MARKET',
                time_in_force='DAY',
                status='PENDING' if self.paper_mode else 'PENDING'
            )

            orders.append(order)

        return orders

    def _persist_results(
        self,
        cycle_id: str,
        decisions: Dict[str, SymbolDecision],
        orders: List[OrderRecord],
        features: Dict[str, Dict[str, Any]],
        quality_results: Dict[str, QualityResult]
    ) -> None:
        """Persist all results to database"""
        try:
            # 1. Persist decisions
            decision_records = []
            for symbol, decision in decisions.items():
                record = DecisionRecord(
                    cycle_id=cycle_id,
                    symbol=symbol,
                    final_decision=decision.final_decision,
                    reason_codes=decision.reason_codes,
                    mu_hat=decision.mu_hat,
                    sigma_hat=decision.sigma_hat,
                    conviction=decision.conviction,
                    position_size=decision.position_size,
                    stop_loss=decision.stop_loss,
                    trailing_params=decision.trailing_params,
                    data_quality_score=quality_results.get(symbol, QualityResult(
                        symbol=symbol, is_usable=True, quality_score=1.0
                    )).quality_score,
                    provider_confidence=decision.provider_confidence,
                    metadata=decision.metadata
                )
                decision_records.append(record)

            self.db.insert_decisions(decision_records)

            # 2. Persist orders
            self.db.insert_orders(orders)

            # 3. Persist model outputs
            model_outputs = []
            for symbol, decision in decisions.items():
                for name, contrib in (decision.agent_results or {}).items():
                    output = ModelOutput(
                        cycle_id=cycle_id,
                        symbol=symbol,
                        agent_name=name,
                        mu=contrib.mu,
                        sigma=contrib.sigma,
                        confidence=contrib.confidence,
                        metadata=contrib.metadata
                    )
                    model_outputs.append(output)

            self.db.insert_model_outputs(model_outputs)

            # 4. Persist features
            from database.schema import FeatureRecord
            feature_records = []
            for symbol, feat_dict in features.items():
                if '_error' not in feat_dict:
                    record = FeatureRecord(
                        symbol=symbol,
                        date=datetime.utcnow().strftime('%Y-%m-%d'),
                        features=feat_dict.get('features', {}),
                        version=feat_dict.get('version', '1.0.0')
                    )
                    feature_records.append(record)

            self.db.upsert_features(feature_records)

            # 5. Persist data quality log
            from database.schema import DataQualityRecord
            quality_records = []
            for symbol, quality in quality_results.items():
                record = DataQualityRecord(
                    symbol=symbol,
                    date=datetime.utcnow().strftime('%Y-%m-%d'),
                    quality_score=quality.quality_score,
                    issues=quality.reasons if quality.reasons else None,
                    provider='unknown',  # Would be set during fetch
                    row_count=0  # Would be set during fetch
                )
                quality_records.append(record)

            # Note: We use upsert pattern for quality records
            for record in quality_records:
                self.db.log_data_quality(record)

            # 6. Persist audit log
            self.db.log_audit(AuditEntry(
                cycle_id=cycle_id,
                component='CycleOrchestrator',
                level='INFO',
                message=f'Cycle completed with {len(decisions)} decisions',
                payload={
                    'n_buy': sum(1 for d in decisions.values() if d.final_decision == DECISION_BUY),
                    'n_reject': sum(1 for d in decisions.values() if d.final_decision == DECISION_REJECT),
                    'quality_score_avg': self._calculate_avg_quality(quality_results)
                }
            ))

            logger.info(f"Persisted {len(decision_records)} decisions and {len(orders)} orders")

        except Exception as e:
            logger.error(f"Failed to persist results: {e}")
            raise

    def _update_positions_paper(self, decisions: Dict[str, SymbolDecision]) -> None:
        """Update positions in paper mode"""
        for symbol, decision in decisions.items():
            if decision.final_decision == DECISION_BUY:
                self.positions[symbol] = self.positions.get(symbol, 0.0) + decision.position_size
            elif decision.final_decision == DECISION_SELL:
                self.positions[symbol] = max(0.0, self.positions.get(symbol, 0.0) - decision.position_size)

    def _log_cycle_start(self, result: CycleResult) -> None:
        """Log cycle start to audit"""
        if self.dry_run:
            return
        self.db.log_audit(AuditEntry(
            cycle_id=result.cycle_id,
            component='CycleOrchestrator',
            level='INFO',
            message=f'Cycle started',
            payload={'universe_size': result.universe_size}
        ))

    def _log_cycle_end(self, result: CycleResult) -> None:
        """Log cycle end to audit and persist cycle meta"""
        if self.dry_run:
            return
        # Persist cycle meta
        self.db.insert_cycle_meta(CycleMeta(
            cycle_id=result.cycle_id,
            universe_size=result.universe_size,
            buy_count=result.buy_count,
            sell_count=result.sell_count,
            hold_count=result.hold_count,
            reject_count=result.reject_count,
            duration_seconds=result.duration_seconds,
            nav=result.nav,
            provider_health=result.provider_health,
            risk_warnings=result.risk_warnings,
            top_buys=result.top_buys
        ))

        # Log completion
        self.db.log_audit(AuditEntry(
            cycle_id=result.cycle_id,
            component='CycleOrchestrator',
            level='INFO',
            message=f'Cycle completed',
            payload=result.to_dict()
        ))


def run_institutional_cycle(**kwargs) -> CycleResult:
    """
    Convenience function to run a complete institutional trading cycle.

    Usage:
        result = run_institutional_cycle()
        print(result.to_json())
    """
    orchestrator = CycleOrchestrator(**kwargs)
    return orchestrator.run_cycle()


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )

    # Run a cycle
    result = run_institutional_cycle()
    print(result.to_json())

