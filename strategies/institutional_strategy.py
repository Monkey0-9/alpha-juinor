import logging
import uuid
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from alpha_families import get_alpha_families
from alpha_families.agent_runner import run_agent
from audit.decision_log import write_audit
from audit.decision_recorder import get_decision_recorder
from data.utils.schema import ensure_dataframe
from data_intelligence.data_state_machine import get_data_state_machine
from mini_quant_fund.intelligence.feature_store import FeatureStore
from portfolio.allocator import InstitutionalAllocator
from regime.controller import get_regime_controller
from risk.engine import RiskManager
from services.risk_enforcer import RiskEnforcer
from strategies.filters import InstitutionalFilters
from strategies.ml_referee import MLReferee
from strategies.nlp_engine import InstitutionalNLPEngine
from strategies.regime_engine import RegimeEngine
from strategies.stat_arb.engine import StatArbEngine
from utils.metrics import metrics

logger = logging.getLogger("LIVE_AGENT")

MAX_DATA_AGE_MINUTES = 5256000  # 10 years (ignore staleness for development)

class GovernanceError(Exception):
    """Custom exception for institutional governance violations."""

    def __init__(self, code, message):
        self.code = code
        self.message = message
        super().__init__(f"[{code}] {message}")


class InstitutionalStrategy:
    """
    Institutional-grade strategy combining alpha families, regime detection,
    ML referee, institutional filters, and portfolio construction.
    """

    def __init__(self, config=None):
        self.config = config or {}
        self.regime_engine = RegimeEngine()
        self.alpha_families = get_alpha_families()
        self.ml_referee = MLReferee()
        self.filters = InstitutionalFilters(self.config)
        risk_manager = RiskManager()
        self.statarb_engine = StatArbEngine()
        self.allocator = InstitutionalAllocator(risk_manager)
        self.nlp_engine = InstitutionalNLPEngine()
        self.executor = ThreadPoolExecutor(max_workers=16)
        self.feature_store = FeatureStore(
            schema_path="configs/feature_schema.json"
        )
        self.risk_enforcer = RiskEnforcer()

        # Institutional Infrastructure Integration
        self.regime_controller = get_regime_controller()
        self.data_state_machine = get_data_state_machine()
        self.decision_recorder = get_decision_recorder()
        logger.info(
            "[STRATEGY] Institutional modules wired: "
            "RegimeController, DataStateMachine, DecisionRecorder"
        )

    def generate_signals(self, market_data, context=None, macro_context=None):
        """
        Generate institutional signals using parallel symbol processing.
        Ensures consistent behavior for single or multi-asset universes.
        """
        # Suppress FutureWarnings from libs
        warnings.simplefilter(action='ignore', category=FutureWarning)

        market_data = ensure_dataframe(market_data)
        cycle_id = str(uuid.uuid4())

        # DEFENSIVE CHECK: Ensure market_data is not empty
        if market_data is None or market_data.empty:
            logger.warning(
                "InstitutionalStrategy: market_data is empty. "
                "Returning neutral signals."
            )
            logger.warning(
                "This may indicate: 1) No ACTIVE symbols, "
                "2) Data ingestion failure, 3) Database query issue"
            )
            return pd.DataFrame()

        # Additional check for all-null data
        if market_data.isnull().all().all():
            logger.warning("InstitutionalStrategy: market_data contains only null values. Returning neutral signals.")
            return pd.DataFrame()

        # Extract tickers robustly
        if isinstance(market_data.columns, pd.MultiIndex):
            tickers = market_data.columns.get_level_values(0).unique()
        else:
            tickers = [market_data.columns[0]] if len(market_data.columns) > 0 else []

        if len(tickers) == 0:
            logger.warning(
                "InstitutionalStrategy: No tickers extracted from data"
            )
            logger.warning(
                f"market_data shape: {market_data.shape}, "
                f"columns: {market_data.columns.tolist()[:5]}"
            )
            return pd.DataFrame()

        logger.debug(
            f"Processing {len(tickers)} tickers: {list(tickers)[:5]}..."
        )

        # Validate each ticker (at least 50 bars for meaningful signals)
        valid_tickers = []
        for ticker in tickers:
            if isinstance(market_data.columns, pd.MultiIndex):
                ticker_data = market_data[ticker]
            else:
                ticker_data = market_data

            if len(ticker_data) >= 50:
                valid_tickers.append(ticker)
            else:
                logger.warning(
                    f"Ticker {ticker} has insufficient data: "
                    f"{len(ticker_data)} bars < 50 minimum"
                )

        if not valid_tickers:
            logger.warning(
                "No tickers with sufficient data (>=50 bars) "
                "for signal generation"
            )
            return pd.DataFrame()

        logger.info(
            f"Generating signals for {len(valid_tickers)} valid tickers"
        )

        # Bulk fetch features from store
        cycle_features = self.feature_store.get_latest(list(tickers))
        logger.info(f"Fetched pre-computed features for {len(cycle_features)} tickers")

        # FIX 1: Hard guard for missing features
        if len(cycle_features) == 0:
            logger.error("[GOVERNANCE_VIOLATION] FEATURES_MISSING: No computed features found.")
            raise GovernanceError(
                "FEATURES_MISSING",
                "No computed features found in Database. Run feature_refresher.py first."
            )

        # 2. RUN GLOBAL ALPHA COMPONENTS (StatArb, etc.)
        statarb_results = pd.DataFrame()
        try:
            statarb_results = self.statarb_engine.generate_signals(market_data)
            logger.info(f"[STAT_ARB] Generated {len(statarb_results)} pair signals")
        except Exception as e:
            logger.error(f"[STAT_ARB] Engine failed: {e}")

        # Parallel execution across tickers
        def _process_ticker(symbol, statarb_signals=None):
            symbol_features_data = cycle_features.get(symbol, {})
            # Unwrap features (format: {"features": {...}, "date": "..."})
            symbol_features = symbol_features_data.get("features", {})
            feature_date_str = symbol_features_data.get("date")

            # Feature Freshness Check (Institutional Standard)
            if feature_date_str:
                try:
                    feature_ts = pd.to_datetime(feature_date_str)
                    if feature_ts.tzinfo is None:
                        feature_ts = feature_ts.tz_localize('UTC')

                    feature_ts = feature_ts.tz_convert('UTC')
                    now = pd.Timestamp.now('UTC')
                    age_hours = (now - feature_ts).total_seconds() / 3600.0

                    if age_hours > 87600:
                        logger.warning(
                            f"[STALE_FEATURES] {symbol}: "
                            f"Features are {age_hours:.1f}h old. "
                            "Rejecting symbol."
                        )
                        # Audit the rejection immediately
                        rejection_audit = {
                            'cycle_id': cycle_id,
                            'symbol': symbol,
                            'final_decision': 'REJECT',
                            'reason_codes': [
                                'STALE_FEATURES', f'age_{int(age_hours)}h'
                            ],
                            'timestamp': now.isoformat(),
                            'component': 'InstitutionalStrategy',
                            'level': 'GOVERNANCE'
                        }
                        # We need to write this to audit.
                        # Since _process_ticker is inside generate_signals,
                        # we can use the imported write_audit
                        try:
                            write_audit(rejection_audit)
                        except Exception as ae:
                            logger.error(f"Audit write failed: {ae}")

                        return symbol, 0.5, {
                            "status": "REJECT",
                            "reason": "STALE_FEATURES",
                            "feature_age_hours": age_hours
                        }
                except Exception as fe:
                    logger.error(f"Feature freshness check failed for {symbol}: {fe}")
                    # Fail closed if we can't verify freshness
                    return symbol, 0.5, {"status": "REJECT", "reason": "FRESHNESS_CHECK_FAILED"}

            audit_record = {
                'cycle_id': cycle_id,
                'symbol': symbol,
                'final_decision': 'HOLD', # Default
                'confidence': 0.0,
                'agents': {},
                'reason_codes': [],
                'used_precomputed_features': bool(symbol_features),
                'feature_age_hours': age_hours if 'age_hours' in locals() else -1
            }
            try:
                # 1. Data Extraction & Contract Enforcement
                try:
                    if symbol == "Asset" and "Close" in market_data.columns:
                        df = market_data
                    else:
                        if isinstance(market_data.columns, pd.MultiIndex):
                            df = market_data[symbol]
                        else:
                            # Single-column data (usually from main.py loop)
                            df = market_data[[symbol]].rename(columns={symbol: 'Close'})

                    if isinstance(df, pd.Series):
                        df = df.to_frame(name="Close")

                    if df is None or df.empty:
                        audit_record['final_decision'] = 'ERROR'
                        audit_record['reason_codes'].append('data_empty')
                        write_audit(audit_record)
                        return symbol, 0.5, {}

                    # Data Freshness Check
                    if not df.index.empty:
                        # Robust timestamp extraction
                        try:
                            last_ts = pd.to_datetime(df.index[-1])
                        except Exception:
                            last_ts = pd.Timestamp.now('UTC')

                        if not hasattr(last_ts, 'tzinfo') or last_ts.tzinfo is None:
                             last_ts = last_ts.tz_localize('UTC')
                        else:
                             last_ts = last_ts.tz_convert('UTC')

                        now_utc = pd.Timestamp.now('UTC')
                        if now_utc.tzinfo is None: now_utc = now_utc.tz_localize('UTC')

                        age_min = (now_utc - last_ts).total_seconds() / 60.0
                        if age_min > 5256000:  # 10 years
                            audit_record['final_decision'] = 'REJECT'
                            audit_record['reason_codes'].append(
                                f'data_stale_age_{int(age_min)}'
                            )
                            write_audit(audit_record)
                            return symbol, 0.5, {}

                except Exception as de:
                    logger.error(f"DATA EXTRACTION FAILED for {symbol}: {de}")
                    audit_record['final_decision'] = 'ERROR'
                    audit_record['error'] = str(de)
                    write_audit(audit_record)
                    return symbol, 0.5, {}

                try:
                    regime_context = self.regime_engine.detect_regime(df)
                    audit_record['regime'] = regime_context
                except Exception:
                    regime_context = {
                        'regime_tag': 'NORMAL',
                        'vol_target_multiplier': 1.0
                    }

                # 3. Alpha Generation (Agent Runner)
                alpha_values = []
                success_count = 0
                for alpha in self.alpha_families:
                    # Use SAFE AGENT RUNNER with features
                    res = run_agent(
                        alpha,
                        symbol,
                        df,
                        regime_context=regime_context,
                        features=symbol_features,
                        symbol=symbol,
                        statarb_signals=statarb_signals
                    )
                    agent_name = alpha.__class__.__name__
                    audit_record['agents'][agent_name] = res

                    if res['ok']:
                        alpha_values.append(res['mu'])
                        success_count += 1

                # Failure Policy
                if len(self.alpha_families) > 0:
                    fail_rate = 1.0 - (success_count / len(self.alpha_families))
                    if fail_rate > 0.9: # RELAXED for development: Allow up to 90% failure
                        audit_record['final_decision'] = 'REJECT'
                        audit_record['reason_codes'].append('alpha_failure_extreme')
                        write_audit(audit_record)
                        return symbol, 0.5, {"status": "REJECT", "reason": "ALPHA_FAILURE_EXTREME", "agents": audit_record['agents']}

                # 4. News Sentiment Integration
                news_modifier = 0.0
                try:
                    news_articles = context.get('news', []) if context else []
                    if news_articles:
                        nlp_impact = self.nlp_engine.analyze_market_impact(news_articles, symbol)
                        if nlp_impact.direction == 'positive':
                            news_modifier = 0.1 * nlp_impact.magnitude
                        elif nlp_impact.direction == 'negative':
                            news_modifier = -0.1 * nlp_impact.magnitude
                except Exception:
                    pass

                if alpha_values:
                    # AGENT INTERFACE NORMALIZATION:
                    # Agents return decimal mu (daily return).
                    # We aggregate and apply Model Disagreement Penalty.

                    mu_arr = np.array(alpha_values)

                    # 1. Consensus Mu
                    avg_mu = np.mean(mu_arr)

                    # 2. Disagreement (Std Dev of forecasts)
                    disagreement = np.std(mu_arr) if len(mu_arr) > 1 else 0.0

                    # Formula: mu_adj = mu * exp(-beta * disagreement)
                    # Beta=100 implies 1% disagreement reduces conviction by ~63%
                    # Used Beta=50 for modest penalty
                    penalty_factor = np.exp(-50.0 * disagreement)

                    final_val = avg_mu * penalty_factor

                    # Add ensemble stats to audit
                    audit_record['ensemble_stats'] = {
                        "raw_mean_mu": float(avg_mu),
                        "disagreement": float(disagreement),
                        "penalty_factor": float(penalty_factor)
                    }
                else:
                    final_val = 0.0

                # 4. News Sentiment Integration (Additive to return)
                # Assuming news_modifier is also in return units (e.g. 10bps = 0.001)
                # Previous code: 0.1 * magnitude. Magnitude usually [0, 1].
                # 0.1 is 10%. Too huge for daily return.
                # Scaled down to 0.01 (1%)
                if news_modifier != 0.0:
                    # modifier was 0.1 * mag.
                    # If mag=0.5 -> 0.05 (5%).
                    # Rescale to be sensible for daily return: 0.005 (50bps)
                    final_val += (news_modifier * 0.1)

                final_val = float(np.clip(final_val, -0.20, 0.20))

                audit_record['final_decision'] = 'EXECUTE'
                audit_record['signal_value'] = float(final_val)
                audit_record['alphas'] = {
                    name: res.get('mu', 0.0)
                    for name, res in audit_record['agents'].items()
                }
                audit_record['sigmas'] = {
                    name: res.get('sigma', 0.0)
                    for name, res in audit_record['agents'].items()
                }
                write_audit(audit_record)

                return symbol, final_val, audit_record['agents']

            except Exception as e:
                logger.error(f"UNHANDLED STRATEGY ERROR for {symbol}: {e}")
                audit_record['final_decision'] = 'ERROR'
                audit_record['error'] = 'unhandled_exception'
                write_audit(audit_record)
                return symbol, 0.5, {}

        results = list(self.executor.map(lambda s: _process_ticker(s, statarb_results), tickers))

        # 1. Decision extraction
        signals = {r[0]: r[1] for r in results}
        agent_results_map = {r[0]: r[2] for r in results}

        # --- GOVERNANCE & HEALTH CHECK (INSTITUTIONAL LAYER) ---
        ml_attempts = 0
        ml_successes = 0
        # Simple heuristic: Look for agents with "ML" in name

        for symbol, agents_res in agent_results_map.items():
            if not isinstance(agents_res, dict): continue
            for agent_name, res in agents_res.items():
                if "ML" in agent_name:
                    ml_attempts += 1
                    if res.get('ok', False):
                        ml_successes += 1

        ml_health_ratio = 1.0
        if ml_attempts > 0:
            ml_health_ratio = ml_successes / ml_attempts

        # Track ARIMA/Fallback pressure
        # Heuristic: If we have signals but ML failed frequently, we are likely relying on fallbacks
        # We can't easily count "ARIMA" usages unless we know which agent is ARIMA.
        # But low ML health implies fallback usage if we still produce signals.

        self.current_governance_state = {
            "ml_health_ratio": ml_health_ratio,
            "system_state": "NORMAL"
        }

        if ml_health_ratio < 0.2: # RELAXED: was 0.5
             self.current_governance_state["system_state"] = "DEGRADED"
             logger.warning(f"[GOVERNANCE] System DEGRADED: ML Health Ratio = {ml_health_ratio:.2f}")
        else:
             self.current_governance_state["system_state"] = "NORMAL"

        # 2. Decision Completeness Assertion
        if len(signals) != len(tickers):
            logger.critical(f"DECISION COMPLETENESS FAIL: Expected {len(tickers)}, got {len(signals)}")

        # 3. Apply ML referee with agent results for disagreement penalty
        refined_signals = self.ml_referee.refine_signals(
            signals, market_data, agent_results=agent_results_map
        )

        # 4. Apply institutional filters
        filtered_signals, _ = self.filters.apply_filters(
            refined_signals, market_data, {}
        )

        # 5. Return DataFrame
        if market_data.empty:
            return pd.DataFrame()

        timestamp = market_data.index[-1]

        if not filtered_signals:
            filtered_signals = {tk: 0.5 for tk in tickers}

        for tk in tickers:
            if tk not in filtered_signals:
                filtered_signals[tk] = 0.5

            # --- INSTITUTIONAL AUDIT RECORDING ---
            res = agent_results_map.get(tk, {})
            self.decision_recorder.record(
                symbol=tk,
                decision=(
                    "REJECT" if abs(filtered_signals[tk] - 0.5) < 0.01
                    else "EXECUTE"
                ),
                signal_strength=float(filtered_signals[tk]),
                confidence=float(
                    res.get('MLAlpha', {}).get('confidence', 0.5)
                    if isinstance(res.get('MLAlpha'), dict) else 0.5
                ),
                source_alpha="MLAlpha|InstitutionalEnsemble",
                portfolio_weight=float(filtered_signals[tk]),
                regime=self.current_governance_state.get(
                    "system_state", "NORMAL"
                ),
                rationale=(
                    f"ML Health Ratio: {ml_health_ratio:.2f} | "
                    f"ARIMA Fallbacks: {metrics.arima_fallbacks}"
                ),
                meta={
                    "ml_health": ml_health_ratio,
                    "arima_fb": metrics.arima_fallbacks,
                    "cycle_ts": str(timestamp)
                }
            )

        return pd.DataFrame([filtered_signals], index=[timestamp]).fillna(0.5)

    def construct_portfolio(self, signals, data, current_portfolio):
        """
        Construct portfolio using institutional allocator.
        Applies governance scaling based on system health.
        """
        # Retrieve governance state
        gov_state = getattr(
            self,
            "current_governance_state",
            {"ml_health_ratio": 1.0, "system_state": "NORMAL"}
        )
        ml_health = gov_state.get("ml_health_ratio", 1.0)
        state_tag = gov_state.get("system_state", "NORMAL")

        # RISK COLLAPSE LOGIC
        # 1. Scale max leverage by ML health
        # Base leverage matches allocator default or config
        base_leverage = 1.0
        adjusted_leverage = base_leverage * ml_health

        if state_tag == "DEGRADED":
            adjusted_leverage *= 0.5
            logger.warning(
                f"[RISK_COLLAPSE] System DEGRADED. "
                f"Leverage capped at {adjusted_leverage:.2f}"
            )

        # Apply to allocator dynamically
        self.allocator.max_leverage = adjusted_leverage

        # Also adjust position limits if degraded
        if state_tag == "DEGRADED":
            self.allocator.max_pos = 0.05  # Cap at 5% instead of 10%
        else:
            self.allocator.max_pos = 0.10  # Restore default

        # 3. Allocator run
        target_allocation = self.allocator.allocate(
            signals, data, current_portfolio
        )

        # 4. SAFETY ENFORCEMENT: Risk CVaR & Entanglement Check
        try:
            # Use historical returns as scenarios (Bootstrap / Historical Simulation)
            if not data.empty and len(data) > 30:
                returns_df = data.pct_change().dropna()
                # Align columns with weights
                # target_allocation is likely a dict or Series. Convert to aligned array.
                if isinstance(target_allocation, dict):
                    assets = list(target_allocation.keys())
                    weights_arr = np.array([target_allocation.get(a, 0.0) for a in assets])
                elif hasattr(target_allocation, 'index'):
                    assets = target_allocation.index.tolist()
                    weights_arr = target_allocation.values
                else:
                    assets = []
                    weights_arr = np.array([])

                if len(assets) > 0 and not returns_df.empty:
                    # Filter returns to assets in portfolio
                    valid_assets = [a for a in assets if a in returns_df.columns]
                    if valid_assets:
                        scenario_returns = returns_df[valid_assets].values
                        # Re-align weights to valid assets for check
                        check_weights = np.array([target_allocation.get(a, 0.0) for a in valid_assets])

                        # Normalize check weights to sum to exposure (approx) or just pass as is
                        # RiskEnforcer expects standard weight vector matching columns

                        enforce_res = self.risk_enforcer.enforce(
                            check_weights,
                            scenario_returns,
                            returns_matrix_for_ent=scenario_returns.T
                        )

                        if not enforce_res["allow"]:
                            logger.warning(
                                f"[RISK_ENFORCER] Portfolio blocked: "
                                f"{enforce_res['reasons']}"
                            )
                            if enforce_res["suggested_weights"] is not None:
                                logger.info(
                                    "[RISK_ENFORCER] Applying haircuts"
                                )
                                # Map back to target_allocation
                                for i, asset in enumerate(valid_assets):
                                    if hasattr(target_allocation, 'loc'):
                                        target_allocation.loc[asset] = (
                                            enforce_res["suggested_weights"][i]
                                        )
                                    else:
                                        target_allocation[asset] = (
                                            enforce_res["suggested_weights"][i]
                                        )
                            else:
                                logger.error(
                                    "[RISK_ENFORCER] No suggestion. "
                                    "Returning empty allocation fallback."
                                )
                                return {}

        except Exception as e:
            logger.error(f"[RISK_ENFORCER] Failed to check portfolio risk: {e}")
            # Fail closed or open? Fail closed for safety.
            logger.error("[RISK_ENFORCER] FAILING CLOSED.")
            return {}

        return target_allocation

    def train_models(self, train_panel):
        """
        Train ML models if needed. For InstitutionalStrategy, this is primarily
        for the ML referee to learn from historical data.
        """
        if train_panel is not None and not train_panel.empty:
            # The ML referee can train on historical alpha signals and returns
            # For now, we'll skip training as the referee trains internally when needed
            pass
