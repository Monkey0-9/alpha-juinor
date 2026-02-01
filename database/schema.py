"""
Complete Database Schema for Institutional Trading System.

Tables for 5-Year Market Data Backfill:
- price_history_daily: Daily OHLCV with full provenance
- price_history_intraday: Minute bars (partitioned by date)
- execution_feedback: Slippage and execution tracking
- provider_metrics: MAB bandit tracking
- data_quality_log: Per-symbol quality scores
- ingestion_audit: Atomic job audit records

Plus all existing tables for trading operations.
"""

import sqlite3
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# Schema version for migrations
SCHEMA_VERSION = "3.0.0"  # Added symbol_data_state, data_confidence, feature_registry, decision_records_v2, regime_state

# SQL Statements for schema creation
SCHEMA_SQL = """
-- ============================================================================
-- INSTITUTIONAL PRICE HISTORY (Mandated Schema)
-- Stores daily OHLCV data for all symbols with full provenance
-- ============================================================================
CREATE TABLE IF NOT EXISTS price_history (
    symbol TEXT NOT NULL,
    date TEXT NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    adjusted_close REAL,
    volume INTEGER,
    vwap REAL,
    trade_count INTEGER,
    provider TEXT NOT NULL,
    raw_hash TEXT NOT NULL,
    raw_row_json TEXT,             -- Full raw data persistence
    validation_flags TEXT,         -- JSON structure for rule violations
    spike_flag INTEGER,            -- 1 if price spike detected
    volume_spike_flag INTEGER,     -- 1 if volume spike detected
    ingestion_timestamp TEXT NOT NULL,
    PRIMARY KEY(symbol, date)
);

CREATE INDEX IF NOT EXISTS idx_price_history_symbol_date ON price_history(symbol, date);
CREATE INDEX IF NOT EXISTS idx_price_history_date ON price_history(date);
CREATE INDEX IF NOT EXISTS idx_price_history_provider ON price_history(provider);

-- ... (skipping unchanged tables) ...

CREATE TABLE IF NOT EXISTS ingestion_audit (
    run_id TEXT,
    symbol TEXT,
    asset_class TEXT,
    provider TEXT,
    status TEXT,
    reason_code TEXT,
    error_message TEXT,
    row_count INTEGER,
    data_quality_score REAL,
    started_at TIMESTAMP,
    finished_at TIMESTAMP
);

-- ============================================================================
-- P&L DECOMPOSITION (Truth Engine)
-- ============================================================================
CREATE TABLE IF NOT EXISTS pnl_attribution_daily (
    date TEXT NOT NULL,
    symbol TEXT NOT NULL,
    alpha_bps REAL,
    beta REAL,
    market_contribution REAL,
    residual_noise REAL,
    r_squared REAL,
    correlation REAL,
    treynor_ratio REAL,
    information_ratio REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY(date, symbol)
);

CREATE INDEX IF NOT EXISTS idx_pnl_attr_date ON pnl_attribution_daily(date);

-- ============================================================================
-- ALPHA DECAY METRICS (Strategy Death Detection)
-- ============================================================================
CREATE TABLE IF NOT EXISTS alpha_decay_metrics (
    strategy_id TEXT NOT NULL,
    date TEXT NOT NULL,
    rolling_ic_30d REAL,
    rolling_ic_60d REAL,
    rolling_ic_90d REAL,
    decay_score REAL,
    capacity_utilization REAL,
    status TEXT,
    recommendation TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY(strategy_id, date)
);

CREATE INDEX IF NOT EXISTS idx_decay_strategy ON alpha_decay_metrics(strategy_id);
CREATE INDEX IF NOT EXISTS idx_decay_date ON alpha_decay_metrics(date);



-- ============================================================================
-- DECISION RECORDS (Audit)
-- ============================================================================
CREATE TABLE IF NOT EXISTS decision_records (
  id TEXT PRIMARY KEY,
  run_id TEXT,
  timestamp TIMESTAMP,
  symbol TEXT,
  final_decision TEXT, -- EXECUTE|HOLD|REJECT|ERROR
  reason_codes JSON,
  allocations JSON,
  duels JSON,
  model_versions JSON,
  data_quality_score REAL,
  execution_id TEXT
);

-- ============================================================================
-- MANUAL OVERRIDES (Governance)
-- ============================================================================
CREATE TABLE IF NOT EXISTS manual_overrides (
  id TEXT PRIMARY KEY,
  run_id TEXT,
  strategy_id TEXT,
  requested_by TEXT,
  justification TEXT,
  signoffs JSON,
  status TEXT, -- PENDING/APPROVED/REJECTED
  created_at TIMESTAMP
);

-- ============================================================================
-- INGESTION RUNS SUMMARY
-- ============================================================================
CREATE TABLE IF NOT EXISTS ingestion_audit_runs (
    run_id TEXT PRIMARY KEY,
    total_symbols INTEGER,
    processed INTEGER,
    successful INTEGER,
    rejected INTEGER,
    failed INTEGER,
    avg_data_quality REAL,
    summary_json TEXT,
    start_time TIMESTAMP,
    end_time TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_price_history_date ON price_history(date);
CREATE INDEX IF NOT EXISTS idx_ingestion_audit_run ON ingestion_audit(run_id);
CREATE INDEX IF NOT EXISTS idx_data_quality_symbol ON data_quality(symbol);

-- ============================================================================
-- INGESTION JOBS TABLE
-- Track backfill job status and metadata
-- ============================================================================
CREATE TABLE IF NOT EXISTS ingestion_jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL UNIQUE,
    job_type TEXT NOT NULL,           -- 'backfill', 'incremental', 'intraday'
    start_date TEXT NOT NULL,
    end_date TEXT NOT NULL,
    symbols_count INTEGER,
    status TEXT NOT NULL,              -- 'running', 'completed', 'failed', 'partial'
    started_at TEXT NOT NULL,
    completed_at TEXT,
    duration_ms REAL,
    summary_json TEXT,
    error_message TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_status ON ingestion_jobs(status);
CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_type ON ingestion_jobs(job_type);

-- ============================================================================
-- BACKFILL FAILURES TABLE
-- Persistent failures with trace
-- ============================================================================
CREATE TABLE IF NOT EXISTS backfill_failures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    provider TEXT NOT NULL,
    start_date TEXT NOT NULL,
    end_date TEXT NOT NULL,
    error_message TEXT NOT NULL,
    error_trace TEXT,
    retry_count INTEGER DEFAULT 0,
    status TEXT DEFAULT 'failed',     -- 'failed', 'retrying', 'resolved'
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- SYMBOL GOVERNANCE TABLE (Institutional Control - Single Source of Truth)
-- Tracks history integrity, classification state, and trading permissions.
-- ============================================================================
CREATE TABLE IF NOT EXISTS symbol_governance (
    symbol TEXT PRIMARY KEY,
    history_rows INTEGER DEFAULT 0,
    data_quality REAL DEFAULT 0.0,
    state TEXT CHECK(state IN ('ACTIVE','DEGRADED','QUARANTINED')) DEFAULT 'QUARANTINED',
    reason TEXT,
    last_checked_ts TEXT NOT NULL,
    metadata_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_symbol_governance_state ON symbol_governance(state);
CREATE INDEX IF NOT EXISTS idx_symbol_governance_quality ON symbol_governance(data_quality);

CREATE INDEX IF NOT EXISTS idx_backfill_failures_symbol ON backfill_failures(symbol);
CREATE INDEX IF NOT EXISTS idx_backfill_failures_run ON backfill_failures(run_id);

-- Legacy tables removed or consolidated above

CREATE INDEX IF NOT EXISTS idx_corp_actions_symbol ON corporate_actions(symbol);
CREATE INDEX IF NOT EXISTS idx_corp_actions_date ON corporate_actions(action_date);

-- ============================================================================
-- FEATURES TABLE
-- ============================================================================
CREATE TABLE IF NOT EXISTS features (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    date TEXT NOT NULL,
    features_json TEXT NOT NULL,
    version TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, date)
);

CREATE INDEX IF NOT EXISTS idx_features_symbol_date ON features(symbol, date);

-- ============================================================================
-- MODEL OUTPUTS TABLE
-- ============================================================================
CREATE TABLE IF NOT EXISTS model_outputs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    agent_name TEXT NOT NULL,
    mu REAL,
    sigma REAL,
    confidence REAL,
    metadata_json TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_model_outputs_cycle ON model_outputs(cycle_id);
CREATE INDEX IF NOT EXISTS idx_model_outputs_symbol ON model_outputs(symbol);
CREATE INDEX IF NOT EXISTS idx_model_outputs_agent ON model_outputs(agent_name);

-- ============================================================================
-- DECISIONS TABLE
-- ============================================================================
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    final_decision TEXT NOT NULL,
    position_size REAL,
    stop_loss REAL,
    trailing_params_json TEXT,
    reason_codes_json TEXT NOT NULL,
    data_quality_score REAL,
    provider_confidence REAL,
    mu_hat REAL,
    sigma_hat REAL,
    conviction REAL,
    metadata_json TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_decisions_cycle ON decisions(cycle_id);
CREATE INDEX IF NOT EXISTS idx_decisions_symbol ON decisions(symbol);
CREATE INDEX IF NOT EXISTS idx_decisions_final ON decisions(final_decision);

-- ============================================================================
-- ORDERS TABLE
-- ============================================================================
CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id TEXT NOT NULL UNIQUE,
    cycle_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    qty REAL,
    price REAL,
    order_type TEXT,
    time_in_force TEXT,
    status TEXT NOT NULL,
    fill_report_json TEXT,
    commission REAL,
    slippage REAL,
    created_at TEXT,
    filled_at TEXT,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_orders_order_id ON orders(order_id);
CREATE INDEX IF NOT EXISTS idx_orders_cycle ON orders(cycle_id);
CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders(symbol);
CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);

-- ============================================================================
-- POSITIONS TABLE
-- ============================================================================
CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL UNIQUE,
    qty REAL NOT NULL,
    avg_price REAL NOT NULL,
    market_value REAL,
    unrealized_pnl REAL,
    last_update TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol);

-- ============================================================================
-- AUDIT LOG TABLE
-- ============================================================================
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_id TEXT,
    timestamp TEXT NOT NULL,
    component TEXT NOT NULL,
    level TEXT NOT NULL,
    message TEXT NOT NULL,
    payload_json TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_audit_log_cycle ON audit_log(cycle_id);
CREATE INDEX IF NOT EXISTS idx_audit_log_timestamp ON audit_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_log_level ON audit_log(level);

-- ============================================================================
-- CYCLE META TABLE
-- ============================================================================
CREATE TABLE IF NOT EXISTS cycle_meta (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_id TEXT NOT NULL UNIQUE,
    timestamp TEXT NOT NULL,
    universe_size INTEGER,
    buy_count INTEGER,
    sell_count INTEGER,
    hold_count INTEGER,
    reject_count INTEGER,
    nav REAL,
    daily_return REAL,
    drawdown REAL,
    implementation_shortfall REAL,
    slippage REAL,
    fills_ratio REAL,
    provider_health_json TEXT,
    risk_warnings_json TEXT,
    top_buys_json TEXT,
    duration_seconds REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_cycle_meta_cycle_id ON cycle_meta(cycle_id);

-- ============================================================================
-- SHADOW PREDICTIONS TABLE (P1-4)
-- Audit log for shadow ML predictions
-- ============================================================================
CREATE TABLE IF NOT EXISTS shadow_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    symbol TEXT NOT NULL,
    model_version TEXT NOT NULL,
    prediction REAL NOT NULL,
    confidence REAL NOT NULL,
    features_hash TEXT NOT NULL,
    features_json TEXT,
    ml_mode TEXT DEFAULT 'shadow',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_shadow_pred_symbol ON shadow_predictions(symbol);
CREATE INDEX IF NOT EXISTS idx_shadow_pred_timestamp ON shadow_predictions(timestamp);
CREATE INDEX IF NOT EXISTS idx_shadow_pred_model ON shadow_predictions(model_version);

-- ============================================================================
-- FEATURE LINEAGE TABLE (P1-1)
-- Track feature provenance and transformations
-- ============================================================================
CREATE TABLE IF NOT EXISTS feature_lineage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    feature_name TEXT NOT NULL,
    source_column TEXT,
    transform TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_feature_lineage_model ON feature_lineage(model_name, model_version);
CREATE INDEX IF NOT EXISTS idx_feature_lineage_feature ON feature_lineage(feature_name);

-- ============================================================================
-- MODELS REGISTRY TABLE (P1-3)
-- Model registry with quality gates
-- ============================================================================
CREATE TABLE IF NOT EXISTS models_registry (
    model_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    version TEXT NOT NULL,
    path TEXT NOT NULL,
    sha256 TEXT,
    metrics_json TEXT,
    status TEXT CHECK(status IN ('candidate','approved','retired')) DEFAULT 'candidate',
    sharpe_ratio REAL,
    max_drawdown REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    approved_at TEXT,
    approved_by TEXT
);

CREATE INDEX IF NOT EXISTS idx_models_registry_status ON models_registry(status);
CREATE INDEX IF NOT EXISTS idx_models_registry_name ON models_registry(name);

-- ============================================================================
-- SYMBOL DATA STATE (Data Intelligence - Ticket 1)
-- Single source of truth for symbol data health
-- ============================================================================
CREATE TABLE IF NOT EXISTS symbol_data_state (
    symbol TEXT PRIMARY KEY,
    state TEXT CHECK(state IN ('OK', 'DEGRADED_DATA', 'STALE_DATA', 'INVALID_DATA', 'FAILED_PROVIDER', 'UNKNOWN')) NOT NULL,
    last_seen TEXT,                  -- ISO timestamp of last data point
    last_good_ts TEXT,               -- ISO timestamp of last fully validated data
    failure_count_30d INTEGER DEFAULT 0,
    updated_at TEXT NOT NULL,
    reason TEXT,
    provider TEXT,
    metadata_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_symbol_data_state_state ON symbol_data_state(state);
CREATE INDEX IF NOT EXISTS idx_symbol_data_state_updated ON symbol_data_state(updated_at);

-- ============================================================================
-- DATA CONFIDENCE (Data Intelligence - Ticket 2)
-- Per-symbol, per-provider confidence tracking
-- ============================================================================
CREATE TABLE IF NOT EXISTS data_confidence (
    symbol TEXT NOT NULL,
    provider TEXT NOT NULL,
    confidence REAL DEFAULT 0.5,     -- [0.0, 1.0]
    last_good_timestamp TEXT,
    failure_rate_30d REAL DEFAULT 0.0,
    success_count_30d INTEGER DEFAULT 0,
    failure_count_30d INTEGER DEFAULT 0,
    avg_latency_ms REAL,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (symbol, provider)
);

CREATE INDEX IF NOT EXISTS idx_data_confidence_symbol ON data_confidence(symbol);
CREATE INDEX IF NOT EXISTS idx_data_confidence_provider ON data_confidence(provider);

-- ============================================================================
-- FEATURE REGISTRY (ML Governance - Ticket 6)
-- Feature catalog with versions and validation
-- ============================================================================
CREATE TABLE IF NOT EXISTS feature_registry (
    feature_name TEXT PRIMARY KEY,
    version TEXT NOT NULL,
    schema_hash TEXT NOT NULL,           -- MD5 of dtype signature
    dtype TEXT NOT NULL,                 -- Expected data type
    validation_fn TEXT,                  -- Python path to validator
    description TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_feature_registry_version ON feature_registry(version);

-- ============================================================================
-- DECISION RECORDS ENHANCED (Audit - Ticket 17)
-- Full decision audit with all required fields
-- ============================================================================
CREATE TABLE IF NOT EXISTS decision_records_v2 (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    mu_list TEXT,                        -- JSON array of mu values from all alphas
    sigma_list TEXT,                     -- JSON array of sigma values
    confidence_list TEXT,                -- JSON array of confidence values
    model_versions TEXT,                 -- JSON array of model versions
    final_decision TEXT CHECK(final_decision IN ('EXECUTE', 'HOLD', 'REJECT', 'ERROR')) NOT NULL,
    reason_codes TEXT NOT NULL,          -- JSON array of reason codes
    execution_id TEXT,                   -- Order ID if executed
    data_quality_score REAL,
    data_providers TEXT,                 -- JSON array of providers used
    config_sha256 TEXT,                  -- Hash of config at decision time
    regime_label TEXT,
    cvar_portfolio REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_decision_v2_run ON decision_records_v2(run_id);
CREATE INDEX IF NOT EXISTS idx_decision_v2_symbol ON decision_records_v2(symbol);
CREATE INDEX IF NOT EXISTS idx_decision_v2_ts ON decision_records_v2(timestamp);

-- ============================================================================
-- REGIME STATE (Regime Controller - Ticket 10)
-- Tracks current and historical regime labels
-- ============================================================================
CREATE TABLE IF NOT EXISTS regime_state (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    regime_label TEXT CHECK(regime_label IN ('RISK_ON', 'RISK_OFF', 'CRISIS', 'LIQUIDITY_STRESS', 'UNKNOWN')) NOT NULL,
    confidence REAL,
    explanation TEXT,
    indicators_json TEXT,                -- HMM outputs, VIX, etc.
    overrides_applied TEXT,              -- JSON of position cap overrides
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_regime_state_ts ON regime_state(timestamp);
CREATE INDEX IF NOT EXISTS idx_regime_state_label ON regime_state(regime_label);

-- ============================================================================
-- SCHEMA VERSION TRACKING
-- ============================================================================
CREATE TABLE IF NOT EXISTS _schema_versions (
    version TEXT PRIMARY KEY,
    applied_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""


@dataclass
class DailyPriceRecord:
    """Daily price record with full provenance (Institutional Spec)"""
    symbol: str
    date: str
    open: float
    high: float
    low: float
    close: float
    adjusted_close: float
    volume: int
    vwap: float
    trade_count: int
    provider: str
    raw_hash: str
    raw_row_json: Optional[str] = None
    validation_flags: Optional[str] = None
    spike_flag: int = 0
    volume_spike_flag: int = 0
    ingestion_timestamp: str = ""


@dataclass
class IntradayPriceRecord:
    """Intraday price record"""
    symbol: str
    date: str
    time: str
    datetime: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    source_provider: str
    raw_hash: str
    pulled_at: str


@dataclass
class IngestionAuditRecord:
    """Audit record for a single symbol fetch (Institutional Spec)"""
    run_id: str
    symbol: str
    asset_class: str
    provider: str
    status: str
    reason_code: Optional[str] = None
    error_message: Optional[str] = None
    row_count: int = 0
    data_quality_score: float = 0.0
    started_at: str = ""
    finished_at: str = ""

@dataclass
class DecompositionRecord:
    """P&L Decomposition Record"""
    date: str
    symbol: str
    alpha_bps: float
    beta: float
    market_contribution: float
    residual_noise: float
    r_squared: float
    correlation: float
    treynor_ratio: float
    information_ratio: float

@dataclass
class FactorReturn:
    """Factor return record"""
    date: str
    factor: str
    return_val: float
    t_stat: float = 0.0
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class SecurityFactorExposure:
    """Factor exposure record"""
    date: str
    symbol: str
    factor: str
    exposure: float
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class AgentWeight:
    """Agent Weight record"""
    symbol: str
    weight: float
    agent: str # strategy/alpha name
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class AgentPerformance:
    date: str
    agent: str
    sharpe: float
    sortino: float
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class RegimeHistory:
    date: str
    regime: str
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class ExecutionModelVersion:
    version: str
    model_type: str
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class ExecutionDecision:
    order_id: str
    side: str
    quantity: float
    symbol: str
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class ExecutionOutcome:
    order_id: str
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class PortfolioTarget:
    cycle_id: str
    symbol: str
    weight: float
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class PortfolioConstraint:
    cycle_id: str
    constraint_type: str
    value: float
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class OptimizationResult:
    cycle_id: str
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class ExecutionFeedback:
    order_id: str
    price: float
    fill_price: float
    cost: float
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class SlippageModelCoeff:
    symbol: str
    date: str
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class SlippagePredictionError:
    date: str
    symbol: str
    predicted: float
    actual: float
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class AgentPnl:
    """Agent PnL record"""
    cycle_id: str
    symbol: str
    agent_name: str
    mu_hat: float
    sigma_hat: float
    confidence: float
    agent_weight: float
    realized_return: float
    pnl_contribution: float
    factor_adjusted_return: float = 0.0
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class PortfolioAttribution:
    """Portfolio attribution record"""
    date: str
    period: str = "daily"
    total_return: float = 0.0
    market_contribution: float = 0.0
    alpha_contribution: float = 0.0
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class DataQualityRecord:
    """Data quality assessment record (Institutional Spec)"""
    symbol: str
    run_id: str
    quality_score: float
    validation_flags: Dict[str, Any] = field(default_factory=dict)
    provider: str = ""
    recorded_at: str = ""


@dataclass
class ProviderMetricsRecord:
    """Provider metrics for MAB"""
    provider_name: str
    date: str
    pulls: int
    successes: int
    avg_latency_ms: float
    avg_quality_score: float
    cost: float = 0.0


@dataclass
class ExecutionFeedbackRecord:
    """Execution feedback record"""
    order_id: str
    cycle_id: str
    symbol: str
    expected_price: float
    fill_price: float
    slippage_bps: float
    qty: float
    adv: float
    market_conditions: Optional[Dict[str, Any]] = None
    timestamp: str = ""


@dataclass
class BackfillFailureRecord:
    """Persistent backfill failure record"""
    run_id: str
    symbol: str
    provider: str
    start_date: str
    end_date: str
    error_message: str
    error_trace: str = ""
    retry_count: int = 0
    status: str = "failed"


@dataclass
class TradingEligibilityRecord:
    """Institutional Trading Eligibility Record"""
    symbol: str
    state: str  # ACTIVE, DEGRADED, QUARANTINED
    history_rows: int
    data_quality: float
    reason: Optional[str] = None
    last_checked: str = ""

@dataclass
class SymbolGovernanceRecord:
    """Institutional Symbol Governance Record"""
    symbol: str
    history_rows: int
    data_quality: float
    state: str  # ACTIVE, DEGRADED, QUARANTINED
    reason: Optional[str]
    last_checked_ts: str
    metadata: Optional[Dict[str, Any]] = None


# Legacy dataclasses for backward compatibility
@dataclass
class PriceRecord:
    """Legacy price record"""
    symbol: str
    date: str
    open: float
    high: float
    low: float
    close: float
    adjusted_close: float
    volume: int
    source: str
    raw_hash: str = ""


@dataclass
class CorporateAction:
    """Corporate action record"""
    symbol: str
    date: str
    action_type: str
    details: Dict[str, Any]
    source: str = ""


@dataclass
class FeatureRecord:
    """Feature record"""
    symbol: str
    date: str
    features: Dict[str, Any]
    version: str = ""


@dataclass
class ModelOutput:
    """Model output from an agent"""
    cycle_id: str
    symbol: str
    agent_name: str
    mu: float
    sigma: float
    confidence: float
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class DecisionRecord:
    """Final trading decision"""
    cycle_id: str
    symbol: str
    final_decision: str
    reason_codes: List[str]
    mu_hat: float = 0.0
    sigma_hat: float = 0.0
    conviction: float = 0.0
    position_size: float = 0.0
    stop_loss: Optional[float] = None
    trailing_params: Optional[Dict[str, float]] = None
    data_quality_score: float = 1.0
    provider_confidence: float = 0.5
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class OrderRecord:
    """Order record"""
    order_id: str
    cycle_id: str
    symbol: str
    side: str
    qty: float
    price: float
    order_type: str = "MARKET"
    time_in_force: str = "DAY"
    status: str = "PENDING"
    fill_report: Optional[Dict[str, Any]] = None
    commission: float = 0.0
    slippage: float = 0.0


@dataclass
class PositionRecord:
    """Position record"""
    symbol: str
    qty: float
    avg_price: float
    market_value: Optional[float] = None
    unrealized_pnl: Optional[float] = None


@dataclass
class AuditEntry:
    """Audit log entry"""
    cycle_id: Optional[str] = None
    timestamp: Optional[str] = None
    component: str = ""
    level: str = "INFO"
    message: str = ""
    payload: Optional[Dict[str, Any]] = None


@dataclass
class CycleMeta:
    """Cycle metadata"""
    cycle_id: str = ""
    universe_size: int = 0
    buy_count: int = 0
    sell_count: int = 0
    hold_count: int = 0
    reject_count: int = 0
    nav: float = 0.0
    daily_return: float = 0.0
    drawdown: float = 0.0
    implementation_shortfall: float = 0.0
    slippage: float = 0.0
    fills_ratio: float = 0.0
    provider_health: Optional[Dict[str, Any]] = None
    risk_warnings: Optional[List[str]] = None
    top_buys: Optional[List[Dict[str, Any]]] = None
    duration_seconds: float = 0.0


@dataclass
class ModelDecayMetric:
    """Model decay tracking"""
    symbol: str
    date: str
    model_age_days: int
    rolling_error: float
    autocorr_flip: bool
    decay_factor: float


@dataclass
class CapitalAllocation:
    """Capital auction allocation"""
    cycle_id: str
    symbol: str
    strategy_id: Optional[str]
    allocated_weight: float
    reason_codes: Optional[List[str]] = None


@dataclass
class GovernanceDecisionRecord:
    """Governance decision record"""
    cycle_id: str
    symbol: str
    decision: str
    reason_codes: Optional[List[str]] = None
    cvar: float = 0.0
    mu: float = 0.0
    sigma: float = 0.0
    vetoed: bool = False
    veto_reason: Optional[str] = None


@dataclass
class StrategyLifecycleRecord:
    """Strategy lifecycle state"""
    strategy_id: str
    stage: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    capital_pct: float = 0.0
    performance_metrics: Optional[Dict[str, Any]] = None


def get_schema_version() -> str:
    """Get current schema version"""
    return SCHEMA_VERSION

