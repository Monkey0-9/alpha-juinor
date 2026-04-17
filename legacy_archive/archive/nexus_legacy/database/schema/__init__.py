"""
Complete Database Schema for Institutional Trading System.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

# Schema version for migrations
SCHEMA_VERSION = "3.0.0"

# SQL Statements for schema creation
SCHEMA_SQL = """
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
    raw_row_json TEXT,
    validation_flags TEXT,
    spike_flag INTEGER,
    volume_spike_flag INTEGER,
    ingestion_timestamp TEXT NOT NULL,
    PRIMARY KEY(symbol, date)
);

CREATE TABLE IF NOT EXISTS corporate_actions (
    symbol TEXT NOT NULL,
    action_date TEXT NOT NULL,
    action_type TEXT NOT NULL,
    action_details TEXT,
    provider TEXT,
    ingestion_timestamp TEXT,
    PRIMARY KEY(symbol, action_date, action_type)
);

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

CREATE TABLE IF NOT EXISTS data_quality (
    symbol TEXT NOT NULL,
    run_id TEXT NOT NULL,
    quality_score REAL NOT NULL,
    validation_flags TEXT,
    provider TEXT,
    recorded_at TEXT NOT NULL,
    PRIMARY KEY(symbol, recorded_at)
);

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

CREATE TABLE IF NOT EXISTS decision_records (
  id TEXT PRIMARY KEY,
  run_id TEXT,
  timestamp TIMESTAMP,
  symbol TEXT,
  final_decision TEXT,
  reason_codes JSON,
  allocations JSON,
  duels JSON,
  model_versions JSON,
  data_quality_score REAL,
  execution_id TEXT
);

CREATE TABLE IF NOT EXISTS features (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    date TEXT NOT NULL,
    features_json TEXT NOT NULL,
    version TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, date)
);

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

CREATE TABLE IF NOT EXISTS symbol_governance (
    symbol TEXT PRIMARY KEY,
    history_rows INTEGER DEFAULT 0,
    data_quality REAL DEFAULT 0.0,
    state TEXT CHECK(state IN ('ACTIVE','DEGRADED','QUARANTINED')) DEFAULT 'QUARANTINED',
    reason TEXT,
    last_checked_ts TEXT NOT NULL,
    metadata_json TEXT
);

CREATE TABLE IF NOT EXISTS _schema_versions (
    version TEXT PRIMARY KEY,
    applied_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

@dataclass
class DailyPriceRecord:
    symbol: str
    date: str
    open: float
    high: float
    low: float
    close: float
    adjusted_close: float
    volume: int
    vwap: Optional[float] = 0.0
    trade_count: Optional[int] = 0
    provider: str = ""
    raw_hash: str = ""
    raw_row_json: Optional[str] = None
    validation_flags: Optional[str] = None
    spike_flag: int = 0
    volume_spike_flag: int = 0
    ingestion_timestamp: str = ""

@dataclass
class IntradayPriceRecord:
    symbol: str
    date: str
    time: str
    datetime: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    source_provider: str = ""
    raw_hash: str = ""
    pulled_at: str = ""

@dataclass
class IngestionAuditRecord:
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
class CorporateAction:
    symbol: str
    date: str
    action_type: str
    details: Dict[str, Any]
    source: str = ""

@dataclass
class DataQualityRecord:
    symbol: str
    run_id: str
    quality_score: float
    validation_flags: Dict[str, Any] = field(default_factory=dict)
    provider: str = ""
    recorded_at: str = ""

@dataclass
class SymbolGovernanceRecord:
    symbol: str
    history_rows: int
    data_quality: float
    state: str
    reason: Optional[str] = None
    last_checked_ts: str = ""
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class PriceRecord:
    symbol: str
    date: str
    open: float
    high: float
    low: float
    close: float
    adjusted_close: float
    volume: int
    source: str = ""
    raw_hash: str = ""

@dataclass
class FeatureRecord:
    symbol: str
    date: str
    features: Dict[str, Any]
    version: str = ""

@dataclass
class ModelOutput:
    cycle_id: str
    symbol: str
    agent_name: str
    mu: float
    sigma: float
    confidence: float
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class DecisionRecord:
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
    symbol: str
    qty: float
    avg_price: float
    market_value: Optional[float] = None
    unrealized_pnl: Optional[float] = None

@dataclass
class AuditEntry:
    cycle_id: Optional[str] = None
    timestamp: Optional[str] = None
    component: str = ""
    level: str = "INFO"
    message: str = ""
    payload: Optional[Dict[str, Any]] = None

@dataclass
class CycleMeta:
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
class FactorReturn:
    date: str
    factor: str
    return_val: float
    t_stat: float = 0.0
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class SecurityFactorExposure:
    date: str
    symbol: str
    factor: str
    exposure: float
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class AgentPnl:
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
    date: str
    period: str = "daily"
    total_return: float = 0.0
    market_contribution: float = 0.0
    alpha_contribution: float = 0.0
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class AgentWeight:
    symbol: str
    weight: float
    agent: str
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
class ExecutionFeedbackRecord:
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
class ProviderMetricsRecord:
    provider_name: str
    date: str
    pulls: int
    successes: int
    avg_latency_ms: float
    avg_quality_score: float
    cost: float = 0.0

@dataclass
class BackfillFailureRecord:
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
    symbol: str
    state: str
    history_rows: int
    data_quality: float
    reason: Optional[str] = None
    last_checked: str = ""

@dataclass
class ModelDecayMetric:
    symbol: str
    date: str
    model_age_days: int
    rolling_error: float
    autocorr_flip: bool
    decay_factor: float

@dataclass
class CapitalAllocation:
    cycle_id: str
    symbol: str
    strategy_id: Optional[str]
    allocated_weight: float
    reason_codes: Optional[List[str]] = None

@dataclass
class GovernanceDecisionRecord:
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
    strategy_id: str
    stage: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    capital_pct: float = 0.0
    performance_metrics: Optional[Dict[str, Any]] = None

def get_schema_version() -> str:
    return SCHEMA_VERSION
