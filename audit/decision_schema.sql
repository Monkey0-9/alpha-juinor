-- SQLite Schema for Decision Audit Log
-- Institutional-grade audit trail with immutable records
-- CRITICAL: Each statement executed separately to avoid migration failures

CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    timestamp TEXT NOT NULL,

    -- Data Providers (JSON)
    data_providers TEXT NOT NULL,

    -- Alpha Agents (JSON)
    alphas TEXT NOT NULL,
    sigmas TEXT NOT NULL,

    -- PM Brain
    conviction REAL NOT NULL,
    conviction_zscore REAL NOT NULL,

    -- Risk
    risk_checks TEXT NOT NULL,
    pm_override TEXT NOT NULL,

    -- Decision
    final_decision TEXT NOT NULL CHECK(final_decision IN ('EXECUTE', 'HOLD', 'REJECT', 'ERROR')),
    reason_codes TEXT NOT NULL,

    -- Order (JSON, nullable)
    order_data TEXT,

    -- Error Tracking
    raw_traceback TEXT,

    -- Audit Metadata
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Indexes (separate statements for safe migration)
CREATE INDEX IF NOT EXISTS idx_cycle_id ON decisions(cycle_id);

CREATE INDEX IF NOT EXISTS idx_symbol ON decisions(symbol);

CREATE INDEX IF NOT EXISTS idx_timestamp ON decisions(timestamp);

CREATE INDEX IF NOT EXISTS idx_final_decision ON decisions(final_decision);

-- View for easy querying
CREATE VIEW IF NOT EXISTS decision_summary AS
SELECT
    cycle_id,
    symbol,
    timestamp,
    final_decision,
    conviction,
    conviction_zscore,
    pm_override,
    json_extract(reason_codes, '$[0]') as primary_reason
FROM decisions;
