-- Data Confidence & Provider Failure Tracking Schema
-- Supports intelligent degradation and circuit-breaking

-- Table 1: Data Confidence per Symbol
CREATE TABLE IF NOT EXISTS data_confidence (
    symbol TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    confidence_score REAL NOT NULL CHECK(confidence_score >= 0.0 AND confidence_score <= 1.0),
    last_good_date TEXT,
    failure_rate_30d REAL DEFAULT 0.0,
    consecutive_failures INTEGER DEFAULT 0,
    last_updated TEXT DEFAULT CURRENT_TIMESTAMP,
    state TEXT DEFAULT 'OK' CHECK(state IN ('OK', 'DEGRADED', 'STALE', 'INVALID'))
);

CREATE INDEX IF NOT EXISTS idx_confidence_state ON data_confidence(state);
CREATE INDEX IF NOT EXISTS idx_confidence_score ON data_confidence(confidence_score);

-- Table 2: Provider Failure Log
CREATE TABLE IF NOT EXISTS provider_failures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    provider TEXT NOT NULL,
    symbol TEXT,
    error_code INTEGER,
    error_message TEXT,
    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
    failure_type TEXT CHECK(failure_type IN ('ENTITLEMENT', 'TIMEOUT', 'INVALID_DATA', 'RATE_LIMIT'))
);

CREATE INDEX IF NOT EXISTS idx_provider_failures_provider ON provider_failures(provider);
CREATE INDEX IF NOT EXISTS idx_provider_failures_timestamp ON provider_failures(timestamp);

-- Trigger: Auto-update last_updated on confidence changes
CREATE TRIGGER IF NOT EXISTS update_confidence_timestamp
AFTER UPDATE ON data_confidence
FOR EACH ROW
BEGIN
    UPDATE data_confidence
    SET last_updated = CURRENT_TIMESTAMP
    WHERE symbol = NEW.symbol;
END;
