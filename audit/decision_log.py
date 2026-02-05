import json
import os
import threading
import sqlite3
import pandas as pd
import logging
from typing import Dict, Any
from contracts import DecisionRecord
from dataclasses import asdict

logger = logging.getLogger(__name__)
lock = threading.Lock()
audit_queue = [] # Simple list for batching if needed, but we'll use a real queue
import queue
_q = queue.Queue()

# Ensure runtime dir exists
AUDIT_DIR = 'runtime'
if not os.path.exists(AUDIT_DIR):
    try:
        os.makedirs(AUDIT_DIR)
    except:
        pass

AUDIT_JSONL_PATH = os.path.join(AUDIT_DIR, 'audit.log')
AUDIT_DB_PATH = os.path.join(AUDIT_DIR, 'audit.db')


class SystemHalt(Exception):
    """
    Critical system failure requiring immediate halt.
    Raised when audit DB cannot be initialized or writes fail.
    """
    pass


# Initialize SQLite database
def _init_db():
    """
    Initialize SQLite database with schema.
    CRITICAL: Raises SystemHalt on failure - trading cannot proceed without audit DB.
    """
    try:
        conn = sqlite3.connect(AUDIT_DB_PATH)
        conn.execute("PRAGMA journal_mode=WAL")
        cursor = conn.cursor()

        # Read schema
        schema_path = os.path.join('audit', 'decision_schema.sql')
        if os.path.exists(schema_path):
            with open(schema_path, 'r') as f:
                schema_content = f.read()

            # Execute schema statements
            cursor.executescript(schema_content)
            logger.info(f"[OK] Schema loaded from file: {schema_path}")
        else:
            # Fallback inline schema
            logger.warning(f"Schema file not found at {schema_path}, using inline schema")
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cycle_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    data_providers TEXT NOT NULL,
                    alphas TEXT NOT NULL,
                    sigmas TEXT NOT NULL,
                    conviction REAL NOT NULL,
                    conviction_zscore REAL NOT NULL,
                    risk_checks TEXT NOT NULL,
                    pm_override TEXT NOT NULL,
                    final_decision TEXT NOT NULL,
                    reason_codes TEXT NOT NULL,
                    order_data TEXT,
                    raw_traceback TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_cycle_id ON decisions(cycle_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_symbol ON decisions(symbol)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_final_decision ON decisions(final_decision)')

        # Enforce Append-Only via triggers
        cursor.execute('''
            CREATE TRIGGER IF NOT EXISTS prevent_decision_delete
            BEFORE DELETE ON decisions
            BEGIN
                SELECT RAISE(FAIL, "Audit log is append-only. Deletion not allowed.");
            END;
        ''')
        cursor.execute('''
            CREATE TRIGGER IF NOT EXISTS prevent_decision_update
            BEFORE UPDATE ON decisions
            BEGIN
                SELECT RAISE(FAIL, "Audit log is append-only. Updates not allowed.");
            END;
        ''')

        # QUANTUM MIGRATION: Add columns if missing
        # We try to add columns one by one. If they exist, it throws, we catch.
        # SQLite doesn't support IF NOT EXISTS in ADD COLUMN standardly in all versions,
        # but ignoring error is safe enough for this script.
        migration_cols = [
            ("quantum_state", "TEXT"),
            ("regime", "TEXT"),
            ("entanglement_score", "REAL")
        ]

        for col, dtype in migration_cols:
            try:
                cursor.execute(f"ALTER TABLE decisions ADD COLUMN {col} {dtype}")
                logger.info(f"[MIGRATION] Added column {col} to decisions")
            except Exception:
                # Column likely exists
                pass

        conn.commit()
        conn.close()
        logger.info(f"[OK] Audit DB initialized (Append-Only) at {AUDIT_DB_PATH}")


    except Exception as e:
        error_msg = f"CRITICAL: Audit DB initialization failed: {e}"
        logger.critical(error_msg)
        logger.critical("SYSTEM HALT: Cannot proceed without audit database")
        raise SystemHalt(error_msg)

def _audit_worker():
    """Background worker to process audit writes without blocking strategy."""
    logger.info("Audit worker started.")
    while True:
        try:
            record = _q.get()
            if record is None: # Shutdown signal
                break

            _internal_write_audit(record)
            _q.task_done()
        except SystemHalt as sh:
            # CRITICAL: Re-raise to crash the entire system
            logger.critical(f"Audit worker detected SystemHalt: {sh}")
            logger.critical("Trading system must terminate immediately")
            import os
            os._exit(1)  # Force immediate termination
        except Exception as e:
            logger.error(f"Audit worker error: {e}")

# Start worker thread
worker = threading.Thread(target=_audit_worker, daemon=True)
worker.start()

def _map_execdecision_to_final(decision_str: str) -> str:
    """Map internal SKIP_* codes to DB-safe final_decision values."""
    if not isinstance(decision_str, str):
        decision_str = str(decision_str)

    if decision_str == "EXECUTE":
        return "EXECUTE"
    if decision_str in ["REJECT", "HOLD", "ERROR"]:
        return decision_str

    # Group SKIP_* as HOLD or REJECT depending on reason
    if decision_str.startswith("SKIP_"):
        # REJECT for fundamental reasons (not tradable, risk zero)
        if any(x in decision_str for x in ["NOT_TRADABLE", "RISK_ZERO", "SHORTING"]):
            return "REJECT"
        # HOLD for transient reasons (too small, market closed, low confidence)
        return "HOLD"

    return "ERROR"

# Initialize on module load
try:
    _init_db()
except:
    pass

def _internal_write_audit(record):
    """Actual synchronous write logic (moved from write_audit)."""
    with lock:
        # 1. Write to JSONL
        try:
            if hasattr(record, '__dataclass_fields__'):
                record_dict = asdict(record)
            elif isinstance(record, dict):
                record_dict = record
            else:
                return

            if 'written_at' not in record_dict:
                record_dict['written_at'] = pd.Timestamp.now('UTC').isoformat()

            line = json.dumps(record_dict, default=str)
            with open(AUDIT_JSONL_PATH, 'a', encoding='utf-8') as f:
                f.write(line + "\n")
        except Exception as e:
            print(f"CRITICAL AUDIT JSONL FAILURE: {e}")

        # 2. Write to SQLite
        try:
            conn = sqlite3.connect(AUDIT_DB_PATH)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout = 5000")
            cursor = conn.cursor()

            if hasattr(record, 'cycle_id'):
                cycle_id = record.cycle_id
                symbol = record.symbol
                timestamp = record.timestamp
                data_providers = json.dumps(record.data_providers)
                alphas = json.dumps(record.alphas)
                sigmas = json.dumps(record.sigmas)
                conviction = record.conviction
                conviction_zscore = record.conviction_zscore
                risk_checks = json.dumps(record.risk_checks)
                pm_override = record.pm_override
                final_decision = _map_execdecision_to_final(record.decision if hasattr(record, 'decision') else record.final_decision)
                reason_codes = json.dumps(record.reason_codes)
                order_data = json.dumps(record.order) if record.order else None
                raw_traceback = record.raw_traceback

                # Quantum Fields
                quantum_state = json.dumps(record.quantum_state) if record.quantum_state else "{}"
                regime = record.regime if isinstance(record.regime, str) else json.dumps(record.regime)
                entanglement_score = record.entanglement_score
            else:
                record_dict = record
                cycle_id = record_dict.get('cycle_id', 'UNKNOWN')
                symbol = record_dict.get('symbol', 'UNKNOWN')
                timestamp = record_dict.get('timestamp', pd.Timestamp.now('UTC').isoformat())
                data_providers = json.dumps(record_dict.get('data_providers', {}))
                alphas = json.dumps(record_dict.get('alphas', {}))
                sigmas = json.dumps(record_dict.get('sigmas', {}))
                conviction = record_dict.get('conviction', 0.0)
                conviction_zscore = record_dict.get('conviction_zscore', 0.0)
                risk_checks = json.dumps(record_dict.get('risk_checks', []))
                pm_override = record_dict.get('pm_override', 'UNKNOWN')

                # Support both naming conventions
                raw_decision = record_dict.get('decision', record_dict.get('final_decision', 'ERROR'))
                final_decision = _map_execdecision_to_final(raw_decision)

                reason_codes = json.dumps(record_dict.get('reason_codes', []))
                order_data = json.dumps(record_dict.get('order')) if record_dict.get('order') else None
                raw_traceback = record_dict.get('raw_traceback')

                # Quantum Fields
                quantum_state = json.dumps(record_dict.get('quantum_state', {}))
                _regime_raw = record_dict.get('regime', 'UNKNOWN')
                regime = _regime_raw if isinstance(_regime_raw, str) else json.dumps(_regime_raw)
                entanglement_score = record_dict.get('entanglement_score', 0.0)

            cursor.execute('''
                INSERT INTO decisions (
                    cycle_id, symbol, timestamp,
                    data_providers, alphas, sigmas,
                    conviction, conviction_zscore,
                    risk_checks, pm_override,
                    final_decision, reason_codes,
                    order_data, raw_traceback,
                    quantum_state, regime, entanglement_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                cycle_id, symbol, timestamp,
                data_providers, alphas, sigmas,
                conviction, conviction_zscore,
                risk_checks, pm_override,
                final_decision, reason_codes,
                order_data, raw_traceback,
                quantum_state, regime, entanglement_score
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            error_msg = f"CRITICAL AUDIT DB FAILURE: {e}"
            logger.critical(error_msg)
            logger.critical("SYSTEM HALT: Cannot proceed without audit database writes")

            # Emergency fallback: Write to local queue log before halting
            try:
                queue_log = os.path.join(AUDIT_DIR, 'audit_queue.log')
                with open(queue_log, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(record, default=str) + "\n")
                logger.critical(f"Emergency audit record saved to {queue_log}")
            except:
                pass

            # HALT THE SYSTEM - Trading cannot proceed without audit trail
            raise SystemHalt(error_msg)

def write_audit(record):
    """Async wrapper for audit writes."""
    if not record:
        return
    _q.put(record)

def shutdown():
    """Wait for all audit records to be written."""
    logger.info("Flushing audit logs...")
    _q.put(None)
    worker.join()
    logger.info("Audit logs flushed.")

def get_cycle_decisions(cycle_id: str) -> list:
    """Retrieve all decisions for a cycle"""
    try:
        conn = sqlite3.connect(AUDIT_DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute('SELECT * FROM decisions WHERE cycle_id = ?', (cycle_id,))
        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]
    except Exception as e:
        print(f"Failed to query audit DB: {e}")
        return []

def get_decision_counts(cycle_id: str = None) -> Dict[str, int]:
    """Get decision counts, optionally filtered by cycle"""
    try:
        conn = sqlite3.connect(AUDIT_DB_PATH)
        cursor = conn.cursor()

        if cycle_id:
            cursor.execute('''
                SELECT final_decision, COUNT(*) as count
                FROM decisions
                WHERE cycle_id = ?
                GROUP BY final_decision
            ''', (cycle_id,))
        else:
            cursor.execute('''
                SELECT final_decision, COUNT(*) as count
                FROM decisions
                GROUP BY final_decision
            ''')

        results = {row[0]: row[1] for row in cursor.fetchall()}
        conn.close()

        return results
    except Exception as e:
        print(f"Failed to query decision counts: {e}")
        return {}
