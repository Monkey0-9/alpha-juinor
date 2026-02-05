"""
Hypothesis Manager
==================

Database and workflow manager for trading hypotheses.
Tracks ideas from 'Backlog' to 'Live'.
"""

import sqlite3
import pandas as pd
import json
import logging
from dataclasses import dataclass
from typing import Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class Hypothesis:
    id: str
    name: str
    description: str
    code_snippet: str  # The actual logic/function name
    status: str       # 'BACKLOG', 'TESTING', 'ACTIVE', 'REJECTED'
    metrics: Dict     # Stored as JSON
    created_at: str
    updated_at: str

class HypothesisManager:
    def __init__(self, db_path: str = "research/hypotheses.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS hypotheses (
                id TEXT PRIMARY KEY,
                name TEXT,
                description TEXT,
                code_snippet TEXT,
                status TEXT,
                metrics TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def add_hypothesis(self, name: str, description: str, code: str) -> str:
        """Register a new hypothesis."""
        hyp_id = f"HYP-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        now = datetime.now().isoformat()

        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute(
                "INSERT INTO hypotheses VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (hyp_id, name, description, code, 'BACKLOG', '{}', now, now)
            )
            conn.commit()

        logger.info(f"Registered Hypothesis {hyp_id}: {name}")
        return hyp_id

    def update_metrics(self, hyp_id: str, metrics: Dict):
        """Update backtest metrics and auto-transition status."""
        status = 'TESTING'
        # Simple Gateway Logic
        if metrics.get('sharpe', 0) > 0.0:
            status = 'PROVEN' # Candidate for production
        elif metrics.get('sharpe', 0) <= 0.0:
             status = 'REJECTED'

        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute(
                "UPDATE hypotheses SET metrics = ?, status = ?, updated_at = ? WHERE id = ?",
                (json.dumps(metrics), status, datetime.now().isoformat(), hyp_id)
            )
            conn.commit()

    def get_hypothesis(self, hyp_id: str) -> Optional[Hypothesis]:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT * FROM hypotheses WHERE id = ?", (hyp_id,))
        row = c.fetchone()
        conn.close()

        if row:
            return Hypothesis(
                id=row[0], name=row[1], description=row[2], code_snippet=row[3],
                status=row[4], metrics=json.loads(row[5]), created_at=row[6], updated_at=row[7]
            )
        return None

    def list_hypotheses(self) -> pd.DataFrame:
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM hypotheses", conn)
        conn.close()
        return df
