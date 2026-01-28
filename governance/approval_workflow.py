"""
governance/approval_workflow.py

Section F: Human Gov & Manual Overrides.
Requires 2 sign-offs to re-enable strategy.
"""

import json
import logging
from datetime import datetime
from database.manager import DatabaseManager

logger = logging.getLogger("GOV_WORKFLOW")

class ApprovalWorkflow:
    def __init__(self, db: DatabaseManager):
        self.db = db
        self._ensure_table()

    def _ensure_table(self):
        with self.db.get_connection() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS manual_overrides (
                  id TEXT PRIMARY KEY,
                  run_id TEXT,
                  strategy_id TEXT,
                  requested_by TEXT,
                  justification TEXT,
                  signoffs JSON,
                  status TEXT,
                  created_at TEXT
                );
            ''')

    def request_reenable(self, run_id: str, strategy_id: str, user: str, justification: str) -> str:
        """Submit a request to re-enable a strategy."""
        import uuid
        req_id = str(uuid.uuid4())

        with self.db.transaction() as conn:
            conn.execute('''
                INSERT INTO manual_overrides
                (id, run_id, strategy_id, requested_by, justification, signoffs, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (req_id, run_id, strategy_id, user, justification, json.dumps([]), "PENDING", datetime.utcnow().isoformat()))

        logger.info(f"Re-enable request {req_id} submitted by {user}")
        return req_id

    def sign_off(self, req_id: str, user: str, decision: str, comment: str):
        """
        Add a sign-off. If 2 APPROVEs, status -> APPROVED.
        """
        if decision not in ["APPROVE", "REJECT"]:
            raise ValueError("Invalid decision")

        with self.db.transaction() as conn:
            cursor = conn.execute("SELECT signoffs, status FROM manual_overrides WHERE id=?", (req_id,))
            row = cursor.fetchone()
            if not row:
                raise ValueError("Request not found")

            signoffs = json.loads(row[0])
            status = row[1]

            if status != "PENDING":
                raise ValueError(f"Request is {status}")

            # Check double signoff
            if any(s['user'] == user for s in signoffs):
                raise ValueError("User already signed off")

            signoffs.append({
                "user": user,
                "decision": decision,
                "comment": comment,
                "ts": datetime.utcnow().isoformat()
            })

            # Check Logic
            approvals = sum(1 for s in signoffs if s['decision'] == "APPROVE")
            rejects = sum(1 for s in signoffs if s['decision'] == "REJECT")

            new_status = "PENDING"
            if rejects > 0:
                new_status = "REJECTED"
            elif approvals >= 2:
                new_status = "APPROVED"

            conn.execute("UPDATE manual_overrides SET signoffs=?, status=? WHERE id=?",
                         (json.dumps(signoffs), new_status, req_id))

            logger.info(f"Request {req_id} signed off by {user}: {decision}. Status: {new_status}")
            return new_status
