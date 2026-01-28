#!/usr/bin/env python3
"""
Institutional Provider Confidence Agent

Responsibility:
Track and score data provider reliability based on:
1. Historical success rates.
2. Data quality scores delivered.
3. Latency and uptime (metrics).
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.manager import DatabaseManager

logging.basicConfig(level=logging.INFO, format='[PROV_CONF] %(message)s')
logger = logging.getLogger("PROV_CONFIDENCE")

class ProviderConfidenceAgent:
    """
    Intelligence layer to track provider reliability.
    """
    def __init__(self):
        self.db = DatabaseManager()

    def get_confidence_scores(self) -> Dict[str, float]:
        """Calculate confidence scores for all providers."""
        scores = {}

        # Query audit table for success/fail per provider
        query = """
        SELECT provider, status, count(*) as count
        FROM ingestion_audit
        GROUP BY provider, status
        """
        conn = self.db.get_connection()
        try:
            df = pd.read_sql_query(query, conn)
        except:
            return {"yahoo": 1.0, "alpaca": 0.5, "polygon": 0.5}

        if df.empty:
            return {"yahoo": 1.0}

        providers = df['provider'].unique()
        for p in providers:
            p_df = df[df['provider'] == p]
            success = p_df[p_df['status'] == 'SUCCESS']['count'].sum()
            total = p_df['count'].sum()

            # Simple Bayesian-like update or frequentist success rate
            rate = success / total if total > 10 else 0.5

            # Incorporate quality scores
            quality_query = f"SELECT quality_score FROM data_quality WHERE provider = '{p}' ORDER BY recorded_at DESC LIMIT 100"
            try:
                q_df = pd.read_sql_query(quality_query, conn)
                avg_q = q_df['quality_score'].mean() if not q_df.empty else 0.5
            except:
                avg_q = 0.5

            final_conf = 0.4 * rate + 0.6 * avg_q
            scores[p] = float(final_conf)

        return scores

    def rank_providers(self, asset_class: str) -> List[str]:
        """Rank providers for a specific asset class based on current confidence."""
        scores = self.get_confidence_scores()
        # Filter by capabilities (this should ideally read from PROVIDER_CAPABILITIES)
        # For now, just rank all known
        sorted_p = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [p[0] for p in sorted_p]

if __name__ == "__main__":
    pca = ProviderConfidenceAgent()
    print(json.dumps(pca.get_confidence_scores(), indent=2))
