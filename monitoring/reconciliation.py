# monitoring/reconciliation.py
import logging
from typing import Dict, Any, List
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class PositionReconciler:
    """
    Institutional Reconciliation Engine.
    Compares Internal Ledger (Source of Truth) vs Broker Report (Execution Reality).
    """

    def __init__(self, threshold_bps: float = 10.0):
        self.threshold = threshold_bps / 10000.0  # 10 bps = 0.1%

    def reconcile(self, internal_portfolio, broker_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        reconciles internal state with broker data.
        internal_portfolio: Portfolio object
        broker_state: {
            'cash': float,
            'positions': {ticker: quantity},
            'equity': float
        }
        """
        discrepancies = []
        
        # 1. Cash Reconciliation
        cash_drift = abs(internal_portfolio.cash - broker_state['cash']) / (broker_state['cash'] + 1e-9)
        if cash_drift > self.threshold:
            msg = f"CASH DRIFT: Internal ${internal_portfolio.cash:,.2f} vs Broker ${broker_state['cash']:,.2f} (Drift: {cash_drift:.2%})"
            discrepancies.append(msg)
            logger.error(msg)
            
        # 2. Position Reconciliation
        internal_pos = internal_portfolio.positions
        broker_pos = broker_state.get('positions', {})
        
        all_tickers = set(internal_pos.keys()) | set(broker_pos.keys())
        for tk in all_tickers:
            iq = internal_pos.get(tk, 0)
            bq = broker_pos.get(tk, 0)
            if iq != bq:
                msg = f"POSITION MISMATCH [{tk}]: Internal {iq} vs Broker {bq}"
                discrepancies.append(msg)
                logger.error(msg)
                
        # 3. Equity Reconciliation
        eq_drift = abs(internal_portfolio.total_equity - broker_state['equity']) / (broker_state['equity'] + 1e-9)
        if eq_drift > self.threshold:
            msg = f"EQUITY DRIFT: Internal ${internal_portfolio.total_equity:,.2f} vs Broker ${broker_state['equity']:,.2f} (Drift: {eq_drift:.2%})"
            discrepancies.append(msg)
            logger.warning(msg)

        status = "PASSED" if not discrepancies else "FAILED"
        
        return {
            "status": status,
            "discrepancies": discrepancies,
            "cash_drift": cash_drift,
            "equity_drift": eq_drift,
            "timestamp": pd.Timestamp.now()
        }
