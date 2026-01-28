"""
Institutional Capital Auction Engine.

Responsibilities:
1. Conduct "Capital Auction" between competing strategy requests.
2. Calculate Opportunity Cost (Mu vs Hurdle Rate).
3. Apply CVaR-adjusted Kelly sizing.
4. Enforce High-Confidence Winner logic.
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from contracts import AllocationRequest, OrderInfo
from maths.financial import fractional_kelly
from risk.cvar import compute_cvar

logger = logging.getLogger("CAPITAL_AUCTION")

class CapitalAuctionEngine:
    """
    Allocates limited capital based on the highest risk-adjusted opportunity.
    Follows institutional "High-Confidence Winner" logic.
    """

    def __init__(self, hurdle_rate: float = 0.02, total_cap_limit: float = 1.0, gamma: float = 0.5):
        self.hurdle_rate = hurdle_rate / 252.0  # Daily hurdle
        self.total_cap_limit = total_cap_limit  # Max leverage limit
        self.gamma = gamma # Kelly fractional divisor

    def auction_capital(self, requests: List[AllocationRequest]) -> Dict[str, float]:
        """
        Conduct per-symbol auction.
        Returns final weights {symbol: weight}.
        """
        if not requests:
            return {}

        results = {}
        candidates = []

        for req in requests:
            # 1. Opportunity Cost Check
            # Expected Mu must exceed Hurdle Rate
            if req.mu <= self.hurdle_rate:
                logger.debug(f"AUCTION_REJECT: {req.symbol} mu={req.mu:.6f} < hurdle={self.hurdle_rate:.6f}")
                continue

            # 2. Risk Pricing (CVaR vs Sigma)
            # If sigma is zero, use a floor to avoid div by zero
            sigma = max(req.sigma, 0.001)

            # Simple CVaR Proxy if not provided in metadata
            # In a real system, we'd use compute_cvar on history
            # For now, we use a multiplier for fat-tail handling if it's a high-vol regime
            cvar_adj = 1.2 if req.regime in ["BEAR_VOLATILE", "BULL_VOLATILE"] else 1.0
            effective_risk = sigma * cvar_adj

            # 3. Confidence-Aware Kelly
            # Kelly = Mu / (Sigma^2)
            raw_kelly = (req.mu - self.hurdle_rate) / (effective_risk ** 2)
            fractional_size = raw_kelly * self.gamma

            # 4. Confidence Scaler
            # Strict Gate: Rejection if confidence < 0.3
            if req.confidence < 0.3:
                logger.info(f"AUCTION_REJECT_LOW_CONFIDENCE: {req.symbol} conf={req.confidence:.2f}")
                continue

            final_size = fractional_size * req.confidence

            candidates.append({
                "symbol": req.symbol,
                "size": final_size,
                "confidence": req.confidence,
                "mu": req.mu
            })

        if not candidates:
            return {}

        # 5. Competition & Normalization
        # If total size exceeds limit, we scale down linearly (Pro-rata)
        total_requested = sum(c['size'] for c in candidates)

        scaler = 1.0
        if total_requested > self.total_cap_limit:
            scaler = self.total_cap_limit / total_requested
            logger.info(f"AUCTION_SCALING: Total requested {total_requested:.2f} > limit {self.total_cap_limit:.2f}. Scaling by {scaler:.4f}")

        for c in candidates:
            final_weight = c['size'] * scaler
            results[c['symbol']] = float(final_weight)
            logger.info(f"AUCTION_WINNER: {c['symbol']} mu={c['mu']:.6f} conf={c['confidence']:.2f} weight={final_weight:.4f}")

        return results
