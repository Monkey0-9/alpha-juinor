import logging
from typing import Dict, Any, List, Optional
import numpy as np
from contracts import AgentResult
from alpha_agents.technical import MomentumAgent, MeanReversionAgent, VolatilityAgent

logger = logging.getLogger(__name__)

class AlphaRouter:
    """
    Tiered Alpha Router.
    Blends ML and non-ML alphas based on readiness.

    Tiers:
    - Tier 0: Momentum (Technical)
    - Tier 1: Mean Reversion / Volatility (Technical)
    - Tier 2: ML Alpha (Machine Learning)
    """

    def __init__(self):
        self.t0_agent = MomentumAgent()
        self.t1_mr_agent = MeanReversionAgent()
        self.t1_vol_agent = VolatilityAgent()

    def route_signals(self,
                     symbol: str,
                     data: Any,
                     ml_ready: bool,
                     ml_result: Optional[Dict[str, Any]] = None,
                     ml_reasons: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Blends signals based on ML readiness and technical tiers.
        """
        # 1. Compute Tier-0 & Tier-1 Alphas
        t0_res = self.t0_agent.evaluate(symbol, data)
        t1_mr_res = self.t1_mr_agent.evaluate(symbol, data)
        t1_vol_res = self.t1_vol_agent.evaluate(symbol, data)

        used_tiers = ["TIER0", "TIER1"]

        # Weighted blend of non-ML alphas
        # Simple weighted average for mu
        mu_nonml = (t0_res.mu * t0_res.confidence +
                    t1_mr_res.mu * t1_mr_res.confidence +
                    t1_vol_res.mu * t1_vol_res.confidence) / \
                   (t0_res.confidence + t1_mr_res.confidence + t1_vol_res.confidence + 1e-12)

        conf_nonml = (t0_res.confidence + t1_mr_res.confidence + t1_vol_res.confidence) / 3.0

        if ml_ready and ml_result and ml_result.get('signal') is not None:
            used_tiers.append("ML")
            ml_mu = ml_result.get('signal', 0.0)
            ml_conf = ml_result.get('confidence', 0.0)

            # Blend ML with non-ML (50/50 blend for Mu)
            final_mu = 0.5 * mu_nonml + 0.5 * ml_mu
            final_conf = (conf_nonml + ml_conf) / 2.0
        else:
            final_mu = mu_nonml
            final_conf = conf_nonml

        logger.info(f"[ALPHA_ROUTER] symbol={symbol} ml_ready={ml_ready} "
                    f"used_tiers={used_tiers} final_mu={final_mu:.5f}")

        return {
            "mu": float(final_mu),
            "confidence": float(final_conf),
            "used_tiers": used_tiers,
            "ml_ready": ml_ready,
            "ml_reasons": ml_reasons or [],
            "metadata": {
                "t0_mu": t0_res.mu,
                "t1_mu": (t1_mr_res.mu + t1_vol_res.mu) / 2.0,
                "ml_mu": ml_result.get('signal') if ml_result else None
            }
        }
