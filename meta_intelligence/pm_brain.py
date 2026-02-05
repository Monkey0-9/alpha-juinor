
import logging
from typing import List, Dict, Any, Tuple, Optional
import math
import numpy as np
import pandas as pd
from contracts import AgentResult, AllocationRequest
from meta_intelligence.disagreement import ModelDisagreement
from meta_intelligence.opportunity_cost import OpportunityCostEngine
from meta_intelligence.bayesian_scorer import BayesianScorer
logger = logging.getLogger(__name__)

try:
    from portfolio.optimizer import optimize_portfolio
except Exception as e:
    logger.error(f"Failed to import optimize_portfolio: {e}")
    optimize_portfolio = None

class PMBrain:
    """
    The Meta-Intelligence Layer (V3 - Institutional Grade).
    Aggregates signals, applies AI Disagreement Penalty, calculates Opportunity Cost,
    computes conviction z-score using MAD, and returns AllocationRequest.
    """
    def __init__(self, config=None, risk_manager=None, portfolio_state=None):
        self.config = config or {}
        self.pm_threshold = self.config.get("pm_threshold", 0.05)
        self.conf_thresh = self.config.get("confidence_threshold", 0.5)

        # Components
        self.disagregator = ModelDisagreement(beta=5.0)
        self.opportunity_cost = OpportunityCostEngine(replacement_threshold=0.20)
        self.risk_manager = risk_manager
        self.portfolio_state = portfolio_state or {}
        self.scorer = BayesianScorer()

        # Universe scores for opportunity cost (updated externally)
        self.universe_scores: Dict[str, float] = {}

    def compute_mad(self, values: List[float]) -> float:
        """Compute Median Absolute Deviation (MAD)"""
        if not values or len(values) < 2:
            return 0.0
        median = np.median(values)
        deviations = [abs(v - median) for v in values]
        mad = np.median(deviations)
        return mad if mad > 0 else 1e-9  # Avoid division by zero

    def compute_conviction_zscore(self, mu_hat: float, all_mus: List[float]) -> float:
        """
        Compute robust conviction z-score using MAD.
        Formula: z = (mu_hat - median(mus)) / MAD(mus)
        """
        if not all_mus or len(all_mus) < 2:
            return 0.0

        median_mu = np.median(all_mus)
        mad = self.compute_mad(all_mus)

        z_score = (mu_hat - median_mu) / mad
        return z_score

    def aggregate(
        self,
        symbol: str,
        results: List[AgentResult],
        cycle_id: str = "UNKNOWN",
        regime: str = "UNCERTAIN",
        liquidity_usd: float = 1e6
    ) -> Tuple[Optional[AllocationRequest], str, List[str], Dict[str, Any]]:
        """
        Aggregate agent results and produce allocation request.

        Returns:
            (allocation_request, final_decision, reason_codes, metadata)
            allocation_request is None if decision is not EXECUTE
        """
        reasons = []
        metadata = {}

        # 1. Validate inputs
        if not results:
            return None, "REJECT", ["NO_ALPHA_SIGNALS"], metadata

        valid_results = [r for r in results if r.confidence > 0.0]
        if not valid_results:
            return None, "REJECT", ["ALL_AGENTS_ZERO_CONFIDENCE"], metadata

        # 2. Aggregation (Performance-Weighted Institutional Approach)
        mus = [r.mu for r in valid_results]
        sigmas = [r.sigma for r in valid_results]
        confidences = [r.confidence for r in valid_results]

        # Bayesian Weights
        raw_weights = [self.scorer.get_weight(r.name) for r in valid_results]
        sum_weights = sum(raw_weights)
        if sum_weights > 0:
            norm_weights = [w / sum_weights for w in raw_weights]
        else:
            norm_weights = [1.0 / len(valid_results)] * len(valid_results)

        mu_hat = sum(mu * w for mu, w in zip(mus, norm_weights))

        # Sigma aggregation (Performance-scaled)
        # Using a weighted quadratic mean
        sigma_hat = math.sqrt(sum((w * s)**2 for s, w in zip(sigmas, norm_weights)))
        avg_confidence = sum(c * w for c, w in zip(confidences, norm_weights))

        # 3. Conviction Z-Score (MAD-based)
        conviction_zscore = self.compute_conviction_zscore(mu_hat, mus)

        # 4. Risk-Adjusted Score (Sharpe-like)
        if sigma_hat <= 0:
            return None, "REJECT", ["ZERO_VOLATILITY"], metadata

        sharpe_score = mu_hat / sigma_hat

        # 5. Disagreement Penalty
        disagreement_penalty = self.disagregator.calculate_penalty(mus)

        # 6. Opportunity Cost
        oc_result = self.opportunity_cost.evaluate_position(
            symbol=symbol,
            mu=mu_hat,
            sigma=sigma_hat,
            confidence=avg_confidence,
            universe_scores=self.universe_scores,
            current_portfolio=self.portfolio_state
        )

        # 7. Final PM Score
        pm_score = sharpe_score * disagreement_penalty

        # Store metadata
        metadata = {
            "pm_score": pm_score,
            "sharpe_score": sharpe_score,
            "disagreement_penalty": disagreement_penalty,
            "opportunity_cost": oc_result,
            "conviction_zscore": conviction_zscore,
            "avg_confidence": avg_confidence,
            "num_agents": len(valid_results)
        }

        # 8. Decision Logic

        # Check confidence threshold
        if avg_confidence < self.conf_thresh:
            reasons.append(f"LOW_CONFIDENCE_{avg_confidence:.2f}")
            return None, "HOLD", reasons, metadata

        # Check opportunity cost
        if not oc_result['should_hold']:
            reasons.append(oc_result['reason'])
            return None, "REJECT", reasons, metadata

        # Check PM score threshold
        if pm_score < self.pm_threshold:
            reasons.append(f"LOW_PM_SCORE_{pm_score:.3f}")
            return None, "REJECT", reasons, metadata

        # Check conviction z-score (optional filter)
        if conviction_zscore < -2.0:  # Significantly below median
            reasons.append(f"LOW_CONVICTION_ZSCORE_{conviction_zscore:.2f}")
            return None, "REJECT", reasons, metadata

        # 9. Risk Manager Check (if available)
        if self.risk_manager:
            # Basic risk check: validate position isn't too large
            # More sophisticated checks can be added later (CVaR, sector limits, etc.)
            try:
                # Simple validation: check if mu/sigma ratio is reasonable
                if sigma_hat > 0:
                    risk_adjusted_score = mu_hat / sigma_hat
                    if risk_adjusted_score < -0.5:  # Negative expected Sharpe
                        reasons.append("RISK_REJECT_NEGATIVE_SHARPE")
                        return None, "REJECT", reasons, metadata
            except Exception as e:
                logger.warning(f"Risk manager check failed for {symbol}: {e}")
                # Don't block on risk check failure
                pass

        # 10. Passed all checks → Create AllocationRequest
        reasons.append("PM_SCORE_PASS")
        reasons.append(f"CONVICTION_ZSCORE_{conviction_zscore:.2f}")

        allocation_request = AllocationRequest(
            symbol=symbol,
            mu=mu_hat,
            sigma=sigma_hat,
            confidence=avg_confidence,
            liquidity=liquidity_usd,
            regime=regime,
            timestamp=str(pd.Timestamp.utcnow()),
            metadata=metadata
        )

        return allocation_request, "EXECUTE", reasons, metadata

        return allocation_request, "EXECUTE", reasons, metadata

    def update_universe_scores(self, scores: Dict[str, float]):
        """Update universe scores for opportunity cost calculation"""
        self.universe_scores = scores

    def optimize_cycle(
        self,
        candidates: List[Any], # List[DecisionRecord]
        w_prev: Optional[np.ndarray] = None,
        config: Optional[Dict[str, Any]] = None,
        historical_returns: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Run global portfolio optimization on a set of candidates.
        """
        if optimize_portfolio is None:
            logger.error("Optimizer not available (ImportError). Returning empty.")
            return {"w": np.array([]), "rejected_assets": [], "error": "Optimizer Import Error"}

        if not candidates:
            return {"w": np.array([]), "rejected_assets": [], "explain": {}}

        n = len(candidates)
        symbols = [c.symbol for c in candidates]

        # 1. Extract Vectors
        mu = np.array([c.mu for c in candidates])
        sigmas = np.array([c.sigma if c.sigma > 0 else 0.01 for c in candidates])

        # 2. Construct Covariance
        # Default: Diagonal Robust Assumption
        Sigma = np.diag(sigmas**2)

        # Upgrade: Ledoit-Wolf Shrinkage (Institutional Standard)
        if historical_returns is not None and not historical_returns.empty:
            try:
                from sklearn.covariance import LedoitWolf
                # Ensure we only use returns for the candidates
                # (handled by caller, but we double check)
                common_cols = [s for s in symbols if s in historical_returns.columns]
                if len(common_cols) == n:
                    lw = LedoitWolf().fit(historical_returns[symbols])
                    Sigma_lw = lw.covariance_

                    # Hybrid: Scale LW correlation by active sigma forecasts
                    # D = diag(sigmas)
                    # Sigma = D * Corr(Sigma_lw) * D
                    diag_lw = np.sqrt(np.diag(Sigma_lw))
                    # Avoid division by zero
                    diag_lw[diag_lw == 0] = 1e-9
                    Corr_lw = Sigma_lw / np.outer(diag_lw, diag_lw)
                    Sigma = np.diag(sigmas) @ Corr_lw @ np.diag(sigmas)
                    logger.info("PMBrain: Used Ledoit-Wolf Shrinkage for covariance")
                else:
                    logger.warning(f"PMBrain: Returns panel missing symbols ({len(common_cols)}/{n}). Using diagonal.")
            except Exception as e:
                logger.error(f"PMBrain: Ledoit-Wolf failed: {e}. Falling back to diagonal.")

        # 3. Optimization Params
        opt_config = config or {}
        params = {
            "lambda": opt_config.get("risk_aversion", 2.0),
            "gamma": opt_config.get("cvar_weight", 5.0),
            "alpha": 0.95,
            "kappa": opt_config.get("turnover_penalty", 0.01),
            "eta": opt_config.get("impact_penalty", 1e-4),
            "uncertainty_radius": opt_config.get("uncertainty_radius", 0.0),
            "data_confidence": np.ones(n), # Start simple, can extract from metadata if needed
            "regime_compatibility": np.ones(n)
        }

        # Extract metadata scalars if available
        # Example: if c.data_providers has 'confidence' map it
        # for i, c in enumerate(candidates):
        #    params["data_confidence"][i] = c.data_providers.get("confidence", 1.0)

        # 4. Generate Synthetic Scenarios for CVaR (Path Integral Proxy)
        # Sample N=1000 from Multivariate Normal(mu, Sigma)
        # Seeded for determinism
        rng_seed = opt_config.get("seed", 42)
        np.random.seed(rng_seed)
        N_s = 1000
        # scenarios = mu + Z * sigma
        Z = np.random.randn(N_s, n)
        scenario_returns = mu + Z @ np.diag(sigmas)

        # 5. Bounds & Init
        w_min = np.zeros(n)
        w_max = np.ones(n) * opt_config.get("max_position_size", 0.10)

        if w_prev is None or len(w_prev) != n:
             w_prev = np.zeros(n) # Assume full cash start if no prev state

        # 6. Run Optimizer
        try:
            result = optimize_portfolio(
                mu=mu,
                Sigma=Sigma,
                scenario_returns=scenario_returns,
                w_prev=w_prev,
                w_min=w_min,
                w_max=w_max,
                sector_map=None, # Add if sector map is available
                params=params,
                rng_seed=rng_seed
            )

            # Decorate rejected assets with symbols
            rejects = []
            for r in result["rejected_assets"]:
                idx = r["asset_index"]
                rejects.append({
                    "symbol": symbols[idx],
                    "reason": r["reason"],
                    "idx": idx
                })
            result["rejected_assets_named"] = rejects
            result["symbols"] = symbols

            return result

        except Exception as e:
            logger.error(f"Optimization FAILED: {e}")
            # Fallback: Equal Weight or Cash
            return {
                "w": np.zeros(n),
                "rejected_assets": [],
                "error": str(e)
            }
